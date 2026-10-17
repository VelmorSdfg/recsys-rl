"""Шаг 3: RL-дообучение GPT-2 методом GRPO с KL-штрафом к исходной модели.

Для каждого контекста (история юзера из train-периода) политика сэмплирует G треков, reward model A
оценивает каждый, преимущество = (r − среднее по группе) / std по группе, и логарифм вероятности
трека умножается на это преимущество (как в GRPO, без отдельного критика). Штраф β·KL(π || π_ref)
считается точно по всему словарю: действие — один трек, распределение известно целиком.

При --horizon H > 1 политика генерирует H треков подряд, каждый раз дописывая сгенерированный трек
в контекст. Доля дослушивания для дописанного трека в контексте reward model берётся из её же
предсказания (юзер на самом деле ничего не слушал — это и есть «модельный» rollout).
Награда последовательности — сумма наград шагов.

Варианты награды (--reward):
    ratio        ожидаемая доля дослушивания
    time         ожидаемое время прослушивания: ratio × длина трека / средняя длина
    like         вероятность лайка
    ratio_like   ratio + 20·like (20 ≈ отношение средних, вклад сопоставимый)
    cont         вероятность, что юзер продолжит слушать (прокси удержания)
    ratio_norep  ratio − 0.5 за трек, который уже есть в истории юзера
    ratio_ens    среднее ratio по reward models A и B (ансамбль против эксплуатации ошибок одной модели)

    python src/rl_grpo.py --reward ratio --beta 0.05
"""
import argparse
import json
import time

import numpy as np
import torch

from common import CTX, DEV, OOV, PAD, RESULTS, Seqs, limit_gpu_memory, peak_gb
from lib import build_gpt, contexts, evaluate, left_pad, last_logits
from rm import Scorer, append_generated, ctx_batch, load_rm

OUT = RESULTS / "rl"
LIKE_W = 20.0


def reward_fn(name, probs, cand, hist_tok, tok_len, mean_len, probs_b=None):
    """probs [B, G, 3] (ratio, like, cont) -> награда [B, G]."""
    ratio, like, cont = probs.unbind(-1)
    if name == "ratio":
        return ratio
    if name == "time":
        return ratio * tok_len[cand] / mean_len
    if name == "like":
        return like
    if name == "ratio_like":
        return ratio + LIKE_W * like
    if name == "cont":
        return cont
    if name == "ratio_norep":
        rep = (hist_tok[:, None, :] == cand[:, :, None]).any(-1).float()
        return ratio - 0.5 * rep
    if name == "ratio_ens":
        return (ratio + probs_b[..., 0]) / 2
    raise ValueError(name)


def train_positions(S, n, seed):
    idx = np.flatnonzero(S.ts < S.t_valid)
    u = np.searchsorted(S.user_start, idx, side="right") - 1
    idx = idx[idx - S.user_start[u] >= 10]
    return np.random.default_rng(seed).choice(idx, n, replace=False)


def rm_extra(S, pos, scorers, tok_len, mean_len):
    """Для evaluate(): средний предсказанный отклик на топ-10 по reward models A и B."""
    def f(i, lg, rows):
        p = pos[i:i + len(lg)]
        top = torch.topk(lg, 10).indices
        out = {}
        for name, sc in scorers.items():
            h = sc.hidden(*ctx_batch(S, p))
            pr = sc(h, top)
            out[f"{name}: дослушивание"] = pr[..., 0].mean(1).sum().item()
            out[f"{name}: лайк"] = pr[..., 1].mean(1).sum().item()
            out[f"{name}: продолжение"] = pr[..., 2].mean(1).sum().item()
            out[f"{name}: время, с"] = (pr[..., 0] * tok_len[top]).mean(1).sum().item()
        return out
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="ratio")
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adv", choices=["group_std", "fixed"], default="group_std",
                    help="group_std — как в GRPO: (r − среднее) / std группы; fixed — (r − среднее) / σ0, σ0 = std "
                         "наград на первом батче (как Dr. GRPO): масштаб не растёт, когда политика сужается")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    limit_gpu_memory()
    tag = args.tag or f"{args.reward}_b{args.beta:g}" + (f"_h{args.horizon}" if args.horizon > 1 else "")         + ("_fixed" if args.adv == "fixed" else "")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    S = Seqs()
    torch.manual_seed(args.seed)
    pol, ref = build_gpt().to(DEV), build_gpt().to(DEV)
    sd = torch.load(RESULTS / "base" / "model.pt")
    pol.load_state_dict(sd)
    ref.load_state_dict(sd)
    ref.eval().requires_grad_(False)
    if args.horizon > 1:          # H проходов по 200 токенов с градиентом: активации пересчитываем, а не храним
        pol.gradient_checkpointing_enable()
        pol.config.use_cache = False
    tok_len = torch.as_tensor(S.tok_len, device=DEV)
    mean_len = float(np.average(S.tok_len[2:], weights=S.tok_cnt[2:]))
    rmA, rmB = load_rm(RESULTS / "reward" / "rm_A.pt"), load_rm(RESULTS / "reward" / "rm_B.pt")
    scA, scB = Scorer(rmA, S.tok_len), Scorer(rmB, S.tok_len)
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=0.0)
    tpos = train_positions(S, args.steps * args.bs, args.seed)
    extra_v = rm_extra(S, S.valid_pos, {"A": scA, "B": scB}, tok_len, mean_len)
    log, t0 = [], time.time()

    def do_eval(step):
        res, _, _ = evaluate(pol, S, "valid", extra=extra_v)
        rec = {"step": step, **res}
        log.append(rec)
        print(f"  шаг {step}: valid NDCG@10 {res['NDCG@10']:.4f}, дослушивание A/B {res['A: дослушивание']:.3f}/"
              f"{res['B: дослушивание']:.3f}, длина {res['длина трека, с']:.0f} c, повторы "
              f"{res['доля повторов из истории']:.2f}, покрытие {res['покрытие каталога']:.3f} "
              f"[{time.time() - t0:.0f} c, пик {peak_gb():.1f} ГБ]", flush=True)
        json.dump({"args": vars(args), "log": log}, open(out / "log.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    do_eval(0)
    hist = []
    for step in range(1, args.steps + 1):
        pol.train()
        p = tpos[(step - 1) * args.bs: step * args.bs]
        rows, _ = contexts(S, p)
        x, m, pp = (t.to(DEV) for t in left_pad(rows))
        rtok, rrb, rm_, rpos = (t.to(DEV) for t in ctx_batch(S, p))
        B, G = len(p), args.G
        if args.horizon == 1:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = last_logits(pol, x, m, pp)
                with torch.no_grad():
                    lr_ = last_logits(ref, x, m, pp)
            lp, lpr = torch.log_softmax(lg, -1), torch.log_softmax(lr_, -1)
            a = torch.multinomial(lp.exp().detach(), G, replacement=True)            # [B, G]
            logp = lp.gather(1, a)
            kl = (lp.exp() * (lp - lpr).nan_to_num(0)).sum(-1).mean()
            hA = scA.hidden(rtok, rrb, rm_, rpos)
            prA = scA(hA, a)
            prB = scB(scB.hidden(rtok, rrb, rm_, rpos), a) if args.reward == "ratio_ens" else None
            r = reward_fn(args.reward, prA, a, x, tok_len, mean_len, prB)
        else:
            # rollout на H шагов: B·G независимых продолжений
            rep = lambda t: t.repeat_interleave(G, 0)
            x, m, pp = rep(x), rep(m), rep(pp)
            rtok, rrb, rm_, rpos = rep(rtok), rep(rrb), rep(rm_), rep(rpos)
            hist_tok = x.clone()
            logp, kl, r = 0, 0, 0
            for t in range(args.horizon):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = last_logits(pol, x, m, pp)
                    with torch.no_grad():
                        lr_ = last_logits(ref, x, m, pp)
                lp, lpr = torch.log_softmax(lg, -1), torch.log_softmax(lr_, -1)
                a = torch.multinomial(lp.exp().detach(), 1)[:, 0]
                logp = logp + lp.gather(1, a[:, None])[:, 0]
                kl = kl + (lp.exp() * (lp - lpr).nan_to_num(0)).sum(-1).mean()
                pr = scA(scA.hidden(rtok, rrb, rm_, rpos), a[:, None])
                prB = scB(scB.hidden(rtok, rrb, rm_, rpos), a[:, None]) if args.reward == "ratio_ens" else None
                r = r + reward_fn(args.reward, pr, a[:, None], hist_tok, tok_len, mean_len, prB)[:, 0]
                x, m, pp = _append_tok(x, a)
                rtok, rrb, rm_, rpos = append_generated(rtok, rrb, rm_, rpos, a, pr[:, 0, 0])
            logp, r = logp.view(B, G), r.view(B, G)
        if args.adv == "group_std":
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
        else:
            if step == 1:
                sigma0 = r.std().item() + 1e-6
            adv = (r - r.mean(1, keepdim=True)) / sigma0
        loss = -(adv.detach() * logp).mean() + args.beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        hist.append((r.mean().item(), kl.item()))
        if step % 100 == 0:
            rr, kk = np.mean(hist, 0)
            print(f"    шаг {step}: награда сэмплов {rr:.4f}, KL {kk:.4f}", flush=True)
            log.append({"step": step, "train_reward": float(rr), "kl": float(kk)})
            hist = []
        if step % args.eval_every == 0:
            do_eval(step)

    torch.save(pol.state_dict(), out / "model.pt")
    final = {"args": vars(args)}
    for split in ["valid", "test"]:
        extra = rm_extra(S, getattr(S, f"{split}_pos"), {"A": scA, "B": scB}, tok_len, mean_len)
        res, top, ndcg = evaluate(pol, S, split, extra=extra)
        final[split] = res
        np.save(out / f"{split}_top.npy", top)
        np.save(out / f"{split}_ndcg.npy", ndcg)
    json.dump(final, open(out / "final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("test:", json.dumps(final["test"], ensure_ascii=False))


def _append_tok(x, a):
    x = torch.cat([x[:, 1:], a[:, None]], 1)
    m = (x != PAD).long()
    return x, m, (m.cumsum(1) - 1).clamp(min=0)


if __name__ == "__main__":
    main()
