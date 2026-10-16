"""Шаг 2: reward model и её оценка на отложенных по времени прослушиваниях.

Обучение — только на прослушиваниях до t_valid (те же окна, что у GPT-2). Оценка — на 200 тыс.
случайных прослушиваний valid и test: для каждого модель видит 200 предыдущих прослушиваний юзера
(с их долями дослушивания) и предсказывает отклик на реальный трек. Сравнение с эвристиками:
среднее по всем, по треку, по юзеру и «юзер + трек».

Две модели с разной архитектурой и сидом:
    A — «рабочая», по ней учится RL;
    B — «независимый судья»: если RL поднимает награду по A, но не по B, политика эксплуатирует ошибки A.

    python src/train_reward.py --name A
    python src/train_reward.py --name B --d 192 --layers 3 --seed 1
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from common import CTX, DEV, PAD, RESULTS, Seqs, limit_gpu_memory, peak_gb
from rm import HEADS, RewardModel, ctx_batch
from train_gpt import train_windows

OUT = RESULTS / "reward"
BS, LR, WARMUP = 32, 1e-3, 500
N_EVAL = 200_000


def window_batch(S, starts, ends):
    L = CTX + 1
    idx = ends[:, None] - L + np.arange(L)[None, :]
    ok = idx >= starts[:, None]
    ci = np.clip(idx, 0, None)
    g = lambda a, fill=0: torch.from_numpy(np.where(ok, a[ci], fill))
    tok = g(S.tok, PAD).long()
    rb = torch.from_numpy(np.where(ok, S.ratio_bin(ci), 0)).long()
    return dict(tok=tok, rb=rb, m=(tok != PAD).long(), ratio=g(S.ratio.astype(np.float32)), like=g(S.like).float(),
                cont=g(S.cont, -1).float(), org=g(S.organic).long(), length=g(S.length).long())


def loss_fn(rm, b):
    b = {k: v.to(DEV) for k, v in b.items()}
    m = b["m"]
    pos = (m.cumsum(1) - 1).clamp(min=0)
    h = rm.encode(b["tok"][:, :-1], b["rb"][:, :-1], m[:, :-1], pos[:, :-1])
    ok = (m[:, :-1] == 1) & (m[:, 1:] == 1)
    lg = rm.score(h[ok], b["tok"][:, 1:][ok], b["length"][:, 1:][ok], b["org"][:, 1:][ok]).float()
    l_ratio = F.binary_cross_entropy_with_logits(lg[:, 0], b["ratio"][:, 1:][ok])
    l_like = F.binary_cross_entropy_with_logits(lg[:, 1], b["like"][:, 1:][ok])
    c = b["cont"][:, 1:][ok]
    l_cont = F.binary_cross_entropy_with_logits(lg[:, 2][c >= 0], c[c >= 0])
    return l_ratio + l_like + l_cont, (l_ratio.item(), l_like.item(), l_cont.item())


def eval_positions(S, lo, hi, n, seed):
    idx = np.flatnonzero((S.ts >= lo) & (S.ts < hi))
    u = np.searchsorted(S.user_start, idx, side="right") - 1
    idx = idx[idx - S.user_start[u] >= 10]
    return np.sort(np.random.default_rng(seed).choice(idx, min(n, len(idx)), replace=False))


@torch.no_grad()
def predict(rm, S, pos, bs=512):
    rm.eval()
    out = []
    for i in range(0, len(pos), bs):
        p = pos[i:i + bs]
        tok, rb, m, pp = (t.to(DEV) for t in ctx_batch(S, p))
        cand = torch.from_numpy(S.tok[p].astype(np.int64)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = rm.encode(tok, rb, m, pp)[:, -1]
            lg = rm.score(h, cand, torch.from_numpy(S.length[p].astype(np.int64)).to(DEV),
                          torch.from_numpy(S.organic[p].astype(np.int64)).to(DEV))
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)


def baselines(S, pos):
    """Эвристики по train: среднее, по треку, по юзеру, юзер + трек (сглаживание к общему среднему)."""
    tr = S.ts < S.t_valid
    u_all = np.searchsorted(S.user_start, np.arange(len(S.ts)), side="right") - 1
    u = np.searchsorted(S.user_start, pos, side="right") - 1
    out = {}
    for h, y in [("ratio", S.ratio.astype(np.float64)), ("like", S.like.astype(np.float64)),
                 ("cont", S.cont.astype(np.float64))]:
        ok = tr & (y >= 0)
        g = y[ok].mean()
        it_s = np.bincount(S.tok[ok], y[ok], minlength=len(S.tok_len))
        it_n = np.bincount(S.tok[ok], minlength=len(S.tok_len))
        us_s = np.bincount(u_all[ok], y[ok], minlength=S.n_users)
        us_n = np.bincount(u_all[ok], minlength=S.n_users)
        item = (it_s + 20 * g) / (it_n + 20)
        user = (us_s + 20 * g) / (us_n + 20)
        it = item[S.tok[pos]]
        it = np.where(S.tok[pos] == 1, g, it)           # вне словаря — общее среднее
        out[h] = {"среднее": np.full(len(pos), g), "по треку": it, "по юзеру": user[u],
                  "юзер + трек": np.clip(user[u] + it - g, 1e-4, 1 - 1e-4)}
    return out


def metrics(S, pos, pred_by_head):
    y = {"ratio": S.ratio[pos].astype(np.float64), "like": S.like[pos].astype(float), "cont": S.cont[pos].astype(float)}
    res = {}
    for h, p in pred_by_head.items():
        t = y[h]
        ok = t >= 0
        t, p = t[ok], np.asarray(p)[ok]
        if h == "ratio":
            full = (t >= 0.9).astype(int)
            res[h] = {"MAE": float(np.abs(p - t).mean()), "RMSE": float(np.sqrt(((p - t) ** 2).mean())),
                      "Spearman": float(spearmanr(p, t).correlation), "AUC дослушал ≥90%": float(roc_auc_score(full, p))}
        else:
            ll = -np.mean(t * np.log(np.clip(p, 1e-6, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-6, 1)))
            res[h] = {"AUC": float(roc_auc_score(t, p)), "LogLoss": float(ll), "доля позитивов": float(t.mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="A")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2)
    args = ap.parse_args()
    limit_gpu_memory()
    OUT.mkdir(parents=True, exist_ok=True)
    S = Seqs()
    starts, ends = train_windows(S)
    vpos = eval_positions(S, S.t_valid, S.t_test, N_EVAL, 1)
    tpos = eval_positions(S, S.t_test, S.ts.max() + 1, N_EVAL, 2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rm = RewardModel(d=args.d, n_layer=args.layers).to(DEV)
    print(f"RM {args.name}: {sum(p.numel() for p in rm.parameters()) / 1e6:.1f} млн параметров", flush=True)
    opt = torch.optim.AdamW(rm.parameters(), lr=LR, weight_decay=0.01)
    steps_ep = len(starts) // BS
    total = int(steps_ep * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    best, log, parts, t0 = 1e9, [], [], time.time()
    perm = np.concatenate([rng.permutation(len(starts)) for _ in range(math.ceil(args.epochs))])
    every = steps_ep // 2
    vsub = vpos[:50_000]
    for step in range(1, total + 1):
        rm.train()
        b = perm[(step - 1) * BS: step * BS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, part = loss_fn(rm, window_batch(S, starts[b], ends[b]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        sched.step()
        parts.append(part)
        if step % every == 0 or step == total:
            p = predict(rm, S, vsub)
            y = np.stack([S.ratio[vsub].astype(np.float32), S.like[vsub], S.cont[vsub]], 1).astype(np.float32)
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            bce = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
            vl = float(bce[:, 0].mean() + bce[:, 1].mean() + bce[:, 2][y[:, 2] >= 0].mean())
            log.append({"step": step, "train": np.mean(parts, 0).tolist(), "valid_loss": vl})
            print(f"  шаг {step}/{total}: train ratio/like/cont {np.round(np.mean(parts, 0), 4).tolist()}, "
                  f"valid loss {vl:.4f} [{time.time() - t0:.0f} c, пик {peak_gb():.1f} ГБ]", flush=True)
            parts = []
            if vl < best:
                best = vl
                torch.save({"state": rm.state_dict(), "cfg": rm.cfg}, OUT / f"rm_{args.name}.pt")
    rm.load_state_dict(torch.load(OUT / f"rm_{args.name}.pt", weights_only=False)["state"])
    summary = {"args": vars(args), "log": log}
    for split, pos in [("valid", vpos), ("test", tpos)]:
        p = predict(rm, S, pos)
        res = {"RM": metrics(S, pos, {h: p[:, i] for i, h in enumerate(HEADS)})}
        bl = baselines(S, pos)
        for name in ["среднее", "по треку", "по юзеру", "юзер + трек"]:
            res[name] = metrics(S, pos, {h: bl[h][name] for h in HEADS})
        org = S.organic[pos] == 0
        res["RM, только рекомендованные (is_organic=0)"] = metrics(S, pos[org], {h: p[org, i] for i, h in enumerate(HEADS)})
        summary[split] = res
        np.save(OUT / f"rm_{args.name}_{split}_pred.npy", p)
        if split == "test":
            np.save(OUT / "test_eval_pos.npy", pos)
    json.dump(summary, open(OUT / f"rm_{args.name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for split in ["valid", "test"]:
        print(split)
        for k, v in summary[split].items():
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
