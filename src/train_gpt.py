"""Шаг 1: базовая модель GPT-2 на next item prediction (токены — id треков).

Обучение только на прослушиваниях до t_valid. Окна по 201 токену, лосс — кросс-энтропия по всем
позициям, кроме тех, где следующий трек вне словаря. Ранняя остановка по NDCG@10 на valid.
Для сравнения считаются две эвристики: популярное в train и «самое частое в истории юзера».

    python src/train_gpt.py
"""
import json
import math
import time

import numpy as np
import torch

from common import CTX, DEV, OOV, PAD, RESULTS, Seqs, limit_gpu_memory, peak_gb
from lib import behaviour, build_gpt, contexts, evaluate, next_item_loss, rank_metrics

OUT = RESULTS / "base"
BS, LR, EPOCHS, WARMUP, PATIENCE = 32, 1e-3, 8, 500, 2
EVAL_EVERY = 0.5            # доля эпохи


def train_windows(S):
    """(start, end) окон по 201 токену из train-части каждого юзера, с конца истории назад."""
    starts, ends = [], []
    for u in range(S.n_users):
        a, b = S.user_start[u], S.user_start[u + 1]
        bt = a + np.searchsorted(S.ts[a:b], S.t_valid)
        for e in range(bt, a + 1, -CTX):
            starts.append(max(a, e - CTX - 1))
            ends.append(e)
    return np.array(starts), np.array(ends)


def batch(S, starts, ends):
    L = CTX + 1
    idx = ends[:, None] - L + np.arange(L)[None, :]
    ok = idx >= starts[:, None]
    x = np.where(ok, S.tok[np.clip(idx, 0, None)], PAD).astype(np.int64)
    x = torch.from_numpy(x)
    m = (x != PAD).long()
    pos = (m.cumsum(1) - 1).clamp(min=0)
    return x, m, pos


def heuristics(S, split):
    pos, fut = getattr(S, f"{split}_pos"), getattr(S, f"{split}_future")
    rows, _ = contexts(S, pos)
    pop = np.argsort(-S.tok_cnt)[:10]
    out = {}
    tops = {"популярное": np.tile(pop, (len(pos), 1))}
    hist = np.zeros((len(pos), 10), np.int64)
    for i, r in enumerate(rows):
        r = r[r >= 2]
        u, inv = np.unique(r[::-1], return_inverse=True)       # ничьи — в пользу недавних
        c = np.bincount(inv)
        first = np.full(len(u), len(r))
        np.minimum.at(first, inv, np.arange(len(r)))
        best = u[np.lexsort((first, -c))][:10]
        fill = [p for p in pop if p not in best][:10 - len(best)]
        hist[i] = np.concatenate([best, fill])
    tops["самое частое в истории"] = hist
    for name, top in tops.items():
        ndcg, hr = rank_metrics(top, fut[:, 0])
        out[name] = {"NDCG@10": float(ndcg.mean()), "Recall@10": float(hr.mean()), **behaviour(S, top, rows)}
    return out


def main():
    limit_gpu_memory()
    OUT.mkdir(parents=True, exist_ok=True)
    S = Seqs()
    starts, ends = train_windows(S)
    print(f"окон train: {len(starts)}, токенов {int((ends - starts).sum())}", flush=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build_gpt().to(DEV)
    print(f"параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} млн", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_ep = len(starts) // BS
    total = steps_ep * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    log, best, bad, step = [], -1, 0, 0
    every = int(steps_ep * EVAL_EVERY)
    t0, losses = time.time(), []
    for ep in range(EPOCHS):
        perm = rng.permutation(len(starts))
        for i in range(steps_ep):
            model.train()
            b = perm[i * BS:(i + 1) * BS]
            x, m, p = batch(S, starts[b], ends[b])
            x, m, p = x.to(DEV), m.to(DEV), p.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = next_item_loss(model, x, m, p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % every == 0:
                res, _, _ = evaluate(model, S, "valid")
                rec = {"step": step, "epoch": step / steps_ep, "loss": float(np.mean(losses)), **res}
                log.append(rec)
                print(f"  шаг {step} (эпоха {step / steps_ep:.1f}): loss {np.mean(losses):.3f}, valid NDCG@10 "
                      f"{res['NDCG@10']:.4f}, Recall@10 {res['Recall@10']:.4f} [{time.time() - t0:.0f} c, "
                      f"пик {peak_gb():.1f} ГБ]", flush=True)
                losses = []
                if res["NDCG@10"] > best:
                    best, bad = res["NDCG@10"], 0
                    torch.save(model.state_dict(), OUT / "model.pt")
                else:
                    bad += 1
                json.dump(log, open(OUT / "train_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                if bad >= PATIENCE * 2:
                    break
        if bad >= PATIENCE * 2:
            print("  ранняя остановка", flush=True)
            break

    model.load_state_dict(torch.load(OUT / "model.pt"))
    summary = {"heuristics_valid": heuristics(S, "valid"), "heuristics_test": heuristics(S, "test")}
    for split in ["valid", "test"]:
        res, top, ndcg = evaluate(model, S, split)
        summary[f"gpt_{split}"] = res
        np.save(OUT / f"{split}_ndcg.npy", ndcg)
        np.save(OUT / f"{split}_top.npy", top)
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in summary.items():
        print(k, json.dumps(v, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
