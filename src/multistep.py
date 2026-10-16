"""Генерация на несколько шагов вперёд: что происходит, когда рекомендации дописываются в контекст.

Для каждой тестовой точки два режима:
    generated  политика жадно выбирает трек, он дописывается в контекст (как будто юзер его послушал),
               reward model получает предсказанную ею же долю дослушивания;
    real       в контекст дописывается трек, который юзер реально послушал (teacher forcing).
На шаге k измеряем: Recall@10 / NDCG@10 для реального k-го следующего трека, предсказанный отклик
reward models A и B на выбранный трек и их расхождение |A − B|, энтропию политики,
долю повторов (выбранный трек уже был в контексте) и «залипание» (тот же трек, что на прошлом шаге).

    python src/multistep.py --models base rl/ratio_b0.05
"""
import argparse
import json

import numpy as np
import torch

from common import DEV, OOV, RESULTS, Seqs, limit_gpu_memory
from lib import build_gpt, contexts, last_logits, left_pad
from rm import Scorer, append_generated, ctx_batch, load_rm

STEPS, N_POINTS, BS = 5, 6000, 250


def _append(x, a):
    x = torch.cat([x[:, 1:], a[:, None]], 1)
    m = (x != 0).long()
    return x, m, (m.cumsum(1) - 1).clamp(min=0)


@torch.no_grad()
def run(model, S, pos, fut, scA, scB, mode):
    rows, _ = contexts(S, pos)
    stats = {k: np.zeros(STEPS) for k in ["Recall@10", "NDCG@10", "A: дослушивание", "B: дослушивание",
                                           "|A−B|", "энтропия", "повтор из контекста", "тот же трек, что шаг назад"]}
    n_ok = np.zeros(STEPS)
    for i in range(0, len(pos), BS):
        p = pos[i:i + BS]
        f = torch.from_numpy(fut[i:i + BS].astype(np.int64)).to(DEV)
        x, m, pp = (t.to(DEV) for t in left_pad(rows[i:i + BS]))
        ra = [t.to(DEV) for t in ctx_batch(S, p)]
        rb = [t.clone() for t in ra]
        prev = torch.full((len(p),), -1, device=DEV)
        for k in range(STEPS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = last_logits(model, x, m, pp)
            lp = torch.log_softmax(lg, -1)
            top = torch.topk(lg, 10).indices
            a = top[:, 0]
            tgt = f[:, k]
            ok = (tgt >= 2)
            hit = (top == tgt[:, None]) & ok[:, None]
            r = hit.float().argmax(1)
            stats["Recall@10"][k] += hit.any(1).float().sum().item()
            stats["NDCG@10"][k] += (hit.any(1).float() / torch.log2(r.float() + 2)).sum().item()
            n_ok[k] += (tgt != -1).sum().item()
            pa = scA(scA.hidden(*ra), a[:, None])[:, 0]
            pb = scB(scB.hidden(*rb), a[:, None])[:, 0]
            stats["A: дослушивание"][k] += pa[:, 0].sum().item()
            stats["B: дослушивание"][k] += pb[:, 0].sum().item()
            stats["|A−B|"][k] += (pa[:, 0] - pb[:, 0]).abs().sum().item()
            stats["энтропия"][k] += (-(lp.exp() * lp.nan_to_num(neginf=0)).sum(-1)).sum().item()
            stats["повтор из контекста"][k] += (x == a[:, None]).any(1).float().sum().item()
            stats["тот же трек, что шаг назад"][k] += (a == prev).float().sum().item()
            prev = a
            if mode == "generated":
                nxt, ra_r, rb_r = a, pa[:, 0], pb[:, 0]
            else:                                   # реальный следующий трек и его реальное дослушивание
                nxt = torch.where(tgt >= 1, tgt, torch.full_like(tgt, OOV))
                idx = torch.from_numpy(np.minimum(p + k, len(S.ratio) - 1)).to(DEV)
                real_r = torch.from_numpy(S.ratio[idx.cpu().numpy()].astype(np.float32)).to(DEV)
                ra_r = rb_r = real_r
            x, m, pp = _append(x, nxt)
            ra = list(append_generated(*ra, nxt, ra_r))
            rb = list(append_generated(*rb, nxt, rb_r))
    n = len(pos)
    out = {}
    for k, v in stats.items():
        den = n_ok if k in ("Recall@10", "NDCG@10") else np.full(STEPS, n)
        out[k] = (v / den).round(4).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base"])
    args = ap.parse_args()
    limit_gpu_memory()
    S = Seqs()
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(S.test_pos), min(N_POINTS, len(S.test_pos)), replace=False))
    pos, fut = S.test_pos[sel], S.test_future[sel]
    scA = Scorer(load_rm(RESULTS / "reward" / "rm_A.pt"), S.tok_len)
    scB = Scorer(load_rm(RESULTS / "reward" / "rm_B.pt"), S.tok_len)
    res = {}
    for name in args.models:
        model = build_gpt().to(DEV)
        model.load_state_dict(torch.load(RESULTS / name / "model.pt"))
        model.eval()
        for mode in ["generated", "real"]:
            res[f"{name} | {mode}"] = r = run(model, S, pos, fut, scA, scB, mode)
            print(name, mode, json.dumps(r, ensure_ascii=False), flush=True)
    (RESULTS / "multistep").mkdir(exist_ok=True)
    json.dump(res, open(RESULTS / "multistep" / "multistep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
