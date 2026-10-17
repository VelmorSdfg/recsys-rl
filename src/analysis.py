"""Шаг 4: сводный анализ.

1. Аналитический оптимум. Для одного шага задача 'max E_π[r] - β*KL(π || π_ref)' решается в
   закрытом виде: π*(a) ~ π_ref(a)*exp(r(a)/β). Значит, оптимальную RL-политику можно получить
   без обучения - переранжировать кандидатов исходной модели по log π_ref + r/β. Берём топ-500 исходной
   модели (за ними почти нулевая вероятность) и проходим по сетке β: получаем полную кривую
   компромисса между релевантностью и наградой для каждой функции награды и опорную точку для GRPO.
2. Таблица всех RL-прогонов с парными бутстрап-интервалами NDCG@10 к исходной модели.
3. Проверка на реальных откликах: среди рекомендованных треков, которые юзер реально послушал в
   следующих 10 прослушиваниях, - реальная доля дослушивания и доля лайков. Это единственная метрика
   награды, которая не зависит от reward model.

    python src/analysis.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DEV, RESULTS, Seqs, limit_gpu_memory
from lib import behaviour, bootstrap_diff, build_gpt, contexts, policy_logits, rank_metrics
from rm import Scorer, ctx_batch, load_rm
from rl_grpo import reward_fn

OUT = RESULTS / "analysis"
TOPN = 500
BETAS = [None, 3, 1, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0]
REWARDS = ["ratio", "time", "like", "ratio_like", "cont", "ratio_norep"]


def real_outcomes(S, pos, top):
    """Реальный отклик на рекомендованные треки, которые юзер действительно послушал в ближайших 10."""
    rat, lik, n_hit = [], [], 0
    for i, p in enumerate(pos):
        u_end = S.user_start[np.searchsorted(S.user_start, p, side="right")]
        j = np.arange(p, min(p + 10, u_end))
        sel = j[np.isin(S.tok[j], top[i]) & (S.tok[j] >= 2)]
        rat.extend(S.ratio[sel].astype(float))
        lik.extend(S.like[sel].astype(float))
        n_hit += len(sel) > 0
    return {"реально послушано рекомендованных, на точку": len(rat) / len(pos),
            "их реальная доля дослушивания": float(np.mean(rat)) if rat else float("nan"),
            "их реальная доля лайков": float(np.mean(lik)) if lik else float("nan")}


@torch.no_grad()
def analytic(S, base, scA, scB, tok_len, mean_len):
    pos, fut = S.test_pos, S.test_future
    rows, _ = contexts(S, pos)
    cand, lpi, rA, rB = [], [], [], []
    for i, lg in policy_logits(base, S, pos):
        lp = torch.log_softmax(lg, -1)
        v, c = torch.topk(lp, TOPN)
        p = pos[i:i + len(lg)]
        hA = scA.hidden(*ctx_batch(S, p))
        hB = scB.hidden(*ctx_batch(S, p))
        cand.append(c)
        lpi.append(v)
        rA.append(scA(hA, c))
        rB.append(scB(hB, c))
    cand, lpi, rA, rB = (torch.cat(t) for t in (cand, lpi, rA, rB))
    hist = torch.from_numpy(np.stack([np.pad(r[-200:], (200 - len(r[-200:]), 0)) for r in rows]).astype(np.int64)).to(DEV)
    res = {}
    for rw in REWARDS:
        # награда в масштабе, где β сопоставимы: делим на std награды по кандидатам
        if rw == "ratio_norep":           # сравнение с историей кусками: [N, 500, 200] целиком не влезает
            rep = torch.cat([(hist[i:i + 1000, None, :] == cand[i:i + 1000, :, None]).any(-1)
                             for i in range(0, len(cand), 1000)]).float()
            r = rA[..., 0] - 0.5 * rep
        else:
            r = reward_fn(rw, rA, cand, None, tok_len, mean_len)
        r = r / r.std()
        curve = []
        for beta in BETAS:
            score = lpi if beta is None else (r if beta == 0 else lpi + r / beta)
            o = torch.argsort(score, 1, descending=True)[:, :10]
            top = cand.gather(1, o)
            ndcg, hr = rank_metrics(top.cpu().numpy(), fut[:, 0])
            ga = lambda t, j: t.gather(1, o[..., None].expand(-1, -1, 3))[..., j].mean().item()
            rec = {"beta": beta, "NDCG@10": float(ndcg.mean()), "Recall@10": float(hr.mean()),
                   "A: дослушивание": ga(rA, 0), "B: дослушивание": ga(rB, 0),
                   "A: лайк": ga(rA, 1), "B: лайк": ga(rB, 1), "A: продолжение": ga(rA, 2), "B: продолжение": ga(rB, 2),
                   "A: время, с": (rA[..., 0].gather(1, o) * tok_len[top]).mean().item(),
                   "B: время, с": (rB[..., 0].gather(1, o) * tok_len[top]).mean().item(),
                   **behaviour(S, top.cpu().numpy(), rows), **real_outcomes(S, pos, top.cpu().numpy())}
            curve.append(rec)
        res[rw] = curve
        print(rw, [(c["beta"], round(c["NDCG@10"], 4), round(c["A: дослушивание"], 3), round(c["B: дослушивание"], 3))
                   for c in curve], flush=True)
    return res


def rl_table(S, base_ndcg):
    users = np.searchsorted(S.user_start, S.test_pos, side="right") - 1
    rows = {}
    for d in sorted((RESULTS / "rl").glob("*/final.json")):
        f = json.load(open(d, encoding="utf-8"))
        nd = np.load(d.parent / "test_ndcg.npy")
        top = np.load(d.parent / "test_top.npy")
        rows[d.parent.name] = {"args": f["args"], **f["test"],
                               "ΔNDCG@10 к исходной, 95% CI": bootstrap_diff(nd, base_ndcg, users),
                               **real_outcomes(S, S.test_pos, top)}
    return rows


REAL_N, REAL_R, REAL_L = ("реально послушано рекомендованных, на точку", "их реальная доля дослушивания",
                          "их реальная доля лайков")


def plot(analytic_res, rl_rows, base):
    colors = dict(zip(REWARDS, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c9a227", "#d6457a"]))
    # 1. компромисс
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, judge in zip(axes, ["A", "B"]):
        for rw in ["ratio", "ratio_norep"]:
            c = analytic_res[rw]
            ax.plot([r["NDCG@10"] for r in c], [r[f"{judge}: дослушивание"] for r in c], "-o", ms=3,
                    color=colors[rw], alpha=0.8, label=f"оптимум π_ref*exp(r/β), награда {rw}")
        for name, r in rl_rows.items():
            a = r["args"]
            if a["horizon"] != 1 or a["reward"] not in ("ratio", "ratio_norep"):
                continue
            mk = "*" if a.get("adv", "group_std") == "group_std" else "D"
            ax.plot(r["NDCG@10"], r[f"{judge}: дослушивание"], mk, ms=10 if mk == "*" else 6,
                    color=colors[a["reward"]], mec="k", mew=0.6)
            ax.annotate(f"β={a['beta']:g}", (r["NDCG@10"], r[f"{judge}: дослушивание"]), fontsize=6,
                        xytext=(4, -2), textcoords="offset points")
        ax.plot(base["NDCG@10"], base[f"{judge}: дослушивание"], "ks", ms=8, label="исходная GPT-2")
        ax.plot([], [], "*", color="gray", mec="k", ms=10, label="GRPO (нормировка на std группы)")
        ax.plot([], [], "D", color="gray", mec="k", ms=6, label="GRPO (фиксированный масштаб)")
        ax.set_xlabel("NDCG@10 на test")
        ax.set_ylabel(f"предсказанное дослушивание топ-10 ({judge})")
        ax.set_title("по рабочей reward model A (по ней учится RL)" if judge == "A"
                     else "по независимой reward model B", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Компромисс между релевантностью и наградой (награда: дослушивание)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=150)
    plt.close(fig)

    # 2. поведение вдоль кривой оптимума (награда ratio)
    c = [r for r in analytic_res["ratio"] if r["beta"] is not None]
    x = np.arange(len(c))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, key, title in zip(axes, ["длина трека, с", "покрытие каталога", "доля повторов из истории", "NDCG@10"],
                              ["средняя длина трека, с", "покрытие каталога", "доля уже слушанного", "NDCG@10"]):
        ax.plot(x, [r[key] for r in c], "-o", color=colors["ratio"], ms=4)
        ax.axhline(base[key], color="k", ls="--", lw=1)
        ax.set_xticks(x, [f"{r['beta']:g}" for r in c], fontsize=7)
        ax.set_xlabel("β (меньше - сильнее оптимизация)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Что меняется в выдаче при оптимизации дослушивания (пунктир - исходная модель)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "behaviour.png", dpi=150)
    plt.close(fig)

    # 3. предсказание reward model против реальности
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (rw, pkey, rkey, title) in zip(axes, [
            ("like", "A: лайк", REAL_L, "награда: лайк"), ("ratio", "A: дослушивание", REAL_R, "награда: дослушивание")]):
        c = [r for r in analytic_res[rw] if r["beta"] is not None and r[REAL_N] > 0.2]
        rel = lambda v, v0: v / v0 - 1
        ax.plot([r["beta"] for r in c], [rel(r[pkey], base[pkey]) * 100 for r in c], "-o", ms=4, color="#2a78d6",
                label="предсказание reward model A для топ-10")
        ax.plot([r["beta"] for r in c], [rel(r[pkey.replace("A:", "B:")], base[pkey.replace("A:", "B:")]) * 100
                                         for r in c], "-o", ms=4, color="#8a5cd1", label="предсказание reward model B")
        ax.plot([r["beta"] for r in c], [rel(r[rkey], base[rkey]) * 100 for r in c], "-s", ms=4, color="#eb6834",
                label="реальный отклик на реально прослушанные рекомендации")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("β (вправо - сильнее оптимизация)")
        ax.set_ylabel("изменение к исходной модели, %")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "reward_vs_real.png", dpi=150)
    plt.close(fig)


def main():
    limit_gpu_memory()
    OUT.mkdir(parents=True, exist_ok=True)
    S = Seqs()
    tok_len = torch.as_tensor(S.tok_len, device=DEV)
    mean_len = float(np.average(S.tok_len[2:], weights=S.tok_cnt[2:]))
    base = build_gpt().to(DEV)
    base.load_state_dict(torch.load(RESULTS / "base" / "model.pt"))
    base.eval()
    scA = Scorer(load_rm(RESULTS / "reward" / "rm_A.pt"), S.tok_len)
    scB = Scorer(load_rm(RESULTS / "reward" / "rm_B.pt"), S.tok_len)
    if (OUT / "analytic.json").exists() and not os.environ.get("RECOMPUTE"):
        an = json.load(open(OUT / "analytic.json", encoding="utf-8"))     # аналитика не зависит от RL-прогонов
    else:
        an = analytic(S, base, scA, scB, tok_len, mean_len)
        json.dump(an, open(OUT / "analytic.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    base_ndcg = np.load(RESULTS / "base" / "test_ndcg.npy")
    rl = rl_table(S, base_ndcg)
    base_row = an["ratio"][0]
    json.dump({"base": base_row, "rl": rl}, open(OUT / "rl_table.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plot(an, rl, base_row)
    for k, v in rl.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "args"})


if __name__ == "__main__":
    main()
