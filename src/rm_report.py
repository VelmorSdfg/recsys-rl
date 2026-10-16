"""Качество reward models «внутри юзера» — то, что важно для RL.

Обычный AUC смешивает два умения: отличать юзеров (кто вообще много лайкает) и отличать треки для
одного юзера. RL выбирает трек для фиксированного юзера, поэтому ему нужно только второе.
Внутри-юзерный AUC: AUC по прослушиваниям каждого юзера отдельно, усреднённый по юзерам с обоими
классами. Эвристика «по юзеру» здесь по определению даёт 0.5. Плюс согласие моделей A и B друг с другом.

    python src/rm_report.py
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import RESULTS, Seqs
from rm import HEADS
from train_reward import baselines


def within_user_auc(users, y, p, min_n=20):
    aucs, w = [], []
    for u in np.unique(users):
        s = users == u
        if s.sum() >= min_n and 0 < y[s].sum() < s.sum():
            aucs.append(roc_auc_score(y[s], p[s]))
            w.append(s.sum())
    return float(np.mean(aucs)), len(aucs)


def main():
    S = Seqs()
    pos = np.load(RESULTS / "reward" / "test_eval_pos.npy")
    users = np.searchsorted(S.user_start, pos, side="right") - 1
    pa = np.load(RESULTS / "reward" / "rm_A_test_pred.npy")
    pb = np.load(RESULTS / "reward" / "rm_B_test_pred.npy")
    bl = baselines(S, pos)
    y = {"ratio": (S.ratio[pos] >= 0.9).astype(int), "like": S.like[pos].astype(int), "cont": S.cont[pos].astype(int)}
    out = {}
    for i, h in enumerate(HEADS):
        ok = y[h] >= 0
        r = {}
        for name, p in [("RM A", pa[:, i]), ("RM B", pb[:, i]), ("эвристика по треку", bl[h]["по треку"])]:
            auc, n = within_user_auc(users[ok], y[h][ok], np.asarray(p)[ok])
            r[name] = auc
        r["юзеров"] = n
        r["корреляция A и B"] = float(np.corrcoef(pa[:, i], pb[:, i])[0, 1])
        # корреляция после вычитания среднего по юзеру: согласны ли A и B в выборе трека
        da = pa[:, i] - np.bincount(users, pa[:, i])[users] / np.bincount(users)[users]
        db = pb[:, i] - np.bincount(users, pb[:, i])[users] / np.bincount(users)[users]
        r["корреляция A и B внутри юзера"] = float(np.corrcoef(da, db)[0, 1])
        out[h] = r
        print(h, json.dumps(r, ensure_ascii=False))
    json.dump(out, open(RESULTS / "reward" / "within_user.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
