"""Офлайн-проверка для долгосрочных целей: связаны ли краткосрочные сигналы сессии с возвращением.

Сессия - прослушивания юзера с перерывами не больше 30 минут. Для каждой сессии считаем
краткосрочные сигналы (средняя доля дослушивания, доля лайков, доля треков, включённых
рекомендательной системой, длина сессии, доля пропусков) и долгосрочный - вернулся ли юзер в течение
24 часов после конца сессии. Все связи - внутри юзера (из сигналов вычитается среднее по юзеру):
иначе получим 'активные юзеры и дослушивают больше, и возвращаются чаще', что для выбора
рекомендаций бесполезно.

    python src/longterm.py
"""
import json

import numpy as np

from common import DAY, RESULTS, Seqs

GAP = 30 * 60


def main():
    S = Seqs()
    n = len(S.ts)
    uid = S.uid.astype(np.int64)
    new = np.ones(n, bool)
    new[1:] = (np.diff(S.ts) > GAP) | (uid[1:] != uid[:-1])
    sid = np.cumsum(new) - 1                                # номер сессии для каждого прослушивания
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1                     # индекс последнего события сессии
    ns = len(starts)
    cnt = np.bincount(sid)
    mean = lambda v: np.bincount(sid, v.astype(np.float64)) / cnt
    s_user = uid[starts]
    nxt_same = np.append(s_user[1:] == s_user[:-1], False)
    gap_next = np.append(S.ts[starts[1:]] - S.ts[ends[:-1]], 0)
    ret = (nxt_same & (gap_next <= DAY)).astype(float)
    known = S.ts[ends] <= S.ts.max() - DAY                  # для последних суток не знаем, вернётся ли
    feats = {
        "средняя доля дослушивания": mean(S.ratio),
        "доля лайков": mean(S.like),
        "доля треков от рекомендаций": 1 - mean(S.organic),
        "log длины сессии": np.log(cnt),
        "доля пропусков (<30 %)": mean(S.ratio < 0.3),
    }
    u, y = s_user[known], ret[known]
    dm = lambda v: v - (np.bincount(u, v) / np.bincount(u))[u]
    out = {"сессий": int(known.sum()), "доля вернувшихся за 24 ч": float(y.mean())}
    for name, v in feats.items():
        x = v[known]
        out[name] = {"корреляция с возвращением (все сессии)": float(np.corrcoef(x, y)[0, 1]),
                     "корреляция внутри юзера": float(np.corrcoef(dm(x), dm(y))[0, 1])}
    x = dm(feats["средняя доля дослушивания"][known])
    q = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(x, q)
    out["возвращаемость по квинтилям дослушивания внутри юзера"] = [float(y[b == k].mean()) for k in range(5)]
    json.dump(out, open(RESULTS / "longterm.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
