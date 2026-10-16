"""Yambda-50M -> data/listens.npz: все прослушивания по (юзер, время) + метки для reward model.

Для каждого прослушивания:
    tok     токен трека (ранг популярности в train + 2; 1 — трек вне словаря из 50 тыс.)
    ratio   доля дослушивания, обрезанная до [0, 1] (played_ratio_pct / 100)
    like    лайк этого трека этим юзером в течение часа после начала прослушивания
    cont    юзер продолжил слушать: следующее прослушивание началось не позже чем через 30 мин
            (-1 — неизвестно: до конца данных меньше 30 мин)
    organic юзер нашёл трек сам (1) или его включила рекомендательная система (0)

Время: train — всё до t_valid, valid — 7 дней [t_valid, t_test), test — последние 7 дней.
Словарь и длины треков считаются только по train.

Точки оценки: у каждого юзера, активного в valid (test), берём до 3 случайных прослушиваний
этого периода. Модель видит 200 прослушиваний до точки и должна угадать трек в точке;
для генерации на несколько шагов хранятся ещё 10 следующих реальных треков.

    python src/prepare.py
"""
import duckdb
import numpy as np

from common import CTX, DATA, DAY, RAW, TEST_DAYS, VALID_DAYS, VOCAB

CONT_GAP = 30 * 60
LIKE_WINDOW = 3600
EVAL_PER_USER, MIN_HIST, FUTURE = 3, 10, 10


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    src = RAW / "multi_event.parquet"
    t_end = con.execute(f"SELECT max(timestamp) FROM '{src}'").fetchone()[0]
    t_test = t_end - TEST_DAYS * DAY
    t_valid = t_test - VALID_DAYS * DAY
    con.execute(f"""
    CREATE TABLE ev AS SELECT row_number() OVER () AS rn, uid, timestamp::BIGINT AS ts, item_id AS item,
           event_type, played_ratio_pct, track_length_seconds, is_organic FROM '{src}'""")
    con.execute(f"""
    CREATE TABLE vocab AS
    SELECT item, (row_number() OVER (ORDER BY n DESC, item) + 1)::INT AS tok, n, len FROM (
        SELECT item, count(*) AS n, median(track_length_seconds) AS len FROM ev
        WHERE event_type = 'listen' AND ts < {t_valid} GROUP BY item)
    QUALIFY tok < {VOCAB + 2}""")
    con.execute(f"""
    CREATE TABLE li AS
    WITH l AS (SELECT * FROM ev WHERE event_type = 'listen'),
    lk AS (SELECT DISTINCT l.rn FROM l JOIN ev k ON k.event_type = 'like' AND k.uid = l.uid AND k.item = l.item
           AND k.ts >= l.ts AND k.ts <= l.ts + {LIKE_WINDOW})
    SELECT l.uid, l.ts, coalesce(v.tok, 1) AS tok, least(l.played_ratio_pct, 100) / 100.0 AS ratio,
           (lk.rn IS NOT NULL)::TINYINT AS "like", l.is_organic::TINYINT AS organic,
           least(l.track_length_seconds, 32000)::SMALLINT AS length,
           CASE WHEN lead(l.ts) OVER w IS NOT NULL THEN (lead(l.ts) OVER w - l.ts <= {CONT_GAP})::TINYINT
                WHEN l.ts > {t_end - CONT_GAP} THEN -1 ELSE 0 END::TINYINT AS cont
    FROM l LEFT JOIN vocab v USING (item) LEFT JOIN lk USING (rn)
    WINDOW w AS (PARTITION BY l.uid ORDER BY l.ts, l.rn)
    ORDER BY l.uid, l.ts, l.rn""")
    d = con.execute("SELECT * FROM li").fetchnumpy()
    uid = d["uid"]
    users, user_start = np.unique(uid, return_index=True)
    user_start = np.append(user_start, len(uid)).astype(np.int64)
    voc = con.execute("SELECT tok, item, n, len FROM vocab ORDER BY tok").fetchnumpy()
    tok_len = np.zeros(VOCAB + 2, np.float32)
    tok_cnt = np.zeros(VOCAB + 2, np.int64)
    tok_len[voc["tok"]] = voc["len"]
    tok_cnt[voc["tok"]] = voc["n"]
    tok_len[:2] = np.median(voc["len"])
    item_ids = np.zeros(VOCAB + 2, np.int64)
    item_ids[voc["tok"]] = voc["item"]

    ts, tok = d["ts"], d["tok"].astype(np.int32)
    rng = np.random.default_rng(0)
    evals = {}
    for name, lo, hi in [("valid", t_valid, t_test), ("test", t_test, t_end + 1)]:
        pos = []
        for u in range(len(users)):
            a, b = user_start[u], user_start[u + 1]
            idx = np.flatnonzero((ts[a:b] >= lo) & (ts[a:b] < hi)) + a
            idx = idx[idx - a >= MIN_HIST]
            if len(idx):
                pos.append(np.sort(rng.choice(idx, min(EVAL_PER_USER, len(idx)), replace=False)))
        pos = np.concatenate(pos)
        u_of = np.searchsorted(user_start, pos, side="right") - 1
        fut = np.full((len(pos), FUTURE), -1, np.int32)
        for j in range(FUTURE):
            ok = pos + j < user_start[u_of + 1]
            fut[ok, j] = tok[pos[ok] + j]
        evals[name] = (pos, fut)
        print(f"{name}: {len(pos)} точек оценки, {len(np.unique(u_of))} юзеров, "
              f"цель вне словаря {np.mean(fut[:, 0] == 1):.1%}")

    np.savez(DATA / "listens.npz", uid=np.searchsorted(users, uid).astype(np.int32), ts=ts, tok=tok,
             ratio=d["ratio"].astype(np.float16), like=d["like"], cont=d["cont"], organic=d["organic"],
             length=d["length"], user_start=user_start, users=users, t_valid=t_valid, t_test=t_test,
             tok_len=tok_len, tok_cnt=tok_cnt, item_ids=item_ids,
             valid_pos=evals["valid"][0], valid_future=evals["valid"][1],
             test_pos=evals["test"][0], test_future=evals["test"][1])
    tr = ts < t_valid
    print(f"прослушиваний {len(ts)}, юзеров {len(users)}; train {tr.sum()}, "
          f"вне словаря в train {np.mean(tok[tr] == 1):.1%}")
    print(f"доля лайков {d['like'].mean():.2%}, продолжение сессии {np.mean(d['cont'][d['cont'] >= 0]):.1%}, "
          f"средняя доля дослушивания {d['ratio'].mean():.3f}, органических {d['organic'].mean():.1%}")
    # повтор: трек уже был среди 200 предыдущих прослушиваний юзера (важно для музыки)
    pos = evals["valid"][0]
    rep = [tok[p] != 1 and tok[p] in tok[max(user_start[np.searchsorted(user_start, p, 'right') - 1], p - CTX):p]
           for p in pos]
    print(f"valid: цель уже была в последних {CTX} прослушиваниях у {np.mean(rep):.1%} точек")


if __name__ == "__main__":
    main()
