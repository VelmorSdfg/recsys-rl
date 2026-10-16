"""Модель GPT-2 для next item, батчи контекстов, метрики релевантности и поведения рекомендаций."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel

from common import CTX, DEV, OOV, PAD, VOCAB


def build_gpt(n_layer=4, n_embd=256, n_head=4, dropout=0.1):
    cfg = GPT2Config(vocab_size=VOCAB + 2, n_positions=CTX + 8, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                     resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
                     bos_token_id=PAD, eos_token_id=PAD)
    return GPT2LMHeadModel(cfg)


def _chunk_ce(h, w, y):
    return F.cross_entropy((h @ w.T).float(), y, reduction="sum")


def next_item_loss(model, x, m, pos, chunk=2048):
    """Кросс-энтропия next item по всем позициям, где следующий трек в словаре.
    Логиты [позиции × 50 тыс.] целиком в память не влезают: считаем кусками и пересчитываем
    на обратном проходе (checkpoint)."""
    h = model.transformer(input_ids=x[:, :-1], attention_mask=m[:, :-1], position_ids=pos[:, :-1]).last_hidden_state
    y = x[:, 1:]
    ok = (y != OOV) & (y != PAD) & (m[:, :-1] == 1)
    h, y = h[ok], y[ok]
    w = model.lm_head.weight
    tot = sum(checkpoint(_chunk_ce, h[i:i + chunk], w, y[i:i + chunk], use_reentrant=False)
              for i in range(0, len(y), chunk))
    return tot / len(y)


def left_pad(rows, L=CTX):
    """Список массивов токенов -> [B, L] с паддингом слева, маска, позиции."""
    x = np.zeros((len(rows), L), np.int64)
    for i, r in enumerate(rows):
        r = r[-L:]
        if len(r):
            x[i, L - len(r):] = r
    x = torch.from_numpy(x)
    m = (x != PAD).long()
    pos = (m.cumsum(1) - 1).clamp(min=0)
    return x, m, pos


def contexts(S, pos, L=CTX):
    """Токены последних L прослушиваний юзера строго до позиции pos (глобальный индекс)."""
    u = np.searchsorted(S.user_start, pos, side="right") - 1
    start = np.maximum(S.user_start[u], pos - L)
    return [S.tok[a:b] for a, b in zip(start, pos)], start


def last_logits(model, x, m, pos):
    h = model.transformer(input_ids=x.to(DEV), attention_mask=m.to(DEV), position_ids=pos.to(DEV)).last_hidden_state
    lg = model.lm_head(h[:, -1]).float()
    lg[:, PAD] = lg[:, OOV] = float("-inf")     # не рекомендуем паддинг и «трек вне словаря»
    return lg


@torch.no_grad()
def policy_logits(model, S, pos, bs=256):
    """Логиты следующего трека для каждой точки (на CPU не тащим, возвращаем по батчам)."""
    model.eval()
    rows, _ = contexts(S, pos)
    for i in range(0, len(pos), bs):
        x, m, p = left_pad(rows[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield i, last_logits(model, x, m, p)


def rank_metrics(top, target, k=10):
    """top [N, >=k] токены, target [N] -> NDCG@k и HitRate@k (= Recall@k при одной цели)."""
    hit = top[:, :k] == target[:, None]
    hit &= (target != OOV)[:, None]
    rank = hit.argmax(1)
    found = hit.any(1)
    ndcg = np.where(found, 1 / np.log2(rank + 2), 0.0)
    return ndcg, found.astype(float)


def behaviour(S, top, rows, k=10):
    """Поведение рекомендаций: что именно советует модель."""
    t = top[:, :k]
    cnt = np.bincount(t.ravel(), minlength=len(S.tok_len)).astype(float)
    srt = np.sort(cnt[cnt > 0])
    gini = 1 - 2 * np.sum(np.cumsum(srt) / srt.sum()) / len(srt) + 1 / len(srt) if len(srt) else 0
    rep = np.mean([np.isin(t[i], r).mean() for i, r in enumerate(rows)])
    return {
        "длина трека, с": float(S.tok_len[t].mean()),
        "популярность, log10 прослушиваний": float(np.log10(S.tok_cnt[t] + 1).mean()),
        "доля повторов из истории": float(rep),
        "покрытие каталога": float((cnt > 0).sum() / (len(cnt) - 2)),
        "Gini рекомендаций": float(gini),
        "доля топ-100 треков в выдаче": float(np.sort(cnt)[::-1][:100].sum() / cnt.sum()),
    }


def evaluate(model, S, split, k=10, extra=None):
    """Релевантность + поведение для топ-k по логитам политики. extra(i, logits, rows) -> dict сумм."""
    pos, fut = getattr(S, f"{split}_pos"), getattr(S, f"{split}_future")
    rows, _ = contexts(S, pos)
    tops, ent = [], []
    acc = {}
    for i, lg in policy_logits(model, S, pos):
        tops.append(torch.topk(lg, k).indices.cpu().numpy())
        lp = torch.log_softmax(lg, -1)
        ent.append(-(lp.exp() * lp.nan_to_num(neginf=0)).sum(-1).cpu().numpy())
        if extra is not None:
            for key, v in extra(i, lg, rows[i:i + len(lg)]).items():
                acc[key] = acc.get(key, 0) + v
    top = np.concatenate(tops)
    ndcg, hr = rank_metrics(top, fut[:, 0], k)
    f = np.where(fut >= 2, fut, -9)
    fut_rec = np.array([len(set(top[i]) & set(f[i])) / max(1, len(set(f[i]) - {-9})) for i in range(len(top))])
    res = {f"NDCG@{k}": float(ndcg.mean()), f"Recall@{k}": float(hr.mean()),
           f"Recall@{k} по 10 следующим": float(fut_rec.mean()), "энтропия политики": float(np.mean(np.concatenate(ent)))}
    res.update(behaviour(S, top, rows, k))
    res.update({key: v / len(pos) for key, v in acc.items()})
    return res, top, ndcg


def bootstrap_diff(a, b, users, n=1000, seed=0):
    """Парная разница средних с кластерным бутстрапом по юзерам: (Δ, low, high)."""
    rng = np.random.default_rng(seed)
    uu, inv = np.unique(users, return_inverse=True)
    d = np.bincount(inv, a - b)
    c = np.bincount(inv)
    boots = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))
        boots.append((w * d).sum() / (w * c).sum())
    return float((a - b).mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
