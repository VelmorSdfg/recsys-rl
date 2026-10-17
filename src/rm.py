"""Reward model: история юзера (треки + как он их дослушивал) + трек-кандидат -> отклик.

Три головы (все - вероятности):
    ratio  ожидаемая доля дослушивания кандидата (обучается как BCE с мягкой меткой ratio in [0, 1])
    like   лайк в течение часа
    cont   юзер продолжит слушать (следующее прослушивание не позже чем через 30 мин) - сигнал удержания

Кодировщик истории - каузальный GPT-2: позиция t видит события <= t и предсказывает отклик на
событие t+1, поэтому одно окно даёт 200 обучающих примеров. Признаки кандидата: свой эмбеддинг,
длина трека и флаг is_organic. Рекомендации RL-политики оцениваются с is_organic = 0: это ровно
ситуация 'трек включила рекомендательная система'.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from common import CTX, DEV, PAD, RATIO_BINS, VOCAB, ratio_to_bin

HEADS = ["ratio", "like", "cont"]


def len_feat(sec):
    return torch.log1p(sec.float().clamp(min=0)) / 6 - 0.9


class RewardModel(nn.Module):
    def __init__(self, d=128, n_layer=2, n_head=4, dropout=0.1):
        super().__init__()
        self.cfg = dict(d=d, n_layer=n_layer, n_head=n_head, dropout=dropout)
        self.item = nn.Embedding(VOCAB + 2, d, padding_idx=PAD)
        self.rbin = nn.Embedding(RATIO_BINS + 1, d, padding_idx=0)
        self.cand = nn.Embedding(VOCAB + 2, d)
        self.gpt = GPT2Model(GPT2Config(vocab_size=1, n_positions=CTX + 8, n_embd=d, n_layer=n_layer, n_head=n_head,
                                        resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout))
        self.head = nn.Sequential(nn.Linear(3 * d + 2, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 3))

    def encode(self, tok, rb, m, pos):
        e = self.item(tok) + self.rbin(rb)
        return self.gpt(inputs_embeds=e, attention_mask=m, position_ids=pos).last_hidden_state

    def score(self, h, cand, length, organic):
        """h [..., d], cand/length/organic [...] -> логиты [..., 3]."""
        c = self.cand(cand)
        z = torch.cat([h, c, h * c, len_feat(length)[..., None], organic.float()[..., None]], -1)
        return self.head(z)


def load_rm(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rm = RewardModel(**ck["cfg"])
    rm.load_state_dict(ck["state"])
    return rm.to(DEV).eval()


def ctx_batch(S, pos, L=CTX):
    """Контексты reward model для точек pos: токены и корзины дослушивания, паддинг слева."""
    u = np.searchsorted(S.user_start, pos, side="right") - 1
    start = np.maximum(S.user_start[u], pos - L)
    idx = pos[:, None] - L + np.arange(L)[None, :]
    ok = idx >= start[:, None]
    ci = np.clip(idx, 0, None)
    tok = np.where(ok, S.tok[ci], PAD).astype(np.int64)
    rb = np.where(ok, S.ratio_bin(ci), 0).astype(np.int64)
    tok, rb = torch.from_numpy(tok), torch.from_numpy(rb)
    m = (tok != PAD).long()
    return tok, rb, m, (m.cumsum(1) - 1).clamp(min=0)


class Scorer:
    """Оценка произвольных треков-кандидатов для батча контекстов (используется в RL и анализе)."""

    def __init__(self, rm, tok_len):
        self.rm = rm
        self.tok_len = torch.as_tensor(tok_len, device=DEV)

    @torch.no_grad()
    def hidden(self, tok, rb, m, pos):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.rm.encode(tok.to(DEV), rb.to(DEV), m.to(DEV), pos.to(DEV))[:, -1].float()

    @torch.no_grad()
    def __call__(self, h, cand):
        """h [B, d], cand [B, G] -> вероятности [B, G, 3] (is_organic = 0)."""
        hb = h[:, None, :].expand(-1, cand.shape[1], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = self.rm.score(hb, cand, self.tok_len[cand], torch.zeros_like(cand))
        return torch.sigmoid(lg.float())

    @torch.no_grad()
    def all_items(self, h, chunk=5000):
        """Отклик на все треки словаря: [B, V+2, 3]."""
        out = []
        for a in range(0, VOCAB + 2, chunk):
            c = torch.arange(a, min(a + chunk, VOCAB + 2), device=DEV)[None].expand(len(h), -1)
            out.append(self(h, c))
        return torch.cat(out, 1)


def append_generated(tok, rb, m, pos, new_tok, new_ratio):
    """Сдвинуть контекст влево и дописать сгенерированный трек с предсказанной долей дослушивания."""
    tok = torch.cat([tok[:, 1:], new_tok[:, None].to(tok.device)], 1)
    rb = torch.cat([rb[:, 1:], ratio_to_bin(new_ratio)[:, None].to(rb.device)], 1)
    m = (tok != PAD).long()
    return tok, rb, m, (m.cumsum(1) - 1).clamp(min=0)
