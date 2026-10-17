"""Общие пути, константы, загрузка последовательностей, лимит видеопамяти."""
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("YAMBDA_RAW", ROOT.parent / "recsys_autofeatures" / "data" / "raw_yambda"))
DATA = ROOT / "data"
RESULTS = ROOT / "results"

VOCAB = 50_000           # самые популярные треки train; остальные -> OOV
PAD, OOV = 0, 1          # токен трека = ранг популярности + 2
CTX = 200                # длина контекста: последние 200 прослушиваний
DAY = 86_400
VALID_DAYS, TEST_DAYS = 7, 7
RATIO_BINS = 8           # дослушивание в истории подаётся reward model корзинами
DEV = "cuda"


def limit_gpu_memory():
    """Лимит видеопамяти на процесс (RECSYS_GPU_GB, по умолчанию 5 ГБ; 0 - без лимита)."""
    gb = float(os.environ.get("RECSYS_GPU_GB", "5"))
    if gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / total))


def peak_gb():
    p = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return p


class Seqs:
    """Все прослушивания, отсортированные по (юзер, время); user_start[u]..user_start[u+1] - юзер u."""

    def __init__(self):
        z = np.load(DATA / "listens.npz")
        self.__dict__.update({k: z[k] for k in z.files})
        self.n_users = len(self.user_start) - 1

    def ratio_bin(self, idx):
        return np.minimum((self.ratio[idx].astype(np.float32) * RATIO_BINS).astype(np.int64), RATIO_BINS - 1) + 1


def ratio_to_bin(r):
    """r в [0, 1] -> корзина 1..RATIO_BINS (0 - паддинг)."""
    return torch.clamp((r * RATIO_BINS).long(), max=RATIO_BINS - 1) + 1
