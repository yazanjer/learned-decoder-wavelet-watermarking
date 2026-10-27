"""Determinism utilities (ml-paper-pipeline section 1)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, strict: bool = True) -> None:
    """Seed every RNG and request deterministic kernels.

    ``strict=False`` is used for third-party baseline code whose ops may lack
    deterministic CUDA kernels; such runs record ``deterministic=warn_only``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def worker_init_fn(seed: int):
    def _init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return _init
