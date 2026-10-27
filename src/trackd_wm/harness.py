"""Common evaluation harness (identical datasets, payload, attacks and metrics for every method).

A watermarker implements ``embed(x, bits) -> x_w`` and ``extract(y) -> bits`` with
images in [0, 1] (B, 3, H, W) and bits in {0, 1} (B, 64). Residual strength
scaling ``x_s = x + s (x_w - x)`` is applied identically to every method to
obtain matched-PSNR operating points.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from . import metrics as M
from .attacks import ATTACKS, quantize_8bit


def eval_messages(n: int, bits: int, seed: int) -> torch.Tensor:
    """Fixed evaluation messages: image i always carries the same bits."""
    rng = np.random.default_rng(10_000 + seed)
    return torch.from_numpy(rng.integers(0, 2, size=(n, bits))).float()


@torch.no_grad()
def embed_all(wm, dataset, msgs: torch.Tensor, batch: int, device) -> Dict[str, torch.Tensor]:
    covers, marked = [], []
    for i in range(0, len(dataset), batch):
        x = torch.stack([dataset[j] for j in range(i, min(i + batch, len(dataset)))]).to(device)
        m = msgs[i:i + x.shape[0]].to(device)
        xw = wm.embed(x, m).clamp(0, 1)
        covers.append(x.cpu())
        marked.append(xw.cpu())
    return {"cover": torch.cat(covers), "marked": torch.cat(marked)}


@torch.no_grad()
def evaluate(wm, dataset, msgs: torch.Tensor, attacks: Iterable[str], strength: float = 1.0,
             batch: int = 50, device="cpu", cache: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, np.ndarray]:
    """Per-image metrics. Returns arrays of length N for every metric and attack."""
    emb = cache if cache is not None else embed_all(wm, dataset, msgs, batch, device)
    cover = emb["cover"]
    xw = quantize_8bit(cover + strength * (emb["marked"] - cover))
    out: Dict[str, np.ndarray] = {
        "psnr_rgb": M.psnr_rgb(xw, cover).numpy(),
        "psnr_y": M.psnr_y(xw, cover).numpy(),
        "ssim": torch.cat([M.ssim(xw[i:i + batch], cover[i:i + batch]) for i in range(0, len(xw), batch)]).numpy(),
    }
    for a in attacks:
        if hasattr(wm, "set_attack"):  # CIN: oracle noise-branch selection (docs/VERIFICATION_P1.md, Section 1)
            wm.set_attack(a)
        accs, t0 = [], time.time()
        for i in range(0, len(xw), batch):
            y = ATTACKS[a](xw[i:i + batch]).to(device)
            bits_hat = wm.extract(y).cpu()
            accs.append(M.bit_accuracy(bits_hat, msgs[i:i + batch]))
        out[f"bitacc_{a}"] = torch.cat(accs).numpy()
    return out


def matched_strength(wm, dataset, msgs, target_psnr: float = 40.0, batch: int = 50, device="cpu",
                     iters: int = 30) -> Dict[str, float]:
    """Bisection on the residual strength s so that the mean RGB PSNR after 8-bit quantisation equals the target."""
    emb = embed_all(wm, dataset, msgs, batch, device)
    cover, marked = emb["cover"], emb["marked"]

    def mean_psnr(s: float) -> float:
        return float(M.psnr_rgb(quantize_8bit(cover + s * (marked - cover)), cover).mean())

    lo, hi = 0.0, 1.0
    while mean_psnr(hi) > target_psnr and hi < 64:
        hi *= 2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_psnr(mid) > target_psnr:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return {"strength": s, "psnr_at_strength": mean_psnr(s), "psnr_native": mean_psnr(1.0), "_cache": emb}


def summarise(per_image: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in per_image.items()}
