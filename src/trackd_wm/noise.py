"""Training noise for the proposed method and its transform controls (identical for all three).

One distortion per mini-batch, drawn uniformly from the pool. Real JPEG is
applied with a straight-through estimator (forward: real libjpeg output;
backward: identity), as in Video Seal and Pixel Seal; it is mixed with
identity in the same pool, which follows the MBRS mixing principle.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .attacks import jpeg


def jpeg_ste(x: torch.Tensor, q: int) -> torch.Tensor:
    y = jpeg(x.detach().cpu(), q).to(x.device)
    return x + (y - x).detach()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = 5
    t = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(x, w, padding=2, groups=3)  # zero padding: deterministic backward on CUDA


def dropout_cover(x: torch.Tensor, cover: torch.Tensor, p: float) -> torch.Tensor:
    mask = (torch.rand_like(x[:, :1]) < p).float()
    return x * (1 - mask) + cover * mask


POOL = ["identity", "jpeg50", "jpeg_rand", "blur", "gauss", "dropout"]


def apply_training_noise(xw: torch.Tensor, cover: torch.Tensor, rng: random.Random) -> tuple:
    name = rng.choice(POOL)
    if name == "identity":
        y = xw
    elif name == "jpeg50":
        y = jpeg_ste(xw, 50)
    elif name == "jpeg_rand":
        y = jpeg_ste(xw, rng.randint(30, 90))
    elif name == "blur":
        y = gaussian_blur(xw, rng.uniform(0.5, 1.5))
    elif name == "gauss":
        y = xw + 0.02 * torch.randn_like(xw)
    else:
        y = dropout_cover(xw, cover, 0.3)
    return y.clamp(0, 1), name
