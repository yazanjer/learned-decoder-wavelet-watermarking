"""Per-image metrics. Images are float tensors in [0, 1], shape (B, 3, H, W)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr_rgb(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma in [0, 1] (full range)."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


def psnr_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return psnr_rgb(rgb_to_y(x), rgb_to_y(y))


def _gauss(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    t = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean SSIM over RGB channels (Wang et al. 2004 constants, 11x11 Gaussian window)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    w = _gauss(device=x.device).repeat(3, 1, 1, 1)
    mu_x = F.conv2d(x, w, groups=3)
    mu_y = F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x * x, w, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=3) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.flatten(1).mean(1)


def bit_accuracy(bits_hat: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Per-image fraction of correctly decoded bits; both tensors in {0, 1}."""
    return (bits_hat.round() == bits.round()).float().mean(1)


def ber(bits_hat: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return 1.0 - bit_accuracy(bits_hat, bits)
