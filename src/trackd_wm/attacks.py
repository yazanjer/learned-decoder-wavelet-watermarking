"""Evaluation attacks, applied identically to every method.

Every evaluation starts by quantising the watermarked image to 8-bit RGB, as a
saved PNG would be. JPEG uses PIL (libjpeg) with 4:2:0 chroma subsampling.
Inputs and outputs are float tensors in [0, 1], shape (B, 3, H, W).
"""
from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
import torch
from PIL import Image


def quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0, 1) * 255.0) / 255.0


def _to_pil(img: torch.Tensor) -> Image.Image:
    a = (quantize_8bit(img).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def _from_pil(im: Image.Image) -> torch.Tensor:
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(255.0)


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    out = []
    for img in x:
        buf = io.BytesIO()
        _to_pil(img).save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
        buf.seek(0)
        with Image.open(buf) as im:
            out.append(_from_pil(im))
    return torch.stack(out).to(x.device)


def identity(x: torch.Tensor) -> torch.Tensor:
    return quantize_8bit(x)


# Pilot attack set (docs/PILOT_P1.md, Section 10). Stage B extends this registry.
ATTACKS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "clean": identity,
    "jpeg50": lambda x: jpeg(x, 50),
    "jpeg30": lambda x: jpeg(x, 30),
}


# Additional evaluation attacks for stage I-2 (docs/PHASE1_EXECUTION_PLAN.md). Deterministic: the Gaussian noise
# uses a fixed generator, so every method receives the same noise realisation for the same batch position.
# Outputs are quantised to 8 bits, as a saved image would be. Parameters lie inside the training noise pool's
# ranges (blur sigma 0.5-1.5, noise sigma 0.02, JPEG Q30-90) except the resize, which is not in the pool.
def _gauss_kernel(sigma: float, k: int = 5) -> torch.Tensor:
    t = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    import torch.nn.functional as F

    w = _gauss_kernel(sigma).to(x.device, x.dtype)
    return quantize_8bit(F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), w, groups=3))


def gaussian_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    g = torch.Generator().manual_seed(12345)
    n = torch.randn(x.shape, generator=g).to(x.device, x.dtype)
    return quantize_8bit(x + sigma * n)


def resize(x: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    y = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False, antialias=True)
    return quantize_8bit(F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False))


ATTACKS.update({
    "jpeg70": lambda x: jpeg(x, 70),
    "jpeg90": lambda x: jpeg(x, 90),
    "blur1": lambda x: blur(x, 1.0),
    "noise002": lambda x: gaussian_noise(x, 0.02),
    "resize05": lambda x: resize(x, 0.5),
})
EXTENDED_ATTACKS = ("clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "blur1", "noise002", "resize05")


def get_attack(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return ATTACKS[name]


# Stage I-5, module M1: extended classical suite (docs/PROTOCOL_I5.md, Sections 2 and 7). Every attack is
# deterministic and ends with 8-bit quantisation; geometric attacks return the original size, so that every
# decoder sees a 128 x 128 input. Parameters are fixed in the frozen protocol, Section 7.
def webp(x: torch.Tensor, quality: int) -> torch.Tensor:
    out = []
    for img in x:
        buf = io.BytesIO()
        _to_pil(img).save(buf, format="WEBP", quality=int(quality), method=4)
        buf.seek(0)
        with Image.open(buf) as im:
            out.append(_from_pil(im))
    return torch.stack(out).to(x.device)


def blur_wide(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur with kernel size 2 * ceil(3 sigma) + 1 (the I-4 attack ``blur1`` keeps its 5 x 5 kernel)."""
    import math

    import torch.nn.functional as F

    k = 2 * math.ceil(3 * sigma) + 1
    w = _gauss_kernel(sigma, k).to(x.device, x.dtype)
    p = k // 2
    return quantize_8bit(F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"), w, groups=3))


def median3(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    b, c, h, w = x.shape
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    patches = F.unfold(p.reshape(b * c, 1, h + 2, w + 2), 3)          # (b*c, 9, h*w)
    return quantize_8bit(patches.median(dim=1).values.reshape(b, c, h, w))


def centre_crop(x: torch.Tensor, side: float) -> torch.Tensor:
    """Centre crop keeping ``side`` of each side length, resized back (bilinear) to the original size."""
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ch, cw = round(h * side), round(w * side)
    t, l = (h - ch) // 2, (w - cw) // 2
    y = x[..., t:t + ch, l:l + cw]
    return quantize_8bit(F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False, antialias=True))


def brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    return quantize_8bit(x * factor)


def contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Contrast about the per-image mean grey level (torchvision convention: luma mean)."""
    m = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean(dim=(-2, -1)).view(-1, 1, 1, 1)
    return quantize_8bit(m + factor * (x - m))


def rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotation about the centre (bilinear, zero fill), as the WAVES rotation distortion."""
    import math

    import torch.nn.functional as F

    a = math.radians(degrees)
    th = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]], dtype=x.dtype)
    grid = F.affine_grid(th.unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device), list(x.shape), align_corners=False)
    return quantize_8bit(F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False))


def erase(x: torch.Tensor, area: float) -> torch.Tensor:
    """Centred square covering ``area`` of the image set to mid-grey (deterministic WAVES-type erasing)."""
    h, w = x.shape[-2:]
    s = round((area * h * w) ** 0.5)
    t, l = (h - s) // 2, (w - s) // 2
    y = x.clone()
    y[..., t:t + s, l:l + s] = 0.5
    return quantize_8bit(y)


ATTACKS.update({f"jpeg{q}": (lambda q: lambda x: jpeg(x, q))(q) for q in (10, 20, 40, 60, 80)})
ATTACKS.update({
    "webp50": lambda x: webp(x, 50),
    "webp75": lambda x: webp(x, 75),
    "blur05": lambda x: blur_wide(x, 0.5),
    "blur2": lambda x: blur_wide(x, 2.0),
    "noise001": lambda x: gaussian_noise(x, 0.01),
    "noise005": lambda x: gaussian_noise(x, 0.05),
    "median3": median3,
    "resize075": lambda x: resize(x, 0.75),
    "crop075": lambda x: centre_crop(x, 0.75),
    "crop09": lambda x: centre_crop(x, 0.9),
    "bright12": lambda x: brightness(x, 1.2),
    "bright08": lambda x: brightness(x, 0.8),
    "contrast12": lambda x: contrast(x, 1.2),
    "contrast08": lambda x: contrast(x, 0.8),
    "rotate10": lambda x: rotate(x, 10.0),
    "erase10": lambda x: erase(x, 0.10),
})
I5_ATTACKS = ("clean",) + tuple(f"jpeg{q}" for q in range(10, 100, 10)) + (
    "webp50", "webp75", "blur05", "blur1", "blur2", "noise001", "noise002", "noise005", "median3",
    "resize05", "resize075", "crop075", "crop09", "bright12", "bright08", "contrast12", "contrast08",
    "rotate10", "erase10")
