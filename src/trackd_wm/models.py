"""Watermarking network of Paper I and its transform controls.

The encoder, decoder, message processing, attention map and training noise are
identical for every variant; only the transform (``transform`` argument of
:func:`build_wmnet`) differs. This isolates the effect of the transform.

Encoder: x -> analysis (L levels) -> the four sub-bands of the deepest level are
concatenated with a message map and a residual is predicted for them -> synthesis
-> residual r = synthesis(bands + delta) - x -> spatial attention A(x) in (0, 1)
-> x_w = x + s * A(x) * r, with strength s = 1 in training and s varied at test
time to reach a matched PSNR.
Decoder: analysis (own transform instance) -> CNN on the deepest-level sub-bands
-> 64 logits.

``freeze_encoder_transform=True`` (stage I-2, docs/PHASE1_EXECUTION_PLAN.md) keeps the encoder's transform at
its initialisation (fixed CDF 9/7 for ``lift_learned``, fixed Haar-mixed arrangement for the coupling
variants, because their learned operators are zero-initialised) while the decoder's transform is learned.
Parameter names, and therefore checkpoint compatibility, are unchanged.
Images are in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import build_transform, n_params


def cbr(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MessageMap(nn.Module):
    """64 bits -> (1, h, w) map, following the MBRS message processor (linear + upsampling)."""

    def __init__(self, bits: int, size: int):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(bits, 16 * 16)
        self.conv = cbr(1, 16)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        z = self.fc(m * 2 - 1).view(-1, 1, 16, 16)
        f = self.size // 16
        z = z.repeat_interleave(f, dim=2).repeat_interleave(f, dim=3)  # nearest upsampling, deterministic
        return self.conv(z)


class Attention(nn.Module):
    """Per-pixel embedding-strength map in (0, 1); a component, not a claimed contribution."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(cbr(3, hidden), cbr(hidden, hidden), nn.Conv2d(hidden, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x * 2 - 1))


class Encoder(nn.Module):
    def __init__(self, transform: nn.Module, bits: int, image_size: int, levels: int,
                 hidden: int = 64, attention: bool = True):
        super().__init__()
        self.t = transform
        self.band_size = image_size // (2 ** levels)
        self.msg = MessageMap(bits, self.band_size)
        self.body = nn.Sequential(cbr(12 + 16, hidden), cbr(hidden, hidden), cbr(hidden, hidden))
        self.out = nn.Conv2d(hidden, 12, 1)
        self.att = Attention() if attention else None

    def residual(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        bands = self.t.analysis(x)
        deep = torch.cat(bands[-1], dim=1)
        delta = self.out(self.body(torch.cat([deep, self.msg(m)], dim=1)))
        new = list(bands)
        new[-1] = tuple((deep + delta).chunk(4, dim=1))
        r = self.t.synthesis(new) - x
        if self.att is not None:
            r = self.att(x) * r
        return r

    def forward(self, x: torch.Tensor, m: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        return x + strength * self.residual(x, m)


class FrontEnd(nn.Module):
    """Capacity control for stage I-3a (V1+): a residual convolution on the deepest-level sub-bands of a fixed
    transform, with as many parameters as the learned decoder transform of V2 and a zero-initialised last
    layer, so that at initialisation V1+ equals V1. It adds decoder capacity at the transform's position
    without being a transform (no perfect-reconstruction structure)."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Decoder(nn.Module):
    def __init__(self, transform: nn.Module, bits: int, hidden: int = 64, frontend_hidden: int = 0):
        super().__init__()
        self.t = transform
        self.front = FrontEnd(12, frontend_hidden) if frontend_hidden else None
        self.body = nn.Sequential(cbr(12, hidden), cbr(hidden, hidden), cbr(hidden, hidden), cbr(hidden, hidden),
                                  nn.Conv2d(hidden, 1, 3, padding=1))
        self.fc = nn.Linear(64, bits)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        deep = torch.cat(self.t.analysis(y * 2 - 1)[-1], dim=1)
        if self.front is not None:
            deep = self.front(deep)
        h = self.body(deep)
        h = F.avg_pool2d(h, kernel_size=h.shape[-1] // 8)  # fixed-kernel pooling to 8x8 (deterministic)
        return self.fc(h.flatten(1))


class WMNet(nn.Module):
    def __init__(self, transform: str, bits: int = 64, image_size: int = 128, levels: int = 2,
                 lift_hidden: int = 16, attention: bool = True, freeze_encoder_transform: bool = False,
                 decoder_frontend_hidden: int = 0):
        super().__init__()
        self.transform_kind = transform
        self.freeze_encoder_transform = freeze_encoder_transform
        self.encoder = Encoder(build_transform(transform, levels, lift_hidden), bits, image_size, levels,
                               attention=attention)
        self.decoder = Decoder(build_transform(transform, levels, lift_hidden), bits,
                               frontend_hidden=decoder_frontend_hidden)
        if freeze_encoder_transform:
            for p in self.encoder.t.parameters():
                p.requires_grad_(False)

    def transform_params(self) -> int:
        return n_params(self.encoder.t) + n_params(self.decoder.t)

    def total_params(self) -> int:
        return n_params(self)


def build_wmnet(cfg) -> WMNet:
    return WMNet(transform=cfg.transform, bits=cfg.bits, image_size=cfg.image_size,
                 levels=cfg.levels, lift_hidden=cfg.lift_hidden, attention=cfg.attention,
                 freeze_encoder_transform=bool(cfg.get("freeze_encoder_transform", False)),
                 decoder_frontend_hidden=int(cfg.get("decoder_frontend_hidden", 0)))
