"""Invertible multi-level 2D transforms used by the Paper I network.

* ``LiftingWavelet``: separable lifting scheme (split / predict / update).
  ``base='cdf97'`` uses the Daubechies-Sweldens factorisation of the CDF 9/7
  wavelet; ``base='haar'`` the Haar lifting. With ``learned=True`` every fixed
  predict/update step is followed by a learned residual operator that depends
  only on the other polyphase branch, so perfect reconstruction holds by
  construction. Learned operators are zero-initialised (the transform starts
  as the fixed wavelet). ``integer=True`` rounds every lifting update
  (straight-through gradient) and omits the scaling step, which yields exact
  integer reconstruction.
* ``CouplingTransform``: control of matched parameter count; a space-to-depth
  squeeze followed by affine coupling blocks (FIN/CIN-type invertible
  couplings).

Both expose ``analysis(x) -> list of levels`` with each level a tuple
``(LL, LH, HL, HH)`` of tensors (B, C, H/2^l, W/2^l), and ``synthesis``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Bands = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# CDF 9/7 lifting coefficients (Daubechies and Sweldens, 1998).
CDF97 = dict(alpha=-1.586134342059924, beta=-0.052980118572961,
             gamma=0.882911075530934, delta=0.443506852043971, zeta=1.149604398860241)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _shift_next(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t[n+1] with symmetric (replicate) extension at the end."""
    n = t.shape[dim]
    return torch.cat([t.narrow(dim, 1, n - 1), t.narrow(dim, n - 1, 1)], dim=dim)


def _shift_prev(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t[n-1] with symmetric (replicate) extension at the start."""
    n = t.shape[dim]
    return torch.cat([t.narrow(dim, 0, 1), t.narrow(dim, 0, n - 1)], dim=dim)


def _split(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Polyphase split by strided slicing (deterministic backward on CUDA)."""
    if dim == -1:
        return x[..., 0::2], x[..., 1::2]
    if dim == -2:
        return x[..., 0::2, :], x[..., 1::2, :]
    raise ValueError(dim)


def _merge(s: torch.Tensor, o: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of :func:`_split` by stacking and reshaping (no index_copy)."""
    if dim == -1:
        return torch.stack([s, o], dim=-1).flatten(-2)
    if dim == -2:
        return torch.stack([s, o], dim=-2).flatten(-3, -2)
    raise ValueError(dim)


class ResidualOp(nn.Module):
    """Small 2D convolutional operator; last layer zero-initialised."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Lifting1D(nn.Module):
    """One-dimensional lifting along ``dim`` (-1: width, -2: height)."""

    def __init__(self, dim: int, base: str, learned: bool, integer: bool, channels: int, hidden: int):
        super().__init__()
        self.dim, self.base, self.learned, self.integer = dim, base, learned, integer
        if base == "cdf97":
            c = CDF97
            # (kind, coefficient): 'p' updates odd from even, 'u' updates even from odd
            self.steps = [("p", c["alpha"]), ("u", c["beta"]), ("p", c["gamma"]), ("u", c["delta"])]
            self.zeta = None if integer else c["zeta"]
        elif base == "haar":
            self.steps = [("p", -1.0), ("u", 0.5)]
            self.zeta = None
        else:
            raise ValueError(base)
        self.ops = nn.ModuleList([ResidualOp(channels, hidden) for _ in self.steps]) if learned else None

    def _pred(self, k: int, src: torch.Tensor, coef: float, kind: str) -> torch.Tensor:
        d = self.dim
        if self.base == "haar":
            v = coef * src
        elif kind == "p":
            v = coef * (src + _shift_next(src, d))
        else:
            v = coef * (_shift_prev(src, d) + src)
        if self.learned:
            v = v + self.ops[k](src)
        if self.integer:
            v = _ste_round(v)
        return v

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dim
        s, o = _split(x, d)
        for k, (kind, coef) in enumerate(self.steps):
            if kind == "p":
                o = o + self._pred(k, s, coef, kind)
            else:
                s = s + self._pred(k, o, coef, kind)
        if self.zeta is not None:
            s, o = s / self.zeta, o * self.zeta
        return s, o

    def inverse(self, s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        d = self.dim
        if self.zeta is not None:
            s, o = s * self.zeta, o / self.zeta
        for k in reversed(range(len(self.steps))):
            kind, coef = self.steps[k]
            if kind == "p":
                o = o - self._pred(k, s, coef, kind)
            else:
                s = s - self._pred(k, o, coef, kind)
        return _merge(s, o, d)


class LiftingLevel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.row = Lifting1D(dim=-1, **kw)
        self.col_l = Lifting1D(dim=-2, **kw)
        self.col_h = Lifting1D(dim=-2, **kw)

    def forward(self, x: torch.Tensor) -> Bands:
        lo, hi = self.row(x)
        ll, lh = self.col_l(lo)
        hl, hh = self.col_h(hi)
        return ll, lh, hl, hh

    def inverse(self, bands: Bands) -> torch.Tensor:
        ll, lh, hl, hh = bands
        lo = self.col_l.inverse(ll, lh)
        hi = self.col_h.inverse(hl, hh)
        return self.row.inverse(lo, hi)


class LiftingWavelet(nn.Module):
    def __init__(self, levels: int = 2, base: str = "cdf97", learned: bool = True,
                 integer: bool = False, channels: int = 3, hidden: int = 16):
        super().__init__()
        self.levels = nn.ModuleList([LiftingLevel(base=base, learned=learned, integer=integer,
                                                  channels=channels, hidden=hidden) for _ in range(levels)])

    def analysis(self, x: torch.Tensor) -> List[Bands]:
        out: List[Bands] = []
        cur = x
        for lvl in self.levels:
            b = lvl(cur)
            out.append(b)
            cur = b[0]
        return out

    def synthesis(self, bands: Sequence[Bands]) -> torch.Tensor:
        cur = None
        for lvl, b in zip(reversed(self.levels), reversed(list(bands))):
            b = (cur if cur is not None else b[0], b[1], b[2], b[3])
            cur = lvl.inverse(b)
        return cur


class AffineCoupling(nn.Module):
    """x2 <- x2 * exp(clamp * tanh(s(x1))) + t(x1); zero-initialised (identity at start)."""

    def __init__(self, c1: int, c2: int, hidden: int, clamp: float = 1.0):
        super().__init__()
        self.clamp = clamp
        self.net = nn.Sequential(nn.Conv2d(c1, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, 2 * c2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x1):
        s, t = self.net(x1).chunk(2, dim=1)
        return self.clamp * torch.tanh(s), t

    def forward(self, x1, x2):
        s, t = self._st(x1)
        return x2 * torch.exp(s) + t

    def inverse(self, x1, y2):
        s, t = self._st(x1)
        return (y2 - t) * torch.exp(-s)


class CouplingLevel(nn.Module):
    """One level: space-to-depth squeeze, optional fixed mixing, then affine coupling blocks.

    ``mode`` selects how the 4C squeezed channels are arranged before the couplings and which group is passed
    to the next level as its low-pass band (chunk 0):

    * ``"legacy"`` (pilot G3, defective): ``pixel_unshuffle`` order (channel-major). Chunk 0 holds three
      polyphase positions of the first colour channel only, so the next level and the embedding cover one
      colour channel. Kept only so that pilot checkpoints remain loadable.
    * ``"reorder"`` (fix 1, channel ordering only): position-major order; chunk 0 is polyphase position (0, 0)
      of all colour channels (a lazy-wavelet low band).
    * ``"haar"`` (fix 2 = fix 1 + fixed Haar mixing): the four positions of each colour channel are mixed by
      the orthonormal 2D Haar matrix; chunk 0 is the Haar LL band of all colour channels. At initialisation
      the level equals a Haar wavelet level.
    The couplings are zero-initialised, so every mode starts as its fixed (lazy or Haar) transform, and every
    mode is exactly invertible.
    """

    MODES = ("legacy", "reorder", "haar")

    def __init__(self, channels: int, hidden: int, blocks: int, mode: str = "legacy"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(mode)
        c4 = 4 * channels
        self.c1 = c4 // 2
        self.mode = mode
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(AffineCoupling(self.c1, c4 - self.c1, hidden))
            self.blocks.append(AffineCoupling(c4 - self.c1, self.c1, hidden))
        self.channels = channels
        # rows: LL, LH, HL, HH; columns: polyphase positions (0,0), (0,1), (1,0), (1,1); orthonormal
        h = torch.tensor([[1., 1., 1., 1.], [1., 1., -1., -1.], [1., -1., 1., -1.], [1., -1., -1., 1.]]) / 2.0
        self.register_buffer("haar_m", h, persistent=False)

    def _arrange(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "legacy":
            return z
        b, _, hh, ww = z.shape
        z = z.view(b, self.channels, 4, hh, ww)  # pixel_unshuffle order: channel-major, then position
        if self.mode == "haar":
            z = torch.einsum("kp,bcphw->bkchw", self.haar_m, z)
        else:
            z = z.transpose(1, 2)
        return z.reshape(b, 4 * self.channels, hh, ww)  # position/band-major

    def _unarrange(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "legacy":
            return z
        b, _, hh, ww = z.shape
        z = z.view(b, 4, self.channels, hh, ww)
        if self.mode == "haar":
            z = torch.einsum("kp,bkchw->bcphw", self.haar_m, z)  # inverse = transpose (orthonormal)
        else:
            z = z.transpose(1, 2)
        return z.reshape(b, 4 * self.channels, hh, ww)

    def forward(self, x: torch.Tensor) -> Bands:
        z = self._arrange(F.pixel_unshuffle(x, 2))
        a, b = z[:, : self.c1], z[:, self.c1:]
        for i in range(0, len(self.blocks), 2):
            b = self.blocks[i](a, b)
            a = self.blocks[i + 1](b, a)
        z = torch.cat([a, b], dim=1)
        return tuple(z.chunk(4, dim=1))  # type: ignore[return-value]

    def inverse(self, bands: Bands) -> torch.Tensor:
        z = torch.cat(list(bands), dim=1)
        a, b = z[:, : self.c1], z[:, self.c1:]
        for i in reversed(range(0, len(self.blocks), 2)):
            a = self.blocks[i + 1].inverse(b, a)
            b = self.blocks[i].inverse(a, b)
        return F.pixel_shuffle(self._unarrange(torch.cat([a, b], dim=1)), 2)


class CouplingTransform(nn.Module):
    def __init__(self, levels: int = 2, channels: int = 3, hidden: int = 16, blocks: int = 2, mode: str = "legacy"):
        super().__init__()
        self.levels = nn.ModuleList([CouplingLevel(channels, hidden, blocks, mode) for _ in range(levels)])

    analysis = LiftingWavelet.analysis
    synthesis = LiftingWavelet.synthesis


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def matched_coupling(target_params: int, levels: int = 2, channels: int = 3, blocks: int = 2,
                     mode: str = "legacy") -> Tuple[CouplingTransform, int]:
    """Coupling transform whose hidden width gives the parameter count closest to ``target_params``."""
    best = None
    for h in range(2, 257):
        n = n_params(CouplingTransform(levels, channels, h, blocks, mode))
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (h, n)
    h, _ = best
    m = CouplingTransform(levels, channels, h, blocks, mode)
    return m, h


def build_transform(kind: str, levels: int = 2, hidden: int = 16) -> nn.Module:
    if kind == "lift_learned":
        return LiftingWavelet(levels, "cdf97", learned=True, hidden=hidden)
    if kind == "lift_learned_int":
        return LiftingWavelet(levels, "cdf97", learned=True, integer=True, hidden=hidden)
    if kind == "lift_cdf97":
        return LiftingWavelet(levels, "cdf97", learned=False)
    if kind == "lift_haar":
        return LiftingWavelet(levels, "haar", learned=False)
    if kind == "coupling_matched":
        target = n_params(LiftingWavelet(levels, "cdf97", learned=True, hidden=hidden))
        return matched_coupling(target, levels)[0]
    if kind in ("coupling_reorder_matched", "coupling_haar_matched"):
        target = n_params(LiftingWavelet(levels, "cdf97", learned=True, hidden=hidden))
        return matched_coupling(target, levels, mode=kind.split("_")[1])[0]
    raise ValueError(kind)
