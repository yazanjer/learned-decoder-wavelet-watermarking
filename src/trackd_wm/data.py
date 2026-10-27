"""Data splits with SHA256 integrity, preprocessing and datasets.

Split files are plain text, one line per image: ``<file name> <sha256 of the
original file>``. They are generated once from the MS-COCO 2017 archives by
:func:`build_coco_splits`, committed under ``src/trackd_wm/splits/`` and
verified on every pod by :func:`verify_split`.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

SPLIT_DIR = Path(__file__).resolve().parent / "splits"

# Pilot split sizes (docs/PILOT_P1.md, Section 10).
N_TRAIN, N_VAL, N_TEST = 10_000, 1_000, 1_000
MIN_SIDE = 128          # no upscaling of originals
MAX_ASPECT = 2.0        # MBRS skips images with aspect ratio > 2; excluded for every method


def sha256_file(path: os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def eligible(path: os.PathLike) -> bool:
    """Eligibility rule applied identically to every method."""
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return False
    return min(w, h) >= MIN_SIDE and max(w, h) / min(w, h) <= MAX_ASPECT


@dataclass(frozen=True)
class SplitEntry:
    name: str
    sha256: str


def read_split(path: os.PathLike) -> List[SplitEntry]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, digest = line.split()
        out.append(SplitEntry(name, digest))
    return out


def write_split(path: os.PathLike, entries: Iterable[SplitEntry], header: str) -> None:
    lines = [f"# {h}" for h in header.splitlines()]
    lines += [f"{e.name} {e.sha256}" for e in entries]
    Path(path).write_text("\n".join(lines) + "\n")


def seeded_order(names: Sequence[str], seed: int) -> List[str]:
    names = sorted(names)
    rng = np.random.default_rng(seed)
    return [names[i] for i in rng.permutation(len(names))]


def select_eligible(order: Sequence[str], root: os.PathLike, n: int, skip: set) -> List[SplitEntry]:
    out: List[SplitEntry] = []
    for name in order:
        if name in skip:
            continue
        p = Path(root) / name
        if p.exists() and eligible(p):
            out.append(SplitEntry(name, sha256_file(p)))
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} eligible images found, {n} required")
    return out


def build_coco_splits(train_root: os.PathLike, val_root: os.PathLike, out_dir: os.PathLike,
                      train_names: Sequence[str], val_names: Sequence[str], seed: int = 0) -> Tuple[Path, Path, Path]:
    """Deterministic selection: seeded permutation of the sorted archive listing,
    then the first eligible images in that order.

    ``train_names``/``val_names`` are the complete archive listings
    (``unzip -Z1``), so the order does not depend on which files were extracted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_order = seeded_order(train_names, seed)
    trval = select_eligible(tr_order, train_root, N_TRAIN + N_VAL, skip=set())
    train, val = trval[:N_TRAIN], trval[N_TRAIN:]
    te_order = seeded_order(val_names, seed)
    test = select_eligible(te_order, val_root, N_TEST, skip=set())
    hdr = (f"MS-COCO 2017; seed={seed}; rule: min side >= {MIN_SIDE}, aspect <= {MAX_ASPECT}; "
           "sha256 of the original JPEG file")
    paths = (out_dir / "coco_train.txt", out_dir / "coco_val.txt", out_dir / "coco_test.txt")
    write_split(paths[0], train, hdr + "\nsource: train2017")
    write_split(paths[1], val, hdr + "\nsource: train2017 (disjoint from coco_train)")
    write_split(paths[2], test, hdr + "\nsource: val2017")
    return paths


def candidate_names(listing: Sequence[str], seed: int, n_candidates: int) -> List[str]:
    """First ``n_candidates`` names of the seeded order (used to extract a subset of the archive)."""
    return seeded_order(listing, seed)[:n_candidates]


def verify_split(entries: Sequence[SplitEntry], root: os.PathLike) -> None:
    bad = [e.name for e in entries if sha256_file(Path(root) / e.name) != e.sha256]
    if bad:
        raise RuntimeError(f"SHA256 mismatch for {len(bad)} files, e.g. {bad[:3]}")


def preprocess_image(src: os.PathLike, dst: os.PathLike, size: int) -> None:
    """Centre square crop on the shorter side, bicubic resize, 8-bit RGB PNG."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)
        im.save(dst, format="PNG")


def prepare(entries: Sequence[SplitEntry], src_root: os.PathLike, dst_root: os.PathLike, size: int) -> List[Path]:
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    out = []
    for e in entries:
        dst = dst_root / (Path(e.name).stem + ".png")
        if not dst.exists():
            preprocess_image(Path(src_root) / e.name, dst, size)
        out.append(dst)
    return out


class ImageFolderDataset:
    """Preprocessed PNGs in [0, 1], CHW float32, ordered as in the split file."""

    def __init__(self, files: Sequence[os.PathLike]):
        self.files = [Path(f) for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        import torch

        with Image.open(self.files[i]) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(a).permute(2, 0, 1).float().div(255.0)


def link_layout(files: Sequence[Path], dst: os.PathLike) -> Path:
    """Symlink preprocessed files into a flat directory (baseline-specific layouts)."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        t = dst / f.name
        if not t.exists():
            t.symlink_to(Path(f).resolve())
    return dst
