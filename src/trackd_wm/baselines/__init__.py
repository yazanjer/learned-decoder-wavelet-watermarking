"""Baseline adapters (docs/BASELINE_INSPECTION.md).

Third-party code is never vendored: each repository is cloned at its pinned
commit at run time (:func:`fetch`). MBRS, CIN and FIN carry no licence file and
are used for research comparison only. All compatibility changes are applied
at run time by the launchers and wrappers in this package, not by editing the
third-party files.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPOS = {
    "hidden": ("https://github.com/ando-khachatryan/HiDDeN", "556f76d"),
    "mbrs": ("https://github.com/jzyustc/MBRS", "2bab0df"),
    "cin": ("https://github.com/rmpku/CIN", "c3643f9"),
    "fin": ("https://github.com/QQiuyp/FIN", "f77222f"),
    "dwtdct": ("https://github.com/ShieldMnt/invisible-watermark", "68d0376"),
}

LICENCE = {"hidden": "MIT", "mbrs": "none (comparison only)", "cin": "none (comparison only)",
           "fin": "none (comparison only)", "dwtdct": "MIT"}


def fetch(name: str, root: Path) -> Path:
    """Clone ``name`` into ``root/name`` and check out the pinned commit (idempotent)."""
    url, commit = REPOS[name]
    dst = Path(root) / name
    if not (dst / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", url, str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", str(dst), "rev-parse", "--short=7", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise RuntimeError(f"{name}: HEAD {head} != pinned {commit}")
    return dst
