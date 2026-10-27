"""Stage I-6: evaluation of public baseline checkpoints (docs/PROTOCOL_I6.md). Evaluation only; no training.

Step ``i6_public`` evaluates three checkpoints released by their authors in the common harness, on the COCO test
split with the 29 conditions of the I-5 suite (``attacks.I5_ATTACKS``), at two operating points:

* ``matched_40dB``: residual strength found by bisection on the COCO validation split so that the mean RGB PSNR
  after 8-bit quantisation is 40 dB (the rule of every other method in the paper);
* ``native``: strength 1, the operating point of the released model.

Checkpoints (decision d-2026-10-27-m07):

* ``fin64``    FIN, ``experiments/JPEG/FED.pt`` in the authors' repository (64 bits, 128 x 128, JPEG-trained on DIV2K).
  Same payload and the same evaluation messages as the WMNet variants (I-4 message seed), so per-image pairing is
  possible.
* ``hidden30`` HiDDeN (community PyTorch implementation), ``experiments/combined-noise`` (30 bits, 128 x 128, epoch
  400, trained on COCO with a JPEG approximation). Native payload 30 bits: a reference row, not a matched payload.
* ``mbrs30``   MBRS, ``MBRS_Diffusion_128_m30`` epoch 114 from the authors' Google Drive folder (30 bits, 128 x 128,
  message diffusion, trained on ImageNet). Native payload 30 bits: a reference row. The Drive download may fail on
  the pod; the model is then recorded as ``unavailable`` and the step continues.

30-bit models receive 30-bit messages drawn by ``harness.eval_messages`` with the I-4 message seed.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .attacks import I5_ATTACKS
from .baselines import LICENCE, REPOS, fetch
from .harness import embed_all, eval_messages
from .pilot import (cpu_threads, device, matched_strength_cached, provenance, set_phase, split_files,
                    write_json)

MBRS_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1A_SAqvU2vMsHxki0s9m9rKa-g8B6aghe"
MBRS30_PROJECT = "MBRS_Diffusion_128_m30"
MBRS30_EPOCH = 114
GDOWN_VERSION = "5.2.0"   # pinned (supervisor d-2026-10-27-i6, required change 3)
REQUIRED = ("fin64",)      # the replacement baseline: the step fails unless it is evaluated (required change 5)

PUBLIC = {
    "fin64": {"repo": "fin", "bits": 64, "checkpoint": "experiments/JPEG/FED.pt", "training_data": "DIV2K",
              "payload_matched": True},
    "hidden30": {"repo": "hidden", "bits": 30, "checkpoint": "experiments/combined-noise/checkpoints/"
                 "combined-noise--epoch-400.pyt", "training_data": "COCO (10,000 images)", "payload_matched": False},
    "mbrs30": {"repo": "mbrs", "bits": 30, "checkpoint": f"{MBRS30_PROJECT}/models/EC_{MBRS30_EPOCH}.pth (Google Drive)",
               "training_data": "ImageNet", "payload_matched": False},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_mbrs30(dest: Path) -> Path:
    """Download the authors' MBRS_Diffusion_128_m30 encoder-decoder checkpoint from their public Drive folder."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"gdown=={GDOWN_VERSION}"], check=True)
    import gdown  # noqa: E402  (installed above, pod only)

    gdown.download_folder(MBRS_DRIVE_FOLDER, output=str(dest), quiet=True, remaining_ok=True)
    return mbrs30_path(dest)


def mbrs30_path(dest: Path) -> Path:
    """The intended checkpoint only: ``<folder>/MBRS_Diffusion_128_m30/models/EC_114.pth`` directly under one
    top-level folder of the download. No fallback to other files of the same name (required change 4)."""
    hits = [h for h in Path(dest).rglob(f"EC_{MBRS30_EPOCH}.pth")
            if h.parent.name == "models" and h.parent.parent.name == MBRS30_PROJECT]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {MBRS30_PROJECT}/models/EC_{MBRS30_EPOCH}.pth in the Drive "
                                f"download, found {len(hits)}: {[str(h) for h in hits]}")
    return hits[0]


def load_public(name: str, repos: Path, dev):
    from .baselines import wrappers as W

    spec = PUBLIC[name]
    repo = fetch(spec["repo"], repos)
    if name == "fin64":
        ck = repo / spec["checkpoint"]
        return W.FINWatermarker(repo, ck, dev), ck
    if name == "hidden30":
        ck = repo / spec["checkpoint"]
        return W.HiDDeNWatermarker(repo, ck.parent.parent, ck, dev), ck
    if name == "mbrs30":
        ck = fetch_mbrs30(repos / "mbrs_drive")
        return W.MBRSWatermarker(repo, ck, dev, message_length=30, diffusion=True), ck
    raise ValueError(name)


def evaluate_public(cfg, wm, bits: int, dev, target_psnr: float = 40.0) -> tuple:
    """Strength matched on validation (``bits``-bit messages), then the test split at that strength and at s = 1."""
    from .i5 import eval_points

    seed = int(cfg.i4.message_seed)
    va = D.ImageFolderDataset(split_files(cfg, "val"))
    cache = embed_all(wm, va, eval_messages(len(va), bits, seed), cfg.eval_batch, dev)
    st = {f"matched_{target_psnr:.0f}dB": matched_strength_cached(cache, float(target_psnr)), "native": 1.0}
    te = D.ImageFolderDataset(split_files(cfg, "test"))
    return st, eval_points(wm, te, eval_messages(len(te), bits, seed), st, cfg.eval_batch, dev, "coco")


def versions() -> dict:
    import importlib.metadata as md

    v = {}
    for pkg in ("torch", "torchvision", "numpy", "scipy", "pillow", "kornia", "gdown"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    return v


def split_provenance(cfg) -> dict:
    """SHA-256 of the split lists (the image list and digests of each split, from step ``data``)."""
    out = {}
    for sp in ("val", "test"):
        lst = Path(D.SPLIT_DIR) / f"coco_{sp}.txt"
        out[sp] = {"split_list": f"src/trackd_wm/splits/{lst.name}", "sha256": sha256(lst),
                   "images": len(split_files(cfg, sp))}
    return out


def preserve(cfg, out: Path, log: list, label: str) -> bool:
    """Push ``out`` to its results branch and, under Orchestron, upload it with scripts/upload_artifacts.sh.
    Both outcomes are appended to ``out/UPLOAD_LOG.json``; a failure also writes the persistent flag
    ``<work>/results/UPLOAD_INCOMPLETE``, which bootstrap.sh turns into STOP_REASON completed_upload_incomplete
    (supervisor d-2026-10-27-i6b, required changes 2-3)."""
    from .pilot import REPO_ROOT

    entry = {"after": label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if os.environ.get("NO_PUSH") == "1":
        entry.update(push_rc=None, upload_rc=None, note="NO_PUSH=1 (local run)")
        log.append(entry)
        write_json(out / "UPLOAD_LOG.json", log)
        return True
    write_json(out / "UPLOAD_LOG.json", log + [entry])  # the log itself travels with this attempt
    entry["push_rc"] = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "pod" / "push_results.sh"), str(out),
                                       "r/pilot-p1-i6_public"], check=False).returncode
    if os.environ.get("ORCH_UPLOAD_URL"):
        entry["upload_rc"] = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "upload_artifacts.sh"), str(out)],
                                            check=False).returncode
    else:
        entry["upload_rc"] = None
    ok = entry["push_rc"] == 0 and entry["upload_rc"] in (0, None)
    entry["ok"] = ok
    log.append(entry)
    write_json(out / "UPLOAD_LOG.json", log)
    if not ok:
        (Path(cfg.work) / "results" / "UPLOAD_INCOMPLETE").write_text(f"i6_public: preservation failed after {label}\n")
    return ok


def pip_freeze(out: Path) -> None:
    r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    (out / "pip_freeze.txt").write_text(r.stdout)


def step_i6_public(cfg):
    from .i5 import smoke_attacks
    from .sysmon import GPUSampler

    torch.set_num_threads(cpu_threads())
    set_phase(cfg, "eval")
    out = Path(cfg.work) / "results" / "i6_public"
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    only = [m for m in os.environ.get("I6_MODELS", "").split() if m] or list(PUBLIC)
    upload_log = []
    pip_freeze(out)
    sampler = GPUSampler(out / "gpu.csv").start()   # GPU monitoring (required change 1); pushed with the results
    res = {"protocol": "docs/PROTOCOL_I6.md", "attacks": list(I5_ATTACKS), "required": list(REQUIRED),
           "message_seed": int(cfg.i4.message_seed), "eval_batch": int(cfg.eval_batch), "versions": versions(),
           "config": {"bits_default": int(cfg.bits), "target_psnr": 40.0, "operating_points": ["matched_40dB", "native"]},
           "models": {}}
    try:
        res["splits"] = split_provenance(cfg)
    except Exception as e:
        res["splits"] = {"error": repr(e)}
    try:
        res["attack_smoke"] = smoke_attacks(D.ImageFolderDataset(split_files(cfg, "val")))
    except Exception as e:
        res["attack_smoke"] = {"status": "failed", "error": repr(e)}
        write_json(out / "i6_public.json", res)
        sampler.stop()
        preserve(cfg, out, upload_log, "attack_smoke_failure")
        raise
    for name in only:
        spec = PUBLIC[name]
        r = {**spec, "pinned_commit": REPOS[spec["repo"]][1], "licence": LICENCE[spec["repo"]]}
        t0 = time.time()
        try:
            wm, ck = load_public(name, Path(cfg.work) / "repos", dev)
            r["checkpoint_sha256"] = sha256(ck)
            st, (points, per_all) = evaluate_public(cfg, wm, spec["bits"], dev)
            np.savez_compressed(out / f"{name}_i6_per_image.npz", **per_all)
            r.update({"status": "ok", "strengths_from_val": st, "points": {"coco": points}})
        except Exception as e:
            r.update({"status": "unavailable" if name == "mbrs30" else "failed", "error": repr(e),
                      "trace": traceback.format_exc()[-3000:]})
        r["seconds"] = time.time() - t0
        r["message_seed"] = int(cfg.i4.message_seed)
        res["models"][name] = r
        res["provenance"] = provenance(cfg)
        write_json(out / "i6_public.json", res)
        r["preserved"] = preserve(cfg, out, upload_log, name)  # metrics, per-image arrays, gpu.csv, versions
        write_json(out / "i6_public.json", res)
    sampler.stop()
    missing = [m for m in REQUIRED if res["models"].get(m, {}).get("status") != "ok"]
    res["status"] = "failed_required_model" if missing else "ok"
    res["missing_required"] = missing
    pip_freeze(out)
    write_json(out / "i6_public.json", res)
    res["final_preserved"] = preserve(cfg, out, upload_log, "final")
    write_json(out / "i6_public.json", res)
    if missing:
        raise SystemExit(f"i6_public: required model(s) not evaluated: {missing}")


STEPS = {"i6_public": step_i6_public}
