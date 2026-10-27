"""Paper I pilot driver (docs/PILOT_P1.md).

Steps::

    python -m trackd_wm.pilot data   # pod A: build (or verify) splits, preprocess, layouts
    python -m trackd_wm.pilot g2     # FIN official checkpoint in the common harness
    python -m trackd_wm.pilot g1     # timing probe: one epoch per trained method, DWT-DCT timing
    python -m trackd_wm.pilot g3     # 20 epochs per method, learning curves, matched-PSNR evaluation
    python -m trackd_wm.pilot smoke  # CPU smoke test on synthetic images (no download)
    python -m trackd_wm.pilot i2     # stage I-2: minimal feasibility experiment (in-pod smoke gate first)
    python -m trackd_wm.pilot i2b    # stage I-2b: GPU evaluation of the archived baseline checkpoints
    python -m trackd_wm.pilot i5_m1  # stage I-5 steps i5_inputs, i5_m1, i5_mbrs, i5_hidden, i5_dwtdct (trackd_wm.i5)

Configuration: ``configs/pilot.yaml`` (OmegaConf; overrides as ``key=value``).
All outputs are written under ``<work>/results/<step>``; nothing is reported
from the pilot in any manuscript (feasibility only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from . import __version__
from . import data as D
from .baselines import LICENCE, REPOS, fetch
from .harness import embed_all, eval_messages, evaluate, matched_strength, summarise
from .sysmon import GPUSampler, Windows, summarise as gpu_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
WMNET = ("proposed", "cdf97", "coupling", "v1", "v2", "v3", "v0", "v1p", "v2l1", "v2l3", "v2o2", "v2s")
BASELINES = ("hidden", "mbrs", "cin", "fin")


# --------------------------------------------------------------------------- helpers
def load_cfg(argv):
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "pilot.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([a for a in argv if "=" in a]))
    return cfg


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance(cfg) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"commit": git_commit(), "trackd_wm": __version__, "torch": torch.__version__, "device": gpu,
            "seed": cfg.seed, "pod": os.environ.get("RUNPOD_POD_ID", "local"), "time_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": OmegaConf.to_container(cfg, resolve=True)}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def set_phase(cfg, phase: str) -> None:
    """Read by the pod's idle watchdog: the idle rule applies only during 'train'."""
    Path(cfg.work).mkdir(parents=True, exist_ok=True)
    (Path(cfg.work) / "phase").write_text(phase)


def push(cfg, step: str) -> None:
    script = REPO_ROOT / "scripts" / "pod" / "push_results.sh"
    if os.environ.get("NO_PUSH") == "1" or not script.exists():
        return
    subprocess.run(["bash", str(script), str(Path(cfg.work) / "results" / step), f"r/pilot-p1-{step}"], check=False)


def cpu_threads() -> int:
    """CPU threads for torch. os.cpu_count() reports the host's cores inside a RunPod container, which
    oversubscribed the evaluation in both I-2b attempts; use the affinity mask, capped by TORCH_THREADS
    (default 8)."""
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1
    return max(1, min(avail, int(os.environ.get("TORCH_THREADS", "8"))))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_files(cfg, split: str):
    root = Path(cfg.data_root) / f"png{cfg.image_size}" / split
    names = [Path(e.name).stem + ".png" for e in D.read_split(D.SPLIT_DIR / f"coco_{split}.txt")]
    return [root / n for n in names]


def layouts(cfg) -> dict:
    """Baseline-specific directory layouts over the same preprocessed PNGs (symlinks)."""
    L = Path(cfg.data_root) / "layouts"
    tr, va = split_files(cfg, "train"), split_files(cfg, "val")
    D.link_layout(tr, L / "hidden" / "train" / "img")
    D.link_layout(va, L / "hidden" / "val" / "img")
    D.link_layout(tr, L / "mbrs" / "train")
    D.link_layout(va, L / "mbrs" / "validation")
    D.link_layout(tr, L / "fin" / "train")
    D.link_layout(va, L / "fin" / "val")
    D.link_layout(tr, L / "cin" / "train")
    D.link_layout(va, L / "cin" / "val")
    return {k: L / k for k in BASELINES}


# --------------------------------------------------------------------------- data
def step_data(cfg):
    set_phase(cfg, "prep")
    out = Path(cfg.work) / "results" / "data"
    raw = Path(cfg.data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for key in ("train", "val"):
        z = raw / f"{key}2017.zip"
        if not z.exists():
            subprocess.run(["aria2c", "-q", "-x", "16", "-s", "16", "-d", str(raw), "-o", z.name, cfg.coco[key]],
                           check=True)
    listing = {k: [n.split("/", 1)[1] for n in subprocess.run(["unzip", "-Z1", str(raw / f"{k}2017.zip")],
                                                              capture_output=True, text=True, check=True).stdout.split()
                   if n.endswith(".jpg")] for k in ("train", "val")}
    have = all((D.SPLIT_DIR / f"coco_{s}.txt").exists() for s in ("train", "val", "test"))
    if have:
        need = {"train": [e.name for s in ("train", "val") for e in D.read_split(D.SPLIT_DIR / f"coco_{s}.txt")],
                "val": [e.name for e in D.read_split(D.SPLIT_DIR / "coco_test.txt")]}
    else:
        need = {"train": D.candidate_names(listing["train"], cfg.seed, cfg.n_candidates.train),
                "val": D.candidate_names(listing["val"], cfg.seed, cfg.n_candidates.val)}
    for k in ("train", "val"):
        lst = raw / f"extract_{k}.txt"
        lst.write_text("\n".join(f"{k}2017/{n}" for n in need[k]) + "\n")
        subprocess.run(f"cat {lst} | xargs -n 500 unzip -q -n -j {raw / f'{k}2017.zip'} -d {raw / k}",
                       shell=True, check=True)
    if not have:
        D.build_coco_splits(raw / "train", raw / "val", D.SPLIT_DIR, listing["train"], listing["val"], cfg.seed)
    rep = {"splits_built_here": not have}
    for s, src in (("train", "train"), ("val", "train"), ("test", "val")):
        entries = D.read_split(D.SPLIT_DIR / f"coco_{s}.txt")
        D.verify_split(entries, raw / src)
        D.prepare(entries, raw / src, Path(cfg.data_root) / f"png{cfg.image_size}" / s, cfg.image_size)
        rep[s] = {"n": len(entries), "sha256_verified": True}
    layouts(cfg)
    for k in ("train", "val"):
        (raw / f"{k}2017.zip").unlink(missing_ok=True)
    rep["seconds"] = time.time() - t0
    rep["provenance"] = provenance(cfg)
    write_json(out / "data.json", rep)
    if not have:  # the split files are pushed for commit to main
        (out / "splits").mkdir(exist_ok=True)
        for s in ("train", "val", "test"):
            (out / "splits" / f"coco_{s}.txt").write_text((D.SPLIT_DIR / f"coco_{s}.txt").read_text())
    push(cfg, "data")


# --------------------------------------------------------------------------- training wrappers
def wmnet_cfg(cfg, name: str):
    return OmegaConf.create({
        "name": name, "seed": cfg.seed, "lr": cfg.wmnet.lr, "batch_size": cfg.wmnet.batch_size,
        "num_workers": cfg.wmnet.num_workers, "strict_determinism": cfg.wmnet.strict_determinism,
        "max_steps": cfg.get("max_steps", None),
        "model": dict(OmegaConf.to_container(cfg.wmnet.common), transform=cfg.wmnet.variants[name],
                      **dict(OmegaConf.to_container(cfg.wmnet.get("model_overrides", {})).get(name, {})),
                      **({"freeze_encoder_transform": True}
                         if name in cfg.wmnet.get("freeze_encoder_transform", []) else {}),
                      **({"decoder_frontend_hidden": int(cfg.wmnet.decoder_frontend_hidden[name])}
                         if name in cfg.wmnet.get("decoder_frontend_hidden", {}) else {})),
    })


def train_method(cfg, name: str, epochs: int, run_dir: Path, lay: dict, log) -> dict:
    """Train one method; returns timing information. Baselines run as subprocesses of their own code."""
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if name in WMNET:
        from .train import train_wmnet

        ds = D.ImageFolderDataset(split_files(cfg, "train"))
        info = train_wmnet(wmnet_cfg(cfg, name), ds, run_dir, device(), epochs, log=log)
    else:
        repo = fetch(name, Path(cfg.work) / "repos")
        data = lay[name]
        cmd = [sys.executable, "-m", "trackd_wm.baselines.launch", name, "--repo", str(repo), "--data", str(data),
               "--work", str(run_dir / "work"), "--epochs", str(epochs), "--seed", str(cfg.seed)]
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        if cfg.get("smoke", False):
            env["TRACKD_SMOKE"] = "1"
        timeout = cfg.get("method_timeout_s", None)
        with open(run_dir / "stdout.log", "w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        info = {"method": name, "returncode": p.returncode, "pinned_commit": REPOS[name][1], "licence": LICENCE[name]}
        if p.returncode != 0:
            raise RuntimeError(f"{name} training failed (see stdout.log)")
    info["wall_seconds"] = time.time() - t0
    info["epoch_seconds"] = epoch_seconds(name, run_dir)
    return info


def checkpoints(name: str, run_dir: Path) -> dict:
    """Map epoch -> checkpoint path for every method's own checkpoint naming."""
    import re

    out = {}
    if name in WMNET:
        for p in (run_dir / "ckpt").glob("epoch_*.pt"):
            out[int(p.stem.split("_")[1])] = p
    elif name == "hidden":
        for p in (run_dir / "work" / "runs").glob("pilot*/checkpoints/pilot--epoch-*.pyt"):
            out[int(re.search(r"epoch-(\d+)", p.name).group(1))] = p
    elif name == "mbrs":
        for p in (run_dir / "work" / "results").glob("MBRS_m64*/models/EC_*.pth"):
            out[int(p.stem.split("_")[1]) + 1] = p  # MBRS counts epochs from 0
    elif name == "fin":
        d = run_dir / "work" / "experiments" / "pilot"
        for p in d.glob("fed_*.pt"):
            out[int(p.stem.rsplit("_", 1)[1]) + 1] = p  # FIN saves after i_epoch (0-based), not after the first
        if not out and (d / "fed.pt").exists():  # fed.pt duplicates the last epoch; used only if nothing else exists
            out[1] = d / "fed.pt"
    elif name == "cin":
        for p in (run_dir / "work").glob("exp_CIN/*/path_checkpoint/epoch-*_step-*-cinNet.pth"):
            out[int(re.search(r"epoch-(\d+)_", p.name).group(1))] = p
    return dict(sorted(out.items()))


def epoch_seconds(name: str, run_dir: Path) -> list:
    """Uniform definition for every method: wall time between consecutive checkpoints divided by the number
    of epochs between them (the run start counts as epoch 0); includes each method's own in-loop validation
    and checkpoint writing. Returns one value per interval."""
    ck = checkpoints(name, run_dir)
    pts = [(e, p.stat().st_mtime) for e, p in ck.items()]
    if (run_dir / "started").exists():
        pts = [(0, (run_dir / "started").stat().st_mtime)] + pts
    return [(t1 - t0) / (e1 - e0) for (e0, t0), (e1, t1) in zip(pts, pts[1:]) if e1 > e0]


def load_watermarker(cfg, name: str, run_dir: Path, ckpt: Path):
    from .baselines import wrappers as W

    dev = device()
    if name in WMNET:
        from .train import load_wmnet

        return load_wmnet(ckpt, dev)
    repo = Path(cfg.work) / "repos" / name
    if name == "hidden":
        return W.HiDDeNWatermarker(repo, ckpt.parent.parent, ckpt, dev)
    if name == "mbrs":
        return W.MBRSWatermarker(repo, ckpt, dev)
    if name == "fin":
        return W.FINWatermarker(repo, ckpt, dev)
    if name == "cin":
        return W.CINWatermarker(repo, run_dir / "work" / "opt_pilot.yml", ckpt, dev)
    raise ValueError(name)


# --------------------------------------------------------------------------- steps
def step_g2(cfg):
    """Harness validation: official FIN 64-bit checkpoint (DIV2K-trained) on the COCO test split."""
    set_phase(cfg, "eval")
    out = Path(cfg.work) / "results" / "g2"
    from .baselines.wrappers import FINWatermarker

    repo = fetch("fin", Path(cfg.work) / "repos")
    wm = FINWatermarker(repo, repo / "experiments" / "JPEG" / "FED.pt", device())
    ds = D.ImageFolderDataset(split_files(cfg, "test"))
    msgs = eval_messages(len(ds), cfg.bits, cfg.seed)
    per = evaluate(wm, ds, msgs, ["clean", "jpeg50", "jpeg30"], 1.0, cfg.eval_batch, device())
    s = summarise(per)
    s["published_fin_q50_bitacc"] = 0.9971
    s["criterion_P2a_pass"] = abs(s["bitacc_jpeg50"] - 0.9971) <= 0.02
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fin_official_per_image.npz", **per)
    write_json(out / "g2.json", {"summary": s, "checkpoint": "experiments/JPEG/FED.pt @ f77222f",
                                 "provenance": provenance(cfg)})
    push(cfg, "g2")


def step_g1(cfg):
    """Timing probe: one epoch per trained method; DWT-DCT embedding/extraction time on 100 images."""
    out = Path(cfg.work) / "results" / "g1"
    out.mkdir(parents=True, exist_ok=True)
    lay = layouts(cfg)
    sampler = GPUSampler(out / "gpu.csv").start()
    win = Windows(out / "windows.json")
    res = {}
    for name in cfg.methods:
        set_phase(cfg, "train")
        run = Path(cfg.work) / "runs" / "g1" / name
        run.mkdir(parents=True, exist_ok=True)
        (run / "started").touch()
        win.start(name, "g1")
        try:
            info = train_method(cfg, name, 1, run, lay, log=lambda s: None)
            win.end("ok")
            info["gpu"] = gpu_summary(out / "gpu.csv", win.items[-1]["start"], win.items[-1]["end"])
            res[name] = info
        except Exception as e:  # recorded, never hidden (condition 6)
            win.end("failed", error=repr(e))
            res[name] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        write_json(out / "g1.json", {"methods": res, "provenance": provenance(cfg)})
        push(cfg, "g1")
    set_phase(cfg, "eval")
    from .baselines.wrappers import DWTDCTWatermarker

    repo = fetch("dwtdct", Path(cfg.work) / "repos")
    wm = DWTDCTWatermarker(repo)
    ds = D.ImageFolderDataset(split_files(cfg, "val")[:100])
    msgs = eval_messages(len(ds), cfg.bits, cfg.seed)
    t0 = time.time()
    per = evaluate(wm, ds, msgs, ["clean"], 1.0, 50, "cpu")
    res["dwtdct"] = {"seconds_100_images_embed_extract": time.time() - t0, "bitacc_clean": float(per["bitacc_clean"].mean()),
                     "psnr_rgb": float(per["psnr_rgb"].mean())}
    sampler.stop()
    write_json(out / "g1.json", {"methods": res, "provenance": provenance(cfg)})
    push(cfg, "g1")


def eval_curve_and_final(cfg, name: str, run: Path, out: Path) -> dict:
    set_phase(cfg, "eval")
    ds = D.ImageFolderDataset(split_files(cfg, "val"))
    msgs = eval_messages(len(ds), cfg.bits, cfg.seed)
    ck = checkpoints(name, run)
    curve = []
    for ep in [e for e in ck if e % cfg.eval_every == 0 or e == max(ck)]:
        wm = load_watermarker(cfg, name, run, ck[ep])
        s = summarise(evaluate(wm, ds, msgs, ["clean", "jpeg50", "jpeg30"], 1.0, cfg.eval_batch, device()))
        s["epoch"] = ep
        curve.append(s)
    last = max(ck)
    wm = load_watermarker(cfg, name, run, ck[last])
    ms = matched_strength(wm, ds, msgs, cfg.target_psnr, cfg.eval_batch, device())
    cache = ms.pop("_cache")
    per_m = evaluate(wm, ds, msgs, ["clean", "jpeg50", "jpeg30"], ms["strength"], cfg.eval_batch, device(), cache)
    per_n = evaluate(wm, ds, msgs, ["clean", "jpeg50", "jpeg30"], 1.0, cfg.eval_batch, device(), cache)
    np.savez_compressed(out / f"{name}_val_per_image_matched40.npz", **per_m)
    np.savez_compressed(out / f"{name}_val_per_image_native.npz", **per_n)
    return {"curve": curve, "final_epoch": last, "matched": {**ms, **summarise(per_m)}, "native": summarise(per_n)}


def step_g3(cfg):
    out = Path(cfg.work) / "results" / "g3"
    out.mkdir(parents=True, exist_ok=True)
    lay = layouts(cfg)
    sampler = GPUSampler(out / "gpu.csv").start()
    win = Windows(out / "windows.json")
    path = out / "g3.json"
    res = json.loads(path.read_text())["methods"] if path.exists() else {}
    for name in cfg.methods:
        if res.get(name, {}).get("status") == "ok":
            continue  # resume after a restart
        run = Path(cfg.work) / "runs" / "g3" / name
        run.mkdir(parents=True, exist_ok=True)
        (run / "started").touch()
        set_phase(cfg, "train")
        win.start(name, "g3-train")
        try:
            info = train_method(cfg, name, cfg.g3_epochs, run, lay, log=lambda s: None)
            win.end("ok")
            info["gpu_train"] = gpu_summary(out / "gpu.csv", win.items[-1]["start"], win.items[-1]["end"])
            win.start(name, "g3-eval")
            info["eval"] = eval_curve_and_final(cfg, name, run, out)
            win.end("ok")
            info["status"] = "ok"
        except Exception as e:
            win.end("failed", error=repr(e))
            info = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res[name] = info
        write_json(path, {"methods": res, "provenance": provenance(cfg)})
        archive_checkpoints(cfg, name, run)
        push(cfg, "g3")
    if res.get("dwtdct", {}).get("status") != "ok":
        res["dwtdct"] = eval_dwtdct(cfg, out)
        write_json(path, {"methods": res, "provenance": provenance(cfg)})
    sampler.stop()
    write_json(path, {"methods": res, "provenance": provenance(cfg)})
    push(cfg, "g3")


def eval_dwtdct(cfg, out: Path) -> dict:
    """Training-free DWT-DCT on the validation split (CPU): native scale 36 and matched 40 dB."""
    set_phase(cfg, "eval")
    try:
        from .baselines.wrappers import DWTDCTWatermarker

        repo = fetch("dwtdct", Path(cfg.work) / "repos")
        wm = DWTDCTWatermarker(repo)
        ds = D.ImageFolderDataset(split_files(cfg, "val"))
        msgs = eval_messages(len(ds), cfg.bits, cfg.seed)
        t0 = time.time()
        ms = matched_strength(wm, ds, msgs, cfg.target_psnr, cfg.eval_batch, "cpu")
        cache = ms.pop("_cache")
        t_embed = time.time() - t0
        per_m = evaluate(wm, ds, msgs, ["clean", "jpeg50", "jpeg30"], ms["strength"], cfg.eval_batch, "cpu", cache)
        per_n = evaluate(wm, ds, msgs, ["clean", "jpeg50", "jpeg30"], 1.0, cfg.eval_batch, "cpu", cache)
        np.savez_compressed(out / "dwtdct_val_per_image_matched40.npz", **per_m)
        np.savez_compressed(out / "dwtdct_val_per_image_native.npz", **per_n)
        return {"status": "ok", "pinned_commit": REPOS["dwtdct"][1], "licence": LICENCE["dwtdct"],
                "embed_seconds_1000_images_cpu": t_embed, "seconds_total": time.time() - t0,
                "eval": {"matched": {**ms, **summarise(per_m)}, "native": summarise(per_n)}}
    except Exception as e:
        return {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}


def archive_checkpoints(cfg, name: str, run: Path) -> None:
    """Even-epoch checkpoints and configs into one tar per method (uploaded to the Zenodo draft, never to git)."""
    import tarfile

    arc = Path(cfg.work) / "archive"
    arc.mkdir(exist_ok=True)
    ck = checkpoints(name, run)
    with tarfile.open(arc / f"g3_{name}_checkpoints.tar", "w") as t:
        for ep, p in ck.items():
            if ep % cfg.eval_every == 0 or ep == max(ck):
                t.add(p, arcname=f"{name}/epoch_{ep:03d}{p.suffix}")
        for extra in list(run.glob("work/runs/**/*.pickle")) + list(run.glob("**/opt_pilot.yml")) + \
                list(run.glob("**/train_settings.json")) + list(run.glob("train_info.json")):
            t.add(extra, arcname=f"{name}/{extra.relative_to(run)}")


# --------------------------------------------------------------------------- strength sweep (evaluation only)
SWEEP_GRID = (0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00)
SWEEP_TARGETS = (30.0, 35.0, 40.0, 45.0)
# Cheapest first, so that the one-hour limit cuts only the most expensive baselines. The pilot coupling control
# is excluded: its training collapsed (zero residual), so there is no watermark to scale.
SWEEP_METHODS = ("proposed", "cdf97", "dwtdct", "fin", "hidden", "mbrs", "cin")
KEEP_ALL_EPOCHS = ("fin",)  # numerical-stability investigation (instruction of 2026-10-21, 20:56 UTC)
SWEEP_EXCLUDED = {"coupling": "invalid control: training collapsed in G3 (zero residual)"}


def prepare_val_only(cfg) -> dict:
    """Validation split only (from train2017): download, SHA256 verification, preprocessing. No push."""
    raw = Path(cfg.data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    entries = D.read_split(D.SPLIT_DIR / "coco_val.txt")
    dst = Path(cfg.data_root) / f"png{cfg.image_size}" / "val"
    if not all((dst / (Path(e.name).stem + ".png")).exists() for e in entries):
        z = raw / "train2017.zip"
        if not z.exists():
            subprocess.run(["aria2c", "-q", "-x", "16", "-s", "16", "-d", str(raw), "-o", z.name, cfg.coco.train],
                           check=True)
        lst = raw / "extract_val.txt"
        lst.write_text("\n".join(f"train2017/{e.name}" for e in entries) + "\n")
        subprocess.run(f"cat {lst} | xargs -n 500 unzip -q -n -j {z} -d {raw / 'train'}", shell=True, check=True)
        D.verify_split(entries, raw / "train")
        D.prepare(entries, raw / "train", dst, cfg.image_size)
        z.unlink(missing_ok=True)
    return {"n": len(entries), "sha256_verified": True, "seconds": time.time() - t0}


def fetch_zenodo_checkpoints(cfg, names, dest: Path, prefix: str = "pilot_p1_g3_", keep_all: bool = False,
                             sha256: dict | None = None) -> dict:
    """Download g3 checkpoint archives from the Paper I Zenodo draft (read only; the token is never printed).

    Only the final-epoch checkpoint and the configuration files of each method are extracted. ``sha256`` maps a
    name to the expected SHA-256 of its archive; a mismatch removes the download and raises."""
    import re
    import tarfile
    import urllib.request

    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        raise RuntimeError("ZENODO_TOKEN not set")
    dep_id = os.environ.get("ZENODO_DEPOSITION", "23534235")
    h = {"Authorization": "Bearer " + tok}

    def get_to(url, path):
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=1800) as r, open(path, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)

    with urllib.request.urlopen(urllib.request.Request(f"https://zenodo.org/api/deposit/depositions/{dep_id}",
                                                       headers=h), timeout=300) as r:
        dep = json.loads(r.read())
    files = {f["filename"]: f for f in dep.get("files", [])}
    dest.mkdir(parents=True, exist_ok=True)
    got = {}
    for n in names:
        # archives written by scripts/upload_artifacts.sh (Orchestron) carry no prefix; accept both names
        key = next((k for k in (f"{prefix}{n}_checkpoints.tar", f"{n}_checkpoints.tar") if k in files), None)
        if key is None:
            got[n] = {"archive": None}
            continue
        p = dest / key
        t_dl = time.time()
        get_to(files[key]["links"]["download"], p)
        t_dl = time.time() - t_dl
        # integrity check against the Zenodo record (instruction of 2026-10-22, 09:36 UTC)
        import hashlib

        md5 = hashlib.md5()  # not "h": h holds the request headers used by get_to
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                md5.update(blk)
        want = str(files[key].get("checksum") or "").split(":")[-1]
        if want and md5.hexdigest() != want:
            p.unlink()
            raise RuntimeError(f"{key}: MD5 mismatch (got {md5.hexdigest()}, Zenodo {want})")
        sha = None
        if sha256 and sha256.get(n):
            from .data import sha256_file

            sha = sha256_file(p)
            if sha != sha256[n]:
                p.unlink()
                raise RuntimeError(f"{key}: SHA-256 mismatch (got {sha}, expected {sha256[n]})")
        with tarfile.open(p) as t:
            members = t.getmembers()
            eps = [int(m.group(1)) for x in members if (m := re.search(r"/epoch_(\d+)\.", x.name))]
            last = max(eps) if eps else None
            keep = [x for x in members if not re.search(r"/epoch_\d+\.", x.name)
                    or (last is not None and f"/epoch_{last:03d}." in x.name)
                    or n in KEEP_ALL_EPOCHS or keep_all]
            # "data" filter: no absolute paths, links out of dest or device files (Python >= 3.10.12 / 3.11.4)
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, members=keep, filter="data")
            else:
                t.extractall(dest, members=keep)
        got[n] = {"archive": key, "bytes": p.stat().st_size, "md5_zenodo": files[key].get("checksum"),
                  "md5_verified": bool(want) and md5.hexdigest() == want, "download_seconds": t_dl,
                  "final_epoch": last, **({"sha256_verified": True} if sha else {})}
        p.unlink()
    return got


def archived_watermarker(cfg, name: str, root: Path, dev=None):
    """Watermarker for the final archived g3 checkpoint of ``name`` (layout written by archive_checkpoints)."""
    from .baselines import wrappers as W

    if name == "dwtdct":
        return W.DWTDCTWatermarker(fetch("dwtdct", Path(cfg.work) / "repos")), "library default (scale 36)"
    d = root / name
    cks = sorted(d.glob("epoch_*"))
    if not cks:
        raise FileNotFoundError(f"no archived checkpoint for {name}")
    ck = cks[-1]
    dev = dev if dev is not None else torch.device("cpu")
    if name in WMNET:
        from .train import load_wmnet

        return load_wmnet(ck, dev), f"{name}/{ck.name}"
    repo = fetch(name, Path(cfg.work) / "repos")
    if name == "hidden":
        opts = next(d.rglob("options-and-config.pickle"))
        return W.HiDDeNWatermarker(repo, opts.parent, ck, dev), f"{name}/{ck.name}"
    if name == "mbrs":
        return W.MBRSWatermarker(repo, ck, dev), f"{name}/{ck.name}"
    if name == "fin":
        return W.FINWatermarker(repo, ck, dev), f"{name}/{ck.name}"
    if name == "cin":
        return W.CINWatermarker(repo, next(d.rglob("opt_pilot.yml")), ck, dev), f"{name}/{ck.name}"
    raise ValueError(name)


def checkpoint_finiteness(root: Path, name: str) -> dict:
    """Count non-finite values in the archived final checkpoint (diagnoses diverged training; no training)."""
    d = root / name
    cks = sorted(d.glob("epoch_*"))
    if not cks:
        return {"checked": False}
    obj = torch.load(cks[-1], map_location="cpu", weights_only=False)
    n_tot, n_bad, bad_keys = 0, 0, []

    def walk(o, key=""):
        nonlocal n_tot, n_bad
        if torch.is_tensor(o):
            if o.is_floating_point():
                n_tot += o.numel()
                b = int((~torch.isfinite(o)).sum())
                if b:
                    n_bad += b
                    if len(bad_keys) < 20:
                        bad_keys.append(key)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{key}[{i}]")

    walk(obj)
    return {"checked": True, "file": cks[-1].name, "float_values": n_tot, "nonfinite_values": n_bad,
            "first_nonfinite_keys": bad_keys}


@torch.no_grad()
def stability_by_checkpoint(cfg, name: str, root: Path, ds, msgs, n_images: int = 50) -> list:
    """For every archived checkpoint of ``name``: non-finite parameters, and non-finite outputs of embedding and
    decoding on the first ``n_images`` validation images (CPU). Separates training divergence (non-finite
    weights) from inference instability (finite weights, non-finite outputs). Evaluation only."""
    from .baselines import wrappers as W

    repo = fetch(name, Path(cfg.work) / "repos")
    x = torch.stack([ds[i] for i in range(n_images)])
    m = msgs[:n_images]
    rows = []
    for ck in sorted((root / name).glob("epoch_*")):
        row = {"checkpoint": ck.name}
        try:
            tmp = root / f"_{name}_one"
            tmp.mkdir(exist_ok=True)
            for old in tmp.glob("*"):
                old.unlink()
            (tmp / ck.name).symlink_to(ck.resolve())
            row["params"] = checkpoint_finiteness(root, f"_{name}_one")
            wm = W.FINWatermarker(repo, ck, torch.device("cpu")) if name == "fin" else None
            if wm is None:
                raise ValueError(name)
            xw = wm.embed(x, m)
            row["embed_nonfinite_frac"] = float((~torch.isfinite(xw)).float().mean())
            xw_f = torch.nan_to_num(xw.clamp(0, 1), nan=0.0)
            bits = wm.extract(xw_f)
            row["extract_nonfinite_frac"] = float((~torch.isfinite(bits)).float().mean())
        except Exception as e:
            row["error"] = repr(e)
        rows.append(row)
    return rows


@torch.no_grad()
def oracle_amplified_accuracy(wm, cache, msgs, s: float, attacks, batch: int = 50) -> dict:
    """Diagnostic separating an unrecoverable signal from a decoder untrained on weak signals (evaluation only).

    With the cover x known (oracle, non-blind), the attacked residual is amplified back to training energy:
    y = Q8(x) + (A(Q8(x + s r)) - A(Q8(x))) / s, and decoded with the unchanged decoder. If accuracy recovers,
    the attack preserved the information and the blind decoder failed because the residual was weak; if not,
    the information was removed by the attack (or by quantisation) at that strength.
    """
    from . import metrics as M
    from .attacks import ATTACKS, quantize_8bit

    cover, marked = cache["cover"], cache["marked"]
    out = {}
    for a in attacks:
        if hasattr(wm, "set_attack"):
            wm.set_attack(a)
        accs = []
        for i in range(0, len(cover), batch):
            c = cover[i:i + batch]
            x0 = quantize_8bit(c)
            xs = quantize_8bit(c + s * (marked[i:i + batch] - c))
            y = (x0 + (ATTACKS[a](xs) - ATTACKS[a](x0)) / max(s, 1e-8)).clamp(0, 1)
            accs.append(M.bit_accuracy(wm.extract(y).cpu(), msgs[i:i + batch]))
        out[f"bitacc_{a}"] = float(torch.cat(accs).mean())
    return out


def step_sweep(cfg, methods=SWEEP_METHODS, step: str = "sweep", dev=None, validity=None, lazy_fetch: bool = False):
    """Embedding-strength sweep on existing g3 checkpoints (evaluation only; Option A, approved 2026-10-21).

    For each valid method with an archived checkpoint (cheapest first), the final-epoch model embeds the
    validation split once; the residual is scaled to the strengths that reach mean RGB PSNR targets of 30-45 dB
    (bisection, as in the matched-40 evaluation), to the native strength (s = 1) and, for the three transform
    variants, to a fixed strength grid. Metrics: PSNR (RGB, Y), SSIM and bit accuracy without attack and after
    real JPEG Q50 and Q30; residual spectrum and JPEG survival; oracle-amplified accuracy (decoder-sensitivity
    diagnostic). Results are pushed after every method. No training.

    Stage I-2b reuses this step on a GPU (``methods``, ``step``, ``dev``); ``validity`` records the documented
    validity status of each method next to its numbers.
    """
    set_phase(cfg, "eval")
    torch.set_num_threads(cpu_threads())
    dev = dev if dev is not None else torch.device("cpu")
    out = Path(cfg.work) / "results" / step
    out.mkdir(parents=True, exist_ok=True)
    attacks = ["clean", "jpeg50", "jpeg30"]
    res = {"data": prepare_val_only(cfg), "grid": list(SWEEP_GRID), "targets_db": list(SWEEP_TARGETS),
           "attacks": attacks, "split": "val", "methods": {}, "excluded": dict(SWEEP_EXCLUDED),
           "device": str(dev), "validity": dict(validity or {})}
    ck_root = Path(cfg.work) / "g3_ckpt"
    # lazy_fetch (stage I-2b rerun): each archive is downloaded and verified just before its method is evaluated
    res["checkpoints"] = {} if lazy_fetch else fetch_zenodo_checkpoints(cfg, [m for m in methods if m != "dwtdct"],
                                                                        ck_root)
    write_json(out / f"{step}.json", res)
    ds = D.ImageFolderDataset(split_files(cfg, "val"))
    msgs = eval_messages(len(ds), cfg.bits, cfg.seed)
    for name in methods:
        t0 = time.time()
        try:
            if lazy_fetch and name != "dwtdct":
                res["checkpoints"][name] = fetch_zenodo_checkpoints(cfg, [name], ck_root)[name]
                write_json(out / f"{step}.json", res)
                if not res["checkpoints"][name].get("archive"):
                    raise FileNotFoundError(f"no Zenodo archive for {name}")
            fin_chk = checkpoint_finiteness(ck_root, name) if name != "dwtdct" else {"checked": False}
            res.setdefault("checkpoint_finiteness", {})[name] = fin_chk
            if name in KEEP_ALL_EPOCHS:
                res.setdefault("stability_by_checkpoint", {})[name] = stability_by_checkpoint(cfg, name, ck_root, ds,
                                                                                           msgs)
                write_json(out / f"{step}.json", res)
            wm, ck = archived_watermarker(cfg, name, ck_root, dev)
            cache = embed_all(wm, ds, msgs, cfg.eval_batch, dev)
            strengths = {"native": 1.0}
            if name in WMNET:
                strengths.update({f"grid_{s:.2f}": s for s in SWEEP_GRID})
            for tgt in SWEEP_TARGETS:
                strengths[f"matched_{tgt:.0f}dB"] = matched_strength_cached(cache, tgt)
            points, per_all = {}, {}
            for label, s in strengths.items():
                per = evaluate(wm, ds, msgs, attacks, s, cfg.eval_batch, dev, cache)
                points[label] = {"strength": s, **summarise(per)}
                per_all.update({f"{label}__{k}": v for k, v in per.items()})
            oracle = {f"matched_{t:.0f}dB": oracle_amplified_accuracy(wm, cache, msgs,
                                                                      strengths[f"matched_{t:.0f}dB"], attacks)
                      for t in SWEEP_TARGETS}
            np.savez_compressed(out / f"{name}_val_per_image_{step}.npz", **per_all)
            res["methods"][name] = {"status": "ok", "checkpoint": ck, "points": points, "oracle_amplified": oracle,
                                    "residual": residual_diagnostics(cache, strengths["matched_40dB"]),
                                    "seconds": time.time() - t0}
        except Exception as e:
            res["methods"][name] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:],
                                    "seconds": time.time() - t0}
        res["provenance"] = provenance(cfg)
        write_json(out / f"{step}.json", res)
        push(cfg, step)


@torch.no_grad()
def residual_diagnostics(cache, s40: float, bins: int = 8, batch: int = 100) -> dict:
    """Where the watermark residual lives and how much of it survives JPEG (evaluation only).

    * radial spectrum: share of residual energy in ``bins`` equal-width bands of normalised radial frequency
      (0 = DC, 1 = Nyquist), mean over images and channels, at native strength (the spectrum shape does not
      depend on the strength);
    * survival: <JPEG_q(x + s r) - JPEG_q(x), d> / <d, d> with d = Q8(x + s r) - Q8(x), i.e. the share of the
      embedded residual that JPEG preserves along its own direction (1 = preserved, 0 = removed), at s = 1 and
      at the matched-40 dB strength, for q = 50 and 30.
    """
    from .attacks import ATTACKS, quantize_8bit

    cover, marked = cache["cover"], cache["marked"]
    r = marked - cover
    n, _, h, w = r.shape
    fy = torch.fft.fftfreq(h).abs().view(-1, 1)
    fx = torch.fft.fftfreq(w).abs().view(1, -1)
    rad = torch.sqrt(fy ** 2 + fx ** 2) / 0.5 / (2 ** 0.5)  # 0..1
    idx = (rad * bins).long().clamp(max=bins - 1)
    spec = torch.zeros(bins)
    for i in range(0, n, batch):
        e = torch.fft.fft2(r[i:i + batch]).abs().pow(2).sum(dim=(0, 1))
        spec += torch.bincount(idx.flatten(), weights=e.flatten(), minlength=bins)
    spec = (spec / spec.sum()).tolist()
    surv = {}
    for label, s in (("native", 1.0), ("matched_40dB", s40)):
        for q in ("jpeg50", "jpeg30"):
            num, den = 0.0, 0.0
            for i in range(0, n, batch):
                c = cover[i:i + batch]
                xs = quantize_8bit(c + s * r[i:i + batch])
                x0 = quantize_8bit(c)
                d = xs - x0
                num += float(((ATTACKS[q](xs) - ATTACKS[q](x0)) * d).sum())
                den += float(d.pow(2).sum())
            surv[f"{label}_{q}"] = num / max(den, 1e-20)
    return {"radial_spectrum_share": spec, "jpeg_survival": surv}


def matched_strength_cached(cache, target_psnr: float, iters: int = 30) -> float:
    """Bisection of :func:`trackd_wm.harness.matched_strength` on an existing embedding cache."""
    from .attacks import quantize_8bit
    from . import metrics as M

    cover, marked = cache["cover"], cache["marked"]

    def mean_psnr(s):
        return float(M.psnr_rgb(quantize_8bit(cover + s * (marked - cover)), cover).mean())

    lo, hi = 0.0, 1.0
    while mean_psnr(hi) > target_psnr and hi < 64:
        hi *= 2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_psnr(mid) > target_psnr else (lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- stage I-2 (feasibility) and I-2b
I2_TARGETS = (30.0, 35.0, 40.0, 45.0)
I2_LABELS = ("native",) + tuple(f"matched_{t:.0f}dB" for t in I2_TARGETS)
I2B_METHODS = ("fin", "hidden", "mbrs", "cin")  # cheapest first, so the stage limit cuts the largest models last
I2B_VALIDITY = {
    "fin": "invalid: training diverged in G3 (non-finite outputs from epoch 2); evaluated for the per-checkpoint "
           "stability investigation only; not evidence of inferiority",
    "hidden": "under-trained (20 epochs; accuracy still rising at epoch 20); preliminary only; not evidence of "
              "inferiority",
    "mbrs": "valid pilot run (P), single seed; about 74 times the parameters of the proposed network",
    "cin": "corrected evaluation: oracle noise-branch selection per attack (disclosed; harness fix ff85ce5); "
           "the defective pilot evaluation is kept for traceability only",
}


def state_dict_finite(ckpt: Path) -> bool:
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    return all(bool(torch.isfinite(v).all()) for v in st["net"].values()
               if torch.is_tensor(v) and v.is_floating_point())


@torch.no_grad()
def reconstruction_error(t, x: torch.Tensor) -> float:
    """Maximum absolute perfect-reconstruction error of analysis followed by synthesis (no embedding)."""
    return float((t.synthesis(t.analysis(x)) - x).abs().max())


def paired_stats(a, b, reps: int, seed: int = 0) -> dict:
    """Paired image-level comparison of two single-seed models. Conditional on the trained models: it captures
    image-level variation only, not training-seed variation (revised protocol, PHASE1_EXECUTION_PLAN.md 6.3)."""
    from scipy.stats import wilcoxon

    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for i in range(0, reps, 500):
        k = min(500, reps - i)
        boot[i:i + k] = d[rng.integers(0, len(d), size=(k, len(d)))].mean(axis=1)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:  # all differences zero
        p = None
    return {"mean_diff": float(d.mean()), "ci95_image_bootstrap": [float(np.percentile(boot, 2.5)),
                                                                  float(np.percentile(boot, 97.5))],
            "wilcoxon_p_conditional": p, "n_images": int(len(d))}


def i2_variant_cfg(cfg, name: str, smoke: bool):
    c = wmnet_cfg(cfg, name)
    c.objective = str(cfg.get("i3b", {}).get("objective_override", {}).get(name, cfg.i2.objective))
    c.target_psnr = float(cfg.i2.target_psnr)
    c.finite_guard = True
    c.max_steps = int(cfg.i2.smoke_steps) if smoke else None
    return c


def i2_run(cfg, name: str, out: Path, run_root: Path, smoke: bool) -> dict:
    """Train one I-2 variant and evaluate it on the validation split (or a subset in the smoke gate)."""
    import shutil

    from .attacks import EXTENDED_ATTACKS
    from .train import load_wmnet, train_wmnet

    t_all = time.time()
    run = run_root / name
    if run.exists():
        shutil.rmtree(run)  # no partial resume: a restarted variant is retrained from the start
    run.mkdir(parents=True)
    (run / "started").touch()
    ds_tr = D.ImageFolderDataset(split_files(cfg, "train"))
    ds_va = D.ImageFolderDataset(split_files(cfg, "val"))
    if smoke:
        ds_va = torch.utils.data.Subset(ds_va, range(int(cfg.i2.smoke_images)))
    msgs = eval_messages(len(ds_va), cfg.bits, cfg.seed)
    dev = device()
    set_phase(cfg, "train")
    t0 = time.time()
    info = train_wmnet(i2_variant_cfg(cfg, name, smoke), ds_tr, run, dev, 1 if smoke else int(cfg.i2.epochs),
                       log=lambda s: None)
    info["wall_seconds_train"] = time.time() - t0
    set_phase(cfg, "eval")
    ck = checkpoints(name, run)
    finite = {e: state_dict_finite(p) for e, p in ck.items()}
    ok_eps = [e for e, f in finite.items() if f]
    if not ok_eps:
        raise RuntimeError(f"{name}: no finite checkpoint")
    curve = []
    for ep in ok_eps:
        if ep % cfg.eval_every == 0 or ep == max(ok_eps):
            wm = load_wmnet(ck[ep], dev)
            c = summarise(evaluate(wm, ds_va, msgs, ["clean", "jpeg50", "jpeg30"], 1.0, cfg.eval_batch, dev))
            c["epoch"] = ep
            curve.append(c)
    last = max(ok_eps)
    wm = load_wmnet(ck[last], dev)
    cache = embed_all(wm, ds_va, msgs, cfg.eval_batch, dev)
    strengths = {"native": 1.0}
    for t in I2_TARGETS:
        strengths[f"matched_{t:.0f}dB"] = matched_strength_cached(cache, t)
    points, per_all = {}, {}
    for label, sv in strengths.items():
        per = evaluate(wm, ds_va, msgs, list(EXTENDED_ATTACKS), sv, cfg.eval_batch, dev, cache)
        points[label] = {"strength": sv, **summarise(per)}
        per_all.update({f"{label}__{k}": v for k, v in per.items()})
    oracle = {f"matched_{t:.0f}dB": oracle_amplified_accuracy(wm, cache, msgs, strengths[f"matched_{t:.0f}dB"],
                                                              ["clean", "jpeg50", "jpeg30"]) for t in I2_TARGETS}
    xs = torch.stack([ds_va[i] for i in range(min(50, len(ds_va)))]).to(dev)
    pr = {"encoder_transform": reconstruction_error(wm.net.encoder.t, xs),
          "decoder_transform": reconstruction_error(wm.net.decoder.t, xs)}
    if not smoke:
        np.savez_compressed(out / f"{name}_val_per_image.npz", **per_all)
    ep_log = info.get("epochs", [])
    e2 = next((c["bitacc_clean"] for c in curve if c["epoch"] == 2), None)
    acc = {"finite": not info.get("stopped_nonfinite", False) and all(finite.values()),
           "train_psnr_last_epoch": ep_log[-1].get("train_psnr_clamped") if ep_log else None,
           "val_bitacc_clean_native_epoch2": e2}
    acc["train_psnr_within_40pm05"] = (acc["train_psnr_last_epoch"] is not None
                                       and abs(acc["train_psnr_last_epoch"] - float(cfg.i2.target_psnr)) <= 0.5)
    acc["no_collapse"] = e2 is not None and e2 > 0.60
    acc["valid"] = bool(acc["finite"] and acc["train_psnr_within_40pm05"] and acc["no_collapse"])
    return {"transform": cfg.wmnet.variants[name], "freeze_encoder_transform":
            name in cfg.wmnet.get("freeze_encoder_transform", []), "train": info, "checkpoint_finite": finite,
            "final_epoch": last, "curve": curve, "points": points, "oracle_amplified": oracle,
            "residual": residual_diagnostics(cache, strengths["matched_40dB"]),
            "perfect_reconstruction_max_abs_error": pr, "acceptance": acc, "seconds": time.time() - t_all}


def archive_i2(cfg, name: str, run: Path, stage: str = "i2") -> None:
    """All epoch checkpoints and the training log of one I-2 variant (Zenodo draft, never git)."""
    import tarfile

    arc = Path(cfg.work) / "archive"
    arc.mkdir(exist_ok=True)
    with tarfile.open(arc / f"{stage}_{name}_checkpoints.tar", "w") as t:
        for ep, pth in checkpoints(name, run).items():
            t.add(pth, arcname=f"{name}/epoch_{ep:03d}{pth.suffix}")
        if (run / "train_info.json").exists():
            t.add(run / "train_info.json", arcname=f"{name}/train_info.json")


def i2_analysis(cfg, out: Path, res: dict) -> dict:
    from .attacks import EXTENDED_ATTACKS

    V = res["variants"]
    ok = [v for v in cfg.i2.variants if V.get(v, {}).get("status") == "ok"]
    per = {v: np.load(out / f"{v}_val_per_image.npz") for v in ok}
    a = {"note": "single seed (0), validation split; image-level intervals and tests are conditional on the trained "
                 "models and do not capture training-seed variation", "pairs": {}}
    for A, B in (("v2", "v1"), ("v2", "v3"), ("v3", "v1")):
        if A in per and B in per:
            a["pairs"][f"{A}-{B}"] = {f"{lab}/{att}": paired_stats(per[A][f"{lab}__bitacc_{att}"],
                                                                   per[B][f"{lab}__bitacc_{att}"],
                                                                   int(cfg.i2.bootstrap))
                                      for lab in I2_LABELS for att in EXTENDED_ATTACKS}
    if "v1" in per and "v2" in per:
        p = a["pairs"]["v2-v1"]["matched_40dB/jpeg50"]
        valid = all(V[v]["acceptance"]["valid"] for v in ("v1", "v2"))
        a["go_no_go"] = {
            "criterion": "go if mean bit accuracy of V2 minus V1 >= +0.010 after JPEG Q50 at matched 40 dB "
                         "(validation split, seed 0); pre-registered",
            "mean_diff": p["mean_diff"], "ci95_image_bootstrap": p["ci95_image_bootstrap"],
            "runs_valid": valid,
            "decision": ("go" if p["mean_diff"] >= float(cfg.i2.go_margin) else "no-go") if valid
            else "not decidable: acceptance criteria not met",
            "interpretation": "preliminary feasibility evidence, not confirmation of superiority"}
    return a


def step_i2(cfg):
    """Stage I-2: V1-V3 under PSNR projection, seed 0, 20 epochs; in-pod smoke gate first."""
    out = Path(cfg.work) / "results" / "i2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i2.json"
    res = json.loads(path.read_text()) if path.exists() else {"variants": {}}
    sampler = GPUSampler(out / "gpu.csv").start()
    if res.get("smoke", {}).get("status") != "ok":
        sm = {}
        try:
            for v in cfg.i2.variants:
                r = i2_run(cfg, v, out, Path(cfg.work) / "runs" / "i2_smoke", smoke=True)
                sm[v] = {"finite": r["acceptance"]["finite"], "points_native": r["points"]["native"],
                         "pr_error": r["perfect_reconstruction_max_abs_error"], "seconds": r["seconds"]}
            res["smoke"] = {"status": "ok", "variants": sm}
        except Exception as e:
            res["smoke"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        push(cfg, "i2")
        if res["smoke"]["status"] != "ok":
            sampler.stop()
            raise SystemExit("I-2 smoke gate failed; full runs not started")
    for v in cfg.i2.variants:
        if res["variants"].get(v, {}).get("status") == "ok":
            continue
        try:
            r = i2_run(cfg, v, out, Path(cfg.work) / "runs" / "i2", smoke=False)
            r["status"] = "ok"
        except Exception as e:
            r = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["variants"][v] = r
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        if (Path(cfg.work) / "runs" / "i2" / v).exists():
            archive_i2(cfg, v, Path(cfg.work) / "runs" / "i2" / v)
        push(cfg, "i2")
    try:
        res["analysis"] = i2_analysis(cfg, out, res)
    except Exception as e:
        res["analysis"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
    sampler.stop()
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    push(cfg, "i2")


def step_i2b(cfg):
    """Stage I-2b (rerun approved 2026-10-22, 09:36 UTC): GPU evaluation of the archived G3 checkpoints of
    HiDDeN, MBRS and CIN (corrected oracle branch). FIN is skipped; each archive is downloaded and MD5-verified
    just before its evaluation."""
    methods = tuple(cfg.get("i2b", {}).get("methods", I2B_METHODS))
    print(f"I-2b torch threads: {cpu_threads()} (affinity {len(os.sched_getaffinity(0))})", flush=True)
    step_sweep(cfg, methods=methods, step="i2b", dev=device(), validity=I2B_VALIDITY, lazy_fetch=True)


# --------------------------------------------------------------------------- stage I-3a
def _leaves(b) -> list:
    if torch.is_tensor(b):
        return [b]
    return [t for q in b for t in _leaves(q)]


def _rebuild(template, it):
    if torch.is_tensor(template):
        return next(it)
    return type(template)(_rebuild(q, it) for q in template)


def _spectral_norm(f, x0: torch.Tensor, iters: int, seed: int = 0) -> float:
    """Largest singular value of the Jacobian of f at x0, float64 power iteration on J^T J.

    J v by a central finite difference (step 1e-6 relative to |x0|), J^T u by reverse-mode autograd; this
    avoids relying on forward-mode support of every operator."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
    v = v / v.norm()
    eps = 1e-6 * max(1.0, float(x0.abs().max()))
    sig = 0.0
    for _ in range(iters):
        with torch.no_grad():
            jv = (f(x0 + eps * v) - f(x0 - eps * v)) / (2 * eps)
        sig = float(jv.norm())
        x = x0.clone().requires_grad_(True)
        (w,) = torch.autograd.grad(f(x), x, grad_outputs=jv)
        v = w / (w.norm() + 1e-300)
    return sig


def recon_diagnostics(t, xs: torch.Tensor, iters: int, n_power: int) -> dict:
    """Theoretical invertibility vs numerical reconstruction of one transform (evaluation only, CPU).

    * reconstruction S(A(x)) - x and inverse consistency A(S(b)) - b, in float32 and float64, per image;
    * largest coefficient magnitude (dynamic range of the sub-bands);
    * conditioning: sigma_max of the analysis Jacobian J_A and of the synthesis Jacobian J_S at b = A(x),
      in float64; because S = A^-1 exactly, sigma_min(J_A) = 1 / sigma_max(J_S) and the condition number of
      J_A is sigma_max(J_A) * sigma_max(J_S).
    """
    import copy

    out = {}
    for dname, dt in (("float32", torch.float32), ("float64", torch.float64)):
        m = copy.deepcopy(t).to("cpu", dt).eval()
        errs, rels, ics, coef = [], [], [], 0.0
        with torch.no_grad():
            for i in range(0, len(xs), 25):
                x = xs[i:i + 25].to(dt)
                b = m.analysis(x)
                r = m.synthesis(b)
                errs.append((r - x).abs().flatten(1).max(1).values)
                rels.append((r - x).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1))
                b2 = m.analysis(r)
                ics.append(torch.stack([(p - q).abs().flatten(1).max(1).values
                                        for p, q in zip(_leaves(b), _leaves(b2))]).max(0).values)
                coef = max(coef, max(float(q.abs().max()) for q in _leaves(b)))
        e, rl, ic = torch.cat(errs).double(), torch.cat(rels).double(), torch.cat(ics).double()
        out[dname] = {"recon_max_abs": float(e.max()), "recon_median_of_max_abs": float(e.median()),
                      "recon_p95_of_max_abs": float(torch.quantile(e, 0.95)), "recon_mean_rel_l2": float(rl.mean()),
                      "worst_images": [int(k) for k in torch.argsort(e, descending=True)[:5]],
                      "inverse_consistency_max_abs": float(ic.max()), "max_abs_coefficient": coef,
                      "max_abs_input": float(xs.abs().max())}
    m = copy.deepcopy(t).to("cpu", torch.float64).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    cond = []
    try:
        cond = _conditioning(m, xs, iters, n_power)
    except Exception as e:  # diagnostics must not abort the stage
        cond = [{"error": repr(e)}]
    out["conditioning_float64"] = cond
    return out


def _conditioning(m, xs, iters: int, n_power: int) -> list:
    cond = []
    for k in range(min(n_power, len(xs))):
        x0 = xs[k:k + 1].double()
        b0 = m.analysis(x0)
        tmpl = b0
        fa = lambda x: torch.cat([q.flatten() for q in _leaves(m.analysis(x))])
        fs = lambda z: m.synthesis(_rebuild_shaped(tmpl, z)).flatten()
        z0 = torch.cat([q.flatten() for q in _leaves(b0)])
        sa = _spectral_norm(fa, x0, iters)
        ss = _spectral_norm(fs, z0, iters)
        cond.append({"image": k, "sigma_max_analysis": sa, "sigma_max_synthesis": ss, "condition_number": sa * ss})
    return cond


def _rebuild_shaped(tmpl, z: torch.Tensor):
    leaves = _leaves(tmpl)
    parts = torch.split(z, [q.numel() for q in leaves])
    return _rebuild(tmpl, iter(p.view_as(q) for p, q in zip(parts, leaves)))


def fetch_i2_results(out: Path) -> dict:
    """Per-image arrays and the summary of stage I-2 from its results branch (read only)."""
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    rec = {}
    subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "-q", f"https://x-access-token@github.com/"
                    f"{os.environ.get('GH_REPO', 'yazanjer/learned-decoder-wavelet-watermarking')}.git",
                    "r/pilot-p1-i2"], check=True, env=env)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "FETCH_HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    for f in ("i2.json", "v1_val_per_image.npz", "v2_val_per_image.npz", "v3_val_per_image.npz"):
        blob = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"FETCH_HEAD:{f}"], capture_output=True,
                              check=True).stdout
        (out / f).write_bytes(blob)
    rec["branch_commit"] = head
    return rec


def i3a_analysis(cfg, out: Path, prev: Path, res: dict) -> dict:
    from .attacks import EXTENDED_ATTACKS

    per = {v: np.load(prev / f"{v}_val_per_image.npz") for v in ("v1", "v2", "v3")}
    for v in cfg.i3a.variants:
        if res["variants"].get(v, {}).get("status") == "ok":
            per[v] = np.load(out / f"{v}_val_per_image.npz")
    a = {"note": "single seed (0), validation split; image-level intervals and tests are conditional on the trained "
                 "models and are preliminary; they are not evidence of cross-seed reproducibility", "pairs": {}}
    for A, B in (("v2", "v1p"), ("v1p", "v1"), ("v2", "v0"), ("v0", "v1"), ("v2", "v3"), ("v3", "v1p")):
        if A in per and B in per:
            a["pairs"][f"{A}-{B}"] = {f"{lab}/{att}": paired_stats(per[A][f"{lab}__bitacc_{att}"],
                                                                   per[B][f"{lab}__bitacc_{att}"],
                                                                   int(cfg.i2.bootstrap))
                                      for lab in I2_LABELS for att in EXTENDED_ATTACKS}
    m = float(cfg.i3a.margin)
    if "v2-v1p" in a["pairs"]:
        d = a["pairs"]["v2-v1p"]["matched_40dB/jpeg50"]["mean_diff"]
        a["capacity_attribution"] = {
            "rule": "decision instruction (9): V1+ matches V2 if |V2 - V1+| < 0.010 after JPEG Q50 at matched 40 dB",
            "mean_diff_v2_minus_v1p": d, "ci95_image_bootstrap":
                a["pairs"]["v2-v1p"]["matched_40dB/jpeg50"]["ci95_image_bootstrap"],
            "outcome": ("V1+ matches V2 (gain not attributable to learned lifting at this stage)" if abs(d) < m
                        else "V2 ahead of V1+ by at least the margin" if d >= m else "V1+ ahead of V2 by at least the margin"),
            "v1p_valid": res["variants"]["v1p"]["acceptance"]["valid"]}
    return a


def reconstruction_study(cfg) -> dict:
    """Numerical reconstruction of the archived I-2 transforms (V2, V3 decoder and encoder; fixed CDF 9/7
    reference): float32/float64 error, inverse consistency, conditioning, and the float32/float64 error over
    the training epochs."""
    rec = {}
    try:
        from .train import load_wmnet

        ck_root = Path(cfg.work) / "i2_ckpt"
        rec["archives"] = {}
        for v in ("v2", "v3"):
            rec["archives"].update(fetch_zenodo_checkpoints(cfg, [v], ck_root, prefix="pilot_p1_i2_", keep_all=True))
        ds = D.ImageFolderDataset(split_files(cfg, "val"))
        xs = torch.stack([ds[i] for i in range(int(cfg.i3a.recon_images))])
        it, npw = int(cfg.i3a.power_iters), int(cfg.i3a.power_images)
        rec["reference_fixed_cdf97"] = recon_diagnostics(build_fixed("lift_cdf97", cfg), xs, it, npw)
        for v in ("v2", "v3"):
            eps = sorted((ck_root / v).glob("epoch_*.pt"))
            wm = load_wmnet(eps[-1], torch.device("cpu"))
            rec[v] = {"checkpoint": eps[-1].name,
                      "decoder_transform": recon_diagnostics(wm.net.decoder.t, xs * 2 - 1, it, npw),
                      "encoder_transform": recon_diagnostics(wm.net.encoder.t, xs, it, 1)}
            traj = []
            for e in eps:  # float32 / float64 reconstruction error over training (50 images)
                w = load_wmnet(e, torch.device("cpu"))
                d = recon_diagnostics(w.net.decoder.t, xs[:50] * 2 - 1, 0, 0)
                traj.append({"checkpoint": e.name, "f32_max": d["float32"]["recon_max_abs"],
                             "f64_max": d["float64"]["recon_max_abs"],
                             "max_abs_coefficient": d["float32"]["max_abs_coefficient"]})
            rec[v]["trajectory"] = traj
        rec["status"] = "ok"
    except Exception as e:
        rec.update({"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]})
    return rec


def prepare_val_subset(cfg, n: int) -> dict:
    """First n validation images only: individual downloads from the COCO image server, SHA256-verified
    against the split file, preprocessed exactly as in the pilot (avoids the 18 GB archive)."""
    import urllib.request

    t0 = time.time()
    entries = D.read_split(D.SPLIT_DIR / "coco_val.txt")[:n]
    raw = Path(cfg.data_root) / "raw" / "train"
    raw.mkdir(parents=True, exist_ok=True)
    for e in entries:
        f = raw / e.name
        if not f.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{e.name}", f)
    D.verify_split(entries, raw)
    D.prepare(entries, raw, Path(cfg.data_root) / f"png{cfg.image_size}" / "val", cfg.image_size)
    return {"n": len(entries), "sha256_verified": True, "seconds": time.time() - t0}


def step_i3a_recon(cfg):
    """Rerun of the I-3a reconstruction study alone (after the variable-shadowing fix), evaluation only, CPU."""
    set_phase(cfg, "eval")
    torch.set_num_threads(cpu_threads())
    out = Path(cfg.work) / "results" / "i3a_recon"
    out.mkdir(parents=True, exist_ok=True)
    res = {"data": prepare_val_subset(cfg, int(cfg.i3a.recon_images))}
    res["reconstruction"] = reconstruction_study(cfg)
    res["provenance"] = provenance(cfg)
    write_json(out / "recon.json", res)
    push(cfg, "i3a_recon")


def step_i3a(cfg):
    """Stage I-3a: V0 (original design) and V1+ (capacity control) under the I-2 protocol, plus the
    numerical-reconstruction investigation of the learned transforms. In-pod smoke gate first."""
    out = Path(cfg.work) / "results" / "i3a"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i3a.json"
    res = json.loads(path.read_text()) if path.exists() else {"variants": {}}
    sampler = GPUSampler(out / "gpu.csv").start()
    prev = out / "i2_reference"
    try:
        res["i2_reference"] = fetch_i2_results(prev)
    except Exception as e:
        res["i2_reference"] = {"status": "failed", "error": repr(e)}
    if res.get("smoke", {}).get("status") != "ok":
        sm = {}
        try:
            for v in cfg.i3a.variants:
                r = i2_run(cfg, v, out, Path(cfg.work) / "runs" / "i3a_smoke", smoke=True)
                sm[v] = {"finite": r["acceptance"]["finite"], "points_native": r["points"]["native"],
                         "params_trainable": r["train"]["params_trainable"], "seconds": r["seconds"]}
            res["smoke"] = {"status": "ok", "variants": sm}
        except Exception as e:
            res["smoke"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        push(cfg, "i3a")
        if res["smoke"]["status"] != "ok":
            sampler.stop()
            raise SystemExit("I-3a smoke gate failed; full runs not started")
    # 1. reconstruction investigation of the archived I-2 learned transforms (cheap; first)
    if "reconstruction" not in res:
        res["reconstruction"] = reconstruction_study(cfg)
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        push(cfg, "i3a")
    # 2. V0 and V1+ under the identical I-2 protocol
    for v in cfg.i3a.variants:
        if res["variants"].get(v, {}).get("status") == "ok":
            continue
        try:
            r = i2_run(cfg, v, out, Path(cfg.work) / "runs" / "i3a", smoke=False)
            r["status"] = "ok"
            if v == "v0":
                from .train import load_wmnet

                ds = D.ImageFolderDataset(split_files(cfg, "val"))
                xs = torch.stack([ds[i] for i in range(int(cfg.i3a.recon_images))])
                wm = load_wmnet(sorted((Path(cfg.work) / "runs" / "i3a" / v / "ckpt").glob("epoch_*.pt"))[-1],
                                torch.device("cpu"))
                r["reconstruction"] = {
                    "decoder_transform": recon_diagnostics(wm.net.decoder.t, xs * 2 - 1, int(cfg.i3a.power_iters),
                                                           int(cfg.i3a.power_images)),
                    "encoder_transform": recon_diagnostics(wm.net.encoder.t, xs, int(cfg.i3a.power_iters),
                                                           int(cfg.i3a.power_images))}
        except Exception as e:
            r = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["variants"][v] = r
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        if (Path(cfg.work) / "runs" / "i3a" / v).exists():
            archive_i2(cfg, v, Path(cfg.work) / "runs" / "i3a" / v, stage="i3a")
        push(cfg, "i3a")
    try:
        res["analysis"] = i3a_analysis(cfg, out, prev, res)
    except Exception as e:
        res["analysis"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
    sampler.stop()
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    push(cfg, "i3a")


def i3b_analysis(cfg, out: Path, prev: Path, res: dict) -> dict:
    from .attacks import EXTENDED_ATTACKS

    per = {"v2": np.load(prev / "v2_val_per_image.npz")}
    for v in cfg.i3b.variants:
        if res["variants"].get(v, {}).get("status") == "ok":
            per[v] = np.load(out / f"{v}_val_per_image.npz")
    a = {"note": "single seed (0), validation split; configuration selection only, never a claim", "pairs": {}}
    for v in cfg.i3b.variants:
        if v in per:
            a["pairs"][f"{v}-v2"] = {f"{lab}/{att}": paired_stats(per[v][f"{lab}__bitacc_{att}"],
                                                                  per["v2"][f"{lab}__bitacc_{att}"],
                                                                  int(cfg.i2.bootstrap))
                                     for lab in I2_LABELS for att in EXTENDED_ATTACKS}
    valid = [v for v in cfg.i3b.variants if res["variants"].get(v, {}).get("acceptance", {}).get("valid")]
    scores = {v: a["pairs"][f"{v}-v2"]["matched_40dB/jpeg50"]["mean_diff"] for v in valid}
    best = max(scores, key=scores.get) if scores else None
    a["selection"] = {"rule": "carry the best valid variant (Q50, matched 40 dB) into I-4 only if it exceeds V2 by "
                              ">= 1.0 pp; otherwise keep V2", "diff_vs_v2": scores, "best": best,
                      "selected": best if best is not None and scores[best] >= float(cfg.i3b.margin) else "v2"}
    return a


def step_i3b(cfg):
    """Stage I-3b: single-seed V2 ablations (one factor each) under the I-2 protocol; in-pod smoke gate first."""
    out = Path(cfg.work) / "results" / "i3b"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i3b.json"
    res = json.loads(path.read_text()) if path.exists() else {"variants": {}}
    sampler = GPUSampler(out / "gpu.csv").start()
    prev = out / "i2_reference"
    try:
        res["i2_reference"] = fetch_i2_results(prev)
    except Exception as e:
        res["i2_reference"] = {"status": "failed", "error": repr(e)}
    if res.get("smoke", {}).get("status") != "ok":
        sm = {}
        try:
            for v in cfg.i3b.variants:
                r = i2_run(cfg, v, out, Path(cfg.work) / "runs" / "i3b_smoke", smoke=True)
                sm[v] = {"finite": r["acceptance"]["finite"], "points_native": r["points"]["native"],
                         "params_trainable": r["train"]["params_trainable"], "seconds": r["seconds"]}
            res["smoke"] = {"status": "ok", "variants": sm}
        except Exception as e:
            res["smoke"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        push(cfg, "i3b")
        if res["smoke"]["status"] != "ok":
            sampler.stop()
            raise SystemExit("I-3b smoke gate failed; full runs not started")
    for v in cfg.i3b.variants:
        if res["variants"].get(v, {}).get("status") == "ok":
            continue
        try:
            r = i2_run(cfg, v, out, Path(cfg.work) / "runs" / "i3b", smoke=False)
            r["status"] = "ok"
        except Exception as e:
            r = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["variants"][v] = r
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        if (Path(cfg.work) / "runs" / "i3b" / v).exists():
            archive_i2(cfg, v, Path(cfg.work) / "runs" / "i3b" / v, stage="i3b")
        push(cfg, "i3b")
    try:
        res["analysis"] = i3b_analysis(cfg, out, prev, res)
    except Exception as e:
        res["analysis"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
    sampler.stop()
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    push(cfg, "i3b")


def hier_bootstrap(d: np.ndarray, reps: int, seed: int = 0, alpha: float = 0.05) -> list:
    """Two-level bootstrap of the mean of d (shape seeds x images): seeds with replacement, then images within
    each resampled seed. Returns the percentile (1 - alpha) interval."""
    rng = np.random.default_rng(seed)
    S, N = d.shape
    out = np.empty(reps)
    for r in range(reps):
        ss = rng.integers(0, S, S)
        out[r] = np.mean([d[s, rng.integers(0, N, N)].mean() for s in ss])
    return [float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))]


def mixed_model_test(a: np.ndarray, b: np.ndarray) -> dict:
    """acc ~ method + (1|image) + (1|seed) + (1|seed:method); a, b shape seeds x images. statsmodels MixedLM
    with seed as group and variance components for image and seed:method. Returns the method effect (A - B),
    its standard error and p-value (normal approximation)."""
    import pandas as pd
    import statsmodels.formula.api as smf

    S, N = a.shape
    rows = []
    for m, arr in (("A", a), ("B", b)):
        for s in range(S):
            for i in range(N):
                rows.append((arr[s, i], m, s, i))
    df = pd.DataFrame(rows, columns=["acc", "method", "seed", "image"])
    df["sm"] = df["seed"].astype(str) + "_" + df["method"]
    md = smf.mixedlm("acc ~ C(method, Treatment('B'))", df, groups="seed",
                     vc_formula={"image": "0 + C(image)", "sm": "0 + C(sm)"})
    f = md.fit(reml=True, method="lbfgs")
    k = [n for n in f.params.index if n.startswith("C(method")][0]
    return {"effect": float(f.params[k]), "se": float(f.bse[k]), "p": float(f.pvalues[k]), "converged": bool(f.converged)}


def i4_run(cfg, name: str, seed: int, out: Path, run_root: Path, smoke: bool = False) -> dict:
    """Train one (variant, seed) under the I-2 protocol, fix the strengths on the validation split, evaluate the
    test split once at those strengths (PROTOCOL_I4.md)."""
    import shutil

    from .attacks import EXTENDED_ATTACKS
    from .train import load_wmnet, train_wmnet

    t_all = time.time()
    run = run_root / f"{name}_s{seed}"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    (run / "started").touch()
    c = i2_variant_cfg(cfg, name, smoke=smoke)
    c.seed = int(seed)
    dev = device()
    set_phase(cfg, "train")
    info = train_wmnet(c, D.ImageFolderDataset(split_files(cfg, "train")), run, dev,
                       1 if smoke else int(cfg.i2.epochs), log=lambda s_: None)
    set_phase(cfg, "eval")
    ck = checkpoints(name, run)
    finite = {e: state_dict_finite(p) for e, p in ck.items()}
    last = max(e for e, f in finite.items() if f)
    wm = load_wmnet(ck[last], dev)
    ms = int(cfg.i4.message_seed)
    va = D.ImageFolderDataset(split_files(cfg, "val"))
    if smoke:
        va = torch.utils.data.Subset(va, range(50))
    cache_v = embed_all(wm, va, eval_messages(len(va), cfg.bits, ms), cfg.eval_batch, dev)
    strengths = {f"matched_{t:.0f}dB": matched_strength_cached(cache_v, t) for t in cfg.i4.targets}
    val_e2 = summarise(evaluate(load_wmnet(ck[min(2, last)], dev), va, eval_messages(len(va), cfg.bits, ms),
                                ["clean"], 1.0, cfg.eval_batch, dev))
    del cache_v
    # first and only test-split access (after freezing); the smoke gate uses validation images instead
    te = va if smoke else D.ImageFolderDataset(split_files(cfg, "test"))
    msgs = eval_messages(len(te), cfg.bits, ms)
    cache = embed_all(wm, te, msgs, cfg.eval_batch, dev)
    points, per_all = {}, {}
    for lab, sv in strengths.items():
        per = evaluate(wm, te, msgs, list(EXTENDED_ATTACKS), sv, cfg.eval_batch, dev, cache)
        points[lab] = {"strength": sv, **summarise(per)}
        per_all.update({f"{lab}__{k}": v for k, v in per.items()})
    if not smoke:
        np.savez_compressed(out / f"{name}_s{seed}_test_per_image.npz", **per_all)
    xs = torch.stack([va[i] for i in range(50)])
    recon = {"decoder_transform": recon_diagnostics(wm.net.decoder.t, xs * 2 - 1, 10, 1)}
    ep = info.get("epochs", [])
    acc = {"finite": not info.get("stopped_nonfinite", False) and all(finite.values()),
           "train_psnr_last_epoch": ep[-1].get("train_psnr_clamped") if ep else None,
           "val_bitacc_clean_native_epoch2": val_e2["bitacc_clean"]}
    acc["train_psnr_within_40pm05"] = (acc["train_psnr_last_epoch"] is not None
                                       and abs(acc["train_psnr_last_epoch"] - float(cfg.i2.target_psnr)) <= 0.5)
    acc["no_collapse"] = acc["val_bitacc_clean_native_epoch2"] > 0.60
    acc["valid"] = bool(acc["finite"] and acc["train_psnr_within_40pm05"] and acc["no_collapse"])
    return {"variant": name, "seed": seed, "train": info, "final_epoch": last, "strengths_from_val": strengths,
            "test_points": points, "reconstruction": recon, "acceptance": acc, "seconds": time.time() - t_all}


def seed_t_interval(sd, alpha: float = 0.05) -> list:
    """Paired t-interval over the seed-level mean differences. With few seeds the hierarchical bootstrap and the
    normal-approximation LMM understate between-seed uncertainty, so this interval is always reported next to the
    frozen decision (HANDOVER_TRACKD_P1.md, Section 5, item 6); it never changes the decision."""
    from scipy import stats

    sd = np.asarray(sd, dtype=float)
    if len(sd) < 2:
        return [None, None]
    t = stats.t.ppf(1 - alpha / 2, len(sd) - 1) * sd.std(ddof=1) / np.sqrt(len(sd))
    return [float(sd.mean() - t), float(sd.mean() + t)]


def holm(ps: dict) -> dict:
    """Holm step-down adjusted p-values for {name: p}; None entries are left out."""
    order = sorted([h for h in ps if ps[h] is not None], key=lambda h: ps[h])
    adj, running = {}, 0.0
    for j, h in enumerate(order):
        running = max(running, min(1.0, (len(order) - j) * ps[h]))
        adj[h] = running
    return adj


# Secondary endpoints (PROTOCOL_I4.md, section 3.5): (label, per-image key in the I-4 test arrays, transform).
# "tpr" turns the per-image bit accuracy into detection at 0.1 % FPR under the analytic null of independent fair
# bits. The stored arrays hold no decodes of unmarked images, so the protocol's empirical TPR is not available; the
# analytic version is reported descriptively and kept out of the Holm family (PROTOCOL_I4.md, notes of 2026-10-24).
I4_SECONDARY_ENDPOINTS = (
    ("q30_40dB", "matched_40dB__bitacc_jpeg30", None),
    ("q70_40dB", "matched_40dB__bitacc_jpeg70", None),
    ("resize_40dB", "matched_40dB__bitacc_resize05", None),
    ("q50_35dB", "matched_35dB__bitacc_jpeg50", None),
    ("q30_35dB", "matched_35dB__bitacc_jpeg30", None),
    ("q50_45dB", "matched_45dB__bitacc_jpeg50", None),
    ("q30_45dB", "matched_45dB__bitacc_jpeg30", None),
    ("clean_40dB", "matched_40dB__bitacc_clean", None),
    ("tpr_fpr1e-3_q50_40dB", "matched_40dB__bitacc_jpeg50", "tpr"),
    ("ssim_40dB", "matched_40dB__ssim", None),
)


def detection_threshold(bits: int, fpr: float = 1e-3) -> int:
    """Smallest number of matching bits k with P(Binomial(bits, 0.5) >= k) <= fpr."""
    from scipy import stats

    k = 0
    while stats.binom.sf(k - 1, bits, 0.5) > fpr:
        k += 1
    return k


def i4_secondary(cfg, out: Path, seeds: list) -> dict:
    """Secondary Holm family over all (comparison, endpoint) pairs; reported only, never used for decisions.
    Needs the mixed models, so it runs only with I4_MIXED=1 (CPU pod, after approval)."""
    bits = int(cfg.bits)
    k = detection_threshold(bits)
    load = lambda v, key: np.stack([np.load(out / f"{v}_s{s}_test_per_image.npz")[key] for s in seeds])
    fam = {}
    for h, (A, B) in {"H1": ("v2", "v1p"), "H2": ("v2", "v3")}.items():
        for lab, key, tr in I4_SECONDARY_ENDPOINTS:
            name = f"{h}:{lab}"
            try:
                x, y = load(A, key), load(B, key)
            except (KeyError, FileNotFoundError) as e:
                fam[name] = {"status": "missing", "error": repr(e)}
                continue
            if tr == "tpr":
                x, y = (np.rint(x * bits) >= k).astype(float), (np.rint(y * bits) >= k).astype(float)
            d = x - y
            r = {"comparison": f"{A} - {B}", "key": key, "theta_hat": float(d.mean()),
                 "ci95": hier_bootstrap(d, int(cfg.i4.bootstrap)), "seed_diffs": d.mean(1).tolist(),
                 "seed_paired_t_ci95": seed_t_interval(d.mean(1))}
            if tr == "tpr":
                r["null"] = f"analytic: >= {k} of {bits} bits correct, Binomial({bits}, 0.5) tail <= 1e-3"
                r["in_holm_family"] = False
            try:
                r["mixed_model"] = mixed_model_test(x, y)
            except Exception as e:
                r["mixed_model"] = {"error": repr(e)}
            fam[name] = r
    for name, p in holm({n: r.get("mixed_model", {}).get("p") for n, r in fam.items()
                         if r.get("in_holm_family", True)}).items():
        fam[name]["p_holm_secondary"] = p
    return {"family": fam, "seeds_used": list(seeds), "used_for_decisions": False,
            "tpr_threshold_bits": k}


def i4_analysis(cfg, out: Path, res: dict) -> dict:
    # whole seeds only (PROTOCOL_I4.md, section 5): a seed enters every comparison only if all variants completed
    seeds = [s for s in cfg.i4.seeds
             if all(res["runs"].get(f"{v}_s{s}", {}).get("status") == "ok" for v in cfg.i4.variants)]
    have = lambda v: len(seeds) >= 2
    load = lambda v, key: np.stack([np.load(out / f"{v}_s{s}_test_per_image.npz")[key] for s in seeds])
    a = {"family_primary": {}, "secondary": {}, "seeds_used": list(seeds)}
    m = float(cfg.i4.margin)
    pairs = {"H1": ("v2", "v1p"), "H2": ("v2", "v3")}
    for h, (A, B) in pairs.items():
        if not (have(A) and have(B)):
            a["family_primary"][h] = {"status": "incomplete"}
            continue
        x, y = load(A, "matched_40dB__bitacc_jpeg50"), load(B, "matched_40dB__bitacc_jpeg50")
        d = x - y
        r = {"theta_hat": float(d.mean()), "ci95": hier_bootstrap(d, int(cfg.i4.bootstrap)),
             "ci90": hier_bootstrap(d, int(cfg.i4.bootstrap), alpha=0.10),
             "seed_means_A": x.mean(1).tolist(), "seed_means_B": y.mean(1).tolist(),
             "seed_diffs": d.mean(1).tolist(), "seed_paired_t_ci95": seed_t_interval(d.mean(1))}
        if os.environ.get("I4_MIXED", "0") == "1":  # 10,000-row model with a 1,000-level component: CPU pod
            try:
                r["mixed_model"] = mixed_model_test(x, y)
            except Exception as e:
                r["mixed_model"] = {"error": repr(e)}
        else:
            r["mixed_model"] = {"deferred": "computed after the run on a CPU pod (I4_MIXED=1), same code"}
        a["family_primary"][h] = r
    ps = {h: r["mixed_model"].get("p") for h, r in a["family_primary"].items() if "mixed_model" in r}
    for h, p in holm(ps).items():
        a["family_primary"][h]["p_holm"] = p
    if os.environ.get("I4_MIXED", "0") == "1" and os.environ.get("I4_SECONDARY", "0") == "1" and len(seeds) >= 2:
        a["secondary"] = i4_secondary(cfg, out, seeds)
    for h, r in a["family_primary"].items():
        if "theta_hat" not in r:
            continue
        p = r.get("p_holm")
        lo, hi = r["ci95"]
        lo90, hi90 = r["ci90"]
        if p is not None and p < 0.05 and lo > 0 and r["theta_hat"] >= m:
            v = "advantage"
        elif p is not None and p < 0.05 and hi < 0 and r["theta_hat"] <= -m:
            v = "disadvantage"
        elif -m < lo90 and hi90 < m:
            v = "equivalent"
        else:
            v = "inconclusive"
        r["decision"] = v
        r["ci95_half_width"] = (hi - lo) / 2
        r["precise_enough"] = r["ci95_half_width"] <= m
    return a


def step_i4(cfg):
    """Stage I-4 confirmatory comparison (PROTOCOL_I4.md). Requires I4_FROZEN=1 in the environment, set only at
    an approved launch; seeds run in order so that a stage-limit stop loses whole seeds."""
    if os.environ.get("I4_FROZEN") != "1":
        raise SystemExit("I-4 is not authorised in this environment (I4_FROZEN != 1)")
    torch.set_num_threads(cpu_threads())  # evaluation runs on CPU threads too (JPEG, SSIM); see cpu_threads()
    print(f"I-4 torch threads: {torch.get_num_threads()} (affinity {len(os.sched_getaffinity(0))})", flush=True)
    out = Path(cfg.work) / "results" / "i4"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i4.json"
    res = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    sampler = GPUSampler(out / "gpu.csv").start()
    if res.get("smoke", {}).get("status") != "ok":  # in-pod smoke gate: every variant, 5 steps, validation subset
        sm = {}
        try:
            for v in cfg.i4.variants:
                r = i4_run(cfg, v, 0, out, Path(cfg.work) / "runs" / "i4_smoke", smoke=True)
                sm[v] = {"finite": r["acceptance"]["finite"], "params": r["train"]["params_trainable"],
                         "strengths": r["strengths_from_val"], "seconds": r["seconds"]}
            res["smoke"] = {"status": "ok", "variants": sm}
        except Exception as e:
            res["smoke"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        push(cfg, "i4")
        if res["smoke"]["status"] != "ok":
            sampler.stop()
            raise SystemExit("I-4 smoke gate failed; full runs not started")
    for s in cfg.i4.seeds:
        for v in cfg.i4.variants:
            key = f"{v}_s{s}"
            if res["runs"].get(key, {}).get("status") == "ok":
                continue
            try:
                r = i4_run(cfg, v, int(s), out, Path(cfg.work) / "runs" / "i4")
                r["status"] = "ok"
            except Exception as e:
                r = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
            res["runs"][key] = r
            res["provenance"] = provenance(cfg)
            write_json(path, res)
            run = Path(cfg.work) / "runs" / "i4" / key
            if run.exists():
                archive_i2(cfg, v, run, stage=f"i4_s{s}")
            push(cfg, "i4")
    try:
        res["analysis"] = i4_analysis(cfg, out, res)
    except Exception as e:
        res["analysis"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
    sampler.stop()
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    push(cfg, "i4")


def step_i4_final(cfg):
    """Final I-4 analysis on a CPU pod (PROTOCOL_I4.md, implementation note of 2026-10-23): fetches i4.json and the
    per-image test arrays from r/pilot-p1-i4, fits the mixed models (I4_MIXED=1), applies Holm and the four-way
    decision, and adds the descriptive analyses (seed-level paired t-interval; conditional Wilcoxon; empirical-
    logit LMM sensitivity analysis). No training, no new evaluation."""
    from scipy import stats

    set_phase(cfg, "eval")
    torch.set_num_threads(cpu_threads())
    out = Path(cfg.work) / "results" / "i4_final"
    src = out / "i4_branch"
    head = fetch_i4_branch(src)
    res = json.loads((src / "i4.json").read_text())
    os.environ["I4_MIXED"] = "1"
    final = {"source_commit": head, "analysis": i4_analysis(cfg, src, res)}
    seeds = list(cfg.i4.seeds)
    done = [s_ for s_ in seeds if all(res["runs"].get(f"{v}_s{s_}", {}).get("status") == "ok" for v in cfg.i4.variants)]
    final["seeds_complete"] = done
    extra = {}
    for h, (A, B) in {"H1": ("v2", "v1p"), "H2": ("v2", "v3")}.items():
        if len(done) < 2:
            continue
        x = np.stack([np.load(src / f"{A}_s{s_}_test_per_image.npz")["matched_40dB__bitacc_jpeg50"] for s_ in done])
        y = np.stack([np.load(src / f"{B}_s{s_}_test_per_image.npz")["matched_40dB__bitacc_jpeg50"] for s_ in done])
        sd = (x - y).mean(1)
        t = stats.t.ppf(0.975, len(sd) - 1) * sd.std(ddof=1) / np.sqrt(len(sd))
        e = {"seed_paired_t_ci95": [float(sd.mean() - t), float(sd.mean() + t)],
             "wilcoxon_seed_averaged_conditional_p": float(stats.wilcoxon((x - y).mean(0)).pvalue)
             if np.any((x - y).mean(0) != 0) else None}
        lg = lambda a: np.log((a * 64 + 0.5) / (64 - a * 64 + 0.5))  # empirical logit of the per-image bit count
        try:
            e["empirical_logit_lmm"] = mixed_model_test(lg(x), lg(y))
        except Exception as ex:
            e["empirical_logit_lmm"] = {"error": repr(ex)}
        extra[h] = e
    final["descriptive_and_sensitivity"] = extra
    final["provenance"] = provenance(cfg)
    write_json(out / "i4_final.json", {k: v for k, v in final.items()})
    import shutil

    shutil.rmtree(src, ignore_errors=True)
    push(cfg, "i4_final")


def step_i4_secondary(cfg):
    """Secondary-endpoint family of I-4 (PROTOCOL_I4.md, section 3.5) on a CPU pod, after approval. Reads the frozen
    I-4 arrays from r/pilot-p1-i4 and writes results/i4_secondary only; the primary analysis and its decisions in
    r/pilot-p1-i4_final are not recomputed or overwritten. Reported, never used for decisions."""
    set_phase(cfg, "eval")
    torch.set_num_threads(cpu_threads())
    out = Path(cfg.work) / "results" / "i4_secondary"
    src = out / "i4_branch"
    head = fetch_i4_branch(src)
    res = json.loads((src / "i4.json").read_text())
    seeds = [s_ for s_ in cfg.i4.seeds
             if all(res["runs"].get(f"{v}_s{s_}", {}).get("status") == "ok" for v in cfg.i4.variants)]
    sec = {"source_commit": head, "secondary": i4_secondary(cfg, src, seeds), "provenance": provenance(cfg)}
    write_json(out / "i4_secondary.json", sec)
    import shutil

    shutil.rmtree(src, ignore_errors=True)
    push(cfg, "i4_secondary")


def compare_per_image(a: Path, b: Path) -> dict:
    """Reproduction check of a retrained run against the frozen per-image arrays: maximum absolute difference per
    array and whether every array is bitwise identical. Descriptive only; the frozen arrays stay authoritative."""
    x, y = np.load(a), np.load(b)
    common = sorted(set(x.files) & set(y.files))
    diff = {k: float(np.abs(x[k].astype(float) - y[k].astype(float)).max()) for k in common}
    return {"arrays_compared": len(common), "missing_in_retrained": sorted(set(y.files) - set(x.files)),
            "max_abs_diff": max(diff.values()) if diff else None,
            "q50_40dB_mean_diff": float(x["matched_40dB__bitacc_jpeg50"].mean() - y["matched_40dB__bitacc_jpeg50"].mean())
            if "matched_40dB__bitacc_jpeg50" in common else None,
            "identical": bool(common) and set(x.files) == set(y.files)
            and all(np.array_equal(x[k], y[k]) for k in common), "per_array": diff}


def step_i4_recover(cfg):
    """Retraining of the lost I-4 checkpoints (decision d-2026-10-24-01 (c); docs/PROTOCOL_I4_RECOVERY.md).
    Requires I4_RECOVER=1. Runs cfg.i4_recover.recover (seeds 2-4 that were trained in I-4) first, then
    cfg.i4_recover.new (V3 seed 4, never trained), so a stage-limit stop loses the new run first. Each recovered
    run is compared with its frozen per-image arrays from r/pilot-p1-i4; the frozen results, r/pilot-p1-i4 and
    r/pilot-p1-i4_final are never overwritten (output: results/i4_recover, branch r/pilot-p1-i4_recover). The new
    run is descriptive and enters no analysis without a separate decision."""
    import shutil

    if os.environ.get("I4_RECOVER") != "1":
        raise SystemExit("I-4 recovery is not authorised in this environment (I4_RECOVER != 1)")
    torch.set_num_threads(cpu_threads())
    print(f"I-4 recovery torch threads: {torch.get_num_threads()}", flush=True)
    out = Path(cfg.work) / "results" / "i4_recover"
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(cfg.work) / "i4_frozen"
    head = fetch_i4_branch(ref)
    frozen = json.loads((ref / "i4.json").read_text())["runs"]
    path = out / "i4_recover.json"
    res = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    res["frozen_source_commit"] = head
    sampler = GPUSampler(out / "gpu.csv").start()
    plan = [(k, "recovery") for k in cfg.i4_recover.recover] + [(k, "new_not_in_frozen_analysis")
                                                                 for k in cfg.i4_recover.new]
    for key, purpose in plan:
        if res["runs"].get(key, {}).get("status") == "ok":
            continue
        v, s = key.rsplit("_s", 1)
        try:
            r = i4_run(cfg, v, int(s), out, Path(cfg.work) / "runs" / "i4_recover")
            r["status"] = "ok"
        except Exception as e:
            r = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        r["purpose"] = purpose
        if r["status"] == "ok" and purpose == "recovery" and (ref / f"{key}_test_per_image.npz").exists():
            r["reproduction"] = compare_per_image(out / f"{key}_test_per_image.npz", ref / f"{key}_test_per_image.npz")
            r["reproduction"]["frozen_strengths"] = frozen.get(key, {}).get("strengths_from_val")
        res["runs"][key] = r
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        run = Path(cfg.work) / "runs" / "i4_recover" / key
        if run.exists():
            archive_i2(cfg, v, run, stage=f"i4_s{s}")
        push(cfg, "i4_recover")
    sampler.stop()
    rec = [res["runs"].get(k, {}) for k in cfg.i4_recover.recover]
    res["summary"] = {"recovered_ok": sum(r.get("status") == "ok" for r in rec),
                      "bitwise_identical": sum(bool(r.get("reproduction", {}).get("identical")) for r in rec),
                      "recovery_planned": len(rec)}
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    shutil.rmtree(ref, ignore_errors=True)
    push(cfg, "i4_recover")


def fetch_i4_branch(src: Path) -> str:
    """Copy i4.json and the per-image test arrays of r/pilot-p1-i4 into ``src``; returns the branch commit."""
    src.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("GH_REPO", "yazanjer/learned-decoder-wavelet-watermarking")
    subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "-q", f"https://x-access-token@github.com/{repo}.git",
                    "r/pilot-p1-i4"], check=True)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "FETCH_HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    names = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-tree", "--name-only", "FETCH_HEAD"], capture_output=True,
                           text=True, check=True).stdout.split()
    for n in names:
        if n.endswith(".npz") or n == "i4.json":
            (src / n).write_bytes(subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"FETCH_HEAD:{n}"],
                                                 capture_output=True, check=True).stdout)
    return head


def build_fixed(kind: str, cfg):
    from .transforms import build_transform

    return build_transform(kind, int(cfg.wmnet.common.levels), int(cfg.wmnet.common.lift_hidden))


# --------------------------------------------------------------------------- smoke
def step_smoke(cfg):
    """CPU smoke test on synthetic images: every method trains a few steps and is evaluated."""
    from PIL import Image

    rng = np.random.default_rng(0)
    root = Path(cfg.data_root) / f"png{cfg.image_size}"
    for split, n in (("train", 64), ("val", 16), ("test", 16)):
        (root / split).mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(n):
            a = (rng.random((cfg.image_size, cfg.image_size, 3)) * 64 + rng.random((1, 1, 3)) * 191).astype(np.uint8)
            name = f"smoke_{split}_{i:03d}"
            Image.fromarray(a).save(root / split / f"{name}.png")
            lines.append(f"{name}.jpg 0")
        (Path(cfg.work) / "splits").mkdir(parents=True, exist_ok=True)
        (Path(cfg.work) / "splits" / f"coco_{split}.txt").write_text("\n".join(lines) + "\n")
    D.SPLIT_DIR = Path(cfg.work) / "splits"
    cfg.smoke = True
    step_g2(cfg)
    step_g3(cfg)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    step, cfg = argv[0], load_cfg(argv[1:])
    steps = {"data": step_data, "g1": step_g1, "g2": step_g2, "g3": step_g3, "sweep": step_sweep,
             "smoke": step_smoke, "i2": step_i2, "i2b": step_i2b, "i3a": step_i3a,
             "i3a_recon": step_i3a_recon, "i3b": step_i3b, "i4": step_i4,
             "i4_final": step_i4_final, "i4_secondary": step_i4_secondary, "i4_recover": step_i4_recover}
    if step.startswith("i5_"):  # stage I-5 (docs/PROTOCOL_I5.md); imported here because i5 imports this module
        from .i5 import STEPS as I5_STEPS

        steps.update(I5_STEPS)
    if step.startswith("i6_"):  # stage I-6 (docs/PROTOCOL_I6.md): public baseline checkpoints, evaluation only
        from .i6 import STEPS as I6_STEPS

        steps.update(I6_STEPS)
    steps[step](cfg)


if __name__ == "__main__":
    main()
