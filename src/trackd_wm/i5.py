"""Stage I-5: full evaluation and baselines for Paper I (docs/PROTOCOL_I5.md, Sections 6 and 7).

Steps (``python -m trackd_wm.pilot <step>``), one pod each (Section 6.1)::

    i5_inputs   archives the pod needs (I5_POD = A, B or C); fails before any paid work if one is missing
    i5_m1       I5-A: extended evaluation of the 12 I-4 models (COCO test, DIV2K tiles, operator analysis)
    i5_mbrs     I5-B: MBRS continuation from the 20-epoch pilot checkpoint to 120 epochs, then evaluation
    i5_hidden   I5-C: HiDDeN continuation from the 20-epoch pilot checkpoint to 120 epochs, then evaluation
    i5_dwtdct   I5-C: DWT-DCT evaluation (no training; CPU)
    i5_mbrs_seed  a further MBRS training seed (``I5_TRAIN_SEED``): 20 epochs from initialisation, the seed-0
                continuation schedule to 120 epochs, then the I5-B evaluation (d-2026-10-25-79;
                docs/PROPOSAL_MBRS_SEEDS.md, Section 6)
    i5_compare  after the three pods (CPU): image-level comparison of the baselines with the WMNet variants

Every evaluation uses the attack suite ``attacks.I5_ATTACKS``, real JPEG and WebP, the fixed evaluation messages
of I-4 (``i4.message_seed``) so that per-image arrays pair across methods, and residual strengths fixed on the
validation split before the test split is read. I-5 tests no hypothesis; nothing here changes the frozen I-4
results or decisions.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .attacks import I5_ATTACKS
from .baselines import LICENCE, REPOS, fetch
from .harness import embed_all, eval_messages, evaluate, summarise
from .pilot import (REPO_ROOT, checkpoints, cpu_threads, device, epoch_seconds, fetch_i4_branch,
                    fetch_zenodo_checkpoints, layouts, matched_strength_cached, provenance, push, recon_diagnostics, seed_t_interval,
                    set_phase, split_files, state_dict_finite, write_json)
from .sysmon import GPUSampler

ARCHIVE_PREFIX = "pilot_p1_"  # name of every archive in the Zenodo draft (scripts/pod/zenodo_upload.py)


# --------------------------------------------------------------------------- inputs
def _extract(tar_path: Path, dest: Path) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        members = t.getmembers()
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, members=members, filter="data")
        else:
            t.extractall(dest, members=members)
    return [m.name for m in members]


def fetch_input(cfg, name: str, root: Path) -> dict:
    """Make archive ``<name>_checkpoints.tar`` available under ``root/<name>``.

    Sources, in order: a directory ``I5_INPUT_DIR`` (for example a network volume) holding the archive under its
    local or its Zenodo name; the Paper I Zenodo draft, read only, with ``ZENODO_READ_TOKEN`` (MD5 verified), under
    either name; then each further deposition in ``I5_INPUT_DEPOSITIONS`` (comma-separated ids, for example the
    Orchestron deposit of R11), read only, under either name. The read token has its own name so that bootstrap.sh,
    which uploads to Zenodo whenever ``ZENODO_TOKEN`` is set, never uploads from an I-5 pod (orchestron/WORKFLOW.md:
    Orchestron makes the deposit). An archive listed in ``cfg.i5.input_sha256`` must match that SHA-256
    whatever its source.
    Every epoch is extracted, so that the last finite checkpoint can be chosen."""
    dest = root / name
    if dest.exists() and any(dest.rglob("epoch_*")):
        return {"source": "already present", "path": str(dest)}
    want = dict(cfg.i5.get("input_sha256") or {})
    local = os.environ.get("I5_INPUT_DIR")
    if local:
        for fn in (f"{name}_checkpoints.tar", f"{ARCHIVE_PREFIX}{name}_checkpoints.tar"):
            p = Path(local) / fn
            if p.exists():
                sha = D.sha256_file(p)
                if want.get(name) and sha != want[name]:
                    raise RuntimeError(f"I5_INPUT_DIR/{fn}: SHA-256 mismatch (got {sha}, expected {want[name]})")
                members = _extract(p, dest)
                return {"source": f"I5_INPUT_DIR/{fn}", "sha256": sha, "bytes": p.stat().st_size,
                        "members": len(members)}
    tok = os.environ.get("ZENODO_READ_TOKEN")
    if tok:
        extra = [d.strip() for d in os.environ.get("I5_INPUT_DEPOSITIONS", "").split(",") if d.strip()]
        primary = os.environ.get("ZENODO_DEPOSITION")
        # the R11 archives keep their upload name (no prefix) if placed in the draft unchanged (job 25)
        tries = [(dep, pf) for dep in [primary] + extra for pf in (ARCHIVE_PREFIX, "")]
        os.environ["ZENODO_TOKEN"] = tok  # this process only (fetch_zenodo_checkpoints reads ZENODO_TOKEN)
        try:
            for dep, prefix in tries:
                if dep:
                    os.environ["ZENODO_DEPOSITION"] = dep
                else:
                    os.environ.pop("ZENODO_DEPOSITION", None)  # the Paper I draft
                got = fetch_zenodo_checkpoints(cfg, [name], dest, prefix=prefix, keep_all=True,
                                               sha256=want)[name]
                if got.get("archive"):
                    src = "zenodo draft (read only)" if dep == primary else f"zenodo deposition {dep} (read only)"
                    return {"source": src, **got}
        finally:
            os.environ.pop("ZENODO_TOKEN", None)
            if primary is None:
                os.environ.pop("ZENODO_DEPOSITION", None)
            else:
                os.environ["ZENODO_DEPOSITION"] = primary
    raise FileNotFoundError(f"archive {name}_checkpoints.tar not found (I5_INPUT_DIR, Zenodo depositions)")


def pod_inputs(cfg, pod: str) -> list:
    return [str(n) for n in cfg.i5.inputs[pod]]


def step_i5_inputs(cfg):
    """Fetch every input archive of pod ``I5_POD`` before the test gate and the data preparation, so that a
    missing input stops the pod within minutes (bootstrap.sh: FATAL_STEPS). Exit code 1 if any is missing."""
    set_phase(cfg, "prep")
    pod = os.environ.get("I5_POD", "")
    if pod not in cfg.i5.inputs:
        raise SystemExit(f"I5_POD must be one of {sorted(cfg.i5.inputs)} (got {pod!r})")
    out = Path(cfg.work) / "results" / "i5_inputs"
    root = Path(cfg.work) / "i5_inputs"
    res = {"pod": pod, "inputs": {}}
    for n in pod_inputs(cfg, pod):
        try:
            res["inputs"][n] = {"status": "ok", **fetch_input(cfg, n, root)}
        except Exception as e:
            res["inputs"][n] = {"status": "missing", "error": repr(e)}
    res["complete"] = all(r["status"] == "ok" for r in res["inputs"].values())
    res["provenance"] = provenance(cfg)
    write_json(out / "i5_inputs.json", res)
    if not res["complete"]:
        raise SystemExit("I-5 inputs incomplete: " + ", ".join(n for n, r in res["inputs"].items()
                                                                if r["status"] != "ok"))


def last_finite(d: Path) -> Path:
    """Last finite archived checkpoint ``<d>/epoch_NNN.*`` (WMNet: as I-4 chose its final epoch)."""
    cks = sorted(d.glob("epoch_*"))
    if not cks:
        raise FileNotFoundError(f"no checkpoint in {d}")
    for ck in reversed(cks):
        if ck.suffix != ".pt" or state_dict_finite(ck):
            return ck
    raise RuntimeError(f"no finite checkpoint in {d}")


# --------------------------------------------------------------------------- evaluation helpers
def smoke_attacks(ds, n: int = 4) -> dict:
    """Every attack of the suite on ``n`` validation images: finite, 8-bit and of the input shape. Run before
    any training or full evaluation so that a missing codec (e.g. WebP) stops the step early."""
    from .attacks import ATTACKS

    x = torch.stack([ds[i] for i in range(n)])
    bad = {}
    for a in I5_ATTACKS:
        try:
            y = ATTACKS[a](x)
            if (y.shape != x.shape or not torch.isfinite(y).all()
                    or not torch.allclose(torch.round(y * 255) / 255, y, atol=1e-6)):
                bad[a] = "shape, finiteness or 8-bit check failed"
        except Exception as e:
            bad[a] = repr(e)
    if bad:
        raise RuntimeError(f"attack smoke test failed: {bad}")
    return {"attacks": len(I5_ATTACKS), "images": n, "status": "ok"}


def eval_points(wm, ds, msgs, strengths: dict, batch: int, dev, prefix: str) -> tuple:
    """Per-image metrics of ``wm`` on ``ds`` at each strength (one embedding pass); keys ``prefix__label__k``."""
    cache = embed_all(wm, ds, msgs, batch, dev)
    points, per_all = {}, {}
    for lab, sv in strengths.items():
        per = evaluate(wm, ds, msgs, list(I5_ATTACKS), sv, batch, dev, cache)
        points[lab] = {"strength": sv, **summarise(per)}
        per_all.update({f"{prefix}__{lab}__{k}": v.astype(np.float32) for k, v in per.items()})
    return points, per_all


def val_strengths(wm, cfg, targets, dev) -> dict:
    va = D.ImageFolderDataset(split_files(cfg, "val"))
    cache = embed_all(wm, va, eval_messages(len(va), cfg.bits, int(cfg.i4.message_seed)), cfg.eval_batch, dev)
    return {f"matched_{t:.0f}dB": matched_strength_cached(cache, float(t)) for t in targets}


def hier_bootstrap_vec(d: np.ndarray, reps: int, seed: int = 0, alpha: float = 0.05) -> list:
    """Vectorised two-level bootstrap of the mean of d (seeds x images), the scheme of ``pilot.hier_bootstrap``
    (seeds with replacement, then images within each resampled seed). A different random stream than I-4;
    I-5 intervals are descriptive."""
    rng = np.random.default_rng(seed)
    S, N = d.shape
    out = np.empty(reps)
    for i in range(0, reps, 250):
        k = min(250, reps - i)
        ss = rng.integers(0, S, (k, S))
        ii = rng.integers(0, N, (k, S, N))
        out[i:i + k] = d[ss[..., None], ii].mean(axis=(1, 2))
    return [float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))]


# --------------------------------------------------------------------------- DIV2K tiles (cross-dataset)
def tile_positions(w: int, h: int, n: int, size: int, rng) -> list:
    """``n`` distinct non-overlapping tiles on the ``size`` grid of a w x h image, drawn with ``rng``."""
    grid = [(x, y) for y in range(0, h - size + 1, size) for x in range(0, w - size + 1, size)]
    if len(grid) < n:
        raise ValueError(f"image {w}x{h} has only {len(grid)} tiles of {size}")
    return [grid[i] for i in sorted(rng.choice(len(grid), size=n, replace=False))]


def prepare_div2k(cfg) -> dict:
    """DIV2K validation (100 HR images): ``i5.div2k.tiles_per_image`` native-resolution 128 x 128 tiles per image,
    positions drawn with seed ``i5.div2k.seed`` in sorted file order; SHA-256 of every source image and tile."""
    from PIL import Image

    c = cfg.i5.div2k
    raw = Path(cfg.data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dst = Path(cfg.data_root) / f"div2k_tiles{cfg.image_size}"
    z = raw / "DIV2K_valid_HR.zip"
    src = raw / "DIV2K_valid_HR"
    t0 = time.time()
    if not src.exists():
        if not z.exists():
            subprocess.run(["aria2c", "-q", "-x", "8", "-s", "8", "-d", str(raw), "-o", z.name, str(c.url)], check=True)
        subprocess.run(["unzip", "-q", "-n", str(z), "-d", str(raw)], check=True)
    files = sorted(src.glob("*.png"))
    if len(files) != int(c.n_images):
        raise RuntimeError(f"DIV2K validation: {len(files)} images, {c.n_images} expected")
    rng = np.random.default_rng(int(c.seed))
    dst.mkdir(parents=True, exist_ok=True)
    tiles, rec = [], []
    s = int(cfg.image_size)
    for f in files:
        with Image.open(f) as im:
            im = im.convert("RGB")
            pos = tile_positions(im.width, im.height, int(c.tiles_per_image), s, rng)
            for x, y in pos:
                p = dst / f"{f.stem}_{x:04d}_{y:04d}.png"
                if not p.exists():
                    im.crop((x, y, x + s, y + s)).save(p, format="PNG")
                tiles.append(p)
                rec.append({"tile": p.name, "sha256": D.sha256_file(p)})
    z.unlink(missing_ok=True)
    return {"n_images": len(files), "n_tiles": len(tiles), "seed": int(c.seed),
            "source_sha256": {f.name: D.sha256_file(f) for f in files}, "tiles": rec,
            "seconds": time.time() - t0, "files": [str(p) for p in tiles]}


# --------------------------------------------------------------------------- operator analysis
@torch.no_grad()
def band_energy(t, xs: torch.Tensor) -> dict:
    """Share of the analysis-coefficient energy per sub-band (level k: LH, HL, HH; deepest LL), FP32, mean over
    ``xs`` (inputs in [-1, 1]). Falls back to the flat leaf order for transforms with another band layout."""
    from .pilot import _leaves

    b = t.analysis(xs.float())
    named = []
    if isinstance(b, (list, tuple)) and all(isinstance(q, (list, tuple)) and len(q) == 4 for q in b):
        for k, q in enumerate(b):
            named += [(f"L{k + 1}_{n}", q[i]) for i, n in ((1, "lh"), (2, "hl"), (3, "hh"))]
        named.append((f"L{len(b)}_ll", b[-1][0]))
    else:
        named = [(f"leaf{k}", q) for k, q in enumerate(_leaves(b))]
    e = {n: float(q.double().pow(2).sum()) for n, q in named}
    tot = sum(e.values())
    return {n: v / tot for n, v in e.items()}


def operator_analysis(cfg, wm) -> dict:
    va = D.ImageFolderDataset(split_files(cfg, "val"))
    xs = torch.stack([va[i] for i in range(int(cfg.i5.m1.recon_images))]) * 2 - 1
    t = wm.net.decoder.t
    import copy

    tc = copy.deepcopy(t).to("cpu").eval()
    return {"band_energy_share_fp32": band_energy(tc, xs),
            "reconstruction": recon_diagnostics(tc, xs, int(cfg.i5.m1.power_iters), 1)}


# --------------------------------------------------------------------------- I5-A: M1
def m1_models(cfg) -> list:
    return [(str(v), int(s)) for s in cfg.i5.m1.seeds for v in cfg.i5.m1.variants]


def m1_analysis(cfg, out: Path, res: dict) -> dict:
    """Robustness profiles: per comparison, dataset, strength and attack, the mean paired difference over seeds
    and images with the hierarchical bootstrap (95 %) and the seed-level paired t-interval. No Holm family, no
    decisions (Section 2, M1). Whole seeds only."""
    from scipy import stats

    V = list(cfg.i5.m1.variants)
    seeds = [s for s in cfg.i5.m1.seeds if all(res["models"].get(f"{v}_s{s}", {}).get("status") == "ok" for v in V)]
    a = {"seeds_used": list(seeds), "used_for_decisions": False, "profiles": {}, "operator_association": {}}
    if len(seeds) < 2:
        a["status"] = "incomplete: fewer than two complete seeds"
        return a
    per = {(v, s): np.load(out / f"{v}_s{s}_i5_per_image.npz") for v in V for s in seeds}
    keys = [k for k in per[(V[0], seeds[0])].files if "__bitacc_" in k or k.endswith("__ssim")
            or k.endswith("__psnr_rgb")]
    reps = int(cfg.i5.bootstrap)
    for A, B in cfg.i5.m1.comparisons:
        A, B = str(A), str(B)
        rows = {}
        for k in keys:
            x = np.stack([per[(A, s)][k] for s in seeds]).astype(np.float64)
            y = np.stack([per[(B, s)][k] for s in seeds]).astype(np.float64)
            d = x - y
            rows[k] = {"mean_A": float(x.mean()), "mean_B": float(y.mean()), "theta_hat": float(d.mean()),
                       "ci95_hier_bootstrap": hier_bootstrap_vec(d, reps), "seed_diffs": d.mean(1).tolist(),
                       "seed_paired_t_ci95": seed_t_interval(d.mean(1))}
        a["profiles"][f"{A}-{B}"] = rows
    key = "coco__matched_40dB__bitacc_jpeg50"
    for v in V:
        q50 = [float(per[(v, s)][key].mean()) for s in seeds]
        ops = [res["models"][f"{v}_s{s}"]["operator"]["reconstruction"] for s in seeds]
        err = [o["float32"]["recon_max_abs"] for o in ops]
        cond = [(o["conditioning_float64"][0] or {}).get("condition_number") for o in ops]
        row = {"seeds": list(seeds), "q50_40dB_coco": q50, "fp32_recon_max_abs": err, "condition_number": cond,
               "n": len(seeds), "note": "descriptive; n = number of seeds"}
        if len(seeds) >= 3 and np.ptp(err) > 0:
            row["spearman_q50_vs_fp32_error"] = float(stats.spearmanr(q50, err).statistic)
        if len(seeds) >= 3 and all(c is not None for c in cond) and np.ptp(cond) > 0:
            row["spearman_q50_vs_condition_number"] = float(stats.spearmanr(q50, cond).statistic)
        a["operator_association"][v] = row
    return a


def step_i5_m1(cfg):
    """I5-A (PROTOCOL_I5.md, M1). Inputs: archives ``i4_s{seed}_{variant}`` (step i5_inputs). Per model: strengths
    fixed on the validation split at the I-4 targets (and compared with the frozen I-4 strengths), then the COCO
    test split and the DIV2K tiles at those strengths with the full suite, then the operator analysis. No
    training. Output: results/i5_m1, branch r/pilot-p1-i5_m1."""
    from .train import load_wmnet

    torch.set_num_threads(cpu_threads())
    set_phase(cfg, "eval")
    out = Path(cfg.work) / "results" / "i5_m1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i5_m1.json"
    res = json.loads(path.read_text()) if path.exists() else {"models": {}}
    dev = device()
    sampler = GPUSampler(out / "gpu.csv").start()
    va = D.ImageFolderDataset(split_files(cfg, "val"))
    res["attack_smoke"] = smoke_attacks(va)
    if "div2k" not in res:
        dv = prepare_div2k(cfg)
        res["div2k"] = {k: v for k, v in dv.items() if k != "files"}
    div_files = sorted((Path(cfg.data_root) / f"div2k_tiles{cfg.image_size}").glob("*.png"))
    ref = Path(cfg.work) / "i4_frozen"
    try:
        res["frozen_source_commit"] = fetch_i4_branch(ref)
        frozen = json.loads((ref / "i4.json").read_text())["runs"]
    except Exception as e:  # the comparison with the frozen strengths is descriptive; it must not stop M1
        res["frozen_source_commit"] = {"error": repr(e)}
        frozen = {}
    write_json(path, res)
    ms = int(cfg.i4.message_seed)
    te = D.ImageFolderDataset(split_files(cfg, "test"))
    dv_ds = D.ImageFolderDataset(div_files)
    for v, s in m1_models(cfg):
        key = f"{v}_s{s}"
        if res["models"].get(key, {}).get("status") == "ok":
            continue
        t0 = time.time()
        try:
            ck = last_finite(Path(cfg.work) / "i5_inputs" / f"i4_s{s}_{v}" / v)
            wm = load_wmnet(ck, dev)
            st = val_strengths(wm, cfg, cfg.i5.m1.targets, dev)
            p_coco, a_coco = eval_points(wm, te, eval_messages(len(te), cfg.bits, ms), st, cfg.eval_batch, dev, "coco")
            p_div, a_div = eval_points(wm, dv_ds, eval_messages(len(dv_ds), cfg.bits, ms), st, cfg.eval_batch, dev,
                                       "div2k")
            np.savez_compressed(out / f"{key}_i5_per_image.npz", **a_coco, **a_div)
            fz = frozen.get(key, {}).get("strengths_from_val")
            r = {"status": "ok", "checkpoint": f"i4_s{s}_{v}/{ck.relative_to(ck.parents[1])}",
                 "strengths_from_val": st, "frozen_i4_strengths": fz,
                 "max_abs_strength_diff_vs_frozen": max(abs(st[k] - fz[k]) for k in st) if fz else None,
                 "points": {"coco": p_coco, "div2k": p_div}, "operator": operator_analysis(cfg, wm)}
        except Exception as e:
            r = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        r["seconds"] = time.time() - t0
        res["models"][key] = r
        res["provenance"] = provenance(cfg)
        write_json(path, res)
        push(cfg, "i5_m1")
    sampler.stop()
    try:
        res["analysis"] = m1_analysis(cfg, out, res)
    except Exception as e:
        res["analysis"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    shutil.rmtree(ref, ignore_errors=True)
    push(cfg, "i5_m1")


# --------------------------------------------------------------------------- I5-B / I5-C: M2 baselines
def _prepare_continuation(cfg, name: str, run: Path) -> tuple:
    """Lay out the archived 20-epoch pilot checkpoint where the baseline's own continuation expects it.
    Returns (continue-from directory, start epoch)."""
    src = Path(cfg.work) / "i5_inputs" / f"g3_{name}" / name
    start = int(cfg.i5.baselines.start_epoch)
    ck = src / (f"epoch_{start:03d}.pyt" if name == "hidden" else f"epoch_{start:03d}.pth")
    if not ck.exists():
        raise FileNotFoundError(f"{ck} (archived epoch {start} checkpoint)")
    cont = run / "cont"
    if name == "hidden":
        opts = next(src.rglob("options-and-config.pickle"))
        (cont / "checkpoints").mkdir(parents=True, exist_ok=True)
        (cont / "images").mkdir(exist_ok=True)  # HiDDeN's train() saves validation images here; only `new` creates it
        shutil.copy2(opts, cont / "options-and-config.pickle")
        shutil.copy2(ck, cont / "checkpoints" / f"pilot--epoch-{start}.pyt")
    else:  # MBRS counts epochs from 0: EC_<start> is loaded and the first new epoch is labelled <start>
        (cont / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ck, cont / "models" / f"EC_{start}.pth")
    return cont, start


def baseline_checkpoints(name: str, run: Path) -> dict:
    """epoch (1-based, total) -> checkpoint of the continued run."""
    import re

    if name == "hidden":
        return {int(re.search(r"epoch-(\d+)", p.name).group(1)): p
                for p in sorted((run / "cont" / "checkpoints").glob("pilot--epoch-*.pyt"))}
    return checkpoints("mbrs", run)


def archive_baseline(cfg, name: str, run: Path, every: int, tag: str = "") -> Path:
    """Checkpoints every ``every`` epochs and the last one, MBRS discriminators of the same epochs, settings and
    logs, into one tar (sent with scripts/upload_artifacts.sh; never to git). A further seed (``tag`` "_s<seed>")
    also archives its phase-1 run (``run/phase1``) under ``<name>/phase1/``."""
    arc = Path(cfg.work) / "archive"
    arc.mkdir(exist_ok=True)
    tp = arc / f"i5_{name}{tag}_checkpoints.tar"

    def add_checkpoints(t, ck: dict, prefix: str):
        for ep, p in ck.items():
            if ep % every == 0 or ep == max(ck):
                t.add(p, arcname=f"{prefix}/epoch_{ep:03d}{p.suffix}")
                d = p.with_name(p.name.replace("EC_", "D_"))
                if name == "mbrs" and d.exists() and d != p:
                    t.add(d, arcname=f"{prefix}/discriminator_epoch_{ep:03d}{d.suffix}")

    with tarfile.open(tp, "w") as t:
        ck = baseline_checkpoints(name, run)
        if ck:
            add_checkpoints(t, ck, name)
        p1 = run / "phase1"
        if p1.exists():
            ck1 = checkpoints(name, p1)
            if ck1:
                add_checkpoints(t, ck1, f"{name}/phase1")
            for extra in (list(p1.glob("work/train_settings.json")) + list(p1.glob("work/results/*/train_*.txt"))
                          + list(p1.glob("work/results/*/val_log.txt")) + list(p1.glob("stdout.log"))):
                t.add(extra, arcname=f"{name}/{extra.relative_to(run)}")
        for extra in (list(run.glob("cont/options-and-config.pickle")) + list(run.glob("cont/*.log"))
                      + list(run.glob("work/train_settings.json")) + list(run.glob("work/results/*/train_*.txt"))
                      + list(run.glob("work/results/*/val_log.txt")) + list(run.glob("stdout.log"))):
            t.add(extra, arcname=f"{name}/{extra.relative_to(run)}")
    return tp


def _fresh_phase1(cfg, name: str, run: Path, repo: Path, data: Path, seed: int, env: dict, timeout) -> tuple:
    """Phase 1 of a further training seed (docs/PROPOSAL_MBRS_SEEDS.md, Section 6): ``i5.baselines.start_epoch``
    epochs from initialisation with the pilot G3 settings (``pilot.train_method``) and training seed ``seed``. The
    epoch-<start> encoder-decoder alone is then laid out for the continuation, as the archived seed-0 checkpoint
    was (``_prepare_continuation``): the discriminator is re-initialised and no optimiser state is restored, so
    the further seed repeats the seed-0 schedule. Returns (continue-from directory, start epoch, phase-1 info)."""
    start = int(cfg.i5.baselines.start_epoch)
    p1 = run / "phase1"
    p1.mkdir(parents=True)
    (p1 / "started").touch()
    cmd = [sys.executable, "-m", "trackd_wm.baselines.launch", name, "--repo", str(repo), "--data", str(data),
           "--work", str(p1 / "work"), "--epochs", str(start), "--seed", str(seed)]
    t0 = time.time()
    rc = None
    with open(p1 / "stdout.log", "w") as f:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                timeout=float(timeout) if timeout else None).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    ck = checkpoints(name, p1)
    info = {"returncode": rc, "wall_seconds": time.time() - t0, "last_epoch": max(ck) if ck else None,
            "epochs": start, "epoch_seconds": epoch_seconds(name, p1)}
    if rc != 0 or start not in ck:
        raise RuntimeError(f"phase 1 reached epoch {info['last_epoch']} of {start} (returncode {rc})")
    cont = run / "cont"
    (cont / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck[start], cont / "models" / f"EC_{start}.pth")  # encoder-decoder only, as for seed 0
    return cont, start, info


def eval_baseline(cfg, name: str, wm, dev) -> tuple:
    """Strength matched to ``i5.baselines.target_psnr`` on the validation split, then the COCO test split once
    at that strength and at the native strength (s = 1) with the full suite."""
    st = val_strengths(wm, cfg, [cfg.i5.baselines.target_psnr], dev)
    st["native"] = 1.0
    te = D.ImageFolderDataset(split_files(cfg, "test"))
    return st, eval_points(wm, te, eval_messages(len(te), cfg.bits, int(cfg.i4.message_seed)), st,
                           cfg.eval_batch, dev, "coco")


def train_seed_from_env() -> int:
    """Training seed of step ``i5_mbrs_seed`` (environment ``I5_TRAIN_SEED``). It is kept apart from ``cfg.seed``,
    which also selects the data split in step ``data`` and must stay 0. Seed 0 is the I5-B model."""
    v = os.environ.get("I5_TRAIN_SEED", "")
    if not v.isdigit() or int(v) == 0:
        raise SystemExit(f"I5_TRAIN_SEED must be a positive integer (seed 0 is the I5-B model); got {v!r}")
    return int(v)


def step_i5_baseline(cfg, name: str, fresh: bool = False):
    """Continue the 20-epoch pilot run of ``name`` (seed 0) to ``i5.baselines.total_epochs`` with the baseline's
    own training code and settings, archive, then evaluate the final checkpoint. A run that does not reach the
    total is archived and reported as incomplete and is not evaluated as a 120-epoch baseline (Section 6.2).
    ``fresh`` (MBRS only): a further training seed ``I5_TRAIN_SEED`` that first trains its own 20 epochs from
    initialisation (``_fresh_phase1``) and then follows the same continuation; outputs are named ``_s<seed>``.
    ``method_timeout_s`` bounds both phases together."""
    from .baselines import wrappers as W

    torch.set_num_threads(cpu_threads())
    if fresh and name != "mbrs":
        raise SystemExit(f"a further training seed is implemented for mbrs only, not {name}")
    seed = train_seed_from_env() if fresh else int(cfg.seed)
    tag = f"_s{seed}" if fresh else ""
    step = f"i5_{name}{tag}"
    out = Path(cfg.work) / "results" / step
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{step}.json"
    total = int(cfg.i5.baselines.total_epochs)
    validity = cfg.i5.baselines.seed_validity[name] if fresh else cfg.i5.baselines.validity[name]
    res = {"method": name, "seed": seed, "pinned_commit": REPOS[name][1], "licence": LICENCE[name],
           "total_epochs": total, "validity": validity}
    sampler = GPUSampler(out / "gpu.csv").start()
    dev = device()
    try:
        res["attack_smoke"] = smoke_attacks(D.ImageFolderDataset(split_files(cfg, "val")))
        lay = layouts(cfg)
        run = Path(cfg.work) / "runs" / "i5" / f"{name}{tag}"
        if run.exists():
            shutil.rmtree(run)
        run.mkdir(parents=True)
        repo = fetch(name, Path(cfg.work) / "repos")
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        timeout = cfg.get("method_timeout_s", None)
        set_phase(cfg, "train")
        if fresh:
            try:
                cont, start, res["phase1"] = _fresh_phase1(cfg, name, run, repo, lay[name], seed, env, timeout)
            except Exception:
                if (run / "phase1").exists():  # whatever phase 1 wrote is archived and reported, not discarded
                    tp = archive_baseline(cfg, name, run, int(cfg.i5.baselines.archive_every), tag)
                    res["archive"] = {"file": tp.name, "bytes": tp.stat().st_size}
                raise
            if timeout:
                timeout = max(60.0, float(timeout) - res["phase1"]["wall_seconds"])
        else:
            cont, start = _prepare_continuation(cfg, name, run)
        res["start_epoch"] = start
        cmd = [sys.executable, "-m", "trackd_wm.baselines.launch", name, "--repo", str(repo), "--data", str(lay[name]),
               "--work", str(run / "work"), "--epochs", str(total), "--seed", str(seed),
               "--continue-from", str(cont), "--start-epoch", str(start)]
        t0 = time.time()
        rc = None
        with open(run / "stdout.log", "w") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                    timeout=float(timeout) if timeout else None).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        set_phase(cfg, "eval")
        ck = baseline_checkpoints(name, run)
        last = max(ck) if ck else None
        res["train"] = {"returncode": rc, "wall_seconds": time.time() - t0, "last_epoch": last,
                        "new_epochs": (last - start) if last else 0,
                        "seconds_per_epoch": (time.time() - t0) / (last - start) if last and last > start else None}
        if ck:
            tp = archive_baseline(cfg, name, run, int(cfg.i5.baselines.archive_every), tag)
            res["archive"] = {"file": tp.name, "bytes": tp.stat().st_size}
        write_json(path, res)
        push(cfg, step)
        if last != total or rc != 0:
            res["status"] = "incomplete"
            res["note"] = f"training reached epoch {last} of {total} (returncode {rc}); not evaluated"
        else:
            if name == "hidden":
                wm = W.HiDDeNWatermarker(repo, cont, ck[last], dev)
            else:
                wm = W.MBRSWatermarker(repo, ck[last], dev)
            st, (points, per_all) = eval_baseline(cfg, name, wm, dev)
            np.savez_compressed(out / f"{name}_s{seed}_i5_per_image.npz", **per_all)
            res.update({"status": "ok", "checkpoint": f"{name}/epoch_{last:03d}", "strengths_from_val": st,
                        "points": {"coco": points}})
    except Exception as e:
        res.update({"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]})
    sampler.stop()
    res["provenance"] = provenance(cfg)
    write_json(path, res)
    push(cfg, step)


def step_i5_mbrs(cfg):
    step_i5_baseline(cfg, "mbrs")


def step_i5_mbrs_seed(cfg):
    step_i5_baseline(cfg, "mbrs", fresh=True)


def step_i5_hidden(cfg):
    step_i5_baseline(cfg, "hidden")


def step_i5_dwtdct(cfg):
    """DWT-DCT (invisible-watermark, library scale 36 as native): matched 40 dB on validation, COCO test split with
    the full suite. No training; CPU."""
    from .baselines.wrappers import DWTDCTWatermarker

    torch.set_num_threads(cpu_threads())
    set_phase(cfg, "eval")
    out = Path(cfg.work) / "results" / "i5_dwtdct"
    out.mkdir(parents=True, exist_ok=True)
    res = {"method": "dwtdct", "pinned_commit": REPOS["dwtdct"][1], "licence": LICENCE["dwtdct"]}
    t0 = time.time()
    try:
        res["attack_smoke"] = smoke_attacks(D.ImageFolderDataset(split_files(cfg, "val")))
        wm = DWTDCTWatermarker(fetch("dwtdct", Path(cfg.work) / "repos"))
        st, (points, per_all) = eval_baseline(cfg, "dwtdct", wm, "cpu")
        np.savez_compressed(out / "dwtdct_i5_per_image.npz", **per_all)
        res.update({"status": "ok", "strengths_from_val": st, "points": {"coco": points}})
    except Exception as e:
        res.update({"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]})
    res["seconds"] = time.time() - t0
    res["provenance"] = provenance(cfg)
    write_json(out / "i5_dwtdct.json", res)
    push(cfg, "i5_dwtdct")


# --------------------------------------------------------------------------- after the pods: CPU comparison
BASELINE_FILES = {"mbrs": ("i5_mbrs.json", "mbrs_s0_i5_per_image.npz"),
                  "hidden": ("i5_hidden.json", "hidden_s0_i5_per_image.npz"),
                  "dwtdct": ("i5_dwtdct.json", "dwtdct_i5_per_image.npz")}


def fetch_branch(branch: str, dest: Path) -> str:
    """Copy the JSON and .npz files at the top of results branch ``branch`` into ``dest``; returns its commit."""
    dest.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("GH_REPO", "yazanjer/learned-decoder-wavelet-watermarking")
    git = ["git", "-C", str(REPO_ROOT)]
    subprocess.run(git + ["fetch", "-q", f"https://x-access-token@github.com/{repo}.git", branch], check=True)
    head = subprocess.run(git + ["rev-parse", "FETCH_HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    names = subprocess.run(git + ["ls-tree", "--name-only", "FETCH_HEAD"], capture_output=True, text=True,
                           check=True).stdout.split()
    for n in names:
        if n.endswith((".npz", ".json")):
            (dest / n).write_bytes(subprocess.run(git + ["show", f"FETCH_HEAD:{n}"], capture_output=True,
                                                  check=True).stdout)
    return head


def compare_analysis(cfg, src: Path) -> dict:
    """Image-level comparison of every baseline with every WMNet variant at the shared matched strength on the
    COCO test split (PROTOCOL_I5.md, Section 7.4): per variant and attack, the mean paired difference
    (variant - baseline) over the whole seeds of M1 and the images, the two-level bootstrap interval (seeds of
    the variant, then images; the baseline is one trained model) and the seed-level t-interval. Every pair shares
    the image and the I-4 message (``i4.message_seed``). Preliminary and conditional on the single trained
    baseline model; no Holm family, no decision. ``src`` holds the subdirectories m1, mbrs, hidden, dwtdct.

    Further training seeds of a baseline (``i5.compare.extra_seeds``, e.g. ``{mbrs: [1]}``; subdirectory
    ``<b>_s<seed>``, written by step ``i5_mbrs_seed``) are compared in the same way as models of their own
    (``<v>-<b>_s<seed>``); the seed-0 rows are unchanged. With at least two complete models of ``b``,
    ``across_models`` adds the estimate over the training runs of both methods (function ``across_models``)."""
    m1 = json.loads((src / "m1" / "i5_m1.json").read_text())
    V = [str(v) for v in cfg.i5.m1.variants]
    seeds = [s for s in cfg.i5.m1.seeds if all(m1["models"].get(f"{v}_s{s}", {}).get("status") == "ok" for v in V)]
    lab = str(cfg.i5.compare.strength)
    reps = int(cfg.i5.bootstrap)
    a = {"strength": lab, "dataset": "coco_test", "seeds_used": list(seeds), "used_for_decisions": False,
         "validity": "preliminary; conditional on one trained model per baseline (seed 0)", "methods": {},
         "differences": {}}
    if len(seeds) < 2:
        a["status"] = "incomplete: fewer than two complete WMNet seeds"
        return a
    per = {(v, s): np.load(src / "m1" / f"{v}_s{s}_i5_per_image.npz") for v in V for s in seeds}
    pre = f"coco__{lab}__"
    keys = [k for k in per[(V[0], seeds[0])].files if k.startswith(pre)]
    for v in V:
        a["methods"][v] = {"kind": "wmnet", "seeds": list(seeds),
                           "mean": {k[len(pre):]: float(np.mean([per[(v, s)][k].mean() for s in seeds])) for k in keys}}
    extra = {str(k): [int(x) for x in v] for k, v in (cfg.i5.compare.get("extra_seeds") or {}).items() if v}
    models = []
    for b in cfg.i5.compare.baselines:
        b = str(b)
        models.append((b, *BASELINE_FILES[b]))
        models += [(f"{b}_s{s}", f"i5_{b}_s{s}.json", f"{b}_s{s}_i5_per_image.npz") for s in extra.get(b, [])]
    for b, jf, nf in models:
        try:
            info = json.loads((src / b / jf).read_text())
        except FileNotFoundError:
            a["methods"][b] = {"kind": "baseline", "status": "missing"}
            continue
        if info.get("status") != "ok" or not (src / b / nf).exists():
            a["methods"][b] = {"kind": "baseline", "status": info.get("status", "unknown"), "note": info.get("note")}
            continue
        y_all = np.load(src / b / nf)
        bk = [k for k in keys if k in y_all.files]
        a["methods"][b] = {"kind": "baseline", "status": "ok", "validity": info.get("validity"),
                           "strengths_from_val": info.get("strengths_from_val"),
                           "missing_keys": sorted(k[len(pre):] for k in set(keys) - set(bk)),
                           "mean": {k[len(pre):]: float(y_all[k].mean()) for k in bk}}
        for v in V:
            rows = {}
            for k in bk:
                x = np.stack([per[(v, s)][k] for s in seeds]).astype(np.float64)
                y = y_all[k].astype(np.float64)
                if y.shape[0] != x.shape[1]:
                    raise ValueError(f"{b} {k}: {y.shape[0]} images, WMNet {x.shape[1]}; arrays do not pair")
                d = x - y[None, :]
                rows[k[len(pre):]] = {"mean_variant": float(x.mean()), "mean_baseline": float(y.mean()),
                                      "theta_hat": float(d.mean()), "ci95_hier_bootstrap": hier_bootstrap_vec(d, reps),
                                      "seed_diffs": d.mean(1).tolist(), "seed_paired_t_ci95": seed_t_interval(d.mean(1)),
                                      "share_images_variant_better": float((d.mean(0) > 0).mean())}
            a["differences"][f"{v}-{b}"] = rows
    for b, seeds_b in extra.items():
        a["across_models"] = a.get("across_models", {})
        a["across_models"].update(across_models(a, per, V, seeds, b, [0] + seeds_b, pre))
    if extra:
        a["validity"] = ("preliminary; per-model rows conditional on one trained baseline model each; "
                         "across_models over the complete training seeds of " + ", ".join(sorted(extra)))
    a["status"] = "ok"
    return a


def welch_interval(x, y, alpha: float = 0.05) -> dict:
    """Welch t-interval for mean(x) - mean(y) of two independent samples (training runs of two methods)."""
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return {"ci95": [None, None], "df": None}
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    d = float(x.mean() - y.mean())
    if se == 0:
        return {"ci95": [d, d], "df": None}
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    t = stats.t.ppf(1 - alpha / 2, df) * se
    return {"ci95": [d - float(t), d + float(t)], "df": float(df)}


def across_models(a: dict, per: dict, V: list, seeds: list, b: str, b_seeds: list, pre: str) -> dict:
    """Comparison over training runs of both methods (docs/PROPOSAL_MBRS_SEEDS.md, Section 6.3), per variant and
    endpoint, over the complete models of baseline ``b`` (seed 0 is ``b``, seed s is ``b_s<s>``):
    theta = mean of the WMNet seed means - mean of the baseline model means (each model mean over the same COCO
    test images and I-4 messages); interval = Welch t-interval over the two sets of model means, which treats the
    training runs of the two methods as independent (unpaired: WMNet seed s and baseline seed s share nothing but
    the seed number), so the training variation of both methods enters. A direction is stated for the baseline
    as a method only if the per-model estimates (``<v>-<b>[_s<seed>]``) all have the same sign and the Welch
    interval excludes 0; otherwise "not resolved across <b> seeds". Descriptive; no test, no Holm family."""
    keys_b = {0: b, **{s: f"{b}_s{s}" for s in b_seeds if s != 0}}
    ok = [s for s in b_seeds if a["methods"].get(keys_b[s], {}).get("status") == "ok"]
    out = {}
    if len(ok) < 2:
        return {f"{v}-{b}": {"status": f"fewer than two complete {b} models", "models": ok} for v in V}
    for v in V:
        rows = {}
        common = set.intersection(*[set(a["differences"][f"{v}-{keys_b[s]}"]) for s in ok])
        for e in sorted(common):
            wm = [float(per[(v, s)][pre + e].astype(np.float64).mean()) for s in seeds]
            bm = [a["differences"][f"{v}-{keys_b[s]}"][e]["mean_baseline"] for s in ok]
            th = [a["differences"][f"{v}-{keys_b[s]}"][e]["theta_hat"] for s in ok]
            w = welch_interval(wm, bm)
            same = all(t > 0 for t in th) or all(t < 0 for t in th)
            lo, hi = w["ci95"]
            if same and lo is not None and (lo > 0 or hi < 0):
                direction = f"{v} higher" if lo > 0 else f"{b} higher"
            else:
                direction = f"not resolved across {b} seeds"
            rows[e] = {"theta_hat": float(np.mean(wm) - np.mean(bm)), "wmnet_seed_means": wm,
                       "baseline_model_means": bm, "theta_per_model": th, "same_sign_all_models": same,
                       "ci95_welch": w["ci95"], "welch_df": w["df"], "direction": direction}
        out[f"{v}-{b}"] = {"status": "ok", "models": ok, "wmnet_seeds": list(seeds), "endpoints": rows}
    return out


def compare_table(a: dict, attacks: list) -> str:
    """Markdown table of the mean bit accuracy per method and attack, then the PSNR and SSIM rows."""
    ms = [m for m, r in a["methods"].items() if "mean" in r]
    many = any("_s" in m for m in ms if a["methods"][m].get("kind") == "baseline")
    lines = [f"Bit accuracy on the COCO test split at {a['strength']} (WMNet: mean over seeds {a['seeds_used']}; "
             + ("baselines: one model per column; <b>_s<seed> is a further training seed" if many
                else "baselines: one model, seed 0") + "). Preliminary; no hypothesis test.", "",
             "| Attack | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for k in [f"bitacc_{t}" for t in attacks] + ["psnr_rgb", "ssim"]:
        vals = [a["methods"][m]["mean"].get(k) for m in ms]
        fmt = "{:.2f}" if k == "psnr_rgb" else "{:.3f}"
        lines.append(f"| {k.replace('bitacc_', '')} | " + " | ".join("-" if v is None else fmt.format(v)
                                                                       for v in vals) + " |")
    out = [m for m, r in a["methods"].items() if r.get("status") not in (None, "ok")]
    if out:
        lines += ["", "Not compared: " + ", ".join(f"{m} ({a['methods'][m]['status']})" for m in out) + "."]
    return "\n".join(lines) + "\n"


def step_i5_compare(cfg):
    """CPU, after the three I-5 pods: fetch the result branches (or read ``I5_COMPARE_DIR``) and compare. Output:
    results/i5_compare (i5_compare.json, i5_compare.md), branch r/pilot-p1-i5_compare."""
    out = Path(cfg.work) / "results" / "i5_compare"
    out.mkdir(parents=True, exist_ok=True)
    res = {"sources": {}}
    local = os.environ.get("I5_COMPARE_DIR")
    src = Path(local) if local else Path(cfg.work) / "i5_compare_inputs"
    if not local:
        for name, br in cfg.i5.compare.branches.items():
            try:
                res["sources"][name] = {"branch": str(br), "commit": fetch_branch(str(br), src / name)}
            except Exception as e:
                res["sources"][name] = {"branch": str(br), "error": repr(e)}
    try:
        res["analysis"] = compare_analysis(cfg, src)
        (out / "i5_compare.md").write_text(compare_table(res["analysis"], list(cfg.i5.compare.table_attacks)))
    except Exception as e:
        res["analysis"] = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
    res["provenance"] = provenance(cfg)
    write_json(out / "i5_compare.json", res)
    push(cfg, "i5_compare")


STEPS = {"i5_inputs": step_i5_inputs, "i5_m1": step_i5_m1, "i5_mbrs": step_i5_mbrs,
         "i5_mbrs_seed": step_i5_mbrs_seed, "i5_hidden": step_i5_hidden,
         "i5_dwtdct": step_i5_dwtdct, "i5_compare": step_i5_compare}
