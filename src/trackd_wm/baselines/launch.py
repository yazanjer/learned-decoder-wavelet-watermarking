"""Run a baseline's own training script with pilot settings.

Usage::

    python -m trackd_wm.baselines.launch <hidden|mbrs|cin|fin> --repo <clone> --data <layout dir>
           --work <run dir> --epochs N --seed S [--continue-from <dir> --start-epoch E]

The clone is copied to ``--work`` (outputs stay inside the run directory); the
seed is set before the third-party code is imported (``deterministic=warn_only``,
because third-party ops may lack deterministic kernels). Settings are written to
the baseline's own configuration mechanism; no third-party file in the clone is
modified.

Continuation (stage I-5, docs/PROTOCOL_I5.md, Section 7): ``--epochs`` is the total number of epochs and
``--start-epoch`` the number already trained in ``--continue-from``. HiDDeN uses its own ``continue`` command
(models, optimisers and discriminator restored). MBRS uses its own ``train_continue`` settings; MBRS restores no
optimiser state in its own continuation, and the pilot archives hold no discriminator (``D_*.pth``), so a missing
discriminator file is initialised afresh and reported in the log.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

HIDDEN_NOISE = ("crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+"
                "dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()")  # the repository's own combined-noise run


def _cuda_shim():
    """CPU smoke tests only: make hard-coded ``.cuda()`` calls no-ops when no GPU is present."""
    import torch

    if torch.cuda.is_available():
        return
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self


SMOKE = os.environ.get("TRACKD_SMOKE") == "1"   # CPU smoke tests only: tiny batches, no worker processes


def _copy(repo: Path, work: Path) -> None:
    if work.exists():
        return
    shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git"))


def _torch_full_float_default():
    """HiDDeN builds BCE labels with ``torch.full(size, 1)``; torch < 1.7 inferred float32 for an integer
    fill value, current torch infers int64, which BCE rejects. Restore the old default for integer fills."""
    import torch

    orig = torch.full

    def full(size, fill_value, *args, **kw):
        if "dtype" not in kw and isinstance(fill_value, int) and not isinstance(fill_value, bool):
            kw["dtype"] = torch.float32
        return orig(size, fill_value, *args, **kw)

    torch.full = full


def _hidden_jpeg_out_of_place():
    """HiDDeN's JpegCompression.apply_conv calls ``unsqueeze_`` in place on views, which current autograd
    rejects; ``unsqueeze`` computes the same tensors out of place. The method is recompiled at run time from the
    repository's own source with these two substitutions (the file itself is not modified)."""
    import inspect
    import textwrap

    import noise_layers.jpeg_compression as jc

    src = textwrap.dedent(inspect.getsource(jc.JpegCompression.apply_conv))
    subs = [("image[:, channel, :, :].unsqueeze_(1)", "image[:, channel, :, :].unsqueeze(1)"),
            ("image_conv.unsqueeze_(1)", "image_conv = image_conv.unsqueeze(1)")]
    for old, new in subs:
        assert src.count(old) == 1, f"unexpected HiDDeN source ({old})"
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, jc.__file__, "exec"), jc.__dict__, ns)
    jc.JpegCompression.apply_conv = ns["apply_conv"]


def _save_image_positional_nrow():
    """HiDDeN passes ``nrow`` as the third positional argument of torchvision's ``save_image``; current
    torchvision reads that position as ``format``. Map an integer third argument back to ``nrow``."""
    import torchvision.utils as tvu

    orig = tvu.save_image

    def save_image(tensor, fp, *args, **kw):
        if args and isinstance(args[0], int):
            kw.setdefault("nrow", args[0])
            args = args[1:]
        return orig(tensor, fp, *args, **kw)

    tvu.save_image = save_image


def run_hidden(a, work: Path):
    _torch_full_float_default()
    _hidden_jpeg_out_of_place()
    _save_image_positional_nrow()
    if a.continue_from:
        sys.argv = ["main.py", "continue", "--folder", str(a.continue_from), "--data-dir", str(a.data),
                    "--epochs", str(a.epochs)]
        runpy.run_path(str(work / "main.py"), run_name="__main__")
        return
    sys.argv = ["main.py", "new", "--data-dir", str(a.data), "--batch-size", "2" if SMOKE else "32", "--epochs", str(a.epochs),
                "--name", "pilot", "--size", "128", "--message", "64", "--noise", HIDDEN_NOISE]
    runpy.run_path(str(work / "main.py"), run_name="__main__")


def _mbrs_discriminator_optional():
    """MBRS's ``load_model`` requires ``D_<epoch>.pth``; the archived pilot checkpoints hold the encoder-decoder
    only. Load the discriminator when it exists, otherwise keep its fresh initialisation (disclosed)."""
    from network.Network import Network

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        if os.path.exists(path_discriminator):
            self.load_model_dis(path_discriminator)
        else:
            print(f"MBRS continuation: {path_discriminator} not found; discriminator initialised afresh", flush=True)

    Network.load_model = load_model


def run_mbrs(a, work: Path):
    import kornia.losses as KL

    # kornia >= 0.5: losses.ssim became a submodule; losses.ssim_loss returns the same (1 - SSIM) / 2 value.
    # Only MBRS's logged SSIM uses it (not its loss).
    if not callable(getattr(KL, "ssim", None)):
        KL.ssim = lambda x, y, window_size, reduction="mean": KL.ssim_loss(x, y, window_size, reduction=reduction)
    settings = {
        "project_name": "MBRS", "with_diffusion": False, "only_decoder": False,
        "epoch_number": a.epochs - a.start_epoch, "batch_size": 2 if SMOKE else 16,
        "train_continue": bool(a.continue_from),
        # MBRS reads results/<path>/models/EC_<epoch>.pth relative to the run directory
        "train_continue_path": os.path.relpath(a.continue_from, work / "results") if a.continue_from else "/",
        "train_continue_epoch": a.start_epoch, "dataset_path": str(a.data) + "/", "save_images_number": 4,
        "H": 128, "W": 128, "message_length": 64, "lr": 1e-3,
        "noise_layers": ["Combined([JpegMask(50),Jpeg(50),Identity()])"],
    }
    (work / "train_settings.json").write_text(json.dumps(settings, indent=1))
    (work / "results").mkdir(exist_ok=True)
    (work / "temp").mkdir(exist_ok=True)
    if a.continue_from:
        _mbrs_discriminator_optional()
    sys.argv = ["train.py"]
    runpy.run_path(str(work / "train.py"), run_name="__main__")


def run_fin(a, work: Path):
    import config as c  # FIN's module-level configuration

    c.epochs = a.epochs
    c.TRAIN_PATH = str(Path(a.data) / "train")
    c.VAL_PATH = str(Path(a.data) / "val")
    c.format_train = c.format_val = "png"
    c.MODEL_PATH = "experiments/pilot/"
    c.SAVE_freq = 1
    if SMOKE:
        c.batch_size, c.batchsize_val = 2, 2
    (work / "experiments" / "pilot").mkdir(parents=True, exist_ok=True)
    (work / "logging").mkdir(exist_ok=True)
    sys.argv = ["train.py"]
    runpy.run_path(str(work / "train.py"), run_name="__main__")


def _make_grid_range_kw():
    """CIN passes ``range=`` to torchvision's ``make_grid`` (diagnostic image saving); torchvision renamed it
    ``value_range``."""
    import torchvision.utils as tvu

    orig = tvu.make_grid

    def make_grid(tensor, *args, **kw):
        if "range" in kw:
            kw["value_range"] = kw.pop("range")
        return orig(tensor, *args, **kw)

    tvu.make_grid = make_grid


def run_cin(a, work: Path):
    import yaml

    _make_grid_range_kw()

    codes = work / "codes"
    opt = yaml.safe_load((codes / "options" / "opt.yml").read_text())
    opt["train/test"] = "train"
    opt["train"].update(manual_seed=a.seed, device_ids=[0], os_environ="0", epoch=a.epochs, num_workers=8,
                        checkpoint_per_epoch=1, saveTrainImgs_per_step=10 ** 9)
    if SMOKE:
        opt["train"].update(batch_size=2, num_workers=0)
    opt["train"]["val"]["per_epoch"] = a.epochs
    opt["train"]["resume"] = {"one_pth": False, "Partial": False, "Empty": True}
    opt["noise"]["option"] = "Combined"
    opt["path"]["logs_folder"] = str(work / "exp_")
    opt["path"]["train_folder"] = str(Path(a.data) / "train")
    opt["path"]["test_folder"] = str(Path(a.data) / "val")
    opt["network"]["message_length"] = 64
    n_train = len([p for p in (Path(a.data) / "train").iterdir()])
    opt["datasets"]["nDatasets"]["num"] = n_train
    yml = work / "opt_pilot.yml"
    yml.write_text(yaml.safe_dump(opt, sort_keys=False))
    src = (codes / "main.py").read_text().replace("'.../codes/options/opt.yml'", repr(str(yml)))
    os.chdir(codes)
    sys.path.insert(0, str(codes))
    g = {"__name__": "__main__", "__file__": str(codes / "main.py")}
    exec(compile(src, str(codes / "main.py"), "exec"), g)


RUNNERS = {"hidden": run_hidden, "mbrs": run_mbrs, "fin": run_fin, "cin": run_cin}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=sorted(RUNNERS))
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--continue-from", type=Path, default=None)
    p.add_argument("--start-epoch", type=int, default=0)
    a = p.parse_args()
    if a.continue_from and a.name not in ("hidden", "mbrs"):
        raise SystemExit(f"continuation is implemented for hidden and mbrs only, not {a.name}")
    from trackd_wm.seed import set_seed

    set_seed(a.seed, strict=False)
    _cuda_shim()
    _copy(a.repo, a.work)
    os.chdir(a.work)
    sys.path.insert(0, str(a.work))
    RUNNERS[a.name](a, a.work)


if __name__ == "__main__":
    main()
