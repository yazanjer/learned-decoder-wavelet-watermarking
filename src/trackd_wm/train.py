"""Deterministic training of the proposed method and its transform controls.

Loss (identical for every variant): 10 * BCE(message) + MSE on images in
[-1, 1], i.e. the MBRS weighting (decoder 10, encoder 1) without a
discriminator. Adam, lr 1e-3, batch 16 (MBRS defaults). A checkpoint is saved
after every epoch; learning curves are evaluated afterwards on the
checkpoints with the common harness, so that every method is evaluated by
the same code.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import build_wmnet
from .noise import apply_training_noise
from .seed import set_seed, worker_init_fn

OBJECTIVES = ("mbrs10", "psnr_projection", "dual_ascent")


def project_to_psnr(x, r, target_psnr: float, eps: float = 1e-12):
    """Rescale each image's residual to the mean-squared error of ``target_psnr`` (images in [0, 1]).

    Proposal O1 (docs/OBJECTIVE_PROPOSAL.md): fidelity becomes a hard constraint at the evaluation point.
    Gradients flow through the residual and through the normalisation.
    """
    mse_t = 10.0 ** (-target_psnr / 10.0)
    cur = r.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    return x + r * torch.sqrt(mse_t / (cur + eps))


def objective_step(cfg, net, x, m, rng, state: dict):
    """One forward pass and loss for the configured objective.

    ``mbrs10`` is the approved pilot objective and the default; the others are proposals that are not used by
    any approved run (docs/OBJECTIVE_PROPOSAL.md)."""
    kind = cfg.get("objective", "mbrs10")
    if kind not in OBJECTIVES:
        raise ValueError(kind)
    if kind == "psnr_projection":
        xw = project_to_psnr(x, net.encoder.residual(x, m), cfg.get("target_psnr", 40.0))
    else:
        xw = net.encoder(x, m)
    y, _ = apply_training_noise(xw, x, rng)
    logits = net.decoder(y)
    bce = F.binary_cross_entropy_with_logits(logits, m)
    mse = F.mse_loss(xw * 2 - 1, x * 2 - 1)
    if kind == "mbrs10":
        loss = 10.0 * bce + mse
    elif kind == "psnr_projection":
        loss = bce
    else:  # dual ascent on log(lambda) towards the target PSNR (average constraint)
        lam = float(torch.exp(torch.tensor(state.setdefault("log_lambda", 0.0))))
        loss = bce + lam * mse
        with torch.no_grad():
            psnr = -10.0 * torch.log10(F.mse_loss(xw.clamp(0, 1), x) + 1e-12)
        state["log_lambda"] += cfg.get("dual_eta", 0.01) * (cfg.get("target_psnr", 40.0) - float(psnr))
    return loss, xw, logits


class WMNetWatermarker:
    """Common harness interface for WMNet checkpoints.

    ``project_psnr`` (models trained with the PSNR-projection objective): the residual of each image is rescaled
    to the target PSNR exactly as in training, so that the native strength (s = 1) is the training fidelity.
    """

    def __init__(self, net, device, project_psnr=None):
        self.net, self.device = net.to(device).eval(), device
        self.project_psnr = project_psnr

    @torch.no_grad()
    def embed(self, x, bits):
        x, bits = x.to(self.device), bits.to(self.device)
        if self.project_psnr is not None:
            return project_to_psnr(x, self.net.encoder.residual(x, bits), float(self.project_psnr))
        return self.net.encoder(x, bits)

    @torch.no_grad()
    def extract(self, y):
        return (self.net.decoder(y.to(self.device)) > 0).float()


def train_wmnet(cfg, train_ds, out_dir: Path, device, epochs: int, log=print) -> dict:
    set_seed(cfg.seed, strict=cfg.strict_determinism)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    net = build_wmnet(cfg.model).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, generator=g, worker_init_fn=worker_init_fn(cfg.seed),
                    persistent_workers=cfg.num_workers > 0)
    rng = random.Random(cfg.seed)
    msg_gen = torch.Generator().manual_seed(cfg.seed + 1)
    info = {"method": cfg.name, "transform": cfg.model.transform, "seed": cfg.seed,
            "params_total": net.total_params(), "params_transform": net.transform_params(), "epochs": []}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    max_steps = cfg.get("max_steps", None)
    obj_state: dict = {}
    info["objective"] = cfg.get("objective", "mbrs10")
    info["params_trainable"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # numerical-stability safeguard (stage I-2): stop at the first non-finite loss or parameter; the last saved
    # checkpoint is then the last finite one. Only enabled when requested, so earlier runs are unchanged.
    guard = bool(cfg.get("finite_guard", False))
    stopped = False
    for ep in range(1, epochs + 1):
        net.train()
        t0, n, tot = time.time(), 0, 0.0
        psnr_sum, acc_sum = 0.0, 0.0
        for x in dl:
            x = x.to(device, non_blocking=True)
            m = torch.randint(0, 2, (x.shape[0], cfg.model.bits), generator=msg_gen).float().to(device)
            loss, xw, logits = objective_step(cfg, net, x, m, rng, obj_state)
            if guard and not bool(torch.isfinite(loss)):
                info["nonfinite"] = {"epoch": ep, "step": n + 1, "what": "loss"}
                stopped = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            if guard:
                with torch.no_grad():
                    mse_b = (xw.detach().clamp(0, 1) - x).pow(2).mean(dim=(1, 2, 3))
                    psnr_sum += float((-10.0 * torch.log10(mse_b + 1e-12)).mean())
                    acc_sum += float(((logits.detach() > 0).float() == m).float().mean())
            n += 1
            if max_steps and n >= max_steps:
                break
        if guard and not stopped and not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
            info["nonfinite"] = {"epoch": ep, "step": n, "what": "parameters"}
            stopped = True
        if stopped:
            info["stopped_nonfinite"] = True
            (out_dir / "train_info.json").write_text(json.dumps(info, indent=1))
            log(json.dumps(info["nonfinite"]))
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        ck = {"net": net.state_dict(), "epoch": ep, "cfg": dict(cfg.model)}
        if info["objective"] != "mbrs10":
            ck["objective"] = {"kind": info["objective"], "target_psnr": float(cfg.get("target_psnr", 40.0))}
        torch.save(ck, out_dir / "ckpt" / f"epoch_{ep:03d}.pt")
        rec = {"epoch": ep, "seconds": dt, "steps": n, "loss": tot / max(n, 1)}
        if guard:
            rec["train_psnr_clamped"] = psnr_sum / max(n, 1)
            rec["train_bitacc_noisy"] = acc_sum / max(n, 1)
        if device.type == "cuda":
            rec["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        info["epochs"].append(rec)
        log(json.dumps(rec))
        (out_dir / "train_info.json").write_text(json.dumps(info, indent=1))
    return info


def load_wmnet(ckpt_path: Path, device):
    from omegaconf import OmegaConf

    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = build_wmnet(OmegaConf.create(st["cfg"]))
    net.load_state_dict(st["net"])
    obj = st.get("objective") or {}
    proj = obj.get("target_psnr") if obj.get("kind") == "psnr_projection" else None
    return WMNetWatermarker(net, device, project_psnr=proj)
