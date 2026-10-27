"""Load trained baseline checkpoints behind the common harness interface.

Conventions are mapped explicitly: all learned baselines work on images in
[-1, 1]; HiDDeN, MBRS and CIN use messages in {0, 1} with a 0.5 decision
threshold (each repository's own rule); FIN uses {-0.5, 0.5} with threshold 0.
CIN has two decoding branches selected by a noise classifier (NSM) whose
training code is not released; the pilot uses an oracle selection (JPEG-type
attacks -> NIAM branch, otherwise the invertible branch), which can only favour
CIN, and this is reported.
"""
from __future__ import annotations

import importlib
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

import torch


@contextmanager
def _on_path(p: Path, purge: List[str]):
    """Import a repository's modules in isolation (several repositories reuse names such as ``utils``)."""
    saved = {k: v for k, v in sys.modules.items() if any(k == m or k.startswith(m + ".") for m in purge)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path.remove(str(p))
        for k in [k for k in sys.modules if any(k == m or k.startswith(m + ".") for m in purge)]:
            del sys.modules[k]
        sys.modules.update(saved)


def _load(path, device):
    return torch.load(path, map_location=device, weights_only=False)


class _Base:
    def __init__(self, device):
        self.device = device

    def set_attack(self, name: str):  # only CIN uses it
        pass


class HiDDeNWatermarker(_Base):
    def __init__(self, repo: Path, run_dir: Path, ckpt: Path, device):
        super().__init__(device)
        with _on_path(repo, ["options", "model", "noise_layers", "utils", "vgg_loss", "average_meter"]):
            from model.encoder_decoder import EncoderDecoder
            from noise_layers.noiser import Noiser
            import options  # noqa: F401  (needed to unpickle the configuration)

            with open(run_dir / "options-and-config.pickle", "rb") as f:
                _train_opts = pickle.load(f)
                _noise = pickle.load(f)
                cfg = pickle.load(f)
            self.net = EncoderDecoder(cfg, Noiser([], device)).to(device).eval()
        self.net.load_state_dict(_load(ckpt, device)["enc-dec-model"])

    @torch.no_grad()
    def embed(self, x, bits):
        return (self.net.encoder(x.to(self.device) * 2 - 1, bits.to(self.device)) + 1) / 2

    @torch.no_grad()
    def extract(self, y):
        return (self.net.decoder(y.to(self.device) * 2 - 1) > 0.5).float()


class MBRSWatermarker(_Base):
    def __init__(self, repo: Path, ckpt: Path, device, message_length: int = 64, diffusion: bool = False):
        """``message_length`` and ``diffusion`` select the released 30-bit diffusion model (stage I-6); the
        defaults are the 64-bit model trained in this work."""
        super().__init__(device)
        with _on_path(repo, ["network", "utils"]):
            mod = importlib.import_module("network.Encoder_MP_Decoder")
            cls = mod.EncoderDecoder_Diffusion if diffusion else mod.EncoderDecoder
            self.net = cls(128, 128, message_length, ["Identity()"]).to(device).eval()
        self.net.load_state_dict(_load(ckpt, device))

    @torch.no_grad()
    def embed(self, x, bits):
        return (self.net.encoder(x.to(self.device) * 2 - 1, bits.to(self.device)) + 1) / 2

    @torch.no_grad()
    def extract(self, y):
        return (self.net.decoder(y.to(self.device) * 2 - 1) > 0.5).float()


class FINWatermarker(_Base):
    def __init__(self, repo: Path, ckpt: Path, device):
        super().__init__(device)
        with _on_path(repo, ["models", "utils", "config"]):
            from models.encoder_decoder import FED

            self.net = FED(False, 64).to(device).eval()
        sd = {k: v for k, v in _load(ckpt, device)["net"].items() if "tmp_var" not in k}
        self.net.load_state_dict(sd)

    @torch.no_grad()
    def embed(self, x, bits):
        stego, _ = self.net([x.to(self.device) * 2 - 1, bits.to(self.device) - 0.5])
        return (stego + 1) / 2

    @torch.no_grad()
    def extract(self, y):
        b = y.shape[0]
        _, msg = self.net([y.to(self.device) * 2 - 1, torch.zeros(b, 64, device=self.device)], rev=True)
        return (msg > 0).float()


class CINWatermarker(_Base):
    def __init__(self, repo: Path, opt_yml: Path, ckpt: Path, device):
        super().__init__(device)
        import yaml

        codes = repo / "codes"
        with _on_path(codes, ["models", "utils"]):
            from utils.yml import dict_to_nonedict
            from models.CIN import CIN

            raw = yaml.safe_load(Path(opt_yml).read_text())
            tmp = Path(opt_yml).parent / "eval_temp"  # CIN's main() sets this at run time; JpegTest needs it
            tmp.mkdir(exist_ok=True)
            raw["path"]["folder_temp"] = str(tmp)
            opt = dict_to_nonedict(raw)
            self.net = CIN(opt, device).to(device).eval()
        sd = _load(ckpt, device)["cinNet"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.net.load_state_dict(sd, strict=True)  # verified: no missing or unexpected keys
        self.jpeg_branch = False

    def set_attack(self, name: str):
        self.jpeg_branch = name.startswith("jpeg")  # oracle noise selection (see module docstring)

    @torch.no_grad()
    def embed(self, x, bits):
        return (self.net.encoder(x.to(self.device) * 2 - 1, bits.to(self.device)) + 1) / 2

    @torch.no_grad()
    def extract(self, y):
        _, _, _, msg = self.net.test_decoder(y.to(self.device) * 2 - 1, 1 if self.jpeg_branch else 0)
        return msg.round().clamp(0, 1)


class DWTDCTWatermarker(_Base):
    """invisible-watermark ``EmbedMaxDct`` called directly (the public API rejects images < 256 x 256).

    The library embeds in the U channel of YUV; its default scale (36) is used as the native operating point.
    CPU only, as in the library.
    """

    def __init__(self, repo: Path, device="cpu", scale: int = 36):
        super().__init__("cpu")
        with _on_path(repo, ["imwatermark"]):
            self.mod = importlib.import_module("imwatermark.maxDct")
        self.scale = scale

    def _cv(self, img):
        import numpy as np

        a = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        return np.ascontiguousarray(a[:, :, ::-1])  # BGR

    def embed(self, x, bits):
        import numpy as np

        out = []
        for img, b in zip(x.cpu(), bits.cpu()):
            e = self.mod.EmbedMaxDct(watermarks=[int(v) for v in b], wmLen=64, scales=[0, self.scale, 0], block=4)
            y = e.encode(self._cv(img).copy())
            y = np.clip(y.astype("float64"), 0, 255)[:, :, ::-1].copy()
            out.append(torch.from_numpy(y).permute(2, 0, 1).float() / 255)
        return torch.stack(out)

    def extract(self, y):
        out = []
        for img in y.cpu():
            d = self.mod.EmbedMaxDct(watermarks=[], wmLen=64, scales=[0, self.scale, 0], block=4)
            out.append(torch.tensor([float(v) for v in d.decode(self._cv(img).copy())]))
        return torch.stack(out)
