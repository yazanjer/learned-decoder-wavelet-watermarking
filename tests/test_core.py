"""Unit tests for properties the pilot relies on (run on CPU: pytest -q tests)."""
import numpy as np
import torch

from trackd_wm import attacks, metrics
from trackd_wm.models import WMNet
from trackd_wm.transforms import Lifting1D, build_transform, n_params


def _randomise(m):
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m


def test_perfect_reconstruction_float():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 128, 128)
    for kind in ["lift_learned", "lift_cdf97", "lift_haar", "coupling_matched", "coupling_reorder_matched",
                 "coupling_haar_matched"]:
        m = _randomise(build_transform(kind))
        assert float((m.synthesis(m.analysis(x)) - x).abs().max()) < 1e-5, kind   # criterion P1


def test_integer_lifting_exact():
    m = _randomise(build_transform("lift_learned_int")).double()
    x = torch.randint(0, 256, (2, 3, 64, 64)).double()
    assert torch.equal(m.synthesis(m.analysis(x)), x)


def test_cdf97_vanishing_moments():
    m = Lifting1D(-1, "cdf97", False, False, 1, 4).double()
    t = torch.arange(64, dtype=torch.float64) / 64
    for deg in range(4):
        _, d = m(t.pow(deg).view(1, 1, 1, -1))
        assert float(d[..., 3:-3].abs().max()) < 1e-10


def test_parameter_matching():
    a = WMNet("lift_learned")
    for kind in ("coupling_matched", "coupling_reorder_matched", "coupling_haar_matched"):
        b = WMNet(kind)
        assert abs(a.total_params() - b.total_params()) / a.total_params() < 0.01, kind


def test_residual_reaches_all_channels():
    """Defect found in the pilot (legacy coupling_matched): the embedding must reach every colour channel equally.

    Full pixel coverage is required for the lifting variants and the Haar-mixed coupling. The channel-reorder
    fix alone is a lazy-wavelet decomposition: the deepest bands cover polyphase position (0, 0) of the first
    level only, i.e. one pixel in four of every channel, which is recorded here as its expected coverage."""
    torch.manual_seed(0)
    x, m = torch.rand(4, 3, 128, 128), torch.randint(0, 2, (4, 64)).float()
    expected = {"lift_learned": 0.99, "lift_cdf97": 0.99, "coupling_haar_matched": 0.99,
                "coupling_reorder_matched": 0.24}
    for kind, lo in expected.items():
        r = WMNet(kind).encoder(x, m) - x
        frac = [(r[:, c].abs() > 1e-6).float().mean().item() for c in range(3)]
        assert min(frac) > lo and max(frac) - min(frac) < 0.01, (kind, frac)


def test_haar_coupling_starts_as_haar():
    """At initialisation the corrected control equals a two-level Haar transform (LL = scaled 2x2 block mean)."""
    t = build_transform("coupling_haar_matched")
    x = torch.rand(1, 3, 32, 32)
    ll = t.analysis(x)[0][0]
    ref = torch.nn.functional.avg_pool2d(x, 2) * 2.0
    assert float((ll - ref).abs().max()) < 1e-5


def test_jpeg_and_metrics():
    x = torch.rand(2, 3, 64, 64)
    y = attacks.jpeg(x, 50)
    assert y.shape == x.shape and float(metrics.psnr_rgb(y, x).mean()) < 60
    assert float(metrics.psnr_rgb(x, x).mean()) > 100
    b = torch.randint(0, 2, (2, 64)).float()
    assert float(metrics.bit_accuracy(b, b).mean()) == 1.0


def test_legacy_coupling_defect_is_reproduced():
    """Regression record: the legacy control confines the embedding to one colour channel (pilot G3 defect)."""
    torch.manual_seed(0)
    x, m = torch.rand(2, 3, 128, 128), torch.randint(0, 2, (2, 64)).float()
    r = WMNet("coupling_matched").encoder(x, m) - x
    assert float(r[:, 1].abs().max()) == 0.0 and float(r[:, 2].abs().max()) == 0.0


def test_gradient_reaches_every_parameter_group():
    """Gradient propagation: the message loss reaches the encoder body, the transforms' learned operators
    (after one step away from zero initialisation) and the decoder, for every trainable variant."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    x, m = torch.rand(2, 3, 128, 128), torch.randint(0, 2, (2, 64)).float()
    for kind in ("lift_learned", "coupling_reorder_matched", "coupling_haar_matched"):
        net = _randomise_transforms(WMNet(kind))
        loss = F.binary_cross_entropy_with_logits(net.decoder(net.encoder(x, m)), m)
        loss.backward()
        for part in (net.encoder.body, net.encoder.t, net.decoder.t, net.decoder.fc):
            g = sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None)
            assert g > 0, (kind, part.__class__.__name__)


def _randomise_transforms(net):
    for t in (net.encoder.t, net.decoder.t):
        _randomise(t)
    return net


def test_checkpoint_compatibility_legacy_coupling():
    """Pilot checkpoints of the legacy control load into the current code (state-dict keys unchanged)."""
    a = WMNet("coupling_matched").state_dict()
    b = WMNet("coupling_matched").load_state_dict(a, strict=True)
    assert not b.missing_keys and not b.unexpected_keys


# --------------------------------------------------------------------------- stage I-2 additions (run in the pod)
def test_frozen_encoder_transform_is_fixed_cdf97_and_decoder_learns():
    """V2: the encoder's learned lifting stays at its CDF 9/7 initialisation; the decoder's transform trains."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    net = WMNet("lift_learned", freeze_encoder_transform=True)
    assert all(not p.requires_grad for p in net.encoder.t.parameters())
    ref = build_transform("lift_cdf97")
    x, m = torch.rand(2, 3, 128, 128), torch.randint(0, 2, (2, 64)).float()
    for a, b in zip(net.encoder.t.analysis(x)[-1], ref.analysis(x)[-1]):
        assert torch.allclose(a, b, atol=1e-6)
    enc0 = [p.clone() for p in net.encoder.t.parameters()]
    dec0 = [p.clone() for p in net.decoder.t.parameters()]
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(2):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(net.decoder(net.encoder(x, m)), m).backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(enc0, net.encoder.t.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(dec0, net.decoder.t.parameters()))


def test_frozen_encoder_coupling_starts_as_haar_and_state_dict_unchanged():
    net = WMNet("coupling_haar_matched", freeze_encoder_transform=True)
    ref = WMNet("coupling_haar_matched")
    assert set(net.state_dict()) == set(ref.state_dict())
    assert all(not p.requires_grad for p in net.encoder.t.parameters())
    assert any(p.requires_grad for p in net.decoder.t.parameters())


def test_psnr_projection_hits_target_and_watermarker_uses_it():
    from trackd_wm.train import WMNetWatermarker, project_to_psnr

    torch.manual_seed(0)
    x, r = torch.rand(4, 3, 128, 128), 0.1 * torch.randn(4, 3, 128, 128)
    mse = (project_to_psnr(x, r, 40.0) - x).pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(-10 * torch.log10(mse), torch.full((4,), 40.0), atol=1e-3)
    net = WMNet("lift_cdf97")
    for p in net.encoder.out.parameters():
        torch.nn.init.normal_(p, std=0.05)
    wm = WMNetWatermarker(net, torch.device("cpu"), project_psnr=40.0)
    m = torch.randint(0, 2, (4, 64)).float()
    mse = (wm.embed(x, m) - x).pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(-10 * torch.log10(mse), torch.full((4,), 40.0), atol=1e-2)


def test_extended_attacks_deterministic_and_8bit():
    torch.manual_seed(0)
    x = attacks.quantize_8bit(torch.rand(3, 3, 128, 128))
    for name in attacks.EXTENDED_ATTACKS:
        a, b = attacks.ATTACKS[name](x), attacks.ATTACKS[name](x)
        assert a.shape == x.shape and torch.equal(a, b), name
        assert float(a.min()) >= 0 and float(a.max()) <= 1
        assert torch.equal(a, attacks.quantize_8bit(a)), name



# --------------------------------------------------------------------------- stage I-3a additions (run in the pod)
def test_v1plus_frontend_is_capacity_matched_and_identity_at_init():
    from trackd_wm.models import FrontEnd

    v2 = WMNet("lift_learned", freeze_encoder_transform=True)
    v1p = WMNet("lift_cdf97", decoder_frontend_hidden=98)
    v1 = WMNet("lift_cdf97")
    extra = n_params(v1p) - n_params(v1)
    learned = n_params(v2.decoder.t)
    assert abs(extra - learned) / learned < 0.01, (extra, learned)
    f = FrontEnd(12, 98)
    z = torch.randn(2, 12, 32, 32)
    assert torch.equal(f(z), z)
    assert set(WMNet("lift_cdf97").state_dict()) < set(v1p.state_dict())


def test_original_design_v0_learns_both_transforms():
    v0 = WMNet("lift_learned")
    assert any(p.requires_grad for p in v0.encoder.t.parameters())
    assert any(p.requires_grad for p in v0.decoder.t.parameters())


def test_reconstruction_diagnostics_on_fixed_and_random_transforms():
    from trackd_wm.pilot import recon_diagnostics

    torch.manual_seed(0)
    xs = torch.rand(4, 3, 64, 64)
    d = recon_diagnostics(build_transform("lift_cdf97"), xs, iters=5, n_power=1)
    assert d["float64"]["recon_max_abs"] < 1e-10 and d["float32"]["recon_max_abs"] < 1e-4
    c = d["conditioning_float64"][0]
    assert "error" not in c and np.isfinite(c["condition_number"]) and c["condition_number"] > 0
    d = recon_diagnostics(_randomise(build_transform("lift_learned")), xs * 2 - 1, iters=3, n_power=1)
    assert d["float64"]["recon_max_abs"] < 1e-8


# --------------------------------------------------------------------------- stage I-3b additions (run in the pod)
def test_i3b_variants_build_and_decode():
    """L = 1, L = 3 and no-attention V2 variants build, embed and decode 64 bits at 128 x 128."""
    torch.manual_seed(0)
    x, m = torch.rand(2, 3, 128, 128), torch.randint(0, 2, (2, 64)).float()
    for kw in ({"levels": 1}, {"levels": 3}, {"attention": False}):
        net = WMNet("lift_learned", freeze_encoder_transform=True, **kw)
        assert net.decoder(net.encoder(x, m)).shape == (2, 64), kw


def test_wmnet_cfg_applies_overrides_only_to_named_variant():
    from trackd_wm.pilot import load_cfg, wmnet_cfg

    cfg = load_cfg([])
    assert wmnet_cfg(cfg, "v2l3").model.levels == 3 and wmnet_cfg(cfg, "v2").model.levels == 2
    assert wmnet_cfg(cfg, "v2s").model.attention is False and wmnet_cfg(cfg, "v2l3").model.freeze_encoder_transform
    assert "freeze_encoder_transform" not in wmnet_cfg(cfg, "proposed").model


# --------------------------------------------------------------------------- stage I-4 additions (run in the pod)
def test_i4_guard_and_hier_bootstrap():
    import os
    import pytest
    from trackd_wm.pilot import hier_bootstrap, load_cfg, step_i4

    os.environ.pop("I4_FROZEN", None)
    with pytest.raises(SystemExit):
        step_i4(load_cfg(["work=/tmp/_i4_guard"]))
    rng = np.random.default_rng(0)
    d = 0.02 + 0.01 * rng.standard_normal((5, 200))
    lo, hi = hier_bootstrap(d, 500)
    assert lo < d.mean() < hi and hi - lo < 0.02


def test_mixed_model_detects_seed_level_effect():
    from trackd_wm.pilot import mixed_model_test

    rng = np.random.default_rng(1)
    img = rng.normal(0.9, 0.05, 60)
    a = np.stack([img + 0.03 + rng.normal(0, 0.002) + rng.normal(0, 0.01, 60) for _ in range(5)])
    b = np.stack([img + rng.normal(0, 0.002) + rng.normal(0, 0.01, 60) for _ in range(5)])
    r = mixed_model_test(a, b)
    assert abs(r["effect"] - 0.03) < 0.01 and r["p"] < 0.05


def test_cpu_threads_uses_affinity_and_cap(monkeypatch):
    import os
    from trackd_wm.pilot import cpu_threads

    monkeypatch.setenv("TORCH_THREADS", "8")
    n = cpu_threads()
    assert 1 <= n <= min(8, len(os.sched_getaffinity(0)))


def _load_zenodo_upload():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "pod" / "zenodo_upload.py"
    spec = importlib.util.spec_from_file_location("zenodo_upload", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_zenodo_upload_retries_skips_and_reports_missing(tmp_path, monkeypatch):
    """Mocked Zenodo API: one archive already present (skipped), one with two transient failures (uploaded after
    retries), one that never succeeds (missing, exit code 1). No network access."""
    import hashlib
    import json
    import urllib.error

    monkeypatch.setenv("ZENODO_TOKEN", "test-token")
    zu = _load_zenodo_upload()
    arc = tmp_path / "archive"
    arc.mkdir()
    for n, b in (("a.tar", b"aaa"), ("b.tar", b"bbbb"), ("c.tar", b"ccccc")):
        (arc / n).write_bytes(b)
    md5 = lambda b: hashlib.md5(b).hexdigest()
    bucket = {"pilot_p1_a.tar": "md5:" + md5(b"aaa")}
    calls = {"b": 0, "c": 0}
    outage = {"c": True}

    def fake_req(url, method="GET", data=None, headers=None, timeout=600):
        if method == "GET":
            return {"submitted": False, "links": {"bucket": "https://bucket"},
                    "files": [{"filename": k, "checksum": v} for k, v in bucket.items()]}
        key = url.rsplit("/", 1)[-1]
        body = data.read()
        if key == "pilot_p1_b.tar":
            calls["b"] += 1
            if calls["b"] < 3:
                raise urllib.error.URLError("connection reset")
        if key == "pilot_p1_c.tar":
            calls["c"] += 1
            if outage["c"]:
                raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        assert key != "pilot_p1_a.tar", "already present archives must not be re-uploaded"
        bucket[key] = "md5:" + md5(body)
        return {"checksum": bucket[key]}

    monkeypatch.setattr(zu, "req", fake_req)
    man = tmp_path / "results" / "ZENODO_MANIFEST.json"
    rc = zu.main(arc, man, retries=4, delay=0.0, sleep=lambda s: None)
    m = json.loads(man.read_text())
    assert rc == 1 and not m["complete"] and m["missing"] == ["pilot_p1_c.tar"]
    assert m["files"]["pilot_p1_a.tar"]["action"] == "skipped_already_present"
    assert m["files"]["pilot_p1_b.tar"]["action"] == "uploaded" and calls["b"] == 3
    assert m["files"]["pilot_p1_c.tar"]["action"] == "failed" and calls["c"] == 4
    assert m["n_verified"] == 2 and m["n_archives"] == 3
    # second run once the outage is over: only c is uploaded, the manifest is complete, exit code 0
    outage["c"] = False
    rc = zu.main(arc, man, retries=2, delay=0.0, sleep=lambda s: None)
    m = json.loads(man.read_text())
    assert rc == 0 and m["complete"] and m["files"]["pilot_p1_b.tar"]["action"] == "skipped_already_present"
    assert m["files"]["pilot_p1_c.tar"]["action"] == "uploaded" and calls["c"] == 5


def test_zenodo_upload_does_not_retry_auth_errors(tmp_path, monkeypatch):
    import urllib.error

    import pytest

    monkeypatch.setenv("ZENODO_TOKEN", "test-token")
    zu = _load_zenodo_upload()
    n = {"calls": 0}

    def denied():
        n["calls"] += 1
        raise urllib.error.HTTPError("u", 403, "forbidden", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        zu.with_retry(denied, "x", 5, 0.0, sleep=lambda s: None)
    assert n["calls"] == 1


def test_holm_seed_t_interval_and_detection_threshold():
    from scipy import stats

    from trackd_wm.pilot import detection_threshold, holm, seed_t_interval

    assert holm({"a": 0.01, "b": 0.04, "c": None}) == {"a": 0.02, "b": 0.04}
    assert holm({"a": 0.03, "b": 0.02}) == {"b": 0.04, "a": 0.04}
    lo, hi = seed_t_interval([0.114, 0.082, 0.037, 0.010])
    assert lo < 0 < hi  # the 4-seed H1 interval of I-4 includes 0 (HANDOVER_TRACKD_P1.md, Section 3)
    assert seed_t_interval([0.1]) == [None, None]
    k = detection_threshold(64)
    assert stats.binom.sf(k - 1, 64, 0.5) <= 1e-3 < stats.binom.sf(k - 2, 64, 0.5)


def test_i4_secondary_family_on_synthetic_arrays(tmp_path):
    from trackd_wm.pilot import I4_SECONDARY_ENDPOINTS, i4_secondary, load_cfg

    cfg = load_cfg([f"work={tmp_path}", "i4.bootstrap=50"])
    rng = np.random.default_rng(0)
    keys = sorted({k for _, k, _ in I4_SECONDARY_ENDPOINTS})
    for v, shift in (("v1p", 0.0), ("v2", 0.03), ("v3", 0.02)):
        for s in range(3):
            np.savez_compressed(tmp_path / f"{v}_s{s}_test_per_image.npz",
                                **{k: np.clip(rng.normal(0.9 + shift, 0.03, 40), 0, 1) for k in keys})
    r = i4_secondary(cfg, tmp_path, [0, 1, 2])
    assert r["used_for_decisions"] is False
    assert len(r["family"]) == 2 * len(I4_SECONDARY_ENDPOINTS)
    for name, row in r["family"].items():
        assert "theta_hat" in row and len(row["seed_paired_t_ci95"]) == 2, name
        if name.endswith("tpr_fpr1e-3_q50_40dB"):
            assert "p_holm_secondary" not in row and row["in_holm_family"] is False
        elif "p" in row["mixed_model"]:
            assert row["p_holm_secondary"] >= row["mixed_model"]["p"]


def test_compare_per_image_identical_and_different(tmp_path):
    from trackd_wm.pilot import compare_per_image

    a = {"matched_40dB__bitacc_jpeg50": np.array([0.9, 1.0]), "matched_40dB__ssim": np.array([0.97, 0.98])}
    np.savez_compressed(tmp_path / "a.npz", **a)
    np.savez_compressed(tmp_path / "b.npz", **a)
    r = compare_per_image(tmp_path / "a.npz", tmp_path / "b.npz")
    assert r["identical"] and r["max_abs_diff"] == 0.0 and r["arrays_compared"] == 2
    np.savez_compressed(tmp_path / "c.npz", **dict(a, matched_40dB__bitacc_jpeg50=np.array([0.9, 0.95])))
    r = compare_per_image(tmp_path / "c.npz", tmp_path / "a.npz")
    assert not r["identical"] and abs(r["max_abs_diff"] - 0.05) < 1e-12 and r["q50_40dB_mean_diff"] < 0


def test_orch_manifest_complete_only_if_every_upload_succeeded(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "orch_manifest", Path(__file__).resolve().parents[1] / "scripts" / "pod" / "orch_manifest.py")
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    d1, d2 = tmp_path / "archive", tmp_path / "archive_uploaded"
    d1.mkdir(), d2.mkdir()
    (d1 / "i4_s2_v2_checkpoints.tar").write_bytes(b"x" * 10)
    (d2 / "i4_s3_v2_checkpoints.tar").write_bytes(b"y" * 5)
    out = tmp_path / "m.json"
    assert om.main([str(out), f"{d1}=0", f"{d2}=0"]) == 0
    m = __import__("json").loads(out.read_text())
    assert m["complete"] and len(m["files"]) == 2 and m["missing"] == []
    assert om.main([str(out), f"{d1}=0", f"{d2}=22"]) == 1
    m = __import__("json").loads(out.read_text())
    assert not m["complete"] and m["missing"] == ["i4_s3_v2_checkpoints.tar"]


def test_i5_attack_suite_deterministic_8bit_and_shape():
    from trackd_wm.attacks import ATTACKS, I5_ATTACKS, EXTENDED_ATTACKS
    from trackd_wm.i5 import smoke_attacks

    assert len(I5_ATTACKS) == len(set(I5_ATTACKS)) == 29 and set(EXTENDED_ATTACKS) <= set(I5_ATTACKS)
    torch.manual_seed(0)
    x = attacks.quantize_8bit(torch.rand(2, 3, 128, 128))
    for a in I5_ATTACKS:
        y1, y2 = ATTACKS[a](x), ATTACKS[a](x)
        assert y1.shape == x.shape and torch.equal(y1, y2), a
        assert torch.allclose(torch.round(y1 * 255) / 255, y1, atol=1e-6) and 0 <= float(y1.min()) <= float(y1.max()) <= 1, a
    c = torch.full((1, 3, 128, 128), 100 / 255)
    for a in ("median3", "crop075", "crop09", "blur2", "contrast12"):
        assert torch.allclose(ATTACKS[a](c), c, atol=1 / 255), a  # constant images are fixed points
    e = ATTACKS["erase10"](c)
    assert abs(float(((e - c).abs() > 1e-3).any(1).float().mean()) - 0.10) < 0.01
    assert torch.allclose(ATTACKS["bright12"](c), attacks.quantize_8bit(c * 1.2))
    assert smoke_attacks([x[0], x[1]], n=2)["status"] == "ok"


def test_i5_tiles_bootstrap_and_band_energy():
    from trackd_wm.i5 import band_energy, hier_bootstrap_vec, tile_positions

    pos = tile_positions(2040, 1356, 10, 128, np.random.default_rng(0))
    assert len(set(pos)) == 10 and all(x % 128 == 0 and y % 128 == 0 and x + 128 <= 2040 and y + 128 <= 1356
                                       for x, y in pos)
    assert pos == tile_positions(2040, 1356, 10, 128, np.random.default_rng(0))
    d = np.random.default_rng(1).normal(0.02, 0.05, (4, 300))
    lo, hi = hier_bootstrap_vec(d, 2000)
    assert lo < d.mean() < hi
    xs = torch.rand(2, 3, 64, 64) * 2 - 1
    for kind in ("lift_learned", "coupling_haar_matched"):
        e = band_energy(build_transform(kind), xs)
        assert abs(sum(e.values()) - 1) < 1e-9 and all(v >= 0 for v in e.values()), kind
    assert set(band_energy(build_transform("lift_cdf97"), xs)) == {"L1_lh", "L1_hl", "L1_hh", "L2_lh", "L2_hl",
                                                                   "L2_hh", "L2_ll"}


def test_i5_m1_analysis_on_synthetic_arrays(tmp_path):
    from trackd_wm.i5 import m1_analysis
    from trackd_wm.pilot import load_cfg

    cfg = load_cfg([f"work={tmp_path}", "i5.bootstrap=200"])
    rng = np.random.default_rng(0)
    keys = ["coco__matched_40dB__bitacc_jpeg50", "div2k__matched_35dB__bitacc_webp50", "coco__matched_45dB__ssim"]
    res = {"models": {}}
    for v, shift in (("v1p", 0.0), ("v2", 0.03), ("v3", 0.02)):
        for s in range(4):
            if v == "v3" and s == 3:
                continue  # incomplete seed: excluded from every comparison
            np.savez_compressed(tmp_path / f"{v}_s{s}_i5_per_image.npz",
                                **{k: np.clip(rng.normal(0.9 + shift, 0.03, 50), 0, 1).astype(np.float32) for k in keys})
            rec = {"float32": {"recon_max_abs": 1e-3 * (s + 1)}, "conditioning_float64": [{"condition_number": 2.0 + s}]}
            res["models"][f"{v}_s{s}"] = {"status": "ok", "operator": {"reconstruction": rec}}
    a = m1_analysis(cfg, tmp_path, res)
    assert a["seeds_used"] == [0, 1, 2] and a["used_for_decisions"] is False
    assert set(a["profiles"]) == {"v2-v1p", "v3-v1p", "v2-v3"}
    row = a["profiles"]["v2-v1p"]["coco__matched_40dB__bitacc_jpeg50"]
    assert row["ci95_hier_bootstrap"][0] < row["theta_hat"] < row["ci95_hier_bootstrap"][1] and row["theta_hat"] > 0
    assert len(a["operator_association"]["v2"]["q50_40dB_coco"]) == 3


def test_i5_inputs_local_dir_and_continuation_layout(tmp_path, monkeypatch):
    import os
    import pickle
    import tarfile

    from trackd_wm.i5 import _prepare_continuation, fetch_input, step_i5_inputs
    from trackd_wm.pilot import load_cfg

    cfg = load_cfg([f"work={tmp_path / 'w'}"])
    src = tmp_path / "src"
    (src / "mbrs").mkdir(parents=True)
    (src / "mbrs" / "epoch_020.pth").write_bytes(b"ec")
    (src / "hidden" / "work" / "runs" / "pilot x").mkdir(parents=True)
    (src / "hidden" / "epoch_020.pyt").write_bytes(b"ck")
    (src / "hidden" / "work" / "runs" / "pilot x" / "options-and-config.pickle").write_bytes(pickle.dumps(1))
    inp = tmp_path / "inputs"
    inp.mkdir()
    for n, fn in (("mbrs", "g3_mbrs_checkpoints.tar"), ("hidden", "pilot_p1_g3_hidden_checkpoints.tar")):
        with tarfile.open(inp / fn, "w") as t:
            t.add(src / n, arcname=n)
    monkeypatch.setenv("I5_INPUT_DIR", str(inp))
    monkeypatch.delenv("ZENODO_READ_TOKEN", raising=False)
    root = tmp_path / "w" / "i5_inputs"
    assert fetch_input(cfg, "g3_mbrs", root)["source"] == "I5_INPUT_DIR/g3_mbrs_checkpoints.tar"
    assert fetch_input(cfg, "g3_hidden", root)["source"].endswith("pilot_p1_g3_hidden_checkpoints.tar")
    run = tmp_path / "w" / "runs" / "i5" / "mbrs"
    cont, start = _prepare_continuation(cfg, "mbrs", run)
    assert start == 20 and (cont / "models" / "EC_20.pth").read_bytes() == b"ec"
    rel = os.path.relpath(cont, run / "work" / "results")  # as written into MBRS's train_continue_path
    assert os.path.normpath(run / "work" / "results" / rel) == str(cont)
    cont, _ = _prepare_continuation(cfg, "hidden", tmp_path / "w" / "runs" / "i5" / "hidden")
    assert (cont / "checkpoints" / "pilot--epoch-20.pyt").exists() and (cont / "options-and-config.pickle").exists()
    assert (cont / "images").is_dir()
    monkeypatch.setenv("I5_POD", "A")  # I5-A archives are not in the local directory: the step must fail
    import pytest

    with pytest.raises(SystemExit):
        step_i5_inputs(cfg)
    r = __import__("json").loads((tmp_path / "w" / "results" / "i5_inputs" / "i5_inputs.json").read_text())
    assert r["complete"] is False and r["inputs"]["i4_s0_v2"]["status"] == "missing"


def test_i5_inputs_extra_deposition_and_sha256(tmp_path, monkeypatch):
    """Seed 2-3 archives may sit in a further deposition (I5_INPUT_DEPOSITIONS) under their unprefixed name; the
    configured SHA-256 is enforced for every source."""
    import tarfile

    import pytest

    import trackd_wm.i5 as I5
    from trackd_wm.pilot import load_cfg

    cfg = load_cfg([f"work={tmp_path / 'w'}"])
    assert len(cfg.i5.input_sha256) == 6
    calls = []

    def fake(cfg_, names, dest, prefix="", keep_all=False, sha256=None):
        import os

        dep = os.environ.get("ZENODO_DEPOSITION")
        calls.append((dep, prefix))
        hit = dep == "999" and prefix == ""
        return {names[0]: {"archive": f"{prefix}{names[0]}_checkpoints.tar" if hit else None}}

    monkeypatch.setattr(I5, "fetch_zenodo_checkpoints", fake)
    monkeypatch.delenv("I5_INPUT_DIR", raising=False)
    monkeypatch.delenv("ZENODO_DEPOSITION", raising=False)
    monkeypatch.setenv("ZENODO_READ_TOKEN", "x")
    monkeypatch.setenv("I5_INPUT_DEPOSITIONS", "999")
    got = I5.fetch_input(cfg, "i4_s2_v2", tmp_path / "r")
    assert got["source"] == "zenodo deposition 999 (read only)"
    assert calls == [(None, "pilot_p1_"), (None, ""), ("999", "pilot_p1_"), ("999", "")]
    import os

    assert "ZENODO_TOKEN" not in os.environ and "ZENODO_DEPOSITION" not in os.environ
    monkeypatch.setenv("I5_INPUT_DEPOSITIONS", "")
    with pytest.raises(FileNotFoundError):
        I5.fetch_input(cfg, "i4_s3_v2", tmp_path / "r")
    inp = tmp_path / "in"
    (inp / "x").mkdir(parents=True)
    (inp / "x" / "epoch_020.pt").write_bytes(b"not the R11 model")
    with tarfile.open(inp / "i4_s2_v3_checkpoints.tar", "w") as t:
        t.add(inp / "x", arcname="i4_s2_v3")
    monkeypatch.setenv("I5_INPUT_DIR", str(inp))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        I5.fetch_input(cfg, "i4_s2_v3", tmp_path / "r")


def test_i5_inputs_zenodo_either_name(tmp_path, monkeypatch):
    """Job 25: I5-A stopped because the R11 archives are not under the pilot_p1_ name; the draft lookup must
    also accept the upload name, and still report a missing archive."""
    import trackd_wm.i5 as I5
    from trackd_wm.pilot import load_cfg

    cfg = load_cfg([f"work={tmp_path / 'w'}"])
    draft = {"i4_s2_v2_checkpoints.tar", "pilot_p1_i4_s0_v2_checkpoints.tar"}
    calls = []

    def fake(cfg, names, dest, prefix="", keep_all=False, sha256=None):
        key = f"{prefix}{names[0]}_checkpoints.tar"
        calls.append(key)
        return {names[0]: {"archive": key} if key in draft else {"archive": None}}

    monkeypatch.setattr(I5, "fetch_zenodo_checkpoints", fake)
    monkeypatch.delenv("I5_INPUT_DIR", raising=False)
    monkeypatch.setenv("ZENODO_READ_TOKEN", "x")
    root = tmp_path / "w" / "i5_inputs"
    assert I5.fetch_input(cfg, "i4_s0_v2", root)["archive"] == "pilot_p1_i4_s0_v2_checkpoints.tar"
    assert I5.fetch_input(cfg, "i4_s2_v2", root)["archive"] == "i4_s2_v2_checkpoints.tar"
    import pytest

    with pytest.raises(FileNotFoundError):
        I5.fetch_input(cfg, "i4_s3_v2", root)
    assert calls[-2:] == ["pilot_p1_i4_s3_v2_checkpoints.tar", "i4_s3_v2_checkpoints.tar"]
    assert "ZENODO_TOKEN" not in __import__("os").environ


def test_zenodo_fetch_accepts_unprefixed_orchestron_name(tmp_path, monkeypatch):
    # R11 archives reach the deposit through scripts/upload_artifacts.sh without the "pilot_p1_" prefix
    import hashlib
    import io
    import json
    import tarfile
    import urllib.request

    from trackd_wm.pilot import fetch_zenodo_checkpoints, load_cfg

    src = tmp_path / "src" / "i4_s2_v2"
    src.mkdir(parents=True)
    (src / "epoch_020.pt").write_bytes(b"ck")
    tar = tmp_path / "i4_s2_v2_checkpoints.tar"
    with tarfile.open(tar, "w") as t:
        t.add(src, arcname="i4_s2_v2")
    blob = tar.read_bytes()
    dep = {"files": [{"filename": "i4_s2_v2_checkpoints.tar", "checksum": hashlib.md5(blob).hexdigest(),
                      "links": {"download": "https://zenodo.invalid/dl"}}]}

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(json.dumps(dep).encode() if "depositions" in req.full_url else blob)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ZENODO_TOKEN", "x")
    cfg = load_cfg([f"work={tmp_path / 'w'}"])
    got = fetch_zenodo_checkpoints(cfg, ["i4_s2_v2", "i4_s3_v2"], tmp_path / "d", prefix="pilot_p1_", keep_all=True)
    assert got["i4_s2_v2"]["archive"] == "i4_s2_v2_checkpoints.tar"
    assert (tmp_path / "d" / "i4_s2_v2" / "epoch_020.pt").exists()
    assert got["i4_s3_v2"]["archive"] is None


def test_i5_compare_on_synthetic_arrays(tmp_path, monkeypatch):
    import json

    from trackd_wm.i5 import compare_analysis, compare_table, step_i5_compare
    from trackd_wm.pilot import load_cfg

    cfg = load_cfg([f"work={tmp_path / 'w'}", "i5.bootstrap=200"])
    rng = np.random.default_rng(0)
    keys = ["coco__matched_40dB__bitacc_jpeg50", "coco__matched_40dB__psnr_rgb", "coco__matched_35dB__bitacc_jpeg50"]
    src = tmp_path / "src"
    (src / "m1").mkdir(parents=True)
    m1 = {"models": {}}
    for v, mu in (("v1p", 0.80), ("v2", 0.86), ("v3", 0.88)):
        for s in range(4):
            if v == "v2" and s == 3:
                continue  # incomplete seed: excluded for every variant
            np.savez_compressed(src / "m1" / f"{v}_s{s}_i5_per_image.npz",
                                **{k: rng.normal(mu, 0.02, 40).astype(np.float32) for k in keys})
            m1["models"][f"{v}_s{s}"] = {"status": "ok"}
    (src / "m1" / "i5_m1.json").write_text(json.dumps(m1))
    (src / "mbrs").mkdir()
    (src / "mbrs" / "i5_mbrs.json").write_text(json.dumps({"status": "ok", "validity": "x"}))
    np.savez_compressed(src / "mbrs" / "mbrs_s0_i5_per_image.npz",
                        **{k: rng.normal(0.95, 0.02, 40).astype(np.float32) for k in keys[:2]})
    (src / "hidden").mkdir()
    (src / "hidden" / "i5_hidden.json").write_text(json.dumps({"status": "incomplete", "note": "epoch 90"}))
    a = compare_analysis(cfg, src)
    assert a["status"] == "ok" and a["seeds_used"] == [0, 1, 2] and a["used_for_decisions"] is False
    assert a["methods"]["hidden"]["status"] == "incomplete" and a["methods"]["dwtdct"]["status"] == "missing"
    assert set(a["differences"]) == {"v1p-mbrs", "v2-mbrs", "v3-mbrs"}
    row = a["differences"]["v2-mbrs"]["bitacc_jpeg50"]
    assert row["theta_hat"] < 0 and row["ci95_hier_bootstrap"][0] < row["theta_hat"] < row["ci95_hier_bootstrap"][1]
    assert len(row["seed_diffs"]) == 3 and "bitacc_jpeg50" not in a["methods"]["mbrs"]["missing_keys"]
    assert "| jpeg50 |" in compare_table(a, ["jpeg50"]) and "hidden (incomplete)" in compare_table(a, ["jpeg50"])
    monkeypatch.setenv("I5_COMPARE_DIR", str(src))
    monkeypatch.setenv("NO_PUSH", "1")
    step_i5_compare(cfg)
    r = json.loads((tmp_path / "w" / "results" / "i5_compare" / "i5_compare.json").read_text())
    assert r["analysis"]["status"] == "ok" and (tmp_path / "w" / "results" / "i5_compare" / "i5_compare.md").exists()


def test_i5_mbrs_seed_phase1_layout_and_archive(tmp_path, monkeypatch):
    """Further MBRS seed (d-2026-10-25-79): phase 1 trains 20 epochs from initialisation with the given seed; only
    the epoch-20 encoder-decoder is laid out for the continuation (discriminator re-initialised, as for seed 0); the
    archive holds both phases. The training seed comes from I5_TRAIN_SEED and never from cfg.seed (data split)."""
    import subprocess
    import tarfile

    import pytest

    from trackd_wm import i5
    from trackd_wm.pilot import load_cfg

    cfg = load_cfg([f"work={tmp_path / 'w'}"])
    calls = []

    def fake_run(cmd, stdout=None, stderr=None, env=None, timeout=None):
        calls.append(cmd)
        work = __import__("pathlib").Path(cmd[cmd.index("--work") + 1])
        models = work / "results" / "MBRS_m64_x" / "models"
        models.mkdir(parents=True)
        for e in range(int(cmd[cmd.index("--epochs") + 1])):
            (models / f"EC_{e}.pth").write_bytes(f"ec{e}".encode())
            (models / f"D_{e}.pth").write_bytes(f"d{e}".encode())
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(i5.subprocess, "run", fake_run)
    run = tmp_path / "w" / "runs" / "i5" / "mbrs_s1"
    run.mkdir(parents=True)
    cont, start, info = i5._fresh_phase1(cfg, "mbrs", run, tmp_path / "repo", tmp_path / "data", 1, {}, None)
    assert start == 20 and info["last_epoch"] == 20 and info["returncode"] == 0
    assert calls[0][calls[0].index("--seed") + 1] == "1" and "--continue-from" not in calls[0]
    assert (cont / "models" / "EC_20.pth").read_bytes() == b"ec19"   # MBRS counts epochs from 0
    assert sorted(p.name for p in (cont / "models").iterdir()) == ["EC_20.pth"]  # no discriminator, as for seed 0
    tp = i5.archive_baseline(cfg, "mbrs", run, 10, "_s1")
    with tarfile.open(tp) as t:
        names = t.getnames()
    assert tp.name == "i5_mbrs_s1_checkpoints.tar"
    assert {"mbrs/phase1/epoch_010.pth", "mbrs/phase1/epoch_020.pth",
            "mbrs/phase1/discriminator_epoch_020.pth"} <= set(names)
    for v in ("", "0", "x"):
        monkeypatch.setenv("I5_TRAIN_SEED", v)
        with pytest.raises(SystemExit):
            i5.train_seed_from_env()
    monkeypatch.setenv("I5_TRAIN_SEED", "1")
    assert i5.train_seed_from_env() == 1 and int(cfg.seed) == 0


def test_i5_compare_across_mbrs_seeds(tmp_path):
    """A further MBRS model adds its own per-model rows and the across-model Welch interval; the seed-0 rows are
    the same as without it."""
    import json

    from scipy import stats

    from trackd_wm.i5 import compare_analysis, compare_table
    from trackd_wm.pilot import load_cfg

    rng = np.random.default_rng(1)
    keys = ["coco__matched_40dB__bitacc_jpeg50", "coco__matched_40dB__bitacc_webp50"]
    src = tmp_path / "src"
    (src / "m1").mkdir(parents=True)
    m1 = {"models": {}}
    for v, mu in (("v1p", 0.80), ("v2", 0.86), ("v3", 0.90)):
        for s in range(4):
            np.savez_compressed(src / "m1" / f"{v}_s{s}_i5_per_image.npz",
                                **{k: rng.normal(mu + 0.01 * s, 0.02, 40).astype(np.float32) for k in keys})
            m1["models"][f"{v}_s{s}"] = {"status": "ok"}
    (src / "m1" / "i5_m1.json").write_text(json.dumps(m1))
    for d, jf, nf, mu in (("mbrs", "i5_mbrs.json", "mbrs_s0_i5_per_image.npz", 0.95),
                          ("mbrs_s1", "i5_mbrs_s1.json", "mbrs_s1_i5_per_image.npz", 0.93)):
        (src / d).mkdir()
        (src / d / jf).write_text(json.dumps({"status": "ok"}))
        np.savez_compressed(src / d / nf, **{keys[0]: rng.normal(mu, 0.02, 40).astype(np.float32),
                                             keys[1]: rng.normal(0.87, 0.02, 40).astype(np.float32)})
    base = load_cfg([f"work={tmp_path / 'w'}", "i5.bootstrap=200", "i5.compare.extra_seeds.mbrs=[]"])
    cfg = load_cfg([f"work={tmp_path / 'w'}", "i5.bootstrap=200"])
    assert list(cfg.i5.compare.extra_seeds.mbrs) == [1]
    a0, a = compare_analysis(base, src), compare_analysis(cfg, src)
    assert "across_models" not in a0 and a["differences"]["v2-mbrs"] == a0["differences"]["v2-mbrs"]
    assert a["methods"]["mbrs_s1"]["status"] == "ok" and "v2-mbrs_s1" in a["differences"]
    r = a["across_models"]["v1p-mbrs"]
    assert r["status"] == "ok" and r["models"] == [0, 1] and r["wmnet_seeds"] == [0, 1, 2, 3]
    e = r["endpoints"]["bitacc_jpeg50"]
    w = stats.ttest_ind(e["wmnet_seed_means"], e["baseline_model_means"], equal_var=False)
    lo, hi = e["ci95_welch"]
    assert abs(w.df - e["welch_df"]) < 1e-9 and lo < e["theta_hat"] < hi
    assert e["direction"] == "mbrs higher" and e["same_sign_all_models"]
    assert a["across_models"]["v2-mbrs"]["endpoints"]["bitacc_webp50"]["direction"].startswith(
        ("not resolved", "v2 higher", "mbrs higher"))
    assert "mbrs_s1" in compare_table(a, ["jpeg50"]) and "further training seed" in compare_table(a, ["jpeg50"])
    (src / "mbrs_s1" / "i5_mbrs_s1.json").write_text(json.dumps({"status": "incomplete", "note": "epoch 90"}))
    a1 = compare_analysis(cfg, src)
    assert a1["across_models"]["v2-mbrs"]["status"].startswith("fewer than two")


def test_i6_public_registry():
    """Stage I-6: the three public checkpoints, their payloads and the step name are registered."""
    from trackd_wm import i6

    assert set(i6.PUBLIC) == {"fin64", "hidden30", "mbrs30"}
    assert i6.PUBLIC["fin64"]["bits"] == 64 and i6.PUBLIC["fin64"]["payload_matched"]
    assert i6.PUBLIC["hidden30"]["bits"] == 30 and i6.PUBLIC["mbrs30"]["bits"] == 30
    assert "i6_public" in i6.STEPS


def test_i6_messages_follow_payload():
    """30-bit models receive 30-bit evaluation messages; the 64-bit messages equal those of I-4."""
    from trackd_wm.harness import eval_messages

    m64, m30 = eval_messages(10, 64, 0), eval_messages(10, 30, 0)
    assert m64.shape == (10, 64) and m30.shape == (10, 30)
    assert torch.equal(m64, eval_messages(10, 64, 0))


def test_i6_mbrs30_path_has_no_fallback(tmp_path):
    """Only <...>/MBRS_Diffusion_128_m30/models/EC_114.pth is accepted; same-named files elsewhere are rejected."""
    import pytest
    from trackd_wm import i6

    (tmp_path / "other" / "models").mkdir(parents=True)
    (tmp_path / "other" / "models" / "EC_114.pth").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        i6.mbrs30_path(tmp_path)
    good = tmp_path / "drive" / i6.MBRS30_PROJECT / "models"
    good.mkdir(parents=True)
    (good / "EC_114.pth").write_bytes(b"y")
    assert i6.mbrs30_path(tmp_path) == good / "EC_114.pth"


def test_i6_split_provenance_and_required():
    """FIN is the required model; the split lists exist and are hashed."""
    from trackd_wm import i6
    from trackd_wm import data as D

    assert i6.REQUIRED == ("fin64",)
    for sp in ("val", "test"):
        assert (D.SPLIT_DIR / f"coco_{sp}.txt").is_file()


def test_orch_manifest_never_complete_when_empty(tmp_path):
    """An archive-only manifest with no files is not complete (d-2026-10-27-i6b, required change 2)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "orch_manifest", Path(__file__).resolve().parents[1] / "scripts" / "pod" / "orch_manifest.py")
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert om.build({str(empty): 0})["complete"] is False
    full = tmp_path / "full"
    full.mkdir()
    (full / "i6_public.json").write_text("{}")
    assert om.build({str(full): 0})["complete"] is True
    assert om.build({str(full): 1})["complete"] is False


def test_i6_preserve_failure_sets_flag(tmp_path, monkeypatch):
    """A failed push leaves results/UPLOAD_INCOMPLETE, which bootstrap.sh turns into an incomplete-upload stop."""
    import subprocess as sp
    import types
    from trackd_wm import i6

    monkeypatch.delenv("NO_PUSH", raising=False)
    monkeypatch.delenv("ORCH_UPLOAD_URL", raising=False)
    monkeypatch.setattr(i6.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 1))
    cfg = types.SimpleNamespace(work=str(tmp_path))
    out = tmp_path / "results" / "i6_public"
    out.mkdir(parents=True)
    log = []
    assert i6.preserve(cfg, out, log, "fin64") is False
    assert (tmp_path / "results" / "UPLOAD_INCOMPLETE").exists()
    assert log[-1]["push_rc"] == 1 and (out / "UPLOAD_LOG.json").exists()
