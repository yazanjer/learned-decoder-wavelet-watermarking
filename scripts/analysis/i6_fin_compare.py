"""Image-level comparison of the WMNet variants with the released FIN checkpoint (stage I-6, docs/PROTOCOL_I6.md,
Section 4), with the I-5 rules: at matched 40 dB on the COCO test split, per variant and condition, the mean paired
difference (variant - FIN) over the four WMNet seeds and the 1,000 images, the two-level bootstrap 95 % interval
(seeds, then images; trackd_wm.i5.hier_bootstrap_vec with the I-5 number of resamples) and the seed-level paired
t-interval. Pairs share the image and the I-4 message. A direction is stated only where both intervals exclude zero
on the same side. Descriptive; no Holm family; one released FIN model.

Usage: python scripts/analysis/i6_fin_compare.py <dir with the i5_m1 per-image .npz> <dir with i6_public outputs>
       <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

from trackd_wm.i5 import hier_bootstrap_vec
from trackd_wm.pilot import load_cfg, seed_t_interval

V, SEEDS, LAB = ["v1p", "v2", "v3"], [0, 1, 2, 3], "matched_40dB"


def main(m1_dir, i6_dir, out):
    cfg = load_cfg([])
    reps = int(cfg.i5.bootstrap)
    m1_dir, i6_dir = Path(m1_dir), Path(i6_dir)
    info = json.loads((i6_dir / "i6_public.json").read_text())
    fin = info["models"]["fin64"]
    if fin.get("status") != "ok":
        raise SystemExit(f"fin64 not evaluated: {fin.get('status')}")
    y_all = np.load(i6_dir / "fin64_i6_per_image.npz")
    per = {(v, s): np.load(m1_dir / f"{v}_s{s}_i5_per_image.npz") for v in V for s in SEEDS}
    pre = f"coco__{LAB}__"
    keys = [k for k in per[(V[0], 0)].files if k.startswith(pre) and k in y_all.files]
    res = {"strength": LAB, "dataset": "coco_test", "seeds_used": SEEDS, "bootstrap_resamples": reps,
           "used_for_decisions": False, "fin_checkpoint_sha256": fin.get("checkpoint_sha256"),
           "fin_strengths_from_val": fin.get("strengths_from_val"),
           "fin_mean": {k[len(pre):]: float(y_all[k].mean()) for k in keys}, "differences": {}}
    for v in V:
        rows = {}
        for k in keys:
            x = np.stack([per[(v, s)][k] for s in SEEDS]).astype(np.float64)
            y = y_all[k].astype(np.float64)
            if y.shape[0] != x.shape[1]:
                raise ValueError(f"{k}: FIN {y.shape[0]} images, WMNet {x.shape[1]}")
            d = x - y[None, :]
            b = hier_bootstrap_vec(d, reps)
            t = seed_t_interval(d.mean(1))
            t = list(t) if not isinstance(t, dict) else [t.get("lo"), t.get("hi")]
            if b[0] > 0 and t[0] is not None and t[0] > 0:
                direction = "variant higher"
            elif b[1] < 0 and t[1] is not None and t[1] < 0:
                direction = "FIN higher"
            else:
                direction = "not resolved"
            rows[k[len(pre):]] = {"mean_variant": float(x.mean()), "mean_fin": float(y.mean()),
                                  "theta_hat": float(d.mean()), "ci95_hier_bootstrap": b, "seed_paired_t_ci95": t,
                                  "seed_diffs": d.mean(1).tolist(), "direction": direction}
        res["differences"][f"{v}-fin64"] = rows
    Path(out).write_text(json.dumps(res, indent=1))
    print("ok", len(keys), "endpoints")


if __name__ == "__main__":
    main(*sys.argv[1:4])
