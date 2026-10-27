"""Figure: pilot learning curves (seed 0) from the r/pilot-p1-g3 results branch.

Usage: python fig_pilot_curves.py <g3.json> <out_stem> [invalid_method ...]
Every value plotted is read from g3.json; nothing is typed by hand.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})
PAL = {"proposed": "#33608c", "cdf97": "#9c6b30", "coupling": "#5b7f4a", "hidden": "#7a5c8a",
       "fin": "#8c3b3b", "mbrs": "#4f7f86", "cin": "#6b6b6b"}
LABEL = {"proposed": "Learned lifting", "cdf97": "Fixed CDF 9/7", "coupling": "Affine coupling",
         "hidden": "HiDDeN", "fin": "FIN", "mbrs": "MBRS", "cin": "CIN"}

d = json.load(open(sys.argv[1]))
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), constrained_layout=True)
for m, r in d["methods"].items():
    if r.get("status") != "ok" or "curve" not in r.get("eval", {}) or m in sys.argv[3:]:
        continue
    c = r["eval"]["curve"]
    ep = [p["epoch"] for p in c]
    ax[0].plot(ep, [p["psnr_rgb"] for p in c], marker="o", ms=2.5, color=PAL[m], label=LABEL[m])
    ax[1].plot(ep, [p["bitacc_jpeg50"] for p in c], marker="o", ms=2.5, color=PAL[m], label=LABEL[m])
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("PSNR, RGB (dB), native strength")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Bit accuracy, real JPEG Q50")
for a in ax:
    a.grid(alpha=0.25, lw=0.4)
ax[1].legend(frameon=False, fontsize=7)
stem = sys.argv[2]
fig.savefig(stem + ".pdf")
fig.savefig(stem + ".png", dpi=300)
