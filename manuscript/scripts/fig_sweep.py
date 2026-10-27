"""Figure: robustness-imperceptibility trade-off from the strength sweep (sweep.json; no hand-typed numbers).

Usage: python fig_sweep.py <sweep.json> <out_stem>
Panels: bit accuracy versus mean RGB PSNR without attack, after JPEG Q50 and after JPEG Q30. Each method's points
(native strength, matched 30/35/40/45 dB and, for the transform variants, the strength grid) are sorted by PSNR.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})
PAL = {"proposed": "#33608c", "cdf97": "#9c6b30", "hidden": "#7a5c8a", "fin": "#8c3b3b", "mbrs": "#4f7f86",
       "cin": "#6b6b6b", "dwtdct": "#a0522d"}
LABEL = {"proposed": "Learned lifting", "cdf97": "Fixed CDF 9/7", "hidden": "HiDDeN", "fin": "FIN", "mbrs": "MBRS",
         "cin": "CIN (oracle branch)", "dwtdct": "DWT-DCT"}
MARK = {"proposed": "o", "cdf97": "s", "hidden": "^", "fin": "v", "mbrs": "D", "cin": "P", "dwtdct": "x"}

d = json.load(open(sys.argv[1]))
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True, constrained_layout=True)
for m, r in d.get("methods", {}).items():
    if r.get("status") != "ok" or m not in PAL:
        continue
    pts = sorted(r["points"].values(), key=lambda p: p["psnr_rgb"])
    pts = [p for p in pts if 28 <= p["psnr_rgb"] <= 50]
    for a, key in zip(ax, ("bitacc_clean", "bitacc_jpeg50", "bitacc_jpeg30")):
        a.plot([p["psnr_rgb"] for p in pts], [p[key] for p in pts], marker=MARK[m], ms=2.5, color=PAL[m],
               label=LABEL[m])
for a, t in zip(ax, ("No attack", "JPEG Q50", "JPEG Q30")):
    a.set_title(t)
    a.set_xlabel("PSNR, RGB (dB)")
    a.axvline(40, color="0.6", lw=0.5, ls="--")
    a.grid(alpha=0.25, lw=0.4)
ax[0].set_ylabel("Bit accuracy")
ax[2].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.savefig(sys.argv[2] + ".pdf")
fig.savefig(sys.argv[2] + ".png", dpi=300)
