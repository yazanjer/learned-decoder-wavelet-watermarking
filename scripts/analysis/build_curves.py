"""Additional curve figures for the IJIES manuscript (V2.1). Read-only on the results JSON; no value is recomputed
beyond means and ranges over seeds that the JSON already lists per seed.

Inputs:  results/i5_interim/summary.json   (seed_means: per-seed values of V1+, V2, V3 at 35/40/45 dB on COCO and
                                            DIV2K; operator_association: per-seed conditioning and FP32 error)
         results/i5_compare/i5_compare.json (MBRS and DWT-DCT at matched 40 dB, COCO)
         results/i6_public/i6_public.json   (released FIN checkpoint at matched 40 dB and at native strength, COCO)
Outputs: manuscript/figures/fig_c1_distortion_curves, fig_c2_fidelity_tradeoff, fig_c3_div2k_jpeg,
         fig_c4_conditioning (.pdf vector and .png 300 dpi)

Usage: python scripts/analysis/build_curves.py
"""
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42})
PAL = {"v1p": "#9c6b30", "v2": "#33608c", "v3": "#4f7f86", "mbrs": "#8c3b3b", "dwtdct": "#6b6b6b",
       "fin64": "#5b7f4a"}
MARK = {"v1p": "s", "v2": "o", "v3": "^", "mbrs": "D", "dwtdct": "x", "fin64": "P"}
NAME = {"v1p": "V1+ (fixed CDF 9/7)", "v2": "V2 (learned lifting)", "v3": "V3 (learned coupling)",
        "mbrs": "MBRS", "dwtdct": "DWT-DCT", "fin64": "FIN (released, 40 dB)"}
V = ["v1p", "v2", "v3"]
JPEG_Q = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


S = load("results/i5_interim/summary.json")
CMP = load("results/i5_compare/i5_compare.json")["analysis"]["methods"]
FIN = load("results/i6_public/i6_public.json")["models"]["fin64"]["points"]["coco"]
SM = S["seed_means"]


def seeds(ds, db, v, key):
    return SM[f"{ds}__matched_{db}dB__{v}"][key]["per_seed"]


def base(m, key):
    if m == "fin64":
        return FIN["matched_40dB"][key]
    return CMP[m]["mean"][key]


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=300)
    plt.close(fig)


def band(ax, x, per_seed_lists, m, label):
    mean = [statistics.mean(p) for p in per_seed_lists]
    ax.fill_between(x, [min(p) for p in per_seed_lists], [max(p) for p in per_seed_lists], color=PAL[m],
                    alpha=0.15, lw=0)
    ax.plot(x, mean, marker=MARK[m], ms=3, color=PAL[m], label=label)


def fig_c1():
    """Bit accuracy against distortion strength (blur, noise, downscaling), COCO, matched 40 dB."""
    panels = [("Gaussian blur, standard deviation", [0, 0.5, 1, 2],
               ["bitacc_clean", "bitacc_blur05", "bitacc_blur1", "bitacc_blur2"]),
              ("Gaussian noise, standard deviation", [0, 0.01, 0.02, 0.05],
               ["bitacc_clean", "bitacc_noise001", "bitacc_noise002", "bitacc_noise005"]),
              ("Downscaling factor", [1.0, 0.75, 0.5], ["bitacc_clean", "bitacc_resize075", "bitacc_resize05"])]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.25), sharey=True, constrained_layout=True)
    for ax, (xl, xs, keys) in zip(axs, panels):
        for m in V:
            band(ax, xs, [seeds("coco", 40, m, k) for k in keys], m, NAME[m])
        for m in ["mbrs", "fin64", "dwtdct"]:
            ax.plot(xs, [base(m, k) for k in keys], marker=MARK[m], ms=3, color=PAL[m], ls="--", label=NAME[m])
        ax.set_xlabel(xl)
        ax.set_xticks(xs)
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
        ax.grid(alpha=0.25, lw=0.4)
    axs[2].invert_xaxis()
    axs[0].set_ylabel("Bit accuracy")
    axs[0].set_ylim(0.45, 1.02)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=6, frameon=False, fontsize=6.5)
    save(fig, "fig_c1_distortion_curves")


def fig_c2():
    """Bit accuracy under JPEG Q50 and Q30 against the PSNR of the evaluation (35, 40, 45 dB), COCO."""
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.35), sharey=True, constrained_layout=True)
    dbs = [35, 40, 45]
    for ax, q in zip(axs, [50, 30]):
        key = f"bitacc_jpeg{q}"
        for m in V:
            xs = [statistics.mean(seeds("coco", d, m, "psnr_rgb")) for d in dbs]
            band(ax, xs, [seeds("coco", d, m, key) for d in dbs], m, NAME[m].split(" (")[0])
        xs = [FIN["matched_40dB"]["psnr_rgb"], FIN["native"]["psnr_rgb"]]
        ax.plot(xs, [FIN["matched_40dB"][key], FIN["native"][key]], marker=MARK["fin64"], ms=3.5,
                color=PAL["fin64"], ls="--", label="FIN (40 dB, native)")
        ax.plot([CMP["mbrs"]["mean"]["psnr_rgb"]], [CMP["mbrs"]["mean"][key]], marker=MARK["mbrs"], ms=3.5,
                color=PAL["mbrs"], ls="none", label="MBRS (40 dB)")
        ax.set_title(f"JPEG quality {q}", fontsize=8)
        ax.set_xlabel("PSNR (dB)")
        ax.set_xticks([35, 40, 45])
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Bit accuracy")
    axs[0].legend(frameon=False, fontsize=5.8, loc="lower left")
    save(fig, "fig_c2_fidelity_tradeoff")


def fig_c3():
    """JPEG-quality curves on COCO and DIV2K (COCO-trained models, strengths matched on COCO)."""
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.35), sharey=True, constrained_layout=True)
    for ax, ds, title in zip(axs, ["coco", "div2k"], ["MS-COCO test", "DIV2K tiles"]):
        for m in V:
            band(ax, JPEG_Q, [seeds(ds, 40, m, f"bitacc_jpeg{q}") for q in JPEG_Q], m, NAME[m].split(" (")[0])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("JPEG quality factor")
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Bit accuracy")
    axs[0].set_ylim(0.45, 1.01)
    axs[1].legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig_c3_div2k_jpeg")


def fig_c4():
    """Per-seed condition number and maximum FP32 reconstruction error against bit accuracy under JPEG Q50."""
    oa = S["operator_association"]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.35), sharey=True, constrained_layout=True)
    for ax, key, xl in zip(axs, ["condition_number", "fp32_recon_max_abs"],
                           ["Condition number", "Max. abs. FP32 error"]):
        for m in V:
            d = oa[m]
            ax.plot(d[key], d["q50_40dB_coco"], marker=MARK[m], ms=4, ls="none", color=PAL[m],
                    label=NAME[m].split(" (")[0])
            if m == "v2":
                for x, y, s in zip(d[key], d["q50_40dB_coco"], d["seeds"]):
                    ax.annotate(f"s{s}", (x, y), xytext=(-9, -6), textcoords="offset points", fontsize=5.5,
                                color=PAL[m])
        ax.set_xscale("log")
        ax.margins(x=0.15)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Bit accuracy, JPEG quality 50, 40 dB")
    axs[1].legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig_c4_conditioning")


def dump_values():
    """Plotted values (means and per-seed ranges) for the text of the manuscript."""
    out = {"source": "results/i5_interim/summary.json (seed_means, operator_association); "
                     "results/i5_compare/i5_compare.json; results/i6_public/i6_public.json"}
    keys = ["bitacc_clean", "bitacc_jpeg30", "bitacc_jpeg50", "bitacc_blur1", "bitacc_blur2", "bitacc_noise005",
            "bitacc_resize05"]
    for ds in ["coco", "div2k"]:
        for db in [35, 40, 45]:
            for m in V:
                out[f"{ds}_{db}dB_{m}"] = {k: {"mean": statistics.mean(seeds(ds, db, m, k)),
                                               "min": min(seeds(ds, db, m, k)), "max": max(seeds(ds, db, m, k))}
                                           for k in keys + ["psnr_rgb"]}
    out["baselines_40dB_coco"] = {m: {k: base(m, k) for k in keys} for m in ["mbrs", "fin64", "dwtdct"]}
    out["fin_native_coco"] = {k: FIN["native"][k] for k in keys + ["psnr_rgb"]}
    with open(os.path.join(ROOT, "results", "analysis", "curves_values.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    dump_values()
    fig_c1()
    fig_c2()
    fig_c3()
    fig_c4()
    print("ok: fig_c1..fig_c4")
