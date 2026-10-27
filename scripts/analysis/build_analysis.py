"""Analysis stage: manuscript tables (LaTeX, booktabs) and figures (PDF + 300 dpi PNG) from the stored result JSON.

Usage: python scripts/analysis/build_analysis.py   (run from the repository root)

Inputs (read only, never recomputed): results/i4/i4_final.json, results/i4/i4_secondary.json,
results/i5_interim/summary.json, results/i5_compare/i5_compare.json.
Outputs: manuscript/tables/tab_a*.tex, manuscript/figures/fig_a*.{pdf,png}, results/analysis/analysis_summary.json,
results/analysis/analysis.md.

The only quantities derived here are descriptive: seed standard deviations (ddof = 1) and minima/maxima of stored
per-seed values, sign counts, and the two-interval direction label. A direction (variant or baseline higher) is
assigned only where the two-level bootstrap 95 % interval and the seed-level paired t 95 % interval both exclude zero
on the same side (d-2026-10-25-80); otherwise the comparison is "not resolved".
For the confirmatory family (I-4) the frozen decision of docs/PROTOCOL_I4.md Section 7 is reported unchanged, with
the qualification required by d-2026-10-25-85 (e.g. "advantage (imprecise; seed-level interval includes zero)"); the
two-interval label is shown beside it as a separate sensitivity reading and never replaces it.
"""
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = {"i4": "results/i4/i4_final.json", "i4s": "results/i4/i4_secondary.json",
       "i5": "results/i5_interim/summary.json", "cmp": "results/i5_compare/i5_compare.json",
       "i6": "results/i6_public/i6_public.json", "i6c": "results/i6_public/i6_fin_compare.json"}
TAB = "manuscript/tables"
FIG = "manuscript/figures"
OUT = "results/analysis"

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42})
# Muted, print-safe palette (same family as manuscript/scripts/fig_sweep.py).
PAL = {"v1p": "#9c6b30", "v2": "#33608c", "v3": "#4f7f86", "mbrs": "#8c3b3b", "hidden": "#7a5c8a",
       "dwtdct": "#6b6b6b", "fin64": "#5b7f4a"}
MARK = {"v1p": "s", "v2": "o", "v3": "^", "mbrs": "D", "hidden": "v", "dwtdct": "x", "fin64": "P"}
NAME = {"v1p": "V1+ (fixed CDF 9/7)", "v2": "V2 (learned lifting)", "v3": "V3 (learned coupling)",
        "mbrs": "MBRS", "hidden": "HiDDeN", "dwtdct": "DWT-DCT", "fin64": "FIN (released)"}
SHORT = {"v1p": "V1+", "v2": "V2", "v3": "V3", "mbrs": "MBRS", "hidden": "HiDDeN", "dwtdct": "DWT-DCT",
         "fin64": "FIN"}
VARIANTS = ["v1p", "v2", "v3"]
# HiDDeN removed from the manuscript by owner decision d-2026-10-27-m07 (I-5 model failed; FIN added after stage I-6).
BASELINES = ["mbrs", "fin64", "dwtdct"]   # fin64: released FIN checkpoint, stage I-6 (docs/PROTOCOL_I6.md)
JPEG_Q = [10, 20, 30, 40, 50, 60, 70, 80, 90]
ATTACKS = (["bitacc_clean"] + [f"bitacc_jpeg{q}" for q in JPEG_Q] +
           ["bitacc_webp50", "bitacc_webp75", "bitacc_blur05", "bitacc_blur1", "bitacc_blur2",
            "bitacc_noise001", "bitacc_noise002", "bitacc_noise005", "bitacc_median3", "bitacc_resize05",
            "bitacc_resize075", "bitacc_bright12", "bitacc_bright08", "bitacc_contrast12", "bitacc_contrast08",
            "bitacc_erase10", "bitacc_crop09", "bitacc_crop075", "bitacc_rotate10"])
ATTACK_LABEL = {"bitacc_clean": "No attack", "bitacc_webp50": "WebP Q50", "bitacc_webp75": "WebP Q75",
                "bitacc_blur05": "Blur 0.5", "bitacc_blur1": "Blur 1", "bitacc_blur2": "Blur 2",
                "bitacc_noise001": "Noise 0.01", "bitacc_noise002": "Noise 0.02", "bitacc_noise005": "Noise 0.05",
                "bitacc_median3": "Median 3", "bitacc_resize05": "Resize 0.5", "bitacc_resize075": "Resize 0.75",
                "bitacc_bright12": "Brightness 1.2", "bitacc_bright08": "Brightness 0.8",
                "bitacc_contrast12": "Contrast 1.2", "bitacc_contrast08": "Contrast 0.8",
                "bitacc_erase10": "Erase 10\\,\\%", "bitacc_crop09": "Crop 0.9", "bitacc_crop075": "Crop 0.75",
                "bitacc_rotate10": "Rotation 10$^\\circ$", "psnr_rgb": "PSNR (dB)", "ssim": "SSIM"}
ATTACK_LABEL.update({f"bitacc_jpeg{q}": f"JPEG Q{q}" for q in JPEG_Q})
SEC_LABEL = {"q30_40dB": "JPEG Q30, 40 dB", "q70_40dB": "JPEG Q70, 40 dB", "resize_40dB": "Resize 0.5, 40 dB",
             "q50_35dB": "JPEG Q50, 35 dB", "q30_35dB": "JPEG Q30, 35 dB", "q50_45dB": "JPEG Q50, 45 dB",
             "q30_45dB": "JPEG Q30, 45 dB", "clean_40dB": "No attack, 40 dB",
             "tpr_fpr1e-3_q50_40dB": "TPR at FPR $10^{-3}$, Q50, 40 dB", "ssim_40dB": "SSIM, 40 dB"}


def plain(s):
    """LaTeX label to plain text (markdown and figure tick labels)."""
    return (s.replace("\\,\\%", " %").replace("$^\\circ$", " deg").replace("$10^{-3}$", "1e-3"))


def load(key):
    with open(os.path.join(ROOT, SRC[key])) as f:
        return json.load(f)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def direction(boot, seed):
    if boot[0] > 0 and seed[0] > 0:
        return "variant"
    if boot[1] < 0 and seed[1] < 0:
        return "baseline"
    return "not resolved"


def framed_decision(decision, precise_enough, seed_ci):
    """Frozen I-4 decision with its uncertainty qualification (d-2026-10-25-85); the decision itself is unchanged."""
    if decision not in ("advantage", "disadvantage"):
        return decision
    q = (["imprecise"] if not precise_enough else []) + \
        (["seed-level interval includes zero"] if seed_ci[0] <= 0 <= seed_ci[1] else [])
    return f"{decision} ({'; '.join(q)})" if q else decision


A1_DIRECTION = {"variant": "V2 higher", "baseline": "V2 lower", "not resolved": "not resolved"}


def pp(x, nd=2):
    return f"{100 * x:+.{nd}f}"


def ci(c, nd=2):
    return f"[{pp(c[0], nd)}, {pp(c[1], nd)}]"


def write(path, text):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)


def save(fig, stem):
    os.makedirs(os.path.join(ROOT, FIG), exist_ok=True)
    fig.savefig(os.path.join(ROOT, FIG, stem + ".pdf"))
    fig.savefig(os.path.join(ROOT, FIG, stem + ".png"), dpi=300)
    plt.close(fig)


def header(src_keys):
    srcs = ", ".join(SRC[k].replace("_", "\\_") for k in src_keys)
    return f"% Generated by scripts/analysis/build_analysis.py from {srcs}; do not edit by hand.\n"


def main():
    i4, i4s, i5, cmp_ = load("i4"), load("i4s"), load("i5"), load("cmp")
    ca = cmp_["analysis"]
    assert ca["strength"] == "matched_40dB" and ca["dataset"] == "coco_test", "unexpected i5_compare scope"
    summary = {"sources": SRC, "rule": "direction only where bootstrap and seed-level 95 % intervals agree in sign",
               "checks": {}}

    # ---- consistency check: i5_compare WMNet means equal the I5-A seed means (COCO, 40 dB), float32 tolerance ----
    worst = 0.0
    for v in VARIANTS:
        sm = i5["seed_means"][f"coco__matched_40dB__{v}"]
        for k in ATTACKS + ["psnr_rgb", "ssim"]:
            a, b = sm[k]["mean"], ca["methods"][v]["mean"][k]
            worst = max(worst, abs(a - b) / max(1.0, abs(a)))
    summary["checks"]["i5_compare_vs_i5a_max_rel_mean_diff"] = worst
    assert worst < 1e-6, f"i5_compare and I5-A seed means disagree ({worst})"

    # ---- Table A1: confirmatory family (I-4) ----
    fam, sens = i4["analysis"]["family_primary"], i4["descriptive_and_sensitivity"]
    rows, a1 = [], {}
    for h, comp in (("H1", "V2 $-$ V1+"), ("H2", "V2 $-$ V3")):
        r, s = fam[h], sens[h]
        a1[h] = {"theta_hat": r["theta_hat"], "ci95_bootstrap": r["ci95"], "ci95_seed_t": s["seed_paired_t_ci95"],
                 "p_holm": r["p_holm"], "decision": r["decision"], "seed_diffs": r["seed_diffs"],
                 "precise_enough": r["precise_enough"],
                 "decision_reported": framed_decision(r["decision"], r["precise_enough"], s["seed_paired_t_ci95"]),
                 "two_interval_direction": direction(r["ci95"], s["seed_paired_t_ci95"])}
        rows.append(f"{h} & {comp} & {pp(r['theta_hat'])} & {ci(r['ci95'])} & {ci(s['seed_paired_t_ci95'])} & "
                    f"{r['p_holm']:.3f} & \\raggedright {a1[h]['decision_reported']} & "
                    f"{A1_DIRECTION[a1[h]['two_interval_direction']]} \\\\")
    seeds = i4["analysis"]["seeds_used"]
    q50 = {"v2": fam["H1"]["seed_means_A"], "v1p": fam["H1"]["seed_means_B"], "v3": fam["H2"]["seed_means_B"]}
    assert all(abs(a - b) < 1e-12 for a, b in zip(fam["H1"]["seed_means_A"], fam["H2"]["seed_means_A"]))
    summary["i4_primary"] = a1
    summary["i4_q50_per_seed"] = {v: {"seeds": seeds, "values": q50[v], "mean": statistics.mean(q50[v]),
                                      "sd": sd(q50[v])} for v in VARIANTS}
    seed_rows = []
    for v in VARIANTS:
        vals = " & ".join(f"{x:.4f}" for x in q50[v])
        seed_rows.append(f"{SHORT[v]} & {vals} & {statistics.mean(q50[v]):.4f} $\\pm$ {sd(q50[v]):.4f} \\\\")
    write(f"{TAB}/tab_a1_confirmatory.tex", header(["i4"]) + "\n".join([
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Confirmatory family (I-4): bit accuracy under JPEG Q50 at matched 40~dB, COCO test split, "
        f"seeds {seeds[0]}--{seeds[-1]}. Differences in percentage points; 95\\,\\% intervals from the two-level "
        "bootstrap and from the seed-level paired $t$ distribution; $p$ Holm-adjusted over the family. Decision: the "
        "frozen pre-registered rule (bootstrap interval, mixed-model $p$, 1.0~pp margin), qualified where the "
        "bootstrap half-width exceeds the 1.0~pp margin (imprecise) or the seed-level interval includes zero. "
        "Both intervals: separate sensitivity reading in which a direction is stated only where both intervals "
        "exclude zero on the same side; it does not replace the frozen decision.}",
        "\\label{tab:confirmatory}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrccrp{2.8cm}l}", "\\toprule",
        "Hyp. & Contrast & $\\hat\\theta$ (pp) & Bootstrap CI & Seed-level CI & "
        "$p_{\\mathrm{Holm}}$ & Decision & Both intervals \\\\", "\\midrule", *rows, "\\midrule",
        "\\multicolumn{8}{l}{Per-seed bit accuracy under JPEG Q50} \\\\",
        "\\multicolumn{2}{l}{Variant} & \\multicolumn{6}{l}{" + ", ".join(f"seed {s}" for s in seeds) +
        "; mean $\\pm$ SD} \\\\",
        *[r"\multicolumn{2}{l}{" + r.split(" & ", 1)[0] + "} & \\multicolumn{6}{l}{" +
          r.split(" & ", 1)[1].replace(" & ", ", ").replace(" \\\\", "") + "} \\\\" for r in seed_rows],
        "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))

    # ---- Table A2: secondary family (R10) ----
    sec = i4s["secondary"]
    rows, a2 = [], {}
    for h in ("H1", "H2"):
        rows.append(f"\\multicolumn{{6}}{{l}}{{{h}: {'V2 $-$ V1+' if h == 'H1' else 'V2 $-$ V3'}}} \\\\")
        for ep, lab in SEC_LABEL.items():
            r = sec["family"][f"{h}:{ep}"]
            p = r.get("p_holm_secondary")
            dirn = direction(r["ci95"], r["seed_paired_t_ci95"])
            a2[f"{h}:{ep}"] = {"theta_hat": r["theta_hat"], "ci95_bootstrap": r["ci95"],
                               "ci95_seed_t": r["seed_paired_t_ci95"], "p_holm_secondary": p,
                               "two_interval_direction": dirn}
            ptxt = "--" if p is None else f"{p:.3f}"
            rows.append(f"\\quad {lab} & {pp(r['theta_hat'])} & {ci(r['ci95'])} & {ci(r['seed_paired_t_ci95'])} & "
                        f"{ptxt} & {dirn} \\\\")
    summary["i4_secondary"] = {"used_for_decisions": sec["used_for_decisions"], "endpoints": a2}
    write(f"{TAB}/tab_a2_secondary.tex", header(["i4s"]) + "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Secondary family (R10), descriptive and not used for decisions. Differences in percentage "
        "points (SSIM $\\times100$); 95\\,\\% bootstrap and seed-level paired $t$ intervals; $p$ Holm-adjusted "
        "within the secondary family (TPR not in the family). Direction is stated only where both intervals "
        "exclude zero on the same side.}",
        "\\label{tab:secondary}", "\\small",
        "\\begin{tabular}{lrccrl}", "\\toprule",
        "Endpoint & $\\hat\\theta$ & Bootstrap 95\\,\\% CI & Seed-level 95\\,\\% CI & $p_{\\mathrm{Holm}}$ & "
        "Direction \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # ---- Table A3: robustness profile at matched 40 dB, COCO (mean +/- SD over seeds; baselines one model) ----
    prof = {}
    for v in VARIANTS:
        sm = i5["seed_means"][f"coco__matched_40dB__{v}"]
        prof[v] = {k: {"mean": sm[k]["mean"], "sd": sd(sm[k]["per_seed"]), "per_seed": sm[k]["per_seed"]}
                   for k in ["psnr_rgb", "ssim"] + ATTACKS}
    i6 = load("i6")
    for b in BASELINES:
        if b == "fin64":  # released checkpoint (I-6): same test split, messages and matched-40 dB rule
            fm = i6["models"]["fin64"]
            assert fm["status"] == "ok", "fin64 not evaluated"
            pt = fm["points"]["coco"]["matched_40dB"]
            prof[b] = {k: {"mean": pt[k]} for k in ["psnr_rgb", "ssim"] + ATTACKS}
            continue
        m = ca["methods"][b]
        assert m["status"] == "ok" and not m["missing_keys"], f"baseline {b} incomplete"
        prof[b] = {k: {"mean": m["mean"][k]} for k in ["psnr_rgb", "ssim"] + ATTACKS}
    summary["i5_profile_coco_40dB"] = {"variants_seeds": ca["seeds_used"], "baseline_validity":
                                       {b: (ca["methods"][b]["validity"] if b in ca["methods"] else
                                            "released checkpoint (I-6), one model") for b in BASELINES},
                                       "values": prof}
    rows = []
    for k in ["psnr_rgb", "ssim"] + ATTACKS:
        nd = 2 if k == "psnr_rgb" else 3
        cells = [f"{prof[v][k]['mean']:.{nd}f} $\\pm$ {prof[v][k]['sd']:.{nd}f}" for v in VARIANTS]
        cells += [f"{prof[b][k]['mean']:.{nd}f}" for b in BASELINES]
        rows.append(f"{ATTACK_LABEL[k]} & " + " & ".join(cells) + " \\\\")
        if k in ("ssim", "bitacc_clean", "bitacc_jpeg90", "bitacc_erase10") and k != ATTACKS[-1]:
            rows.append("\\midrule")
    write(f"{TAB}/tab_a3_robustness.tex", header(["i5", "cmp", "i6"]) + "\n".join([
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Fidelity and bit accuracy at matched 40~dB on the COCO test split (64 bits, $128\\times128$). "
        f"WMNet variants: mean $\\pm$ SD over seeds {ca['seeds_used'][0]}--{ca['seeds_used'][-1]}. MBRS: one "
        "trained model (seed 0); FIN: checkpoint released by its authors (trained on DIV2K), residual scaled by "
        f"{i6['models']['fin64']['strengths_from_val']['matched_40dB']:.2f} to reach 40~dB; DWT-DCT: fixed transform without training, which embeds in the U channel only and "
        "was designed for images of at least $256\\times256$ pixels; baselines descriptive only.}",
        "\\label{tab:robustness}", "\\small",
        "\\begin{tabular}{l" + "c" * len(VARIANTS + BASELINES) + "}", "\\toprule",
        "Condition & " + " & ".join(SHORT[m] for m in VARIANTS + BASELINES) + " \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))

    # ---- Table A4 and Figure A2: variant - MBRS with both intervals ----
    diffs, a4, rows = ca["differences"], {}, []
    for k in ["psnr_rgb", "ssim"] + ATTACKS:
        cells = []
        for v in VARIANTS:
            d = diffs[f"{v}-mbrs"][k]
            dirn = direction(d["ci95_hier_bootstrap"], d["seed_paired_t_ci95"])
            a4.setdefault(v, {})[k] = {"theta_hat": d["theta_hat"], "ci95_bootstrap": d["ci95_hier_bootstrap"],
                                       "ci95_seed_t": d["seed_paired_t_ci95"], "two_interval_direction": dirn}
            mark = {"variant": "$^{+}$", "baseline": "$^{-}$", "not resolved": ""}[dirn]
            if k == "psnr_rgb":
                cells.append(f"{d['theta_hat']:+.2f}{mark}")
            else:
                cells.append(f"{pp(d['theta_hat'])}{mark}")
        rows.append(f"{ATTACK_LABEL[k]} & " + " & ".join(cells) + " \\\\")
    counts = {}
    for v in VARIANTS:
        c = {"variant": 0, "baseline": 0, "not resolved": 0}
        for k in ATTACKS:
            c[a4[v][k]["two_interval_direction"]] += 1
        counts[v] = c
    summary["i5_vs_mbrs"] = {"validity": ca["validity"], "endpoints": a4, "direction_counts_bitacc": counts,
                             "n_bitacc_endpoints": len(ATTACKS)}
    count_row = ("Counts ($+$/$-$/n.r.) & " +
                 " & ".join(f"{counts[v]['variant']}/{counts[v]['baseline']}/{counts[v]['not resolved']}"
                            for v in VARIANTS) + " \\\\")
    write(f"{TAB}/tab_a4_vs_mbrs.tex", header(["cmp"]) + "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Image-level paired difference, variant $-$ MBRS, at matched 40~dB on the COCO test split "
        "(PSNR in dB, otherwise percentage points). $^{+}$: variant higher and $^{-}$: MBRS higher, where the "
        "two-level bootstrap and the seed-level paired $t$ 95\\,\\% intervals both exclude zero; unmarked: not "
        "resolved. One trained MBRS model (seed 0); descriptive only. Intervals in "
        "results/i5\\_compare/summary.md.}",
        "\\label{tab:vsmbrs}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Condition & V1+ $-$ MBRS & V2 $-$ MBRS & V3 $-$ MBRS \\\\", "\\midrule", *rows, "\\midrule",
        count_row, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # ---- Table A7: variant - released FIN checkpoint with both intervals (stage I-6) ----
    i6c = load("i6c")
    fdiff, a7, rows = i6c["differences"], {}, []
    dmap = {"variant higher": "variant", "FIN higher": "baseline", "not resolved": "not resolved"}
    for k in ["psnr_rgb", "ssim"] + ATTACKS:
        cells = []
        for v in VARIANTS:
            d = fdiff[f"{v}-fin64"][k]
            dirn = direction(d["ci95_hier_bootstrap"], d["seed_paired_t_ci95"])
            assert dirn == dmap[d["direction"]], f"direction mismatch {v} {k}"
            a7.setdefault(v, {})[k] = {"theta_hat": d["theta_hat"], "ci95_bootstrap": d["ci95_hier_bootstrap"],
                                       "ci95_seed_t": d["seed_paired_t_ci95"], "two_interval_direction": dirn}
            mark = {"variant": "$^{+}$", "baseline": "$^{-}$", "not resolved": ""}[dirn]
            cells.append((f"{d['theta_hat']:+.2f}" if k == "psnr_rgb" else pp(d["theta_hat"])) + mark)
        rows.append(f"{ATTACK_LABEL[k]} & " + " & ".join(cells) + " \\\\")
    counts7 = {v: {c: sum(a7[v][k]["two_interval_direction"] == c for k in ATTACKS)
                   for c in ("variant", "baseline", "not resolved")} for v in VARIANTS}
    summary["i6_vs_fin"] = {"endpoints": a7, "direction_counts_bitacc": counts7, "n_bitacc_endpoints": len(ATTACKS),
                            "fin_strength_matched_40dB": i6["models"]["fin64"]["strengths_from_val"]["matched_40dB"]}
    write(f"{TAB}/tab_a7_vs_fin.tex", header(["i6", "i6c"]) + "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Image-level paired difference, variant $-$ FIN, at matched 40~dB on the COCO test split (PSNR in "
        "dB, otherwise percentage points). $^{+}$: variant higher and $^{-}$: FIN higher, where the two-level "
        "bootstrap and the seed-level paired $t$ 95\\,\\% intervals both exclude zero; unmarked: not resolved. FIN: "
        "the checkpoint released by its authors (64 bits, trained on DIV2K), residual scaled to 40~dB; one model; "
        "descriptive only.}",
        "\\label{tab:vsfin}", "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
        "Condition & V1+ $-$ FIN & V2 $-$ FIN & V3 $-$ FIN \\\\", "\\midrule", *rows, "\\midrule",
        "Counts ($+$/$-$/n.r.) & " + " & ".join(f"{counts7[v]['variant']}/{counts7[v]['baseline']}/"
                                               f"{counts7[v]['not resolved']}" for v in VARIANTS) + " \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # ---- Table A8: released checkpoints at matched 40 dB and at native strength (stage I-6) ----
    ref_keys = ["psnr_rgb", "ssim", "bitacc_clean", "bitacc_jpeg50", "bitacc_jpeg30", "bitacc_webp50",
                "bitacc_blur1", "bitacc_resize05", "bitacc_crop09", "bitacc_rotate10"]
    a8, rows = {}, []
    names8 = {"fin64": "FIN (64 bits)", "hidden30": "HiDDeN (30 bits)", "mbrs30": "MBRS (30 bits)"}
    for m, r in i6["models"].items():
        if r["status"] != "ok":
            rows.append(f"{names8[m]} & \\multicolumn{{{len(ref_keys) + 1}}}{{l}}{{not evaluated: checkpoint "
                        f"unavailable}} \\\\")
            a8[m] = {"status": r["status"]}
            continue
        for lab, tag in (("native", "native"), ("matched_40dB", "40 dB")):
            pt = r["points"]["coco"][lab]
            a8.setdefault(m, {})[lab] = {k: pt[k] for k in ref_keys}
            cells = [f"{pt['psnr_rgb']:.2f}"] + [f"{pt[k]:.3f}" for k in ref_keys[1:]]
            rows.append(f"{names8[m]} & {tag} & " + " & ".join(cells) + " \\\\")
    summary["i6_released_checkpoints"] = a8
    write(f"{TAB}/tab_a8_released.tex", header(["i6"]) + "\n".join([
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Publicly released checkpoints (FIN: authors' release; HiDDeN: community implementation), evaluated without retraining on the COCO test split at "
        "their native strength and with the residual scaled to 40~dB. One model each; HiDDeN carries 30 bits, so "
        "its row is a reference at its native payload and is not comparable bit for bit. Descriptive only.}",
        "\\label{tab:released}", "\\small", "\\begin{tabular}{ll" + "c" * len(ref_keys) + "}", "\\toprule",
        "Model & Point & " + " & ".join(ATTACK_LABEL[k] for k in ref_keys) + " \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))

    # ---- Table A5: cross-dataset transfer (COCO-trained models, matched 40 dB) ----
    cross_keys = ["psnr_rgb", "ssim", "bitacc_clean", "bitacc_jpeg30", "bitacc_jpeg50", "bitacc_jpeg70",
                  "bitacc_webp50", "bitacc_blur1", "bitacc_noise005", "bitacc_resize05", "bitacc_crop09",
                  "bitacc_rotate10"]
    a5, rows = {}, []
    for k in cross_keys:
        cells = []
        for v in VARIANTS:
            for ds in ("coco", "div2k"):
                sm = i5["seed_means"][f"{ds}__matched_40dB__{v}"][k]
                a5.setdefault(v, {}).setdefault(ds, {})[k] = {"mean": sm["mean"], "sd": sd(sm["per_seed"])}
                nd = 2 if k == "psnr_rgb" else 3
                cells.append(f"{sm['mean']:.{nd}f}")
        rows.append(f"{ATTACK_LABEL[k]} & " + " & ".join(cells) + " \\\\")
    summary["cross_dataset_40dB"] = a5
    write(f"{TAB}/tab_a5_cross_dataset.tex", header(["i5"]) + "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Cross-dataset evaluation of the COCO-trained WMNet variants at matched 40~dB: COCO test split "
        "against DIV2K. Means over four seeds (SDs in results/analysis/analysis\\_summary.json); descriptive.}",
        "\\label{tab:crossdataset}", "\\small",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        " & \\multicolumn{2}{c}{V1+} & \\multicolumn{2}{c}{V2} & \\multicolumn{2}{c}{V3} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Condition & COCO & DIV2K & COCO & DIV2K & COCO & DIV2K \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # ---- Table A6: numerical invertibility and conditioning (descriptive, n = 4) ----
    op = i5["operator_association"]
    rows, a6 = [], {}
    for v in VARIANTS:
        o = op[v]
        a6[v] = {k: o[k] for k in ("seeds", "q50_40dB_coco", "fp32_recon_max_abs", "condition_number")}
        for k in ("spearman_q50_vs_fp32_error", "spearman_q50_vs_condition_number"):
            if k in o:
                a6[v][k] = o[k]
        for s, q, e, c in zip(o["seeds"], o["q50_40dB_coco"], o["fp32_recon_max_abs"], o["condition_number"]):
            rows.append(f"{SHORT[v]} & {s} & {q:.4f} & {e:.1e} & {c:.2e} \\\\")
        rows.append("\\midrule")
    summary["operator_association"] = a6
    write(f"{TAB}/tab_a6_invertibility.tex", header(["i5"]) + "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Decoder-side analysis transform: maximum absolute FP32 reconstruction error and condition number "
        "per seed, with bit accuracy under JPEG Q50 at 40~dB (COCO). Descriptive, $n=4$ seeds per variant.}",
        "\\label{tab:invertibility}", "\\small",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Variant & Seed & Bit acc.\\ Q50 & Max.\\ abs.\\ FP32 error & Condition number \\\\", "\\midrule",
        *rows[:-1], "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))

    # ---- Figure A1: bit accuracy against JPEG quality ----
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    for m in VARIANTS + BASELINES:
        ys = [prof[m][f"bitacc_jpeg{q}"]["mean"] for q in JPEG_Q]
        ls = "-" if m in VARIANTS else "--"
        ax.plot(JPEG_Q, ys, marker=MARK[m], ms=3, color=PAL[m], ls=ls, label=NAME[m])
        if m in VARIANTS:
            lo = [min(prof[m][f"bitacc_jpeg{q}"]["per_seed"]) for q in JPEG_Q]
            hi = [max(prof[m][f"bitacc_jpeg{q}"]["per_seed"]) for q in JPEG_Q]
            ax.fill_between(JPEG_Q, lo, hi, color=PAL[m], alpha=0.15, lw=0)
    ax.set_xlabel("JPEG quality factor")
    ax.set_ylabel("Bit accuracy")
    ax.set_xticks(JPEG_Q)
    ax.set_ylim(0.45, 1.01)
    ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.24))
    save(fig, "fig_a1_jpeg_quality")

    # ---- Figure A2: forest plot, V2 and V3 minus MBRS, both intervals ----
    keys = [k for k in ATTACKS if k != "bitacc_clean"]
    TICKS = [-10, -5, -2, 0, 2, 5, 10]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 4.4), sharey=True, constrained_layout=True)
    for a, v in zip(axs, ("v2", "v3")):
        for i, k in enumerate(keys):
            e = a4[v][k]
            y = len(keys) - 1 - i
            t, b, s = 100 * e["theta_hat"], [100 * x for x in e["ci95_bootstrap"]], [100 * x for x in e["ci95_seed_t"]]
            a.plot(s, [y - 0.17] * 2, color="0.55", lw=0.8)
            a.plot(b, [y + 0.17] * 2, color=PAL[v], lw=1.6)
            filled = e["two_interval_direction"] != "not resolved"
            a.plot(t, y, marker="o", ms=3.5, color=PAL[v], mfc=PAL[v] if filled else "white", mew=0.8)
        a.axvline(0, color="0.3", lw=0.6)
        a.set_xscale("symlog", linthresh=2)
        a.set_xticks(TICKS)
        a.set_xticklabels([f"{t:g}" for t in TICKS])
        a.set_title(f"{SHORT[v]} $-$ MBRS")
        a.set_xlabel("Difference in bit accuracy (pp)")
        a.grid(alpha=0.25, lw=0.4, axis="x")
    axs[0].set_yticks(range(len(keys)))
    axs[0].set_yticklabels([plain(ATTACK_LABEL[k]) for k in reversed(keys)], fontsize=6.5)
    handles = [Line2D([], [], color="0.4", lw=1.6, label="Bootstrap 95 % CI"),
               Line2D([], [], color="0.55", lw=0.8, label="Seed-level paired t 95 % CI"),
               Line2D([], [], marker="o", ls="", color="0.3", ms=3.5, label="Both intervals exclude zero"),
               Line2D([], [], marker="o", ls="", color="0.3", mfc="white", ms=3.5, label="Not resolved")]
    fig.legend(handles=handles, frameon=False, fontsize=6.5, loc="outside lower center", ncol=4)
    save(fig, "fig_a2_vs_mbrs")

    # ---- Figure A3: per-seed Q50 bit accuracy (I-4 confirmatory endpoint) ----
    fig, ax = plt.subplots(figsize=(3.5, 2.3), constrained_layout=True)
    xs = range(len(VARIANTS))
    for j, s in enumerate(seeds):
        ax.plot(xs, [q50[v][j] for v in VARIANTS], color="0.65", lw=0.6, marker="o", ms=2.5, zorder=1)
    for i, v in enumerate(VARIANTS):
        ax.errorbar(i + 0.12, statistics.mean(q50[v]), yerr=sd(q50[v]), fmt=MARK[v], color=PAL[v], ms=4,
                    capsize=2, lw=0.8, zorder=2)
    ax.axhline(0.96, color="0.4", lw=0.5, ls="--")
    ax.text(2.35, 0.962, "0.96", fontsize=6, ha="right", va="bottom", color="0.3")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([NAME[v] for v in VARIANTS], fontsize=6.5)
    ax.set_xlim(-0.3, 2.4)
    ax.set_ylabel("Bit accuracy, JPEG Q50, 40 dB")
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    save(fig, "fig_a3_seeds_q50")

    # ---- Figure A4: attack profile heat map (methods x attacks) ----
    import numpy as np
    meths = VARIANTS + BASELINES
    mat = np.array([[prof[m][k]["mean"] for k in ATTACKS] for m in meths])
    fig, ax = plt.subplots(figsize=(7.16, 2.2), constrained_layout=True)
    im = ax.imshow(mat, cmap="cividis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(meths)))
    ax.set_yticklabels([SHORT[m] for m in meths])
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([plain(ATTACK_LABEL[k]) for k in ATTACKS], rotation=60, ha="right", fontsize=6)
    for (i, j), val in np.ndenumerate(mat):
        txt = f"{val:.2f}"
        ax.text(j, i, txt[1:] if txt.startswith("0") else txt, ha="center", va="center", fontsize=4.5,
                color="white" if val < 0.75 else "black")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Bit accuracy")
    save(fig, "fig_a4_attack_profile")

    write(f"{OUT}/analysis_summary.json", json.dumps(summary, indent=1) + "\n")
    write(f"{OUT}/analysis.md", render_md(summary))
    print("ok: tables", sorted(f for f in os.listdir(os.path.join(ROOT, TAB)) if f.startswith("tab_a")))


def render_md(s):
    L = ["# Analysis stage: derived tables and figures", "",
         "Generated by `scripts/analysis/build_analysis.py` from " + ", ".join(f"`{p}`" for p in SRC.values()) +
         ". Descriptive aggregation only; no statistic is re-estimated. A direction is stated only where the "
         "bootstrap and seed-level 95 % intervals both exclude zero on the same side.", "",
         f"Consistency check: largest relative difference between `i5_compare` WMNet means and the I5-A seed "
         f"means (COCO, 40 dB) = {s['checks']['i5_compare_vs_i5a_max_rel_mean_diff']:.1e}.", "",
         "## Confirmatory family (I-4, JPEG Q50, 40 dB)", "",
         "| | theta (pp) | bootstrap 95 % CI | seed-level 95 % CI | Holm p | frozen decision (qualified) | "
         "both intervals (sensitivity) |",
         "|---|---|---|---|---|---|---|"]
    for h, r in s["i4_primary"].items():
        L.append(f"| {h} | {pp(r['theta_hat'])} | {ci(r['ci95_bootstrap'])} | {ci(r['ci95_seed_t'])} | "
                 f"{r['p_holm']:.3f} | {r['decision_reported']} | {r['two_interval_direction']} |")
    L += ["", "## Secondary family (R10; not used for decisions): endpoints where both intervals agree", ""]
    agree = [f"{k} ({pp(v['theta_hat'])} pp, {v['two_interval_direction']})"
             for k, v in s["i4_secondary"]["endpoints"].items() if v["two_interval_direction"] != "not resolved"]
    L.append("- " + ("; ".join(agree) if agree else "none"))
    L += ["", "## Variant - MBRS (one MBRS model; descriptive)", "",
          f"Counts over {s['i5_vs_mbrs']['n_bitacc_endpoints']} bit-accuracy conditions (variant higher / MBRS "
          "higher / not resolved):", ""]
    for v, c in s["i5_vs_mbrs"]["direction_counts_bitacc"].items():
        L.append(f"- {SHORT[v]}: {c['variant']} / {c['baseline']} / {c['not resolved']}")
        up = [plain(ATTACK_LABEL[k]) for k, e in s["i5_vs_mbrs"]["endpoints"][v].items()
              if k.startswith("bitacc") and e["two_interval_direction"] == "variant"]
        dn = [plain(ATTACK_LABEL[k]) for k, e in s["i5_vs_mbrs"]["endpoints"][v].items()
              if k.startswith("bitacc") and e["two_interval_direction"] == "baseline"]
        L.append(f"  - variant higher: {', '.join(up) or 'none'}")
        L.append(f"  - MBRS higher: {', '.join(dn) or 'none'}")
    L += ["", "## Outputs", "",
          "- Tables: `manuscript/tables/tab_a1_confirmatory.tex`, `tab_a2_secondary.tex`, `tab_a3_robustness.tex`, "
          "`tab_a4_vs_mbrs.tex`, `tab_a5_cross_dataset.tex`, `tab_a6_invertibility.tex`.",
          "- Figures (PDF + 300 dpi PNG): `manuscript/figures/fig_a1_jpeg_quality`, `fig_a2_vs_mbrs`, "
          "`fig_a3_seeds_q50`, `fig_a4_attack_profile`.",
          "- All values: `results/analysis/analysis_summary.json`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
