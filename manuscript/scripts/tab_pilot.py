"""Table: pilot results (seed 0, validation split) at native and matched-40 dB points, generated from g3.json.

Usage: python tab_pilot.py <g3.json> <out.tex> [invalid_method ...] [-omitted_method ...]
A method given with a leading minus is left out of the table (V1.1: HiDDeN removed by owner decision
d-2026-10-27-m07).
Rows appear only for methods whose status is 'ok'; methods listed as invalid (documented defects, see
docs/PILOT_P1_REPORT.md) are shown without numbers. Nothing is typed by hand.
"""
import json
import sys

LABEL = {"proposed": "Learned lifting", "cdf97": "Fixed CDF 9/7", "coupling": "Affine coupling",
         "hidden": "HiDDeN", "fin": "FIN", "mbrs": "MBRS", "cin": "CIN", "dwtdct": "DWT--DCT"}
d = json.load(open(sys.argv[1]))
invalid = {a for a in sys.argv[3:] if not a.startswith("-")}
omit = {a[1:] for a in sys.argv[3:] if a.startswith("-")}
REASON = {"coupling": "invalid control: training collapsed (Section~\\ref{sec:pilot})",
          "fin": "invalid run: non-finite outputs from epoch 2 (Section~\\ref{sec:pilot})",
          "cin": "pilot evaluation superseded (decoding branch not selected; Section~\\ref{sec:pilot})"}
rows = []
for m in ["proposed", "cdf97", "coupling", "hidden", "fin", "mbrs", "cin", "dwtdct"]:
    r = d["methods"].get(m)
    if not r or r.get("status") != "ok" or m in omit:
        continue
    if m in invalid:
        rows.append(f"{LABEL[m]} & \\multicolumn{{10}}{{c}}{{{REASON.get(m, 'invalid run')}}} \\\\")
        continue
    n, t = r["eval"]["native"], r["eval"]["matched"]
    rows.append(f"{LABEL[m]} & {n['psnr_rgb']:.2f} & {n['ssim']:.3f} & {n['bitacc_clean']:.3f} & {n['bitacc_jpeg50']:.3f} & "
                f"{n['bitacc_jpeg30']:.3f} & {t['strength']:.3f} & {t['ssim']:.3f} & {t['bitacc_clean']:.3f} & "
                f"{t['bitacc_jpeg50']:.3f} & {t['bitacc_jpeg30']:.3f} \\\\")
tex = r"""\begin{table*}[t]
\centering
\caption{Pilot results (single seed, 20 epochs, validation split of 1{,}000 images). Native: strength $s=1$. Matched: residual scaled to a mean RGB PSNR of 40~dB. Bit accuracy without attack (clean) and after real JPEG at quality 50 and 30. Preliminary; not used for any claim. Source: \texttt{r/pilot-p1-g3}, \texttt{g3.json}.}
\label{tab:pilot}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccccc}
\toprule
 & \multicolumn{5}{c}{Native operating point} & \multicolumn{5}{c}{Matched 40~dB} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Method & PSNR & SSIM & Clean & Q50 & Q30 & $s$ & SSIM & Clean & Q50 & Q30 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(sys.argv[2], "w").write(tex)
