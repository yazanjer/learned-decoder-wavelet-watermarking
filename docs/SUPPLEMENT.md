# Supplementary material

The article is accompanied by the following material in this repository. Tables A1 to A8 are generated from the
evaluation outputs by `scripts/analysis/build_analysis.py` (run from the repository root); Table S4 lists the per-model
means of the two MBRS models, which are also given, with the across-model comparison, in
`results/i5_compare_mbrs_s1/across_models.md`.

| Material | File | Source outputs |
|---|---|---|
| Confirmatory family (H1, H2) with per-seed values | `manuscript/tables/tab_a1_confirmatory.tex` | `results/i4/i4_final.json` |
| Secondary endpoints (35 and 45 dB, further attacks, SSIM) | `manuscript/tables/tab_a2_secondary.tex` | `results/i4/i4_secondary.json` |
| Robustness profile at matched 40 dB (29 conditions) | `manuscript/tables/tab_a3_robustness.tex` | `results/i5_interim/summary.json`, `results/i5_compare/`, `results/i6_public/` |
| Paired differences against MBRS | `manuscript/tables/tab_a4_vs_mbrs.tex` | `results/i5_compare/i5_compare.json` |
| Transfer to DIV2K (both-interval decisions on both data sets) | `manuscript/tables/tab_a5_cross_dataset.tex` | `results/i5_interim/summary.json` |
| Invertibility and conditioning per seed | `manuscript/tables/tab_a6_invertibility.tex` | `results/i5_interim/summary.json` |
| Paired differences against the released FIN checkpoint | `manuscript/tables/tab_a7_vs_fin.tex` | `results/i6_public/i6_fin_compare.json` |
| Released FIN and HiDDeN checkpoints at native strength and at 40 dB | `manuscript/tables/tab_a8_released.tex` | `results/i6_public/i6_public.json` |
| Second MBRS model (training seed 1) | `manuscript/tables/tab_s4_mbrs_two_models.tex` | `results/i5_compare_mbrs_s1/across_models.md` |
| Figures (JPEG-quality curves, per-seed accuracy, distortion profile, architecture) | `manuscript/figures/` | as above; `manuscript/scripts/fig_architecture.py` |

Protocols: `docs/PROTOCOL_I4.md` (pre-registered confirmatory analysis), `docs/PROTOCOL_I5.md` (extended
distortions and DIV2K transfer) and `docs/PROTOCOL_I6.md` (released checkpoints). The FIN comparison is recomputed by
`python scripts/analysis/i6_fin_compare.py <dir with the i5 per-image arrays> results/i6_public <out.json>`; the
per-image arrays of the configurations are archived with the trained models on Zenodo
(https://doi.org/10.5281/zenodo.23534235).
