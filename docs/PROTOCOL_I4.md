# PROTOCOL_I4 - Confirmatory multi-seed comparison, Paper I (to be frozen before launch)

Status: **frozen, 2026-10-22, after I-3b** (docs/STAGE_I3B_REPORT.md).
- I-3b selected V2 unchanged: no valid variant exceeded V2 by >= 1.0 pp at Q50, matched 40 dB. The variants are therefore exactly the I-2/I-3a definitions in configs/pilot.yaml (`i4.variants: [v1p, v2, v3]`).
- Launch still requires the supervisor's separate approval.
- Stage I-4 was approved in session on 2026-10-22, 12:51 UTC, with a hard cap of USD 6.70.
- This protocol satisfies decision (5) of 09:36 UTC: the estimand, pairing structure, confidence intervals and treatment of seed and image variability are specified before the confirmatory experiments.
- It is frozen, by a commit and a tag in docs/DECISIONS_P1.md, after I-3b has fixed the configuration and before any test-split evaluation.
- Changes after freezing are recorded as deviations.

## 1. Questions and hypotheses

- **H1 (learned analysis vs capacity):** at matched 40 dB mean RGB PSNR, the bit accuracy after real JPEG Q50 of V2 exceeds that of V1+.
  - V2: decoder-side learned lifting with a fixed CDF 9/7 encoder transform.
  - V1+: fixed CDF 9/7 with a capacity-matched decoder front end.
- **H2 (lifting vs coupling at equal capacity):** under the same condition, V2 exceeds V3 (decoder-side learned affine coupling with a fixed Haar-mixed encoder transform).
- **H0 (practical equivalence):** for each hypothesis, the difference lies within +/- 1.0 percentage point.

The configuration of V2 (and the matching settings of V1+ and V3) is the one selected by the I-3b selection rule on the validation split. If I-3b selects V2 unchanged, these are exactly the I-2/I-3a variants.

## 2. Units, estimand and pairing

- **Unit of replication for the training method:** the training seed s in {0, 1, 2, 3, 4}.
- **Unit of measurement:** the test image i in {1, ..., 1000} of the untouched COCO test split. Images are nested in seeds as repeated measurements.
- **Outcome:** acc_M(i, s) = the fraction of the 64 bits correctly decoded for image i after real JPEG Q50 (libjpeg via Pillow, 4:2:0). The image is embedded by method M trained with seed s, at the strength that gives mean RGB PSNR 40 dB on the validation split for that (M, s).
- **Estimand** (for comparison A - B):
  theta = E_s E_i [ acc_A(i, s) - acc_B(i, s) ].
  This is the mean paired difference over the distribution of training seeds and test images. It is estimated by the average over the five seeds and 1,000 images.
- **Pairing:**
  - the same seed index s gives the same data order, training noise sequence and training messages for every method;
  - image i carries the same 64-bit message for every method and seed;
  - so the differences d(i, s) = acc_A(i, s) - acc_B(i, s) are paired at the (i, s) level.
- **Fidelity matching:**
  - the strength factor is fixed on the validation split for each (method, seed) and then applied unchanged to the test split;
  - achieved test-split PSNR and SSIM are reported per (method, seed), so residual quality differences are visible.

## 3. Statistical analysis

1. **Point estimate:** theta_hat = mean over s and i of d(i, s).
2. **Primary interval:** two-level hierarchical bootstrap. Seeds are resampled with replacement, then images within each resampled seed; 10,000 replicates.
   - Percentile 95 % CI for superiority.
   - 90 % CI for the equivalence test.
3. **Primary test:** linear mixed model.
   - Model: acc = method + (1 | image) + (1 | seed) + (1 | seed:method).
   - Fitted to the two methods of each comparison; REML; Satterthwaite degrees of freedom.
   - The method effect is tested against the seed-by-method variance, so seed variation, not image noise, governs the test.
   - A binomial GLMM over bits (logit link) is reported as a sensitivity analysis.
4. **Multiplicity:** Holm correction over the primary family {H1, H2} (alpha = 0.05).
5. **Secondary endpoints:** a separate Holm family; reported, but not used for decisions.
   - Q30, Q70 and resize at 40 dB;
   - Q50 and Q30 at 35 and 45 dB;
   - clean accuracy;
   - TPR at 0.1 % FPR;
   - SSIM.
6. **Equivalence:** TOST at +/- 1.0 pp. Equivalence is declared if the 90 % hierarchical-bootstrap CI lies within the margin. It is applied when the superiority test is not significant.
7. **Decision per hypothesis:**
   - **Advantage:** Holm-adjusted p < 0.05, the 95 % CI excludes 0, and theta_hat >= +1.0 pp.
   - **Equivalent:** the TOST succeeds.
   - **Disadvantage:** Holm-adjusted p < 0.05, the CI excludes 0, and theta_hat <= -1.0 pp.
   - **Inconclusive:** anything else.
8. **Also reported in every case:**
   - per-seed means +/- SD per method;
   - the seed-level paired differences, with a paired t-interval over the 5 seeds as a descriptive check;
   - the image-level Wilcoxon on seed-averaged differences, labelled "conditional on the trained models";
   - null and negative results.
9. **Precision assessment** (decision 5: "assess whether five seeds provide sufficient precision"):
   - The half-width of the 95 % hierarchical CI for each primary comparison is reported against the 1.0 pp margin.
   - If it exceeds 1.0 pp, the comparison is reported as imprecise. Any proposal for more seeds is submitted with its cost and is not run without approval.
   - Seeds are never added after unblinding without approval.

## 4. Design

| Item | Setting |
|---|---|
| Variants | V1+, V2, V3 (configuration from I-3b) |
| Seeds | 0, 1, 2, 3, 4 |
| Training | 20 epochs, PSNR projection to 40 dB, I-2 noise pool, Adam lr 1e-3, batch 16, strict determinism, default PyTorch precision settings (as in I-2/I-3a) |
| Selection data | validation split (1,000 images); used only for strength matching and reporting |
| Test data | COCO test split (1,000 images from val2017), evaluated once, after freezing |
| Attacks | none; JPEG Q90/Q70/Q50/Q30; blur sigma 1; Gaussian noise sigma 0.02; resize 0.5 |
| Fidelity points | 35, 40 (primary) and 45 dB |
| Safeguards | unit tests and smoke gate in the pod; finite-loss and finite-parameter guards; checkpoint finiteness; stage time limit; MAX_HOURS; idle stop; external watchdog; ledger before launch |
| Numerical checks | reconstruction error (float32 and float64, CPU) and conditioning of every learned transform at the final epoch of every seed |
| Artefacts | all epoch checkpoints to the Zenodo draft (not published); per-image arrays and JSON to `r/pilot-p1-i4` |

## 5. Cost (estimate) and limits

- **Per seed** (measured I-2/I-3a times):
  - training V1+ 1,376 s + V2 1,230 s + V3 1,058 s;
  - evaluation (validation strength matching plus test evaluation) about 3 x 700 s;
  - total about 1.60 h.
- **Five seeds:** 8.0 h + 0.15 h setup = 8.15 h.
- **Cost:** about 1.85 h per seed (measured), so 9.4 h in total; 9.4 x 0.59 x 1.2 = **USD 6.66**; hard cap USD 6.70 (stage limit 10.9 h x USD 0.59 = USD 6.43, plus stop overhead).
- **Order:** seed 0 of all three variants first, then seed 1, and so on. A stage-limit stop therefore loses whole seeds, never a variant.
- **Optional I-2b completion** (requires approval): appended after I-4 with its own 0.5 h limit, cap USD 0.35.

## 6. What is frozen, and when

**Freeze record (2026-10-22):**
- configuration as above;
- analysis implemented in `trackd_wm.pilot.i4_analysis`: hierarchical bootstrap; MixedLM with seed groups and image and seed:method variance components; Holm; four-way decision; precision check;
- test messages fixed with `message_seed` 0;
- the freeze commit is the commit that adds this record.

Deviations after this point are recorded in this section.

Known implementation details, stated before the run:
- the mixed model uses statsmodels MixedLM (REML, normal-approximation p-value). Satterthwaite degrees of freedom are not available there, so the seed-level paired t-interval is reported alongside;
- the binomial GLMM sensitivity analysis is computed after the run from the per-image arrays, on a CPU pod.


- **Frozen at the freeze commit:** this document, the I-3b-selected configuration, the code commit, the config file, the evaluation messages and the attack parameters.
- **Before freezing,** the test split may not be loaded by any pod. Code paths that read it are enabled only by the I-4 step.
- **After the results:** no change of endpoint, margin or analysis. Exploratory analyses are labelled as such.

### Deviations and implementation notes after freezing

- **2026-10-23 07:20 UTC (before launch; no data seen):**
  - an in-pod smoke gate was added: each variant runs 5 steps on 50 validation images, and the test split is not touched;
  - the mixed-model fit moves to a CPU pod after the run. It uses the same code (`mixed_model_test`, `I4_MIXED=1`), because a 10,000-row model with a 1,000-level image component is too slow for the GPU pod;
  - the in-pod `analysis` therefore reports the bootstrap results and a provisional decision without p-values. The final four-way decision is computed after the mixed model has been fitted.
  - The statistical method is unchanged.
- **2026-10-23 07:50 UTC (during the run; no I-4 result read):** the binomial GLMM sensitivity analysis over the 64 bits per image is implemented as a linear mixed model on the empirical logit of each image's correct-bit count, with the same random-effects structure. A full bit-level GLMM over about 640,000 rows per comparison is not practical in statsmodels. This is a sensitivity analysis only, and the primary analysis is unchanged.
- **2026-10-23 17:40 UTC (during the run; no I-4 outcome value read):** `i4_analysis` previously required every configured seed and would have returned "incomplete" if the stage limit stopped seed 4. It now restricts every comparison to the seeds for which all three variants completed, which is the whole-seed rule of Section 5. The seeds used are recorded in the output (`seeds_used`). The statistical method is unchanged.
- **2026-10-24 (after the I-4 report; no secondary-endpoint value computed or read):** implementation notes from the handover (docs/HANDOVER_TRACKD_P1.md, Section 5, items 4-6). The primary analysis, the frozen decisions and `r/pilot-p1-i4_final` are unchanged.
  - The secondary family of Section 3.5 is implemented in `trackd_wm.pilot.i4_secondary` and run only by the separate step `i4_secondary` (CPU pod, after approval; output `r/pilot-p1-i4_secondary`). Endpoints: Q30, Q70 and resize at 40 dB; Q50 and Q30 at 35 and 45 dB; clean accuracy at 40 dB; SSIM at 40 dB; each for H1 (V2 - V1+) and H2 (V2 - V3). Each endpoint uses the same estimators as the primary family (hierarchical bootstrap, `mixed_model_test`) and one Holm correction over all of them. The family is reported and is not used for decisions.
  - TPR at 0.1 % FPR: the stored per-image arrays contain no decodes of unmarked images, so the empirical TPR of the protocol cannot be computed. A TPR under the analytic null of independent fair bits (at least k of 64 bits correct, with the Binomial(64, 0.5) tail at most 1e-3) at Q50 / 40 dB is reported descriptively and is kept out of the Holm family.
  - The seed-level paired t-interval is now part of every primary result (`seed_paired_t_ci95`). With 4 seeds the hierarchical bootstrap and the normal-approximation MixedLM understate between-seed uncertainty; the interval is reported beside every frozen decision and never changes it. No re-analysis is carried out to obtain a different verdict.
  - The I-4 path now sets `torch.set_num_threads(cpu_threads())` (the evaluation did not set it). Run-time estimates use about 2,500 s per (variant, seed) run, not 1,850 s.

