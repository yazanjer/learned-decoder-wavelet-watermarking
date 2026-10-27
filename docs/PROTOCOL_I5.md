# PROTOCOL_I5 - Full evaluation and baselines, Paper I (proposal with costing)

Status: **approved 2026-10-24 (decision d-2026-10-24-04: M1 + M2, three pods, limit USD 12.74 at list price; CIN and FIN excluded). Frozen by the commit that adds Section 7 (recorded in docs/DECISIONS_P1.md), before any I-5 test-split evaluation.** Section 7 fixes the implementation, the attack parameters, the ceilings and the pod requests; where it is more specific than Sections 2-6 it governs.

## 1. Position in the plan

- G-I.4 is closed with the claim adopted in decision (a): *learned invertible decoder-side analysis transforms for watermark decoding*; lifting (V2) and coupling (V3) are instantiations; there is no lifting-superiority claim; seed variability and FP32 conditioning are limitations.
- I-5 characterises this claim beyond the I-4 primary endpoint and places it against external baselines (docs/PHASE1_EXECUTION_PLAN.md, I-5). Its output is CHECKPOINT 1B: docs/RESULTS.md with every number traced to commit, configuration and seed.
- I-5 tests **no new confirmatory hypothesis** about H1 or H2. The I-4 decisions stay as frozen. All I-5 comparisons are reported as estimates with uncertainty, labelled descriptive or preliminary where the design requires it.

## 2. Modules

Each module can be approved separately. Costs use A40 secure at USD 0.59/h and billable = hours x 0.59 x 1.2 = hours x USD 0.708. Epoch times are those measured in the pilot G3 and in I-4 (about 2,500 s per WMNet run including evaluation; HiDDeN 70 s, FIN 93 s, MBRS 283 s and CIN 312 s per epoch; docs/PILOT_P1_REPORT.md).

### M1. Extended evaluation of the I-4 models (evaluation only)

- **Models:** V1+, V2 and V3 at seeds 0-3 (12 checkpoints: seeds 0-1 from Zenodo draft 23534235, seeds 2-3 from the recovery stage, docs/PROTOCOL_I4_RECOVERY.md). Seed 4 models are evaluated descriptively if available.
- **Fidelity points:** strengths fixed on the validation split at 35, 40 and 45 dB, as in I-4.
- **Attacks (extended classical suite):** JPEG Q10-Q90 in steps of 10; WebP Q50 and Q75; Gaussian blur sigma 0.5, 1, 2; Gaussian noise sigma 0.01, 0.02, 0.05; median filter 3 x 3; resize 0.5 and 0.75; centre crop 0.75 and 0.9 with resize back; brightness and contrast +/- 20 %; and the distortion attacks of the WAVES benchmark that need no generative model. Generative regeneration attacks (VAE or diffusion) are outside M1 because their cost is not measured; they need a timing probe (at most USD 2) and a separate decision.
- **Data:** the untouched COCO test split (1,000 images; already evaluated once in I-4 for the I-4 attack set) and a cross-dataset test on DIV2K validation (100 images, 1,000 fixed 128 x 128 tiles chosen with seed 0; SHA256-recorded).
- **Operator analysis (CPU, same pod):** spectra and per-level energy of the learned decoder transforms, FP32/FP64 reconstruction error and condition number per seed, and their association with Q50 accuracy (descriptive; n = 4 seeds).
- **Statistics:** per comparison (V2 - V1+, V3 - V1+, V2 - V3), mean paired difference over seeds and images with the I-4 hierarchical bootstrap (95 % CI) and the seed-level paired t-interval. No Holm family and no decisions; reported as robustness profiles.
- **Cost:** 12 runs x about 1,400 s (twice the I-4 evaluation time per run, for about twice the attack-and-dataset load) = 4.67 h. **USD 3.31.** This estimate is checked against the first run in the pod log; the stage limit caps it.

### M2. External baselines at a stated training budget

The I-2b results (single seed, validation, 20 epochs) are preliminary and are not comparable to I-4. M2 brings the baselines that can be trained correctly to a stated budget under the matching rules of docs/PHASE1_EXECUTION_PLAN.md, Section 6.1 (64 bits, 128 x 128, the same 10,000 COCO training images, real JPEG at evaluation, strength matched to 40 dB on the validation split, the M1 attack suite on the test split).

| Baseline | Proposal | Hours | Billable (USD) |
|---|---|---|---|
| MBRS | 100 additional epochs (120 in total), seed 0 | 7.86 | 5.57 |
| HiDDeN | 100 additional epochs (120 in total), seed 0 | 1.94 | 1.37 |
| DWT-DCT | no training; evaluation only | 0.10 | 0.07 |
| Evaluation of MBRS and HiDDeN (test split, M1 attacks) | | 0.30 | 0.21 |
| CIN | **not proposed** until the at-chance JPEG result is explained from code and logs (handover item 10). Cost if later approved: 100 epochs, seed 0, 8.67 h | (8.67) | (6.14) |
| FIN | **not proposed**: diverged in the pilot; a corrected configuration is needed first. Cost if later approved: 100 epochs, seed 0, 2.58 h | (2.58) | (1.83) |

- Single-seed baseline results are labelled preliminary. Under-trained, divergent or defective baselines are not evidence of inferiority, and no superiority claim is made from an unmatched or invalid comparison.
- Statistics: image-level paired differences against each WMNet variant at matched 40 dB, with image-level bootstrap CIs labelled "conditional on the trained models".
- Parameter counts and training budgets are tabulated beside every baseline (MBRS has about 74 times the parameters of WMNet).
- Each extra MBRS + HiDDeN seed: 9.80 h, **USD 6.94**.

### M3. 256 x 256 secondary setting (optional)

- V1+, V2 and V3 at seed 0, trained and evaluated at 256 x 256 (same 64-bit payload). About four times the pixels, so about 4 x 2,500 s = 2.78 h per run; 3 runs = 8.33 h. **USD 5.90.** Single seed, descriptive.

### M4. Precision of H1 and H2 (optional, outside I-5 proper)

- One additional seed of the three variants (3 runs, 2.08 h): **USD 1.48 per seed.** Adding seeds after unblinding needs the owner's approval and would be analysed as a pre-specified extension, never as a re-analysis to change the frozen verdict.

## 3. Recommended minimal package

| Item | Hours | Billable (USD) |
|---|---|---|
| Test gate, data preparation (COCO and DIV2K), smoke gate | 0.30 | 0.21 |
| M1 extended evaluation (12 models) | 4.67 | 3.31 |
| M2 MBRS and HiDDeN (seed 0, 120 epochs), DWT-DCT, evaluation | 10.20 | 7.22 |
| Upload, manifest, status push | 0.15 | 0.11 |
| **Total** | **15.32** | **10.85** |

- Proposed hard limit for I-5 as a whole: USD 11.68 billable (16.5 A40 hours). Below the USD 50 client-approval threshold.
- Execution: **three independent A40 pods** (Section 6, standing instruction of 2026-10-24), not one pod. The table above is the single-pod estimate; the three-pod split repeats the test gate, data preparation and upload in each pod and is costed in Section 6.3 (expected 16.22 h, USD 11.48; ceilings 16.48 h, USD 11.67).
- Cumulative Paper I after this package, at most: 15.89 (spent) + 0.06 (stage b) + 5.24 (recovery, billable at `MAX_HOURS` 7.4) + 11.67 = **USD 32.86**, within the USD 50 Paper I cap.
- Optional additions: M3 USD 5.90; one extra baseline seed USD 6.94; CIN USD 6.14 and FIN USD 1.83 (each only after its defect is resolved); M4 USD 1.48 per seed.

## 4. Dependencies and prerequisites

1. Recovery stage completed and its upload manifest complete (M1 needs the seed 2-3 checkpoints).
2. Secondary-endpoint analysis completed (stage b), so that I-5 does not duplicate endpoints already reported.
3. Code for M1 (extended attack suite, WebP, median, crop, colour attacks, DIV2K tiles) and M2 (MBRS and HiDDeN continuation training through the existing wrappers) written and covered by unit tests before the pod is requested. These are code-only steps at USD 0.
4. CIN investigation (handover item 10) before CIN is proposed; FIN only with a corrected configuration.
5. A positive GPU budget in `orchestron/budget.yaml` (currently USD 0.0).

## 5. What the owner is asked to approve

- The module set (recommended: M1 + M2 as in Section 3) and the hard limit (USD 11.68 billable).
- Whether M3, extra baseline seeds or M4 are added.
- After approval, the protocol is frozen and the M1/M2 code is written; the three pods are then requested as three separate `gpu_run` decisions (Section 6).
- If the approved I-5 limit differs from USD 11.68, the three ceilings are rescaled as stated in Section 6.3.

## 6. Execution split: three independent pods (standing instruction)

Owner's standing instruction for I-5 (Orchestron job 6, 2026-10-24):

> When you submit the I-5 GPU requests, split the work into three independent pods of roughly equal length (for example, the extended evaluation of the 12 I-4 models in one, MBRS in one, and HiDDeN plus DWT-DCT in one), each as its own gpu_run decision with its own MAX_HOURS, upload step and manifest check, and with the three ceilings summing to no more than the approved I-5 limit.

This section is binding for the I-5 `gpu_run` requests. Nothing is launched by writing it; the costing in Section 3 still awaits the owner's approval.

### 6.1 Pods

| Pod | Content | Inputs | Steps (`STEPS`) | Result branch / status branch |
|---|---|---|---|---|
| **I5-A** | M1: extended evaluation of the 12 I-4 models (V1+, V2, V3 x seeds 0-3), DIV2K cross-dataset tiles, operator analysis (CPU part in the same pod) | 12 checkpoints: seeds 0-1 from Zenodo draft 23534235, seeds 2-3 from the recovery stage (manifest must be complete); COCO test split; DIV2K validation | `tests data i5_m1` | `r/pilot-p1-i5_m1` / `r/pilot-p1-i5-m1-status` |
| **I5-B** | M2: MBRS continuation training to 120 epochs, seed 0, then its test-split evaluation (M1 attack suite, strength matched to 40 dB on validation) | I-2b MBRS checkpoint (20 epochs); COCO train/validation/test | `tests data i5_mbrs` | `r/pilot-p1-i5_mbrs` / `r/pilot-p1-i5-mbrs-status` |
| **I5-C** | M2: HiDDeN continuation training to 120 epochs, seed 0, and its evaluation; DWT-DCT evaluation (no training) | I-2b HiDDeN checkpoint (20 epochs); COCO train/validation/test | `tests data i5_hidden i5_dwtdct` | `r/pilot-p1-i5_hidden` / `r/pilot-p1-i5-hidden-status` |

- **Independence.** No pod reads the output of another; each runs the unit-test gate and prepares its own data. I5-B and I5-C depend only on the I-2b checkpoints and can be requested before the recovery stage completes; I5-A requires the recovery stage (c) to be complete with `results/ORCH_UPLOAD_MANIFEST.json` reporting `complete: true` (Section 4, item 1). The pods can run in any order or concurrently.
- **Step names** (`i5_m1`, `i5_mbrs`, `i5_hidden`, `i5_dwtdct`) are to be implemented in `src/trackd_wm/pilot.py` and `scripts/pod/bootstrap.sh` with the M1/M2 code (Section 4, item 3); they do not exist yet.
- **Shared rules.** Real JPEG at evaluation, the M1 attack suite, the matching rules of docs/PHASE1_EXECUTION_PLAN.md, Section 6.1, and the statistics of Section 2 (M1 and M2) are identical in all pods. The image-level paired comparisons of each baseline against the WMNet variants (M2 statistics) are computed on CPU after all three pods have finished, from the per-image arrays pushed to the result branches; they need no pod.

### 6.2 Per-pod safeguards

Each pod is a separate `gpu_run` decision and carries its own:

1. **`MAX_HOURS`** hard stop (Section 6.3) and `STEP_TIMEOUTS` for its steps (each step timeout = expected step time + 10 %, truncated so that the steps plus 0.15 h for upload fit within `MAX_HOURS`).
2. **Upload step:** `bootstrap.sh` calls `scripts/upload_artifacts.sh <output dir>` for the pod's results directory and for its archives (I5-B and I5-C: the 120-epoch baseline checkpoints; I5-A: per-image arrays and operator-analysis outputs; no new model is trained in I5-A).
3. **Manifest check:** `results/ORCH_UPLOAD_MANIFEST.json` (name, size, SHA-256, upload status per archive) must report `complete: true` before the pod is terminated; otherwise STOP_REASON is `completed_upload_incomplete` and the pod is only stopped, with its disk kept (as in docs/PROTOCOL_I4_RECOVERY.md).
4. **Status push:** POD_LOG, TESTS_LOG, STOP_REASON and the manifest to the pod's own status branch after every step.
5. **Partial results:** if a ceiling stops a pod, completed steps are reported as obtained, the incomplete step is reported as incomplete, and nothing is extrapolated. In I5-B and I5-C the training step writes a checkpoint every 10 epochs; a baseline stopped before 120 epochs is not evaluated as a 120-epoch baseline.

### 6.3 Lengths and ceilings

A40 secure, USD 0.59/h; billable = hours x 0.708. Per pod: test gate, data preparation and smoke gate 0.30 h; upload, manifest and status push 0.15 h (Section 3 figures, now incurred once per pod). The evaluation of MBRS and HiDDeN (0.30 h in Section 3) is split equally between I5-B and I5-C.

| Pod | Gate + data | Work | Upload | Expected hours | Expected billable (USD) | `MAX_HOURS` | Ceiling billable (USD) |
|---|---|---|---|---|---|---|---|
| I5-A | 0.30 | M1, 12 x 1,400 s = 4.67 | 0.15 | 5.12 | 3.62 | 5.20 | 3.68 |
| I5-B | 0.30 | MBRS 100 epochs 7.86 + evaluation 0.15 | 0.15 | 8.46 | 5.99 | 8.60 | 6.09 |
| I5-C | 0.30 | HiDDeN 100 epochs 1.94 + evaluation 0.15 + DWT-DCT 0.10 | 0.15 | 2.64 | 1.87 | 2.68 | 1.90 |
| **Total** | 0.90 | | 0.45 | **16.22** | **11.48** | **16.48** | **11.67** |

- **Sum of ceilings:** 16.48 h, USD 11.67 billable, not above the proposed I-5 limit of USD 11.68 (16.5 h). Each request states its own ceiling; Orchestron stops each pod at its own `MAX_HOURS`.
- **Rescaling rule.** If the owner approves a limit L (USD billable) other than 11.68, each pod's `MAX_HOURS` = expected hours x (L / 0.708) / 16.22, rounded down to 0.01 h, so that the three ceilings sum to no more than L.
- **Margins.** The split repeats the gate, data preparation and upload twice more than the single-pod plan (+0.90 h, USD 0.64). Under the USD 11.68 limit this leaves margins of only 0.08 h (I5-A), 0.14 h (I5-B) and 0.04 h (I5-C), about 2 % each; pod start-up and image pull also count against `MAX_HOURS`. A limit of USD 12.74 (17.99 h) would give each pod about 10 % margin (I5-A 5.68 h, I5-B 9.38 h, I5-C 2.92 h by the rescaling rule; sum 17.98 h). The owner is asked to choose between the two.
- **Equal length.** The pods are not of equal length: the MBRS training (7.86 h) cannot be divided between independent pods without a checkpoint hand-off, which would make them dependent, so I5-B fixes the longest pod at about 8.5 h. Moving M1 models from I5-A into I5-C would equalise I5-A and I5-C (about 3.9 h each) but would not shorten the longest pod or the wall-clock time, and would mix main-model evaluation with baseline training in one pod. The module split of the instruction's example is therefore kept.

## 7. Frozen implementation, ceilings and requests (Orchestron job 8, 2026-10-24)

Approval (decision d-2026-10-24-04, owner, 2026-10-24 11:43 UTC):

> Approved: run I-5 as specified in docs/PROTOCOL_I5.md Section 6: three independent A40 pods (I5-A, I5-B, I5-C) with ceilings summing to USD 12.74 actual (list price); CIN and FIN excluded. [...] Submit the three I-5 gpu_run requests now; they may run in parallel with R11 because they do not depend on its checkpoints. If any I-5 analysis needs the recovered seed 2-4 checkpoints, hold that part until R11 has finished and its manifest is complete.

Code: module `src/trackd_wm/i5.py` (steps `i5_inputs`, `i5_m1`, `i5_mbrs`, `i5_hidden`, `i5_dwtdct`), attack suite `attacks.I5_ATTACKS`, continuation options of `baselines/launch.py`, configuration `configs/pilot.yaml`, section `i5`; `FATAL_STEPS` in `scripts/pod/bootstrap.sh`; tests `test_i5_*` in `tests/test_core.py`.

### 7.1 Limit and ceilings

- The limit is read as approved: USD 12.74 **at list price** (uptime x USD 0.59/h), i.e. 21.59 A40 hours, USD 15.29 billable. Section 6.3 had costed USD 12.74 as a billable figure (17.98 h); the owner's wording ("actual (list price)") is followed. The rescaling rule of Section 6.3 is applied with L at list price: `MAX_HOURS` = expected hours x (12.74 / 0.59) / 16.22, rounded down to 0.01 h.

| Pod | Expected h | Expected billable (USD) | `MAX_HOURS` | Ceiling, list (USD) | Ceiling, billable (USD) |
|---|---|---|---|---|---|
| I5-A | 5.12 | 3.62 | 6.81 | 4.02 | 4.82 |
| I5-B | 8.46 | 5.99 | 11.26 | 6.64 | 7.97 |
| I5-C | 2.64 | 1.87 | 3.51 | 2.07 | 2.49 |
| **Total** | **16.22** | **11.48** | **21.58** | **12.73** | **15.28** |

- Budget check against `orchestron/budget.yaml` (`gpu_budget_usd` 40.50, the project's total billable budget, cumulative): before I-5, at most 15.89 x 1.2 = 19.07 (pre-Orchestron runs, if recorded billable) + 0.06 (R10, stage b) + 5.24 (R11, recovery, `MAX_HOURS` 7.4) = USD 24.37; with the I-5 ceilings USD 39.65, within USD 40.50. The authoritative spend is in the Orchestron ledger, which the worker cannot read. Expected I-5 spend is unchanged (USD 11.48 billable); the ceilings are hard stops, not targets.
- Within each pod the step time limits (Section 7.6) end the paid work early enough for the upload to finish before `MAX_HOURS`, because the `MAX_HOURS` watchdog stops the pod without uploading.

### 7.2 Attack suite (M1, used by every pod)

Every attack starts from the 8-bit watermarked image and ends with 8-bit quantisation; outputs keep the 128 x 128 size. 29 conditions:

| Key | Attack |
|---|---|
| `clean` | none (8-bit quantisation only) |
| `jpeg10` ... `jpeg90` | JPEG (PIL/libjpeg, 4:2:0), quality 10, 20, ..., 90 |
| `webp50`, `webp75` | WebP (PIL/libwebp, method 4), quality 50 and 75 |
| `blur05`, `blur1`, `blur2` | Gaussian blur, sigma 0.5, 1, 2; kernel 2 ceil(3 sigma) + 1 (5, 5, 13) except `blur1`, which keeps the I-4 5 x 5 kernel; reflect padding |
| `noise001`, `noise002`, `noise005` | additive Gaussian noise, sigma 0.01, 0.02, 0.05 (fixed generator, seed 12345, as in I-4) |
| `median3` | 3 x 3 median filter, reflect padding |
| `resize05`, `resize075` | bilinear down-scaling by 0.5 and 0.75 (antialiased) and back |
| `crop075`, `crop09` | centre crop keeping 0.75 and 0.9 of each side, resized back (bilinear, antialiased) |
| `bright12`, `bright08` | brightness x 1.2 and x 0.8 |
| `contrast12`, `contrast08` | contrast x 1.2 and x 0.8 about the per-image mean luma (BT.601) |
| `rotate10` | rotation by 10 degrees about the centre, bilinear, zero fill (WAVES rotation) |
| `erase10` | centred square of 10 % of the area set to mid-grey (deterministic WAVES-type erasing) |

The WAVES distortions that need no generative model are covered by rotation, crop, erasing, brightness, contrast, blur, noise and JPEG; regeneration attacks remain outside I-5 (Section 2, M1). Every pod runs an attack smoke test (all 29 attacks on 4 validation images) before any training or full evaluation.

### 7.3 M1 (pod I5-A)

- Models: V1+, V2, V3 at seeds 0-3, last finite checkpoint of each archive `i4_s{seed}_{variant}_checkpoints.tar` (as I-4 chose its final epoch). Seed 4 is outside I5-A: V3 seed 4 was not part of I-4 and the pod ceiling has no room for it.
- Strengths: matched to 35, 40 and 45 dB on the validation split (bisection, 30 iterations) with the I-4 messages (`i4.message_seed` 0), then compared with the frozen I-4 strengths (descriptive check of the recovered seed 2-3 models).
- COCO test split (1,000 images) and DIV2K validation tiles at those strengths, full suite. The DIV2K strengths are the COCO-validation strengths (no re-matching); the PSNR obtained on DIV2K is reported.
- DIV2K tiles: the 100 DIV2K validation HR images in sorted order; for each, 10 distinct non-overlapping native-resolution 128 x 128 tiles on the 128-pixel grid, positions drawn with one generator seeded 0; SHA-256 of every source image and tile recorded in `i5_m1.json`.
- Operator analysis (CPU): per model, the energy share of each decoder-transform sub-band (FP32) and the reconstruction diagnostics of I-4 (FP32/FP64 reconstruction and inverse-consistency errors, condition number by power iteration) on 50 validation images; per variant, the Spearman correlation over seeds of COCO Q50 accuracy at 40 dB with the FP32 reconstruction error and the condition number (descriptive, n = 4).
- Statistics: for V2 - V1+, V3 - V1+ and V2 - V3, for every dataset, strength and attack (bit accuracy, SSIM, PSNR): mean paired difference, two-level bootstrap 95 % interval (10,000 resamples; vectorised implementation of the I-4 scheme with its own random stream) and seed-level paired t-interval. Whole seeds only. No Holm family, no decision.

### 7.4 M2 (pods I5-B and I5-C)

- Continuation from the archived pilot G3 checkpoints (seed 0, epoch 20) to 120 epochs with each baseline's own training code and settings of the pilot (`baselines/launch.py`):
  - HiDDeN: its own `continue` command; encoder-decoder, discriminator and both optimiser states are restored.
  - MBRS: its own `train_continue` settings (epochs labelled from 20, 100 further epochs). MBRS's own continuation restores no optimiser state, and the pilot archive holds no discriminator file, so the discriminator (loss weight 1e-4) restarts from initialisation. Both points are disclosed with the result.
- Archives `i5_{mbrs,hidden}_checkpoints.tar`: checkpoints every 10 epochs and the last, MBRS discriminators of the same epochs (so that a later continuation is complete), settings and logs; sent with `scripts/upload_artifacts.sh`.
- Evaluation of the 120-epoch checkpoint only: strength matched to 40 dB on the validation split, then the COCO test split at that strength and at the native strength, full suite, I-4 messages (pairs per image with the M1 arrays). A run that stops before epoch 120 is archived and reported as incomplete and is not evaluated (Section 6.2, item 5).
- DWT-DCT (I5-C, CPU): library scale 36 as the native strength; matched 40 dB on validation; COCO test split, full suite. Result branch `r/pilot-p1-i5_dwtdct`.
- The image-level comparisons with the WMNet variants are computed on CPU after all three pods, from the per-image arrays (Section 6.1); labelled preliminary and conditional on the trained models.

### 7.5 Inputs

- Step `i5_inputs` (first step of every pod) fetches the archives listed in `configs/pilot.yaml`, `i5.inputs.<pod>`: I5-A the 12 `i4_s{0-3}_{v1p,v2,v3}` archives, I5-B `g3_mbrs`, I5-C `g3_hidden`. Sources in order: a directory `I5_INPUT_DIR` (for example a network volume), then the Paper I Zenodo draft 23534235 read-only with `ZENODO_READ_TOKEN` (MD5 verified against the record). The read token has its own name so that `bootstrap.sh`, which uploads to Zenodo whenever `ZENODO_TOKEN` is set, never uploads from an I-5 pod (orchestron/WORKFLOW.md).
- `FATAL_STEPS="i5_inputs data"`: if an input is missing or the data preparation fails, the pod stops at once (STOP_REASON `i5_inputs_failed` or `data_failed`), at a cost of minutes.

### 7.6 Pod environment

All pods: A40 secure, image `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204`, clone and `bootstrap.sh` as in the recovery request, `TORCH_THREADS=8`, `GH_TOKEN` and `ZENODO_READ_TOKEN` as RunPod secret references.

| Pod | `STEPS` | `STEP_TIMEOUTS` (s) | `METHOD_TIMEOUT_S` (training) | `MAX_HOURS` | Status branch |
|---|---|---|---|---|---|
| I5-A | `i5_inputs tests data i5_m1` | i5_inputs 1800, data 900, i5_m1 20160 | - | 6.81 | `r/pilot-p1-i5-m1-status` |
| I5-B | `i5_inputs tests data i5_mbrs` | i5_inputs 1800, data 900, i5_mbrs 35280 | 33840 (9.4 h; expected 7.86 h) | 11.26 | `r/pilot-p1-i5-mbrs-status` |
| I5-C | `i5_inputs tests data i5_hidden i5_dwtdct` | i5_inputs 600, data 900, i5_hidden 8820, i5_dwtdct 900 | 8100 (2.25 h; expected 1.94 h) | 3.51 | `r/pilot-p1-i5-hidden-status` |

Worst case with every limit reached (start-up 0.1 h, test gate 0.02 h, upload 0.15-0.3 h): I5-A 6.62 h, I5-B 10.97 h, I5-C 3.39 h, each below its `MAX_HOURS`.

Post-freeze implementation fixes (no change to the design, the attack parameters or the ceilings), commit recorded in docs/DECISIONS_P1.md: HiDDeN's `continue` command needs the run's `images` folder, which only its `new` command creates, so the continuation layout now creates it; the DWT-DCT step limit is 900 s instead of 720 s (CPU decoding of 58,000 attacked images).

### 7.7 Order of the requests

- I5-B and I5-C need only the pilot archives and are requested now (decisions d-2026-10-24-08-i5b and d-2026-10-24-08-i5c).
- I5-A needs the seed 2-3 checkpoints of R11 (recovery). It is **held**, as instructed, until R11 has finished with `results/ORCH_UPLOAD_MANIFEST.json` reporting `complete: true`, and until the seed 2-3 archives can be reached by the pod (they were sent to Orchestron with `scripts/upload_artifacts.sh`, not to draft 23534235): either Orchestron places them in the Zenodo draft or on a network volume (`I5_INPUT_DIR`). The I5-A request follows with the same specification.

### 7.8 Input credential (Orchestron job 14, 2026-10-24)

- The RunPod secret that gives the I-5 pods read access to Zenodo draft 23534235 is named **`zenodo`** (confirmed by the owner). Every I-5 pod (I5-A, I5-B, I5-C) sets `ZENODO_READ_TOKEN` = `{{ RUNPOD_SECRET_zenodo }}`. The name `zenodo_token` assumed in Sections 7.5-7.7 and in decisions d-2026-10-24-08-i5b and d-2026-10-24-08-i5c was wrong.
- First launch of I5-B (pod ehft2d14nmhtah) and I5-C (pod omoxft13foqrcx), 2026-10-24 12:08 UTC: step `i5_inputs` received HTTP 403 from Zenodo, and `FATAL_STEPS` stopped each pod about one minute after start (STOP_REASON `i5_inputs_failed`; branches `r/pilot-p1-i5-mbrs-status`, `r/pilot-p1-i5-hidden-status`). No test-split data were read, no training or evaluation ran, and nothing from these attempts is reported.
- I5-B and I5-C are resubmitted as decisions d-2026-10-24-14-i5b and d-2026-10-24-14-i5c. Only `ZENODO_READ_TOKEN` changes; the pod commit `b7dbf16e110aa2be89320ae76cb05af2865ec73b` (no change to `src/`, `scripts/`, `configs/` or `tests/` since), the steps, the step limits, `MAX_HOURS` and the ceilings of Sections 7.1 and 7.6 are unchanged.

### 7.9 Relaunch, reconciliation of the I-5 limit and I5-A ceiling (Orchestron job 17, 2026-10-24)

- **Input access verified.** The relaunched pods read their inputs with `RUNPOD_SECRET_zenodo`: I5-B (pod vg6z2zj5qruj99, started 12:57:47 UTC) downloaded `pilot_p1_g3_mbrs_checkpoints.tar` (834,805,760 bytes, MD5 verified, 228 s); I5-C (pod wr30k4ocflh928, started 12:57:49 UTC) downloaded `pilot_p1_g3_hidden_checkpoints.tar` (64,819,200 bytes, MD5 verified, 10 s). Both passed the test gate (32 passed). Source: `i5_inputs/i5_inputs.json` and `POD_LOG` on `r/pilot-p1-i5-mbrs-status` and `r/pilot-p1-i5-hidden-status`.
- **Approval records.** I5-B: decisions/resolved/d-2026-10-24-08-i5b.json and d-2026-10-24-14-i5b.json. I5-C: d-2026-10-24-14-i5c resolved "approve" (branch `claude/modest-clarke-dq8rfk`, decisions/resolved/d-2026-10-24-14-i5c.json, 12:57:31 UTC).
- **Failed first attempts.** Pods ehft2d14nmhtah and omoxft13foqrcx ran about 35 s of bootstrap each (12:08:46 to 12:09:21 UTC). Their billed uptime, including image pull, is not visible to the worker; an allowance of 0.10 h per pod (USD 0.059 list each, USD 0.12 for both) is charged against the I-5 limit. The authoritative figure is the Orchestron ledger. Both pods ran only `i5_inputs`, which failed before writing anything; their disks hold no result or checkpoint (status branches as of 12:09 UTC), so they can be terminated without loss. While stopped, they accrue volume-storage charges only.
- **Reconciled limit (list price).** 12.74 - 0.12 (failed attempts) - 6.64 (I5-B ceiling) - 2.07 (I5-C ceiling) = USD 3.91 for I5-A, i.e. `MAX_HOURS` = 3.91 / 0.59 = 6.62 h (USD 4.69 billable). The I5-A ceiling is reduced from 6.81 h to 6.62 h. To keep the worst case below the new ceiling, the `i5_m1` step limit is reduced from 20,160 s to 19,800 s (5.50 h; expected M1 work 4.67 h, margin 18 %). Worst case with every limit reached: 0.10 (start-up) + 0.50 (`i5_inputs`) + 0.02 (tests) + 0.25 (data) + 5.50 (`i5_m1`) + 0.15 (upload) = 6.52 h < 6.62 h. Expected I5-A cost is unchanged (5.12 h, USD 3.62 billable).
- **Ceilings after reconciliation (list / billable, USD):** I5-A 3.91 / 4.69, I5-B 6.64 / 7.97, I5-C 2.07 / 2.49, failed attempts 0.12 / 0.14; total 12.74 / 15.29, equal to the approved limit.
- **I5-A inputs.** Seeds 0-1 (`pilot_p1_i4_s{0,1}_{v1p,v2,v3}_checkpoints.tar`) are in Zenodo draft 23534235 from I-4. Seeds 2-3 exist only in the R11 upload to Orchestron (`i4_s{2,3}_{v1p,v2,v3}_checkpoints.tar`). Step `i5_inputs` accepts either name from `I5_INPUT_DIR` and the `pilot_p1_` name from the draft. If neither source holds the six seed 2-3 archives, `FATAL_STEPS` stops the pod within minutes (STOP_REASON `i5_inputs_failed`).

### 7.10 First I5-A attempt and resubmission (Orchestron job 28, 2026-10-24)

- Pod 82oimt2ynvang4 (13:40:08 to 13:41:30 UTC, `MAX_HOURS` 6.62) stopped at `i5_inputs` (`STOP_REASON` `i5_inputs_failed`): seeds 0-1 found in draft 23534235 (MD5 verified), seed 2-3 archives missing. R11 had delivered them through `scripts/upload_artifacts.sh` as `i4_s{2,3}_{v1p,v2,v3}_checkpoints.tar`, without the `pilot_p1_` prefix.
- Commit 772a490: the Zenodo lookup accepts the prefixed name and then the unprefixed upload name; nothing else differs from b7dbf16 in code, configuration, scripts or tests.
- Allowance of 0.10 h (USD 0.059 list) for the failed attempt; the I5-A ceiling becomes `MAX_HOURS` 6.52 (USD 3.85 list, 4.62 billable). I-5 total at list price: 0.12 + 0.059 + 3.847 + 6.643 + 2.071 = USD 12.74, equal to the approved limit.
- Resubmitted as d-2026-10-24-28 (supersedes d-2026-10-24-17 and d-2026-10-24-19).

### 7.10a I5-A request (Orchestron job 18, 2026-10-24)

- Numbering: this section was numbered 7.11 until 2026-10-25, a duplicate of the job 27 section below introduced by a merge (reports/experiments-job54.md). It is renumbered 7.10a so that Section 7.11, as cited in d-2026-10-24-27 and reports/experiments-job27.md, denotes the job 27 section only; Sections 7.12 and 7.13 are unchanged. The text is otherwise unchanged; the request it describes (d-2026-10-24-18) is superseded by d-2026-10-24-27 (Section 7.11).

- R11 finished at 13:25 UTC with `ORCH_UPLOAD_MANIFEST.json` `complete: true` (9 archives); the hold of Section 7.7 is lifted. I5-A is requested as d-2026-10-24-18 with the specification of Sections 7.1, 7.3 and 7.6, commit b7dbf16 and `ZENODO_READ_TOKEN` = `{{ RUNPOD_SECRET_zenodo }}` (Section 7.8).
- Input prerequisite: the six R11 archives `i4_s{2,3}_{v1p,v2,v3}_checkpoints.tar` must be in draft 23534235 under the `pilot_p1_` prefix (or all 12 inputs in `I5_INPUT_DIR`); otherwise `FATAL_STEPS` stops the pod at `i5_inputs`.

### 7.11 Failed I5-A launch, input sources and the single surviving I5-A request (Orchestron job 27, 2026-10-24)

- **Failed launch.** Pod 82oimt2ynvang4 (I5-A, `MAX_HOURS` 6.62, pod commit `b7dbf16`) started at 13:40:08 UTC and stopped at step `i5_inputs` at about 13:41 UTC (STOP_REASON `i5_inputs_failed`, branch `r/pilot-p1-i5-m1-status`). The six seed 0-1 archives were read from draft 23534235 with MD5 verified; the six seed 2-3 archives were missing, because they exist only in the Orchestron upload of R11. No test-split data were read, no evaluation ran, and nothing from this attempt is reported. Its disk holds no result, so the pod can be terminated without loss.
- **Cost of the failed launch.** As in Section 7.9, an allowance of 0.10 h (USD 0.059 list, USD 0.07 billable) is charged against the I-5 limit; the Orchestron ledger is authoritative.
- **Reconciled I5-A ceiling.** 3.91 - 0.06 = USD 3.85 list, i.e. `MAX_HOURS` = 3.85 / 0.59 = **6.52 h** (USD 4.62 billable). The `i5_m1` step limit stays at 19,800 s (Section 7.9). The `i5_inputs` step limit is reduced from 1,800 s to 1,200 s (the twelve archives took about 8 s each). Worst case with every limit reached: 0.10 (start-up) + 0.33 (`i5_inputs`) + 0.02 (tests) + 0.25 (data) + 5.50 (`i5_m1`) + 0.15 (upload) = 6.35 h < 6.52 h. Expected cost is unchanged (5.12 h, USD 3.62 billable). Total of the I-5 ceilings: I5-A 3.85 + I5-B 6.64 + I5-C 2.07 + failed attempts 0.18 = USD 12.74 list, equal to the approved limit.
- **Input sources (code commit `2869809`).** Step `i5_inputs` now also reads, read only and with the same `ZENODO_READ_TOKEN`, every Zenodo deposition listed in `I5_INPUT_DEPOSITIONS` (comma-separated ids), matching both `pilot_p1_<name>_checkpoints.tar` and `<name>_checkpoints.tar`. The seed 2-3 archives can therefore be read directly from the deposition that Orchestron created from the R11 upload, without copying or renaming. For the six seed 2-3 archives the SHA-256 recorded by R11 (configs/pilot.yaml, `i5.input_sha256`; docs/STAGE_I4_RECOVERY_REPORT.md) is checked whatever the source; a mismatch stops the pod at `i5_inputs`. The sources `I5_INPUT_DIR` and draft 23534235 behave as before. The change touches input fetching only (no change to the design, the attacks, the analysis or the step `i5_m1`).
- **Single surviving request.** I5-A was requested three times: d-2026-10-24-17 (`MAX_HOURS` 6.62), d-2026-10-24-18 (6.81, branch `claude/modest-clarke-clpqrs`, approved) and d-2026-10-24-19 (6.81, approved; the launch used 6.62). All three are superseded by **d-2026-10-24-27**, which is the only executable I5-A request (`MAX_HOURS` 6.52, `i5_inputs` 1,200 s, `i5_m1` 19,800 s, pod commit `2869809`). Their approval records stay in `decisions/resolved/` unchanged; none of them may be launched again.
- **Launch condition.** I5-A is launched only after input availability has been verified: either `I5_INPUT_DEPOSITIONS` names a deposition that lists the six files `i4_s{2,3}_{v1p,v2,v3}_checkpoints.tar` (or their `pilot_p1_` names), readable with the `zenodo` secret, or the files are in draft 23534235 under their `pilot_p1_` names, or on a network volume mounted as `I5_INPUT_DIR`. Otherwise the pod stops again at `i5_inputs`.

### 7.12 First I5-A launch stopped at the input check (Orchestron job 25, 2026-10-24)

- **What happened.** Pod I5-A (82oimt2ynvang4, A40, `MAX_HOURS` 6.62, commit `b7dbf16`) started 13:40:08 UTC. Step `i5_inputs` downloaded the six seed 0-1 archives from draft 23534235 (MD5 verified) and found none of the six seed 2-3 archives under any accepted name; `FATAL_STEPS` stopped the pod at 13:41:30 UTC (STOP_REASON `i5_inputs_failed`; `r/pilot-p1-i5-m1-status`: POD_LOG, i5_inputs/i5_inputs.json). No test, data preparation, evaluation or training step ran; nothing from this attempt is reported. The precondition of d-2026-10-24-17 (seed 2-3 archives in the draft or on a volume) was not met at launch.
- **Lookup widened (code, job 25).** `fetch_input` now also looks in the draft for the unprefixed upload name (`i4_s{2,3}_{v1p,v2,v3}_checkpoints.tar`), in addition to the `pilot_p1_` name, so that the R11 archives can be added to the draft without renaming. MD5 verification, the local `I5_INPUT_DIR` path and the failure behaviour are unchanged; a missing archive still stops the pod. Test: `tests/test_core.py::test_i5_inputs_zenodo_either_name`. No change to evaluation, attacks, statistics or step order.
- **Limit (list price).** The failed attempt is charged an allowance of 0.10 h (USD 0.059 list), as in Section 7.9. I5-A: 3.91 - 0.06 = USD 3.85, i.e. `MAX_HOURS` 6.52 h (USD 4.62 billable). The `i5_m1` step limit is reduced from 19,800 s to 19,440 s (5.40 h; expected M1 work 4.67 h, margin 16 %). Worst case with every limit reached: 0.10 + 0.50 + 0.02 + 0.25 + 5.40 + 0.15 = 6.42 h < 6.52 h. Expected I5-A cost unchanged (5.12 h, USD 3.62 billable). I-5 total at list price: 3.85 + 6.64 + 2.07 + 0.18 (three failed attempts) = 12.74, equal to the approved limit.
- **Relaunch precondition (unchanged in substance).** The six R11 archives (SHA-256 in `r/pilot-p1-i4-recover-status:ORCH_UPLOAD_MANIFEST.json`) must be in draft 23534235 under either name, or on a network volume given as `I5_INPUT_DIR`, before the pod is started.

### 7.13 Comparison step after the three pods (Orchestron job 29, 2026-10-24)

- Step `i5_compare` (`src/trackd_wm/i5.py`, configuration `i5.compare` in `configs/pilot.yaml`, commit 61a330b) implements the last item of Section 7.4. It reads `r/pilot-p1-i5_m1`, `r/pilot-p1-i5_mbrs`, `r/pilot-p1-i5_hidden` and `r/pilot-p1-i5_dwtdct` (or a local directory `I5_COMPARE_DIR`) and writes `results/i5_compare/i5_compare.json` and `i5_compare.md` to `r/pilot-p1-i5_compare`. No training, no GPU, no new data.
- Operating point: `matched_40dB`, COCO test split (1,000 images), the only strength shared by M1 and M2. Pairs share the image and the I-4 message (`i4.message_seed`); a mismatch in the number of images stops the analysis.
- Per baseline (MBRS, HiDDeN, DWT-DCT), variant (V1+, V2, V3) and endpoint (bit accuracy under each of the 29 attacks, PSNR, SSIM): mean of each method, mean paired difference (variant - baseline) over the whole M1 seeds and the images, two-level bootstrap 95 % interval (variant seeds, then images; `i5.bootstrap` resamples), seed-level t-interval and the share of images on which the variant is better. Whole seeds only, as in M1.
- Validity: preliminary and conditional on one trained model per baseline (seed 0); the intervals do not include baseline training-seed variation. No Holm family, no decision (`used_for_decisions` false). A baseline whose run is incomplete or failed is listed as not compared, never evaluated (Section 6.2, item 5).
- The step is run on RunPod (decision of 2026-10-22, item 7), on a CPU pod, after all three I-5 pods have finished; its request follows when the result branches exist. Expected cost below USD 0.05 billable.
- Test `test_i5_compare_on_synthetic_arrays` (`tests/test_core.py`) runs in the next pod test gate. Outside RunPod only a syntax check was run.
