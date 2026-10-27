# PROTOCOL I-6: public baseline checkpoints (evaluation only)

Status: frozen by the commit that adds this file, before any I-6 evaluation (owner decisions d-2026-10-27-m07, session of 2026-10-27).

## 1. Purpose

The HiDDeN model trained in I-5 reached a bit accuracy of 0.72 without attack (cause not established; the community implementation trains with a JPEG approximation only). By owner decision it is removed from the manuscript and replaced by the FIN checkpoint released by its authors. Two further released checkpoints are evaluated as reference rows at their native payload. No model is trained in this stage, and no training seed is added to the WMNet variants.

## 2. Models

| Key | Method | Source | Payload / size | Training data | Role |
|---|---|---|---|---|---|
| fin64 | FIN (AAAI 2023) | `experiments/JPEG/FED.pt` in QQiuyp/FIN @ f77222f | 64 bits, 128 x 128 | DIV2K | Replacement baseline; same payload and evaluation messages as the WMNet variants |
| hidden30 | HiDDeN (community PyTorch) | `experiments/combined-noise/checkpoints/combined-noise--epoch-400.pyt` in ando-khachatryan/HiDDeN @ 556f76d | 30 bits, 128 x 128 | COCO (10,000) | Reference row, native payload |
| mbrs30 | MBRS (ACM MM 2021) | `MBRS_Diffusion_128_m30/models/EC_114.pth`, authors' Google Drive folder linked in jzyustc/MBRS @ 2bab0df | 30 bits, 128 x 128, message diffusion | ImageNet | Reference row, native payload; recorded as unavailable if the download fails |

The SHA-256 of every checkpoint is recorded in the output.

## 3. Evaluation

- Data: the COCO validation split (strength matching) and test split (1,000 images each) of I-4/I-5, prepared by step `data`.
- Messages: `harness.eval_messages(n, bits, i4.message_seed)`; for fin64 identical to the I-4/I-5 messages, so per-image pairing with the WMNet variants is possible.
- Operating points: `matched_40dB` (residual strength found by bisection on validation so that the mean RGB PSNR after 8-bit quantisation is 40 dB, the rule of every other method) and `native` (strength 1).
- Conditions: the 29 conditions of the I-5 suite (`attacks.I5_ATTACKS`), with the attack smoke test of I-5 first.
- Metrics: RGB PSNR, luma PSNR, SSIM, bit accuracy per image (`*_i6_per_image.npz`), means in `i6_public.json`.

## 4. Reporting

- All I-6 results are descriptive (one released model each; `used_for_decisions` false; no Holm family).
- fin64 enters the main comparison table and, after the run, a CPU paired comparison against the WMNet variants with the I-5 rules (two-level bootstrap and seed-level t-interval; direction only where both exclude zero).
- hidden30 and mbrs30 are reported as reference rows at their native payload, with the payload difference stated; no paired comparison is made with them.
- The training data of each released model (DIV2K, COCO, ImageNet) is stated next to its results.

## 5. Compute

One A40 pod (`STEPS="tests data i6_public"`, `FATAL_STEPS="data"`), expected about 0.8 h, `MAX_HOURS` 2.0 (USD 1.42 billable at most; owner cap USD 3). Results branch `r/pilot-p1-i6_public`; status branch `r/pilot-p1-i6-status`.
