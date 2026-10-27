# Learned decoder-side wavelet transforms for deep image watermarking

Code, configurations, split lists and evaluation outputs for the article
*Learned Decoder-Side Wavelet Transforms for Deep Image Watermarking: a Controlled Study at Matched Image Fidelity*
(submitted to *Signal Processing: Image Communication*).

The encoder embeds a 64-bit message in the deepest sub-bands of a fixed wavelet transform (CDF 9/7 or Haar);
the decoder analyses the received image with a learned multi-level transform that is initialised at the encoder's
wavelet and constrained to remain invertible (learned residual lifting, `V2`, or affine coupling, `V3`).
A capacity control (`V1+`) adds the same number of decoder parameters on the deepest sub-bands without an invertible
structure. All methods are compared at a matched PSNR of 40 dB on 128 x 128 MS-COCO images under real JPEG
compression and 28 further distortions.

## Contents

| Path | Content |
|---|---|
| `src/trackd_wm/` | Package: transforms (`transforms.py`), models (`models.py`), training (`train.py`), attacks (`attacks.py`), matched-fidelity harness (`harness.py`), stage drivers (`pilot.py`, `i5.py`, `i6.py`), baseline wrappers (`baselines/`) |
| `src/trackd_wm/splits/` | MS-COCO split lists (10,000 train, 1,000 validation from `train2017`; 1,000 test from `val2017`) with the SHA-256 of every source JPEG |
| `configs/pilot.yaml` | Configuration of every stage |
| `docs/PROTOCOL_I4.md` | Pre-registered confirmatory protocol (frozen before the test split was evaluated) |
| `docs/PROTOCOL_I5.md`, `docs/PROTOCOL_I6.md` | Extended evaluation, retrained baselines, released baseline checkpoints |
| `results/i4/` | Confirmatory analysis (`i4_final.json`), secondary family (`i4_secondary.json`), per-run records (`i4.json`) |
| `results/i5_*`, `results/model_budget/` | Extended evaluation (29 conditions, COCO and DIV2K), baselines, image-level comparisons, parameter counts |
| `results/analysis/` | Summary used for the tables and figures of the article |
| `scripts/analysis/build_analysis.py` | Regenerates the article's Tables A1-A6 and Figures A1-A4 from `results/` |
| `scripts/pod/` | Pod entry point and upload helpers used for the GPU runs (RunPod) |
| `tests/` | Unit tests |

## Installation

```bash
pip install -e ".[train,analysis,baselines,test]" -c constraints/recorded-versions.txt
pytest -q tests
```

`constraints/recorded-versions.txt` gives the versions behind the reported results; `constraints/i6-pod.txt`
gives the exact environment of the released-checkpoint evaluation.

## Reproducing the tables and figures

```bash
python scripts/analysis/build_analysis.py   # writes manuscript/tables/tab_a*.tex and manuscript/figures/fig_a*.{pdf,png}
```

Every number is read from the JSON files in `results/`; nothing is typed by hand.

## Re-running the experiments

Data preparation, training and evaluation are run by `python -m trackd_wm.pilot <step> work=<dir>`, for example
`data`, `i4`, `i4_final`, `i5_m1`, `i6_public`. GPU runs were executed on single NVIDIA A40 pods through
`scripts/pod/bootstrap.sh`. Trained models (seeds 0-3 of V1+, V2, V3) are deposited on Zenodo under DOI
[10.5281/zenodo.23534235](https://doi.org/10.5281/zenodo.23534235) (the link resolves once the record is published).

## Baselines

Baseline code is not redistributed. Each repository is cloned at a pinned commit at run time
(`src/trackd_wm/baselines/__init__.py`): MBRS (jzyustc/MBRS), FIN (QQiuyp/FIN), HiDDeN (ando-khachatryan/HiDDeN),
DWT-DCT (ShieldMnt/invisible-watermark). MBRS and FIN carry no licence file and are used for research comparison
only. The HiDDeN model retrained in stage I-5 is not used in the article (its accuracy without attack was 0.72; cause
not established); the released HiDDeN 30-bit checkpoint is reported as a reference row instead.

## Statistical reporting

The confirmatory family (two contrasts, JPEG quality 50 at 40 dB) uses a two-level hierarchical bootstrap, a linear
mixed model with Holm adjustment and equivalence tests, as pre-registered. Four complete training seeds entered the
analysis instead of the five pre-registered; seed-level paired t-intervals are reported beside every decision.
All other comparisons are descriptive and state a direction only where the bootstrap and seed-level intervals both
exclude zero.

## Licence

MIT (see `LICENSE`).
