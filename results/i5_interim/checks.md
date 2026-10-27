## Upload manifests (ORCH_UPLOAD_MANIFEST.json on the status branches)

| Pod | Pod id | STOP_REASON | Manifest time (UTC) | complete | missing | File | Bytes | SHA-256 | uploaded |
|---|---|---|---|---|---|---|---|---|---|
| I5-A | eeey38qlod0phy | completed | 2026-10-24T14:47:45Z | True | 0 | (none; evaluation only) | - | - | - |
| I5-C | wr30k4ocflh928 | completed | 2026-10-24T14:55:48Z | True | 0 | i5_hidden_checkpoints.tar | 73021440 | f915998d54c35ba4806b2d43b9eee7bed638a5582040b83e2db63629d4d67e37 | True |
| I5-B | vg6z2zj5qruj99 | completed | 2026-10-24T20:46:34Z | True | 0 | i5_mbrs_checkpoints.tar | 839884800 | c609a47af41000691eceeb91450592a00f7cb05d6cf976667a16b54ffbf1d84d | True |

## M1 against the frozen I-4 results (COCO test, matched 35/40/45 dB)

| Run | Run-level values compared | Largest abs. diff. | Strength diff. | Per-image source | Arrays (x images) | Bitwise-equal arrays | Largest abs. diff. |
|---|---|---|---|---|---|---|---|
| v1p_s0 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v1p_s1 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v1p_s2 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v1p_s2 | 36 | 0.0e+00 | 0.0e+00 | i4_recover | 33 x 1000 | 33 | 0.0e+00 |
| v1p_s3 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v1p_s3 | 36 | 0.0e+00 | 0.0e+00 | i4_recover | 33 x 1000 | 33 | 0.0e+00 |
| v2_s0 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v2_s1 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v2_s2 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v2_s2 | 36 | 0.0e+00 | 0.0e+00 | i4_recover | 33 x 1000 | 33 | 0.0e+00 |
| v2_s3 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v2_s3 | 36 | 0.0e+00 | 0.0e+00 | i4_recover | 33 x 1000 | 33 | 0.0e+00 |
| v3_s0 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v3_s1 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v3_s2 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v3_s2 | 36 | 0.0e+00 | 0.0e+00 | i4_recover | 33 x 1000 | 33 | 0.0e+00 |
| v3_s3 | 36 | 0.0e+00 | 0.0e+00 | i4 | 33 x 1000 | 33 | 0.0e+00 |
| v3_s3 | 36 | 0.0e+00 | 0.0e+00 | i4_recover | 33 x 1000 | 33 | 0.0e+00 |

## MBRS (I5-B), COCO test, single seed; WMNet seed means (seeds 0-3) for reference (descriptive)

| Metric | MBRS s0 matched 40 dB | MBRS s0 native | V1+ 40 dB | V2 40 dB | V3 40 dB |
|---|---|---|---|---|---|
| psnr_rgb | 39.9418 | 42.0124 | 40.0159 | 40.0078 | 40.0058 |
| ssim | 0.9853 | 0.9906 | 0.9696 | 0.9748 | 0.9740 |
| clean | 1.0000 | 1.0000 | 0.9995 | 0.9995 | 0.9994 |
| jpeg30 | 0.9803 | 0.9555 | 0.7430 | 0.8820 | 0.9289 |
| jpeg50 | 0.9990 | 0.9935 | 0.9000 | 0.9608 | 0.9866 |
| jpeg70 | 1.0000 | 0.9997 | 0.9805 | 0.9862 | 0.9951 |
| webp50 | 0.9494 | 0.9155 | 0.9293 | 0.9444 | 0.9770 |
| blur1 | 0.9901 | 0.9657 | 0.9863 | 0.9979 | 0.9959 |
| blur2 | 0.6966 | 0.6585 | 0.5093 | 0.6891 | 0.6096 |
| noise005 | 0.9593 | 0.9223 | 0.9966 | 0.9957 | 0.9958 |
| median3 | 0.9810 | 0.9545 | 0.9431 | 0.9888 | 0.9819 |
| resize05 | 0.9705 | 0.9314 | 0.8422 | 0.9810 | 0.9937 |
| crop09 | 0.4987 | 0.4987 | 0.4983 | 0.4900 | 0.4956 |
| crop075 | 0.5044 | 0.5015 | 0.5000 | 0.4952 | 0.4929 |
| rotate10 | 0.4997 | 0.5004 | 0.5015 | 0.4971 | 0.4926 |
| erase10 | 0.9845 | 0.9841 | 0.9908 | 0.9921 | 0.9900 |

- Training: 100 new epochs to epoch 120, 27317 s (273 s per epoch), return code 0.
- Strength matched to 40 dB on validation: 1.2757.
- Validity label: preliminary: single seed (0); 120 epochs (20 pilot + 100 continued); discriminator re-initialised at epoch 20 and optimiser state not restored (MBRS's own continuation restores none); about 74 times the parameters of WMNet.
