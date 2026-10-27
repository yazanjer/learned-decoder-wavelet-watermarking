Across both MBRS training runs (seed 0, seed 1) and the four WMNet seeds: theta = mean of the WMNet seed means - mean of the two MBRS model means; Welch 95 % t-interval over the model means; per-model estimates (seed 0 | seed 1). Direction only if both per-model estimates share a sign and the Welch interval excludes 0 (docs/PROPOSAL_MBRS_SEEDS.md, Section 6.3). Descriptive; no test, no Holm family.

| Variant | Endpoint | MBRS s0 | MBRS s1 | theta | Welch 95 % [df] | per model (s0, s1) | Direction |
|---|---|---|---|---|---|---|---|
| V1+ | psnr_rgb | 39.9418 | 39.9235 | +0.0833 | [+0.0086, +0.1579] [1.3] | +0.0741, +0.0924 | v1p higher |
| V1+ | ssim | 0.9853 | 0.9879 | -0.0170 | [-0.0229, -0.0111] [3.3] | -0.0157, -0.0183 | mbrs higher |
| V1+ | bitacc_blur05 | 1.0000 | 0.9992 | -0.0000 | [-0.0028, +0.0027] [1.5] | -0.0004, +0.0004 | not resolved across mbrs seeds |
| V1+ | bitacc_blur1 | 0.9901 | 0.9282 | +0.0271 | [-0.3579, +0.4122] [1.0] | -0.0038, +0.0581 | not resolved across mbrs seeds |
| V1+ | bitacc_blur2 | 0.6966 | 0.6667 | -0.1724 | [-0.3565, +0.0118] [1.0] | -0.1873, -0.1574 | not resolved across mbrs seeds |
| V1+ | bitacc_bright08 | 1.0000 | 0.9997 | -0.0003 | [-0.0010, +0.0004] [3.0] | -0.0004, -0.0001 | not resolved across mbrs seeds |
| V1+ | bitacc_bright12 | 0.9887 | 0.9772 | +0.0087 | [-0.0638, +0.0812] [1.0] | +0.0029, +0.0144 | not resolved across mbrs seeds |
| V1+ | bitacc_clean | 1.0000 | 0.9998 | -0.0004 | [-0.0009, +0.0001] [3.7] | -0.0005, -0.0003 | not resolved across mbrs seeds |
| V1+ | bitacc_contrast08 | 1.0000 | 0.9997 | -0.0003 | [-0.0009, +0.0003] [3.5] | -0.0004, -0.0002 | not resolved across mbrs seeds |
| V1+ | bitacc_contrast12 | 0.9941 | 0.9848 | +0.0076 | [-0.0503, +0.0656] [1.0] | +0.0030, +0.0122 | not resolved across mbrs seeds |
| V1+ | bitacc_crop075 | 0.5044 | 0.4965 | -0.0004 | [-0.0437, +0.0429] [1.1] | -0.0043, +0.0036 | not resolved across mbrs seeds |
| V1+ | bitacc_crop09 | 0.4987 | 0.4972 | +0.0004 | [-0.0069, +0.0076] [3.5] | -0.0004, +0.0011 | not resolved across mbrs seeds |
| V1+ | bitacc_erase10 | 0.9845 | 0.9680 | +0.0146 | [-0.0854, +0.1145] [1.0] | +0.0064, +0.0228 | not resolved across mbrs seeds |
| V1+ | bitacc_jpeg10 | 0.7679 | 0.7466 | -0.1980 | [-0.3252, -0.0707] [1.0] | -0.2086, -0.1873 | mbrs higher |
| V1+ | bitacc_jpeg20 | 0.9287 | 0.8811 | -0.2544 | [-0.4988, -0.0101] [1.1] | -0.2783, -0.2306 | mbrs higher |
| V1+ | bitacc_jpeg30 | 0.9802 | 0.9400 | -0.2171 | [-0.3736, -0.0606] [1.3] | -0.2372, -0.1970 | mbrs higher |
| V1+ | bitacc_jpeg40 | 0.9949 | 0.9676 | -0.1503 | [-0.2182, -0.0824] [2.2] | -0.1640, -0.1366 | mbrs higher |
| V1+ | bitacc_jpeg50 | 0.9990 | 0.9850 | -0.0920 | [-0.1245, -0.0596] [3.6] | -0.0990, -0.0850 | mbrs higher |
| V1+ | bitacc_jpeg60 | 0.9998 | 0.9939 | -0.0468 | [-0.0661, -0.0276] [3.9] | -0.0498, -0.0439 | mbrs higher |
| V1+ | bitacc_jpeg70 | 1.0000 | 0.9984 | -0.0187 | [-0.0295, -0.0080] [3.3] | -0.0195, -0.0180 | mbrs higher |
| V1+ | bitacc_jpeg80 | 1.0000 | 0.9994 | -0.0075 | [-0.0118, -0.0031] [3.2] | -0.0077, -0.0072 | mbrs higher |
| V1+ | bitacc_jpeg90 | 1.0000 | 0.9997 | -0.0037 | [-0.0057, -0.0018] [3.3] | -0.0039, -0.0036 | mbrs higher |
| V1+ | bitacc_median3 | 0.9810 | 0.9395 | -0.0171 | [-0.2229, +0.1886] [1.1] | -0.0378, +0.0036 | not resolved across mbrs seeds |
| V1+ | bitacc_noise001 | 1.0000 | 0.9996 | -0.0004 | [-0.0012, +0.0004] [2.6] | -0.0006, -0.0002 | not resolved across mbrs seeds |
| V1+ | bitacc_noise002 | 0.9999 | 0.9941 | +0.0022 | [-0.0343, +0.0388] [1.0] | -0.0007, +0.0052 | not resolved across mbrs seeds |
| V1+ | bitacc_noise005 | 0.9593 | 0.9172 | +0.0583 | [-0.2084, +0.3251] [1.0] | +0.0373, +0.0794 | not resolved across mbrs seeds |
| V1+ | bitacc_resize05 | 0.9705 | 0.8859 | -0.0860 | [-0.2929, +0.1209] [2.2] | -0.1283, -0.0437 | not resolved across mbrs seeds |
| V1+ | bitacc_resize075 | 1.0000 | 0.9832 | +0.0064 | [-0.0982, +0.1110] [1.0] | -0.0020, +0.0148 | not resolved across mbrs seeds |
| V1+ | bitacc_rotate10 | 0.4998 | 0.5003 | +0.0015 | [-0.0022, +0.0051] [3.3] | +0.0018, +0.0012 | not resolved across mbrs seeds |
| V1+ | bitacc_webp50 | 0.9494 | 0.9145 | -0.0026 | [-0.1231, +0.1179] [1.5] | -0.0201, +0.0149 | not resolved across mbrs seeds |
| V1+ | bitacc_webp75 | 0.9846 | 0.9590 | +0.0087 | [-0.1133, +0.1307] [1.2] | -0.0041, +0.0215 | not resolved across mbrs seeds |
| V2 | psnr_rgb | 39.9418 | 39.9235 | +0.0752 | [-0.0219, +0.1722] [1.1] | +0.0660, +0.0843 | not resolved across mbrs seeds |
| V2 | ssim | 0.9853 | 0.9879 | -0.0118 | [-0.0222, -0.0014] [1.3] | -0.0105, -0.0131 | mbrs higher |
| V2 | bitacc_blur05 | 1.0000 | 0.9992 | +0.0001 | [-0.0043, +0.0045] [1.1] | -0.0003, +0.0005 | not resolved across mbrs seeds |
| V2 | bitacc_blur1 | 0.9901 | 0.9282 | +0.0387 | [-0.3531, +0.4306] [1.0] | +0.0077, +0.0697 | not resolved across mbrs seeds |
| V2 | bitacc_blur2 | 0.6966 | 0.6667 | +0.0074 | [-0.2582, +0.2730] [3.2] | -0.0076, +0.0224 | not resolved across mbrs seeds |
| V2 | bitacc_bright08 | 1.0000 | 0.9997 | -0.0002 | [-0.0010, +0.0006] [1.9] | -0.0003, -0.0000 | not resolved across mbrs seeds |
| V2 | bitacc_bright12 | 0.9887 | 0.9772 | +0.0074 | [-0.0462, +0.0610] [1.2] | +0.0017, +0.0132 | not resolved across mbrs seeds |
| V2 | bitacc_clean | 1.0000 | 0.9998 | -0.0004 | [-0.0009, +0.0001] [3.4] | -0.0005, -0.0003 | not resolved across mbrs seeds |
| V2 | bitacc_contrast08 | 1.0000 | 0.9997 | -0.0002 | [-0.0009, +0.0005] [2.0] | -0.0003, -0.0001 | not resolved across mbrs seeds |
| V2 | bitacc_contrast12 | 0.9941 | 0.9848 | +0.0067 | [-0.0448, +0.0581] [1.1] | +0.0021, +0.0113 | not resolved across mbrs seeds |
| V2 | bitacc_crop075 | 0.5044 | 0.4965 | -0.0052 | [-0.0260, +0.0155] [2.0] | -0.0092, -0.0013 | not resolved across mbrs seeds |
| V2 | bitacc_crop09 | 0.4987 | 0.4972 | -0.0080 | [-0.0208, +0.0048] [3.2] | -0.0087, -0.0072 | not resolved across mbrs seeds |
| V2 | bitacc_erase10 | 0.9845 | 0.9680 | +0.0158 | [-0.0807, +0.1123] [1.0] | +0.0076, +0.0240 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg10 | 0.7679 | 0.7466 | -0.1284 | [-0.2285, -0.0283] [3.6] | -0.1391, -0.1177 | mbrs higher |
| V2 | bitacc_jpeg20 | 0.9287 | 0.8811 | -0.1193 | [-0.2997, +0.0610] [3.8] | -0.1432, -0.0955 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg30 | 0.9802 | 0.9400 | -0.0781 | [-0.2328, +0.0765] [3.7] | -0.0983, -0.0580 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg40 | 0.9949 | 0.9676 | -0.0485 | [-0.1442, +0.0472] [3.8] | -0.0622, -0.0348 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg50 | 0.9990 | 0.9850 | -0.0312 | [-0.0865, +0.0240] [3.7] | -0.0382, -0.0243 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg60 | 0.9998 | 0.9939 | -0.0208 | [-0.0517, +0.0102] [3.5] | -0.0237, -0.0178 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg70 | 1.0000 | 0.9984 | -0.0131 | [-0.0293, +0.0032] [3.1] | -0.0138, -0.0123 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg80 | 1.0000 | 0.9994 | -0.0084 | [-0.0186, +0.0018] [3.0] | -0.0087, -0.0082 | not resolved across mbrs seeds |
| V2 | bitacc_jpeg90 | 1.0000 | 0.9997 | -0.0047 | [-0.0118, +0.0023] [3.0] | -0.0049, -0.0046 | not resolved across mbrs seeds |
| V2 | bitacc_median3 | 0.9810 | 0.9395 | +0.0285 | [-0.1970, +0.2541] [1.1] | +0.0078, +0.0493 | not resolved across mbrs seeds |
| V2 | bitacc_noise001 | 1.0000 | 0.9996 | -0.0003 | [-0.0012, +0.0006] [2.0] | -0.0005, -0.0001 | not resolved across mbrs seeds |
| V2 | bitacc_noise002 | 0.9999 | 0.9941 | +0.0023 | [-0.0345, +0.0390] [1.0] | -0.0007, +0.0052 | not resolved across mbrs seeds |
| V2 | bitacc_noise005 | 0.9593 | 0.9172 | +0.0574 | [-0.2077, +0.3225] [1.0] | +0.0364, +0.0785 | not resolved across mbrs seeds |
| V2 | bitacc_resize05 | 0.9705 | 0.8859 | +0.0528 | [-0.4592, +0.5648] [1.0] | +0.0105, +0.0951 | not resolved across mbrs seeds |
| V2 | bitacc_resize075 | 1.0000 | 0.9832 | +0.0075 | [-0.0982, +0.1131] [1.0] | -0.0009, +0.0158 | not resolved across mbrs seeds |
| V2 | bitacc_rotate10 | 0.4998 | 0.5003 | -0.0029 | [-0.0142, +0.0084] [3.0] | -0.0026, -0.0032 | not resolved across mbrs seeds |
| V2 | bitacc_webp50 | 0.9494 | 0.9145 | +0.0124 | [-0.0826, +0.1074] [1.9] | -0.0050, +0.0299 | not resolved across mbrs seeds |
| V2 | bitacc_webp75 | 0.9846 | 0.9590 | +0.0121 | [-0.0733, +0.0974] [1.5] | -0.0007, +0.0248 | not resolved across mbrs seeds |
| V3 | psnr_rgb | 39.9418 | 39.9235 | +0.0731 | [-0.0345, +0.1808] [1.0] | +0.0640, +0.0823 | not resolved across mbrs seeds |
| V3 | ssim | 0.9853 | 0.9879 | -0.0126 | [-0.0185, -0.0066] [2.9] | -0.0113, -0.0139 | mbrs higher |
| V3 | bitacc_blur05 | 1.0000 | 0.9992 | -0.0001 | [-0.0020, +0.0017] [2.7] | -0.0005, +0.0003 | not resolved across mbrs seeds |
| V3 | bitacc_blur1 | 0.9901 | 0.9282 | +0.0368 | [-0.3551, +0.4287] [1.0] | +0.0058, +0.0678 | not resolved across mbrs seeds |
| V3 | bitacc_blur2 | 0.6966 | 0.6667 | -0.0720 | [-0.2699, +0.1258] [3.3] | -0.0870, -0.0571 | not resolved across mbrs seeds |
| V3 | bitacc_bright08 | 1.0000 | 0.9997 | -0.0004 | [-0.0015, +0.0007] [3.8] | -0.0005, -0.0002 | not resolved across mbrs seeds |
| V3 | bitacc_bright12 | 0.9887 | 0.9772 | +0.0037 | [-0.0519, +0.0592] [1.2] | -0.0021, +0.0094 | not resolved across mbrs seeds |
| V3 | bitacc_clean | 1.0000 | 0.9998 | -0.0005 | [-0.0016, +0.0007] [3.5] | -0.0006, -0.0004 | not resolved across mbrs seeds |
| V3 | bitacc_contrast08 | 1.0000 | 0.9997 | -0.0004 | [-0.0014, +0.0007] [3.7] | -0.0005, -0.0002 | not resolved across mbrs seeds |
| V3 | bitacc_contrast12 | 0.9941 | 0.9848 | +0.0031 | [-0.0365, +0.0427] [1.2] | -0.0015, +0.0077 | not resolved across mbrs seeds |
| V3 | bitacc_crop075 | 0.5044 | 0.4965 | -0.0076 | [-0.0278, +0.0126] [2.1] | -0.0115, -0.0036 | not resolved across mbrs seeds |
| V3 | bitacc_crop09 | 0.4987 | 0.4972 | -0.0024 | [-0.0098, +0.0050] [3.5] | -0.0031, -0.0017 | not resolved across mbrs seeds |
| V3 | bitacc_erase10 | 0.9845 | 0.9680 | +0.0137 | [-0.0679, +0.0953] [1.1] | +0.0055, +0.0220 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg10 | 0.7679 | 0.7466 | -0.0716 | [-0.1925, +0.0493] [3.4] | -0.0823, -0.0609 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg20 | 0.9287 | 0.8811 | -0.0538 | [-0.2118, +0.1041] [3.9] | -0.0777, -0.0300 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg30 | 0.9802 | 0.9400 | -0.0313 | [-0.1288, +0.0663] [3.8] | -0.0514, -0.0111 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg40 | 0.9949 | 0.9676 | -0.0095 | [-0.0836, +0.0647] [1.9] | -0.0231, +0.0042 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg50 | 0.9990 | 0.9850 | -0.0054 | [-0.0637, +0.0530] [1.3] | -0.0124, +0.0016 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg60 | 0.9998 | 0.9939 | -0.0052 | [-0.0294, +0.0191] [1.3] | -0.0081, -0.0022 | not resolved across mbrs seeds |
| V3 | bitacc_jpeg70 | 1.0000 | 0.9984 | -0.0041 | [-0.0076, -0.0006] [2.7] | -0.0049, -0.0033 | mbrs higher |
| V3 | bitacc_jpeg80 | 1.0000 | 0.9994 | -0.0022 | [-0.0040, -0.0004] [3.9] | -0.0025, -0.0019 | mbrs higher |
| V3 | bitacc_jpeg90 | 1.0000 | 0.9997 | -0.0009 | [-0.0023, +0.0006] [3.5] | -0.0010, -0.0007 | not resolved across mbrs seeds |
| V3 | bitacc_median3 | 0.9810 | 0.9395 | +0.0217 | [-0.1541, +0.1975] [1.3] | +0.0010, +0.0424 | not resolved across mbrs seeds |
| V3 | bitacc_noise001 | 1.0000 | 0.9996 | -0.0005 | [-0.0018, +0.0008] [3.8] | -0.0007, -0.0003 | not resolved across mbrs seeds |
| V3 | bitacc_noise002 | 0.9999 | 0.9941 | +0.0021 | [-0.0314, +0.0357] [1.1] | -0.0008, +0.0050 | not resolved across mbrs seeds |
| V3 | bitacc_noise005 | 0.9593 | 0.9172 | +0.0575 | [-0.2045, +0.3195] [1.0] | +0.0364, +0.0785 | not resolved across mbrs seeds |
| V3 | bitacc_resize05 | 0.9705 | 0.8859 | +0.0655 | [-0.4695, +0.6005] [1.0] | +0.0232, +0.1078 | not resolved across mbrs seeds |
| V3 | bitacc_resize075 | 1.0000 | 0.9832 | +0.0072 | [-0.0968, +0.1112] [1.0] | -0.0011, +0.0156 | not resolved across mbrs seeds |
| V3 | bitacc_rotate10 | 0.4998 | 0.5003 | -0.0074 | [-0.0206, +0.0057] [3.0] | -0.0072, -0.0077 | not resolved across mbrs seeds |
| V3 | bitacc_webp50 | 0.9494 | 0.9145 | +0.0450 | [-0.1692, +0.2592] [1.0] | +0.0275, +0.0625 | not resolved across mbrs seeds |
| V3 | bitacc_webp75 | 0.9846 | 0.9590 | +0.0233 | [-0.1360, +0.1825] [1.0] | +0.0105, +0.0360 | not resolved across mbrs seeds |
