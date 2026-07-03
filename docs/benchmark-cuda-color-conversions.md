# CUDA color conversion benchmark

## System

| | |
|---|---|
| Date (UTC) | 2026-07-03 04:22 |
| Host | nvidia-orin00 |
| Machine | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| Kernel / arch | 5.15.148-tegra aarch64 |
| CPU | Cortex-A78AE x6 |
| L4T | # R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025 |
| GPU | nvidia,ga10b |
| CUDA | 12.6.11 |
| Power mode | MAXN_SUPER |
| rustc | rustc 1.93.0 (254b59607 2026-01-19) |
| OpenCV (py) | 4.13.0 |
| VPI | 3.2.4 |
| Git commit | f865362e3a (dirty) |


## 640x480 (min ms per call)

| op | kornia-cpu | kornia-cuda-kernel | kornia-cuda-e2e | kornia-cuda-e2e-pinned | opencv-cpu | vpi-cpu | vpi-cuda | cuda vs best-lib |
|---|---|---|---|---|---|---|---|---|
| apply_colormap_jet_u8 | 0.3095 | 0.0377 | 0.2494 | - | 0.4186 | - | - | 11.11x |
| bgr_from_rgb_u8 | 0.0890 | 0.0242 | 0.6250 | 0.1309 | 0.0467 | 0.6491 | 0.4462 | 1.93x |
| gray_from_rgb_u8 | 0.0823 | 0.0165 | 0.3229 | 0.0909 | 0.0542 | 1.6251 | 0.3393 | 3.28x |
| hsv_from_rgb_f32 | 1.2118 | 0.0829 | 1.1495 | - | 0.1784 | - | - | 2.15x |
| lab_from_rgb_f32 | 5.6509 | 0.0840 | 1.1542 | - | 1.8506 | - | - | 22.02x |
| nv12_from_rgb_u8 | 0.2026 | 0.0218 | 0.2418 | - | - | 1.5213 | 0.3562 | 16.37x |
| rgb_from_bayer_rggb_u8 | 0.6308 | 0.0356 | 0.2708 | - | 0.0697 | - | - | 1.96x |
| rgb_from_nv12_u8 | 0.6210 | 0.0443 | 0.2929 | - | 0.1129 | 0.9408 | 0.3615 | 2.55x |
| rgb_from_rgba_u8 | 0.0549 | 0.0377 | 0.6734 | - | 0.0556 | - | - | 1.47x |
| rgb_from_ycbcr_u8 | 0.6733 | 0.0247 | 0.3285 | - | 0.1066 | - | - | 4.31x |
| rgb_from_yuyv_u8 | 0.6205 | 0.0374 | 0.3052 | - | 0.1059 | - | - | 2.83x |
| rgba_from_rgb_u8 | 0.0961 | 0.0425 | 0.6861 | - | 0.0383 | 0.9315 | 0.4809 | 0.90x |
| sepia_from_rgb_u8 | 0.5779 | 0.0245 | 0.3314 | - | 1.0011 | - | - | 40.81x |
| ycbcr_from_rgb_u8 | 0.5528 | 0.0244 | 0.3290 | 0.1364 | 0.1855 | - | - | 7.61x |
| yuyv_from_rgb_u8 | 0.2557 | 0.0232 | 0.2949 | - | - | - | - | - |

**Fused camera preprocessing** (frame → 640×640 CHW tensor, one kernel)

| pipeline | fused | chained (decode + preprocess) | speedup |
|---|---|---|---|
| preprocess_bgr_640 | **0.0968** | 0.1220 | 1.26x |
| preprocess_nv12_640 | **0.1600** | 0.1636 | 1.02x |
| preprocess_yuyv_640 | **0.1578** | 0.1553 | 0.98x |

## 1280x720 (min ms per call)

| op | kornia-cpu | kornia-cuda-kernel | kornia-cuda-e2e | kornia-cuda-e2e-pinned | opencv-cpu | vpi-cpu | vpi-cuda | cuda vs best-lib |
|---|---|---|---|---|---|---|---|---|
| apply_colormap_jet_u8 | 0.9285 | 0.1086 | 0.7231 | - | 0.6412 | - | - | 5.91x |
| bgr_from_rgb_u8 | 0.2751 | 0.0634 | 1.1598 | 0.3384 | 0.1298 | 0.5689 | 0.5458 | 2.05x |
| gray_from_rgb_u8 | 0.2353 | 0.0444 | 0.9062 | 0.2339 | 0.1605 | 2.8155 | 0.4566 | 3.61x |
| hsv_from_rgb_f32 | 3.6344 | 0.2346 | 3.6276 | - | 0.6401 | - | - | 2.73x |
| lab_from_rgb_f32 | 16.9564 | 0.2378 | 3.6289 | - | 5.9993 | - | - | 25.23x |
| nv12_from_rgb_u8 | 0.6046 | 0.0538 | 0.6344 | - | - | 2.8804 | 0.4536 | 8.43x |
| rgb_from_bayer_rggb_u8 | 1.3058 | 0.0937 | 0.6938 | - | 0.2112 | - | - | 2.25x |
| rgb_from_nv12_u8 | 1.8609 | 0.1195 | 0.7493 | - | 0.3156 | 1.3229 | 0.4604 | 2.64x |
| rgb_from_rgba_u8 | 0.1952 | 0.1013 | 1.2940 | - | 0.1404 | - | - | 1.39x |
| rgb_from_ycbcr_u8 | 2.0205 | 0.0636 | 0.8627 | - | 0.3281 | - | - | 5.16x |
| rgb_from_yuyv_u8 | 1.8569 | 0.1026 | 0.7872 | - | 0.2959 | - | - | 2.88x |
| rgba_from_rgb_u8 | 0.2976 | 0.1148 | 1.3501 | - | 0.1733 | 1.0337 | 0.4734 | 1.51x |
| sepia_from_rgb_u8 | 1.7335 | 0.0631 | 0.8613 | - | 3.0024 | - | - | 47.57x |
| ycbcr_from_rgb_u8 | 1.6602 | 0.0649 | 0.8624 | 0.3372 | 0.5814 | - | - | 8.95x |
| yuyv_from_rgb_u8 | 0.7648 | 0.0657 | 0.7028 | - | - | - | - | - |

**Fused camera preprocessing** (frame → 640×640 CHW tensor, one kernel)

| pipeline | fused | chained (decode + preprocess) | speedup |
|---|---|---|---|
| preprocess_bgr_640 | **0.0896** | 0.1528 | 1.70x |
| preprocess_nv12_640 | **0.1320** | 0.2208 | 1.67x |
| preprocess_yuyv_640 | **0.1303** | 0.2027 | 1.56x |

## 1920x1080 (min ms per call)

| op | kornia-cpu | kornia-cuda-kernel | kornia-cuda-e2e | kornia-cuda-e2e-pinned | opencv-cpu | vpi-cpu | vpi-cuda | cuda vs best-lib |
|---|---|---|---|---|---|---|---|---|
| apply_colormap_jet_u8 | 2.0903 | 0.2364 | 1.5067 | - | 1.2769 | - | - | 5.40x |
| bgr_from_rgb_u8 | 0.3533 | 0.1335 | 2.2057 | 0.6865 | 0.3072 | 1.2188 | 0.6879 | 2.30x |
| gray_from_rgb_u8 | 0.2132 | 0.0917 | 1.4721 | 0.4528 | 0.3132 | 5.8174 | 0.5589 | 3.42x |
| hsv_from_rgb_f32 | 1.4073 | 0.5190 | 7.5353 | - | 1.3509 | - | - | 2.60x |
| lab_from_rgb_f32 | 6.4143 | 0.5260 | 7.5297 | - | 11.2366 | - | - | 21.36x |
| nv12_from_rgb_u8 | 0.3160 | 0.1128 | 1.4068 | - | - | 5.7345 | 0.5842 | 5.18x |
| rgb_from_bayer_rggb_u8 | 2.4798 | 0.2000 | 1.4578 | - | 0.4767 | - | - | 2.38x |
| rgb_from_nv12_u8 | 0.8693 | 0.2608 | 1.9570 | - | 0.6941 | 2.6492 | 0.5901 | 2.26x |
| rgb_from_rgba_u8 | 0.3821 | 0.2201 | 2.2791 | - | 0.4134 | - | - | 1.88x |
| rgb_from_ycbcr_u8 | 0.7672 | 0.1346 | 1.9113 | - | 0.6486 | - | - | 4.82x |
| rgb_from_yuyv_u8 | 0.7331 | 0.2232 | 1.7364 | - | 0.6536 | - | - | 2.93x |
| rgba_from_rgb_u8 | 0.3714 | 0.2504 | 2.3376 | - | 0.3498 | 2.1490 | 0.5965 | 1.40x |
| sepia_from_rgb_u8 | 0.6618 | 0.1338 | 1.9238 | - | 6.7554 | - | - | 50.48x |
| ycbcr_from_rgb_u8 | 0.6449 | 0.1341 | 1.9140 | 0.6914 | 1.3351 | - | - | 9.96x |
| yuyv_from_rgb_u8 | 0.3231 | 0.1387 | 1.5903 | - | - | - | - | - |

**Fused camera preprocessing** (frame → 640×640 CHW tensor, one kernel)

| pipeline | fused | chained (decode + preprocess) | speedup |
|---|---|---|---|
| preprocess_bgr_640 | **0.1005** | 0.2330 | 2.32x |
| preprocess_nv12_640 | **0.1322** | 0.3631 | 2.75x |
| preprocess_yuyv_640 | **0.1303** | 0.3254 | 2.50x |

## 3840x2160 (min ms per call)

| op | kornia-cpu | kornia-cuda-kernel | kornia-cuda-e2e | kornia-cuda-e2e-pinned | opencv-cpu | vpi-cpu | vpi-cuda | cuda vs best-lib |
|---|---|---|---|---|---|---|---|---|
| apply_colormap_jet_u8 | 8.3798 | 0.9250 | 5.8950 | - | 3.4171 | - | - | 3.69x |
| bgr_from_rgb_u8 | 1.1584 | 0.5156 | 7.5598 | 2.5579 | 1.2432 | 3.2410 | 1.4103 | 2.41x |
| gray_from_rgb_u8 | 0.9206 | 0.3470 | 5.0281 | 1.6403 | 1.1582 | 21.3851 | 1.2135 | 3.34x |
| hsv_from_rgb_f32 | 5.5265 | 2.1225 | 80.3585 | - | 21.5649 | - | - | 10.16x |
| lab_from_rgb_f32 | 25.6215 | 2.1775 | 76.6834 | - | 61.4865 | - | - | 28.24x |
| nv12_from_rgb_u8 | 1.2204 | 0.4298 | 5.9166 | - | - | 20.5331 | 1.2704 | 2.96x |
| rgb_from_bayer_rggb_u8 | 8.0914 | 0.7702 | 5.7208 | - | 1.7247 | - | - | 2.24x |
| rgb_from_nv12_u8 | 3.6675 | 1.0094 | 6.5151 | - | 2.3263 | 8.5530 | 1.3992 | 1.39x |
| rgb_from_rgba_u8 | 1.5835 | 0.8635 | 8.9861 | - | 1.9615 | - | - | 2.27x |
| rgb_from_ycbcr_u8 | 3.0527 | 0.5206 | 7.5167 | - | 2.3428 | - | - | 4.50x |
| rgb_from_yuyv_u8 | 3.3398 | 0.8763 | 6.8664 | - | 2.1264 | - | - | 2.43x |
| rgba_from_rgb_u8 | 1.4542 | 0.9825 | 9.1447 | - | 1.6680 | 5.4069 | 1.3015 | 1.32x |
| sepia_from_rgb_u8 | 2.6391 | 0.5160 | 7.5681 | - | 27.1050 | - | - | 52.53x |
| ycbcr_from_rgb_u8 | 3.0260 | 0.5172 | 7.5360 | 2.5930 | 4.2067 | - | - | 8.13x |
| yuyv_from_rgb_u8 | 1.4570 | 0.5391 | 6.4142 | - | - | - | - | - |

**Fused camera preprocessing** (frame → 640×640 CHW tensor, one kernel)

| pipeline | fused | chained (decode + preprocess) | speedup |
|---|---|---|---|
| preprocess_bgr_640 | **0.1444** | 0.6583 | 4.56x |
| preprocess_nv12_640 | **0.1328** | 1.1498 | 8.65x |
| preprocess_yuyv_640 | **0.1344** | 1.0147 | 7.55x |
