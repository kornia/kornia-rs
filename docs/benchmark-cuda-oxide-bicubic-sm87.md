# cuda-oxide Bicubic Resize: Jetson Orin Nano (sm_87)

**Date:** 2026-10-02  
**GPU:** NVIDIA Jetson Orin Nano Super (Ampere, sm_87)  
**CUDA:** 12.6  
**Test Count:** 7 cases (2× downscale, 2× upscale, HD, 4K, non-dyadic)

## Summary

Oxide bicubic outperforms NVRTC across all cases on Ampere architecture:

| Case | NVRTC (ms) | oxide (ms) | Speedup |
|---|---|---|---|
| 1024×1024→512×512 | 0.188 | 0.183 | 2.7% |
| 512×512→1024×1024 | 0.471 | 0.417 | 11.4% |
| 1920×1080→960×540 | 0.355 | 0.347 | 2.3% |
| 960×540→1920×1080 | 0.922 | 0.818 | 11.3% |
| 3840×2160→1920×1080 | 1.408 | 1.356 | 3.7% |

Upscale cases show the largest wins (11%), indicating that unified memory and register occupancy favor the oxide implementation on this architecture.

## Backend Note

Backend gap #4: `f32::mul_add` triggers NVVM IR mode in c8b3103 snapshot. Bicubic kernel uses plain multiplication and addition throughout (Horner-form weights + accumulation). Results are bit-exact with the CPU path but differ ~1 ULP from NVRTC bicubic (which uses fmaf).

## Test Coverage

- `bicubic_oxide_matches_cpu_bit_exact`: 7 cases with 7 test inputs each, no tolerance
- Result: All pass, confirmed on sm_87

## Reproduction

```bash
ssh -i ~/.ssh/id_ed25519_kornia inchara@nvidia-orin01
cd ~/kornia-rs && git checkout feat/cuda-oxide-interp-sweep
cargo test --release -p kornia-imgproc --features cuda --lib cuda::resize_oxide::tests::bicubic_oxide_matches_cpu_bit_exact
cargo run -p kornia-imgproc --example bench_cuda_resize --features cuda --release 2>&1 | grep -A 20 "cuda-oxide bicubic sweep"
```
