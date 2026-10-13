# cuda-oxide Bicubic Resize: GTX 1650 (sm_75)

**Date:** 2026-10-02  
**GPU:** NVIDIA GeForce GTX 1650 (Turing, sm_75)  
**CUDA:** 12.4  
**Test Count:** 7 cases (2× downscale, 2× upscale, HD, 4K, non-dyadic)

## Summary

Oxide bicubic underperforms NVRTC on Turing architecture:

| Case | NVRTC (ms) | oxide (ms) | Penalty |
|---|---|---|---|
| 1024×1024→512×512 | 0.120 | 0.124 | 3.3% |
| 512×512→1024×1024 | 0.204 | 0.214 | 4.9% |
| 1920×1080→960×540 | 0.243 | 0.247 | 1.6% |
| 960×540→1920×1080 | 0.415 | 0.429 | 3.4% |
| 3840×2160→1920×1080 | 0.946 | 0.964 | 1.9% |

Penalty is consistent across all cases (1.6–4.9%), explained by backend gap #2: clamp operations (4×4 tap neighborhood = 8 clamp sites) compile to setp+selp branch pairs instead of native min.f32/max.f32 instructions.

## Backend Note

Backend gap #4: `f32::mul_add` triggers NVVM IR mode in c8b3103 snapshot. Bicubic kernel uses plain multiplication and addition throughout (Horner-form weights + accumulation). Results are bit-exact with the CPU path but differ ~1 ULP from NVRTC bicubic (which uses fmaf).

Backend gap #2 (clamping): Lack of native min.f32/max.f32 support on Turing means each min/max becomes setp+selp (predicate set + select). For compute-bound kernels like bicubic with many clamps, this is measurable. Bandwidth-bound kernels (bilinear, nearest) show no penalty.

## Test Coverage

- `bicubic_oxide_matches_cpu_bit_exact`: 7 cases with 7 test inputs each, no tolerance
- Result: All pass, confirmed on sm_75

## Reproduction

```bash
cd /path/to/kornia-rs && git checkout feat/cuda-oxide-interp-sweep
cargo test --release -p kornia-imgproc --features cuda --lib cuda::resize_oxide::tests::bicubic_oxide_matches_cpu_bit_exact
cargo run -p kornia-imgproc --example bench_cuda_resize --features cuda --release 2>&1 | grep -A 20 "cuda-oxide bicubic sweep"
```

## Comparison to sm_87

On Jetson Orin (sm_87), oxide bicubic is 2–11% faster. The architecture flip (win on Ampere, penalty on Turing) is explained by:
- Ampere: Better unified memory, different occupancy constraints
- Turing: No native clamping instructions, branch penalties are visible

For mobile-first workloads, oxide is favorable. For desktop, NVRTC remains the baseline.
