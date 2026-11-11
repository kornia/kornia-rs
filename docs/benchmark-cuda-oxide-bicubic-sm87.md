# cuda-oxide Bicubic Resize: Jetson Orin Nano (sm_87)

> **Historical snapshot, not reproducible on current `main`.** Measured on the
> `feat/cuda-oxide-interp-sweep` branch before the oxide-kernels refactor, when
> the bicubic kernel lived in `cuda::resize_oxide` with its own benchmark arms.
> That module has been removed: bicubic stays NVRTC-only on current `main`
> until cuda-oxide can lower `f32::mul_add` to a native FMA instruction (see
> "Backend Note" below and `benchmark-cuda-oxide-bicubic-sm75.md`). The
> reproduction commands are kept as written for the commit they were run
> against; they do not run on current `main`.

**Date:** 2026-10-02
**GPU:** NVIDIA Jetson Orin Nano Super (Ampere, sm_87)
**CUDA:** 12.6
**Test Count:** 7 cases (2× downscale, 2× upscale, HD, 4K, non-dyadic)

## Summary

Oxide bicubic outperformed the NVRTC kernel across all cases on this
snapshot:

| Case | NVRTC (ms) | oxide (ms) | Speedup |
|---|---|---|---|
| 1024×1024→512×512 | 0.188 | 0.183 | 2.7% |
| 512×512→1024×1024 | 0.471 | 0.417 | 11.4% |
| 1920×1080→960×540 | 0.355 | 0.347 | 2.3% |
| 960×540→1920×1080 | 0.922 | 0.818 | 11.3% |
| 3840×2160→1920×1080 | 1.408 | 1.356 | 3.7% |

Upscale cases show the largest wins (11%), indicating that unified memory and
register occupancy favor the oxide implementation on this architecture. See
`benchmark-cuda-oxide-bicubic-sm75.md` for the same snapshot on Turing, where
the oxide kernel trailed NVRTC instead — the clamp-overhead codegen
characteristic of this cuda-oxide snapshot (`c8b3103`) affects the two
architectures' occupancy and memory subsystems differently.

## Backend Note

`f32::mul_add` triggers NVVM IR mode on the `c8b3103` snapshot (no native FMA
lowering), so the bicubic kernel used plain multiplication and addition
throughout (Horner-form weights + accumulation) instead. Its output was
bit-exact with a **no-FMA CPU reference implementation** written specifically
for this comparison, not with the production CPU `bicubic_sample`, which uses
`mul_add` and therefore differs by up to 1 ULP from both the no-FMA reference
and this oxide kernel. The NVRTC bicubic kernel uses `fmaf` and also differs
from the oxide kernel by about 1 ULP.

## Test Coverage

- `bicubic_oxide_matches_cpu_bit_exact` (removed from current `main` with the
  `cuda::resize_oxide` module): 7 cases with 7 test inputs each, no tolerance
  against the no-FMA CPU reference described above
- Result at the time: all pass, confirmed on sm_87

## Reproduction (historical commands, do not run on current `main`)

```bash
ssh <jetson-host>
cd ~/kornia-rs && git checkout feat/cuda-oxide-interp-sweep
cargo test --release -p kornia-imgproc --features cuda --lib cuda::resize_oxide::tests::bicubic_oxide_matches_cpu_bit_exact
cargo run -p kornia-imgproc --example bench_cuda_resize --features cuda --release 2>&1 | grep -A 20 "cuda-oxide bicubic sweep"
```
