# cuda-oxide Bicubic Resize: GTX 1650 (sm_75)

> **Historical snapshot, not reproducible on current `main`.** Measured on the
> `feat/cuda-oxide-interp-sweep` branch before the oxide-kernels refactor, when
> the bicubic kernel lived in `cuda::resize_oxide` with its own benchmark arms.
> That module has been removed: bicubic stays NVRTC-only on current `main`
> until cuda-oxide can lower `f32::mul_add` to a native FMA instruction (see
> "Backend Note" below). The reproduction commands are kept as written for the
> commit they were run against; they do not run on current `main`.

**Date:** 2026-10-02
**GPU:** NVIDIA GeForce GTX 1650 (Turing, sm_75)
**CUDA:** 12.4
**Test Count:** 7 cases (2× downscale, 2× upscale, HD, 4K, non-dyadic)

## Summary

Oxide bicubic underperformed the NVRTC kernel on this snapshot:

| Case | NVRTC (ms) | oxide (ms) | Penalty |
|---|---|---|---|
| 1024×1024→512×512 | 0.120 | 0.124 | 3.3% |
| 512×512→1024×1024 | 0.204 | 0.214 | 4.9% |
| 1920×1080→960×540 | 0.243 | 0.247 | 1.6% |
| 960×540→1920×1080 | 0.415 | 0.429 | 3.4% |
| 3840×2160→1920×1080 | 0.946 | 0.964 | 1.9% |

Penalty is consistent across all cases (1.6-4.9%). The bicubic kernel's 4x4 tap
neighborhood needs 8 clamp sites per pixel; on this cuda-oxide snapshot
(`c8b3103`) those clamps compiled to setp+selp branch pairs rather than native
`min.f32`/`max.f32` instructions. This is specific to that snapshot's codegen,
not a Turing hardware limitation: sm_75 has native `min.f32`/`max.f32`, and
the bilinear and nearest kernels on this same GPU show no such penalty because
they have far fewer clamp sites per pixel.

## Backend Note

`f32::mul_add` triggers NVVM IR mode on the `c8b3103` snapshot (no native FMA
lowering), so the bicubic kernel used plain multiplication and addition
throughout (Horner-form weights + accumulation) instead. Its output was
bit-exact with a **no-FMA CPU reference implementation** written specifically
for this comparison, not with the production CPU `bicubic_sample`, which uses
`mul_add` and therefore differs by up to 1 ULP from both the no-FMA reference
and this oxide kernel. The NVRTC bicubic kernel uses `fmaf` and also differs
from the oxide kernel by about 1 ULP.

The setp+selp clamp overhead measured above is specific to this cuda-oxide
snapshot's codegen for the 8-clamp bicubic case. It is not present in the
bilinear and nearest kernels on the same GPU, which have fewer clamp sites per
pixel.

## Test Coverage

- `bicubic_oxide_matches_cpu_bit_exact` (removed from current `main` with the
  `cuda::resize_oxide` module): 7 cases with 7 test inputs each, no tolerance
  against the no-FMA CPU reference described above
- Result at the time: all pass, confirmed on sm_75

## Reproduction (historical commands, do not run on current `main`)

```bash
cd /path/to/kornia-rs && git checkout feat/cuda-oxide-interp-sweep
cargo test --release -p kornia-imgproc --features cuda --lib cuda::resize_oxide::tests::bicubic_oxide_matches_cpu_bit_exact
cargo run -p kornia-imgproc --example bench_cuda_resize --features cuda --release 2>&1 | grep -A 20 "cuda-oxide bicubic sweep"
```

## Comparison to sm_87

On Jetson Orin (sm_87), the same oxide bicubic snapshot was 2-11% faster than
NVRTC (see `benchmark-cuda-oxide-bicubic-sm87.md`) — the architecture flip is
consistent with the clamp overhead being a codegen characteristic of this
cuda-oxide snapshot rather than a fixed hardware cost: it showed up
differently on the two architectures' occupancy and memory subsystems rather
than being present on one and absent on the other for hardware reasons.
