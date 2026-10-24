---
name: kornia-simd
description: Use when adding or optimizing SIMD (NEON/AVX2) or performance-critical kernels in kornia-imgproc — module layout, dispatch, cross-arch testing, benchmarking and correctness-vs-scalar checks. Trigger on "SIMD", "NEON", "AVX2", "benchmark", "speed up".
---

# SIMD / performance in kornia-imgproc

The authoritative guide is `crates/kornia-imgproc/SIMD.md`; read it first.

## Layout

- Multi-kernel op: subdirectory with `mod.rs` (public Image API + sealed-trait
  dispatch) and `kernels.rs` (slice-level NEON/AVX2/scalar). Canonical example:
  `crates/kornia-imgproc/src/color/gray/`.
- Single-dtype op: a flat `.rs` file is fine.
- Shared helpers live in `crates/kornia-imgproc/src/simd/` and `parallel.rs`.

## Rules

- Always keep a scalar fallback; SIMD output must match scalar (assert in tests).
- Every `unsafe` intrinsic block has `// SAFETY:`; gate with
  `#[cfg(target_arch = ...)]`; runtime dispatch reads
  `crate::simd::cpu_features()`, never `is_x86_feature_detected!` directly
  (it is compile-time-sensitive under `-Ctarget-cpu`, see SIMD.md).
- Test cross-arch with qemu/binfmt as described in SIMD.md or
  `pixi run rust-cross-test-aarch64`.
- Benchmark before and after (`cargo bench`, best-of-N rounds). Report numbers
  in the PR. iai-callgrind runner version must exactly match the
  `iai-callgrind` dev-dependency version.
- Avoid allocations in hot paths; mind rayon row chunking.
