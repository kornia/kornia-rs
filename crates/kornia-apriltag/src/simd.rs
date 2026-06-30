//! Runtime SIMD feature detection for kernel dispatch.
//!
//! The hot kernels in this crate provide three implementations selected at the
//! call site by a uniform pattern:
//!   - **NEON** on `aarch64` (baseline on ARMv8-A, taken unconditionally),
//!   - **AVX2** on `x86_64` when [`has_avx2`] reports it present at runtime,
//!   - **scalar** fallback otherwise.
//!
//! Compile-time `#[cfg]` gates ensure exactly one architecture's branches are
//! built per target. The probes delegate to `kornia_imgproc::simd::cpu_features`,
//! which caches the `cpuid` result once, so each call is a single field read.

/// Returns `true` if the running CPU supports AVX2 (always `false` off x86_64).
#[inline]
pub fn has_avx2() -> bool {
    kornia_imgproc::simd::cpu_features().has_avx2
}

/// Returns `true` if the running CPU supports both AVX2 and FMA.
///
/// Kernels declared `#[target_feature(enable = "avx2,fma")]` must gate on this.
#[inline]
pub fn has_avx2_fma() -> bool {
    let cpu = kornia_imgproc::simd::cpu_features();
    cpu.has_avx2 && cpu.has_fma
}
