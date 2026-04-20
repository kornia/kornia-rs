//! SIMD dispatch primitives — the single source of truth for "what CPU
//! features are available at runtime" across every hot kernel in the crate.
//!
//! # Why a central probe
//!
//! The alternative — scattering `is_x86_feature_detected!("avx2")` checks
//! inside every kernel — has two problems:
//!
//! 1. **Cost**: the macro touches a process-global atomic on every call.
//!    Cheap, but adds up at millions of dispatch decisions per frame.
//! 2. **Consistency**: if one kernel checks "avx2" and another checks
//!    "avx2,fma" you can end up with mismatched fast paths. Probing once
//!    into a single [`CpuFeatures`] struct keeps every kernel's
//!    feature-selection logic aligned.
//!
//! # Usage pattern
//!
//! ```ignore
//! use crate::simd::{cpu_features, CpuFeatures};
//!
//! pub fn my_op(src: &[u8], dst: &mut [u8]) {
//!     let cpu = cpu_features();
//!
//!     #[cfg(target_arch = "x86_64")]
//!     if cpu.has_avx2 {
//!         return unsafe { x86::my_op_avx2(src, dst) };
//!     }
//!
//!     #[cfg(target_arch = "aarch64")]
//!     if cpu.has_neon {
//!         return unsafe { aarch64::my_op_neon(src, dst) };
//!     }
//!
//!     scalar::my_op(src, dst);
//! }
//! ```
//!
//! The `cfg(target_arch = ...)` gates dead-code-eliminate unreachable
//! branches per build target, so the compiled binary on aarch64 never
//! carries the x86 call site (and vice versa). The runtime `cpu.has_*`
//! checks only discriminate across feature levels *within* an ISA
//! (e.g. AVX2 vs AVX-512, or NEON vs SVE2).
//!
//! # Adding a new kernel
//!
//! 1. Place the dispatch function and scalar reference in
//!    `<op>/kernels.rs` (mirror the layout in `warp/kernels.rs`).
//! 2. Read `cpu_features()` once at dispatch time; branch on
//!    `cpu.has_<feature>` for feature-gated variants.
//! 3. Mark per-ISA kernels `unsafe fn` with `#[target_feature(enable = "...")]`.
//! 4. Test: every backend must produce bit-identical output vs the scalar
//!    reference on a randomized input batch — pin this in a unit test.

use std::sync::OnceLock;

/// Runtime CPU feature snapshot, probed once per process.
///
/// Fields are grouped by ISA; only the fields relevant to the current
/// build's `target_arch` carry runtime signal — cross-arch fields are
/// always `false` and should be gated out of hot paths via `cfg!` or
/// `#[cfg(target_arch = ...)]`.
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    // ---- x86_64 ----
    /// AVX2 (256-bit integer/float lanes, gather, vpshufb, FMA co-req).
    pub has_avx2: bool,
    /// FMA3 (fused-multiply-add; pairs with AVX2 on every CPU that matters).
    pub has_fma: bool,
    /// AVX-512F (baseline AVX-512 — 512-bit lanes, masking).
    pub has_avx512f: bool,
    /// AVX-512BW (byte + word ops on 512-bit lanes — needed for u8/u16 kernels).
    pub has_avx512bw: bool,

    // ---- aarch64 ----
    /// NEON (Advanced SIMD) — baseline on every aarch64 target we support,
    /// but kept as a field so the dispatch pattern stays uniform across ISAs.
    pub has_neon: bool,
    /// FP16 arithmetic (ARMv8.2-A `fp16` feature) — enables `float16x8_t`
    /// throughput for ML-preprocess kernels.
    pub has_fp16: bool,
    /// Dot-product (`udot`, `sdot`) — 4× u8→u32 MAC per lane, big win for
    /// convolution-style kernels.
    pub has_dotprod: bool,
    /// Scalable Vector Extension. Present on Neoverse-V1/V2, Apple M-series
    /// don't have it. SVE lane width is runtime-variable.
    pub has_sve: bool,
}

impl CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    fn probe() -> Self {
        Self {
            has_avx2: std::is_x86_feature_detected!("avx2"),
            has_fma: std::is_x86_feature_detected!("fma"),
            has_avx512f: std::is_x86_feature_detected!("avx512f"),
            has_avx512bw: std::is_x86_feature_detected!("avx512bw"),
            has_neon: false,
            has_fp16: false,
            has_dotprod: false,
            has_sve: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn probe() -> Self {
        Self {
            has_avx2: false,
            has_fma: false,
            has_avx512f: false,
            has_avx512bw: false,
            // NEON is architectural on aarch64 — no runtime flag needed.
            has_neon: true,
            has_fp16: std::arch::is_aarch64_feature_detected!("fp16"),
            has_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            has_sve: std::arch::is_aarch64_feature_detected!("sve"),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn probe() -> Self {
        Self {
            has_avx2: false,
            has_fma: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_neon: false,
            has_fp16: false,
            has_dotprod: false,
            has_sve: false,
        }
    }
}

/// Returns the cached `CpuFeatures` snapshot for this process.
///
/// The first call probes the CPU and stores the result in a `OnceLock`;
/// every subsequent call is a non-atomic load. Safe to call from any
/// thread, including inside rayon workers.
#[inline]
pub fn cpu_features() -> &'static CpuFeatures {
    static CPU: OnceLock<CpuFeatures> = OnceLock::new();
    CPU.get_or_init(CpuFeatures::probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_is_stable_across_calls() {
        let a = *cpu_features();
        let b = *cpu_features();
        assert_eq!(a.has_avx2, b.has_avx2);
        assert_eq!(a.has_neon, b.has_neon);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_is_always_on_for_aarch64() {
        assert!(cpu_features().has_neon);
    }
}
