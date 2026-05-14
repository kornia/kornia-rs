//! Runtime CPU feature snapshot shared by every SIMD-dispatching kernel.
//!
//! Probing once into a single struct (vs scattering `is_x86_feature_detected!`
//! per call site) avoids both the per-call atomic and the chance of two
//! kernels disagreeing on which feature set is "available".

use std::sync::OnceLock;

/// Runtime CPU feature snapshot, probed once per process.
///
/// Cross-arch fields are always `false` on a build that targets a different
/// ISA — gate hot paths with `#[cfg(target_arch = ...)]` rather than the
/// runtime flag alone.
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    /// AVX2 (256-bit integer/float lanes).
    pub has_avx2: bool,
    /// FMA3 (fused-multiply-add).
    pub has_fma: bool,
    /// AVX-512F (baseline AVX-512).
    pub has_avx512f: bool,
    /// AVX-512BW (byte + word ops on 512-bit lanes).
    pub has_avx512bw: bool,

    /// NEON (Advanced SIMD) — architectural on aarch64.
    pub has_neon: bool,
    /// FP16 arithmetic (ARMv8.2-A `fp16`).
    pub has_fp16: bool,
    /// Dot-product (`udot`, `sdot`).
    pub has_dotprod: bool,
    /// Scalable Vector Extension.
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
/// The first call probes the CPU; every subsequent call is a `OnceLock` load.
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
