//! Local copy of the kornia-imgproc CPU-feature probe.
//!
//! Duplicated rather than depending on `kornia-imgproc` to keep the dependency
//! tree of `kornia-xfeat` small. The struct is tiny and only probed once per
//! process via `OnceLock`.

use std::sync::OnceLock;

/// Runtime CPU feature snapshot, probed once per process.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // fields read by SIMD dispatch arms gated by target_arch
pub(crate) struct CpuFeatures {
    /// x86-64 AVX2 (256-bit integer + float SIMD).
    pub has_avx2: bool,
    /// x86-64 fused multiply-add.
    pub has_fma: bool,
    /// x86-64 AVX-512 foundation.
    pub has_avx512f: bool,
    /// aarch64 Advanced SIMD (NEON) — always true on aarch64.
    pub has_neon: bool,
    /// aarch64 dot-product instructions (`udot` / `sdot`).
    pub has_dotprod: bool,
    /// aarch64 FP16 arithmetic.
    pub has_fp16: bool,
}

impl CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    fn probe() -> Self {
        Self {
            has_avx2: std::is_x86_feature_detected!("avx2"),
            has_fma: std::is_x86_feature_detected!("fma"),
            has_avx512f: std::is_x86_feature_detected!("avx512f"),
            has_neon: false,
            has_dotprod: false,
            has_fp16: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn probe() -> Self {
        Self {
            has_avx2: false,
            has_fma: false,
            has_avx512f: false,
            has_neon: true,
            has_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            has_fp16: std::arch::is_aarch64_feature_detected!("fp16"),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn probe() -> Self {
        Self {
            has_avx2: false,
            has_fma: false,
            has_avx512f: false,
            has_neon: false,
            has_dotprod: false,
            has_fp16: false,
        }
    }
}

#[inline]
pub(crate) fn cpu_features() -> &'static CpuFeatures {
    static CPU: OnceLock<CpuFeatures> = OnceLock::new();
    CPU.get_or_init(CpuFeatures::probe)
}
