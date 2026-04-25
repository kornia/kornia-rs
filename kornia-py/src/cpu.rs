use pyo3::prelude::*;

use kornia_imgproc::simd::cpu_features as imgproc_cpu_features;

/// Runtime CPU SIMD feature snapshot, as observed by the loaded wheel.
///
/// Mirrors `kornia_imgproc::simd::CpuFeatures`. Cross-arch fields are always
/// `False` — e.g. on aarch64 `has_avx2` is `False`, on x86_64 `has_neon` is
/// `False` — so callers can probe both without arch guards.
///
/// Intended use: confirm at runtime that the wheel installed on this machine
/// will actually dispatch through the SIMD paths the build was supposed to
/// ship. A wheel built with the right intrinsics but loaded on a CPU that
/// doesn't expose them will report `False` and fall back to scalar.
#[pyclass(name = "CpuFeatures", module = "kornia_rs.cpu", frozen)]
pub struct PyCpuFeatures {
    /// AVX2 (256-bit integer/float lanes). Always `False` off x86_64.
    #[pyo3(get)]
    pub has_avx2: bool,
    /// FMA3 (fused-multiply-add). Always `False` off x86_64.
    #[pyo3(get)]
    pub has_fma: bool,
    /// AVX-512F (baseline AVX-512). Always `False` off x86_64.
    #[pyo3(get)]
    pub has_avx512f: bool,
    /// AVX-512BW (byte + word ops on 512-bit lanes). Always `False` off x86_64.
    #[pyo3(get)]
    pub has_avx512bw: bool,
    /// NEON (Advanced SIMD). Architectural on aarch64 — always `True` there
    /// and always `False` elsewhere.
    #[pyo3(get)]
    pub has_neon: bool,
    /// FP16 arithmetic (ARMv8.2-A `fp16`). Always `False` off aarch64.
    #[pyo3(get)]
    pub has_fp16: bool,
    /// Dot-product (`udot`, `sdot`). Always `False` off aarch64.
    #[pyo3(get)]
    pub has_dotprod: bool,
    /// Scalable Vector Extension. Always `False` off aarch64.
    #[pyo3(get)]
    pub has_sve: bool,
}

#[pymethods]
impl PyCpuFeatures {
    fn __repr__(&self) -> String {
        format!(
            "CpuFeatures(avx2={}, fma={}, avx512f={}, avx512bw={}, neon={}, fp16={}, dotprod={}, sve={})",
            self.has_avx2,
            self.has_fma,
            self.has_avx512f,
            self.has_avx512bw,
            self.has_neon,
            self.has_fp16,
            self.has_dotprod,
            self.has_sve,
        )
    }
}

/// Probe and return the runtime CPU SIMD feature snapshot.
///
/// First call probes the CPU; subsequent calls hit a process-wide
/// `OnceLock`-backed cache. Use the returned `CpuFeatures` to verify which
/// SIMD paths the loaded wheel will actually dispatch through.
#[pyfunction]
pub fn cpu_features() -> PyCpuFeatures {
    let f = imgproc_cpu_features();
    PyCpuFeatures {
        has_avx2: f.has_avx2,
        has_fma: f.has_fma,
        has_avx512f: f.has_avx512f,
        has_avx512bw: f.has_avx512bw,
        has_neon: f.has_neon,
        has_fp16: f.has_fp16,
        has_dotprod: f.has_dotprod,
        has_sve: f.has_sve,
    }
}
