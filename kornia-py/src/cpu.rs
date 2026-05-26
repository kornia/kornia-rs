use pyo3::prelude::*;

use kornia_imgproc::simd::{cpu_features as imgproc_cpu_features, CpuFeatures};

/// Runtime CPU SIMD feature snapshot, as observed by the loaded wheel.
///
/// Mirrors `kornia_imgproc::simd::CpuFeatures`. Cross-arch fields are always
/// `False` — e.g. on aarch64 `has_avx2` is `False`, on x86_64 `has_neon` is
/// `False` — so callers can probe both without arch guards.
#[pyclass(name = "CpuFeatures", module = "kornia_rs.cpu", frozen)]
pub struct PyCpuFeatures(CpuFeatures);

#[pymethods]
impl PyCpuFeatures {
    #[getter]
    fn has_avx2(&self) -> bool {
        self.0.has_avx2
    }
    #[getter]
    fn has_fma(&self) -> bool {
        self.0.has_fma
    }
    #[getter]
    fn has_avx512f(&self) -> bool {
        self.0.has_avx512f
    }
    #[getter]
    fn has_avx512bw(&self) -> bool {
        self.0.has_avx512bw
    }
    #[getter]
    fn has_neon(&self) -> bool {
        self.0.has_neon
    }
    #[getter]
    fn has_fp16(&self) -> bool {
        self.0.has_fp16
    }
    #[getter]
    fn has_dotprod(&self) -> bool {
        self.0.has_dotprod
    }
    #[getter]
    fn has_sve(&self) -> bool {
        self.0.has_sve
    }

    fn __repr__(&self) -> String {
        let f = &self.0;
        format!(
            "CpuFeatures(avx2={}, fma={}, avx512f={}, avx512bw={}, neon={}, fp16={}, dotprod={}, sve={})",
            f.has_avx2, f.has_fma, f.has_avx512f, f.has_avx512bw,
            f.has_neon, f.has_fp16, f.has_dotprod, f.has_sve,
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
    PyCpuFeatures(*imgproc_cpu_features())
}
