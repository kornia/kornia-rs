//! Loader for the CUDA kernels written in Rust under `kernels-oxide/` and
//! compiled to PTX by `build.rs` with
//! [cuda-oxide](https://github.com/NVIDIA/cuda-rust); see `docs/cuda-oxide.md`.
//!
//! This is a kernel source, not an API: each op keeps its public launcher and
//! fetches its kernel with [`kernel`]. All entries live in one module, loaded
//! once; the PTX targets `sm_75`, so the driver JITs it for any Turing-or-newer
//! GPU.

use std::sync::{Arc, OnceLock};

use cudarc::driver::CudaStream;
use kornia_tensor::CudaKernel;

use super::color::CudaColorError;

// `ENTRIES` and `entry::<NAME>` index consts, generated from kernels-oxide/src.
include!(concat!(env!("OUT_DIR"), "/oxide_entries.rs"));

/// The PTX module holding every entry in [`ENTRIES`].
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

static MODULE: OnceLock<Result<Vec<CudaKernel>, OxideError>> = OnceLock::new();

/// Failure to get a cuda-oxide kernel.
#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum OxideError {
    /// The driver rejected the PTX module, or an entry is missing from it.
    #[error("cuda-oxide module load: {0}")]
    Load(String),
    /// Built with the stub PTX (docs.rs or `KORNIA_OXIDE_STUB=1`).
    #[error("cuda-oxide: built with stub PTX; rebuild with the cuda-oxide toolchain")]
    Stub,
}

impl From<OxideError> for CudaColorError {
    fn from(e: OxideError) -> Self {
        CudaColorError::Cuda(e.to_string())
    }
}

/// The kernel at `index` (an `entry::*` const), loading the module on first
/// use. A load failure is cached and returned on every call.
pub(crate) fn kernel(
    stream: &Arc<CudaStream>,
    index: usize,
) -> Result<&'static CudaKernel, OxideError> {
    if PTX.is_empty() {
        return Err(OxideError::Stub);
    }
    let suite = MODULE
        .get_or_init(|| {
            let kernels = CudaKernel::load_ptx_many(stream.context(), PTX, ENTRIES)
                .map_err(|e| OxideError::Load(e.to_string()))?;
            for k in &kernels {
                // Same carveout as kornia's NVRTC kernels; best effort.
                let _ = k.prefer_l1_cache();
            }
            Ok(kernels)
        })
        .as_ref()
        .map_err(Clone::clone)?;
    suite
        .get(index)
        .ok_or_else(|| OxideError::Load(format!("no entry at index {index}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_entries_load() {
        let stream = crate::cuda::color::test_utils::default_stream();
        assert!(!ENTRIES.is_empty());
        for (i, name) in ENTRIES.iter().enumerate() {
            kernel(&stream, i).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
