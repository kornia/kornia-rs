//! Loader for the CUDA kernels written in Rust in the `kornia-oxide-kernels`
//! crate and compiled to PTX by its `build.rs` with
//! [cuda-oxide](https://github.com/NVIDIA/cuda-rust); see `docs/cuda-oxide.md`.
//!
//! This is a kernel source, not an API: each op keeps its public launcher and
//! fetches its kernel with [`kernel`]. All entries live in one module, loaded
//! once per device; the PTX targets `sm_75`, so the driver JITs it for any
//! Turing-or-newer GPU.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use cudarc::driver::CudaStream;
use kornia_tensor::CudaKernel;

pub(crate) use kornia_oxide_kernels::entry;
use kornia_oxide_kernels::{ENTRIES, IS_STUB, PTX};

use super::color::CudaColorError;

/// One loaded module per device ordinal: a `CUfunction` belongs to the context
/// that loaded it, so a module cached for device 0 cannot launch on device 1.
type Module = Result<Arc<[Arc<CudaKernel>]>, OxideError>;
static MODULES: OnceLock<Mutex<HashMap<usize, Module>>> = OnceLock::new();

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
        CudaColorError::Compile(e.to_string())
    }
}

/// The kernel named `name` (an `entry::*` const) on `stream`'s device, loading
/// the module on first use there. A load failure is cached per device and
/// returned on every call.
pub(crate) fn kernel(stream: &Arc<CudaStream>, name: &str) -> Result<Arc<CudaKernel>, OxideError> {
    if IS_STUB {
        return Err(OxideError::Stub);
    }
    let ctx = stream.context();
    let modules = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
    // Clone out and drop the guard before loading: holding a kernel-cache lock
    // across the miss path deadlocked SIFT (`sift/kernels.rs`).
    let cached = modules
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&ctx.ordinal())
        .cloned();
    let module = match cached {
        Some(m) => m,
        None => {
            let loaded: Module = CudaKernel::load_ptx_many(ctx, PTX, ENTRIES)
                .map(|kernels| {
                    kernels
                        .into_iter()
                        .map(|k| {
                            // Same carveout as kornia's NVRTC kernels; best effort.
                            let _ = k.prefer_l1_cache();
                            Arc::new(k)
                        })
                        .collect()
                })
                .map_err(|e| OxideError::Load(e.to_string()));
            modules
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(ctx.ordinal())
                .or_insert(loaded)
                .clone()
        }
    }?;
    ENTRIES
        .iter()
        .position(|e| *e == name)
        .and_then(|i| module.get(i).cloned())
        .ok_or_else(|| OxideError::Load(format!("no entry `{name}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_entries_load() {
        let stream = crate::cuda::color::test_utils::default_stream();
        assert!(!ENTRIES.is_empty());
        for name in ENTRIES {
            kernel(&stream, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
