//! Loader for the CUDA kernels written in Rust in the `kornia-oxide-kernels`
//! crate and compiled to PTX by its `build.rs` with
//! [cuda-oxide](https://github.com/NVIDIA/cuda-rust); see `docs/cuda-oxide.md`.
//!
//! This is a kernel source, not an API: each op keeps its public launcher and
//! fetches its kernel with [`kernel`]. All entries live in one module, loaded
//! once per device; the PTX targets `sm_75`, so the driver JITs it for any
//! Turing-or-newer GPU.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};
use kornia_tensor::CudaKernel;

pub(crate) use kornia_oxide_kernels::entry;
use kornia_oxide_kernels::{ENTRIES, IS_STUB, PTX};

use super::color::CudaColorError;

/// One slot per device ordinal: a `CUfunction` belongs to the context that
/// loaded it, so a module cached for device 0 cannot launch on device 1. After
/// the first load a lookup is two atomic loads, no lock and no refcount.
type Module = Result<Box<[CudaKernel]>, OxideError>;
static MODULES: OnceLock<Box<[OnceLock<Module>]>> = OnceLock::new();

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
pub(crate) fn kernel(
    stream: &Arc<CudaStream>,
    name: &str,
) -> Result<&'static CudaKernel, OxideError> {
    if IS_STUB {
        return Err(OxideError::Stub);
    }
    let ctx = stream.context();
    // A live stream means the driver is initialized, so the count is known.
    let slots = MODULES.get_or_init(|| {
        let n = CudaContext::device_count().unwrap_or(0).max(0) as usize;
        (0..n).map(|_| OnceLock::new()).collect()
    });
    let slot = slots
        .get(ctx.ordinal())
        .ok_or_else(|| OxideError::Load(format!("device {} not enumerated", ctx.ordinal())))?;
    // Concurrent first calls on one device wait for a single load; the load
    // never re-enters `kernel`, so this cannot deadlock.
    let module = slot
        .get_or_init(|| {
            CudaKernel::load_ptx_many(ctx, PTX, ENTRIES)
                .map(|kernels| {
                    kernels
                        .into_iter()
                        .inspect(|k| {
                            // Same carveout as kornia's NVRTC kernels; best effort.
                            let _ = k.prefer_l1_cache();
                        })
                        .collect()
                })
                .map_err(|e| OxideError::Load(e.to_string()))
        })
        .as_ref()
        .map_err(Clone::clone)?;
    // ENTRIES is sorted by build.rs.
    ENTRIES
        .binary_search(&name)
        .ok()
        .and_then(|i| module.get(i))
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

    #[test]
    fn concurrent_calls_share_one_module() {
        let stream = crate::cuda::color::test_utils::default_stream();
        let name = ENTRIES[0];
        let ptrs: Vec<usize> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(|| kernel(&stream, name).map(|k| k as *const CudaKernel as usize)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap().unwrap())
                .collect()
        });
        assert!(ptrs.windows(2).all(|w| w[0] == w[1]), "{ptrs:?}");
    }
}
