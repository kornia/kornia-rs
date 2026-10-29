//! Loader for the CUDA kernels written in Rust in the `kornia-oxide-kernels`
//! crate, which ships them as PTX compiled by
//! [cuda-oxide](https://github.com/NVIDIA/cuda-rust); see `docs/cuda-oxide.md`.
//!
//! This is a kernel source, not an op API: each op keeps its public launcher
//! and fetches its kernel with [`kernel`]. All entries live in one module,
//! loaded once per device; the PTX targets `sm_75`, so the driver JITs it for
//! any Turing-or-newer GPU. It lives here, not in a consumer crate, so every
//! crate with kernels (this one included) reaches it without a cycle.
//!
//! The PTX is real only when some crate in the build enables
//! `kornia-oxide-kernels/ptx` (kornia-imgproc's `cuda` does); otherwise
//! [`kernel`] returns [`OxideError::Stub`].

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};
pub use kornia_oxide_kernels::entry;
use kornia_oxide_kernels::{ENTRIES, IS_STUB, PTX};

use super::CudaKernel;

/// One slot per device ordinal: a `CUfunction` belongs to the context that
/// loaded it, so a module cached for device 0 cannot launch on device 1. After
/// the first load a lookup is two atomic loads, no lock and no refcount.
type Module = Result<Box<[CudaKernel]>, OxideError>;
static MODULES: OnceLock<Result<Box<[OnceLock<Module>]>, OxideError>> = OnceLock::new();

/// Failure to get a cuda-oxide kernel.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OxideError {
    /// The driver rejected the PTX module, or an entry is missing from it.
    #[error("cuda-oxide module load: {0}")]
    Load(String),
    /// Built with the stub PTX: no crate in the build enabled
    /// `kornia-oxide-kernels/ptx`.
    #[error("cuda-oxide: built with stub PTX; enable kornia-oxide-kernels/ptx (kornia-imgproc's `cuda` does)")]
    Stub,
}

/// The kernel named `name` (an [`entry`] const) on `stream`'s device, loading
/// the module on first use there. A load failure is cached per device and
/// returned on every call.
///
/// # Arguments
///
/// * `stream` - Stream whose device the kernel is loaded on.
/// * `name` - The entry name, an [`entry`] const.
///
/// # Returns
///
/// The kernel, valid for the life of the process.
///
/// # Errors
///
/// [`OxideError::Stub`] without real PTX; [`OxideError::Load`] if the driver
/// rejects the module or `name` is not an entry.
///
/// # Example
///
/// ```no_run
/// use cudarc::driver::CudaContext;
/// use kornia_tensor::cuda::oxide::{entry, kernel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let stream = CudaContext::new(0)?.default_stream();
/// let gray = kernel(&stream, entry::GRAY_FROM_RGB_U8)?;
/// # let _ = gray;
/// # Ok(())
/// # }
/// ```
pub fn kernel(stream: &Arc<CudaStream>, name: &str) -> Result<&'static CudaKernel, OxideError> {
    if IS_STUB {
        return Err(OxideError::Stub);
    }
    let ctx = stream.context();
    // A live stream means the driver is initialized, so the count is known.
    let slots = MODULES
        .get_or_init(|| {
            let n = CudaContext::device_count().map_err(|e| OxideError::Load(e.to_string()))?;
            Ok((0..n.max(0)).map(|_| OnceLock::new()).collect())
        })
        .as_ref()
        .map_err(Clone::clone)?;
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

// Real PTX: `cargo test -p kornia-tensor --features cuda,kornia-oxide-kernels/ptx`
// (scripts/cuda-oxide/orin-test-bench.sh); without `ptx` they check the stub.
#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Arc<CudaStream> {
        CudaContext::new(0).unwrap().default_stream()
    }

    #[test]
    fn all_entries_load() {
        let stream = stream();
        assert!(!ENTRIES.is_empty());
        for name in ENTRIES {
            match kernel(&stream, name) {
                Err(OxideError::Stub) if IS_STUB => {}
                r => {
                    r.unwrap_or_else(|e| panic!("{name}: {e}"));
                }
            }
        }
        if !IS_STUB {
            assert!(matches!(
                kernel(&stream, "no_such_entry"),
                Err(OxideError::Load(_))
            ));
        }
    }

    #[test]
    fn concurrent_calls_share_one_module() {
        if IS_STUB {
            return;
        }
        let stream = stream();
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
