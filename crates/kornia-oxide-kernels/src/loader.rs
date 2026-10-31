//! Loader for this crate's PTX (`cuda` feature).
//!
//! This is a kernel source, not an op API: each op keeps its public launcher
//! and fetches its kernel with [`kernel`]. All entries live in one module,
//! loaded once per device; the PTX targets `sm_75`, so the driver JITs it for
//! any Turing-or-newer GPU.
//!
//! The PTX is real only when some crate in the build enables
//! `kornia-oxide-kernels/ptx` (kornia-imgproc's `cuda` does); otherwise
//! [`kernel`] returns [`OxideError::Stub`].

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};
use kornia_tensor::CudaKernel;

use crate::{ENTRIES, IS_STUB, PTX};

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

/// Load `ptx` **once** and return one [`CudaKernel`] per entry in
/// `fn_names`, in order. The driver JITs the PTX for `ctx`'s device, so PTX
/// targeting an older `sm_XY` runs on newer GPUs.
///
/// # Arguments
///
/// * `ctx` - CUDA context to load the module on.
/// * `ptx` - PTX assembly text.
/// * `fn_names` - Names of the `.entry` functions to load.
///
/// # Returns
///
/// One kernel per name, in the order of `fn_names`.
///
/// # Errors
///
/// Returns [`OxideError::Load`] if the driver rejects the PTX (e.g. its
/// `.version` is newer than the driver supports) or an entry is missing.
///
/// # Example
///
/// ```no_run
/// use cudarc::driver::CudaContext;
/// use kornia_oxide_kernels::{load_ptx_many, ENTRIES, PTX};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ctx = CudaContext::new(0)?;
/// let kernels = load_ptx_many(&ctx, PTX, ENTRIES)?;
/// assert_eq!(kernels.len(), ENTRIES.len());
/// # Ok(())
/// # }
/// ```
pub fn load_ptx_many(
    ctx: &Arc<CudaContext>,
    ptx: &str,
    fn_names: &[&str],
) -> Result<Vec<CudaKernel>, OxideError> {
    let load_err = |e: cudarc::driver::DriverError| OxideError::Load(e.to_string());
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
        .map_err(load_err)?;
    fn_names
        .iter()
        .map(|name| {
            module
                .load_function(name)
                .map(CudaKernel::from)
                .map_err(load_err)
        })
        .collect()
}

/// The kernel named `name` (an [`entry`](crate::entry) const) on `stream`'s device, loading
/// the module on first use there. A load failure is cached per device and
/// returned on every call.
///
/// # Arguments
///
/// * `stream` - Stream whose device the kernel is loaded on.
/// * `name` - The entry name, an [`entry`](crate::entry) const.
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
/// use kornia_oxide_kernels::{entry, kernel};
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
            load_ptx_many(ctx, PTX, ENTRIES).map(|kernels| {
                kernels
                    .into_iter()
                    .inspect(|k| {
                        // Same carveout as kornia's NVRTC kernels; best effort.
                        let _ = k.prefer_l1_cache();
                    })
                    .collect()
            })
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

// Real PTX: `cargo test -p kornia-oxide-kernels --features cuda,ptx`
// (scripts/cuda-oxide/orin-test-bench.sh); without `ptx` they check the stub.
#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Arc<CudaStream> {
        CudaContext::new(0).unwrap().default_stream()
    }

    /// `load_ptx_many` loads entries by name, in order, and returns
    /// `OxideError::Load` for PTX the driver rejects and for a missing entry.
    #[test]
    fn load_ptx_many_entries_and_errors() {
        const PTX: &str = ".version 6.3\n.target sm_75\n.address_size 64\n\
                           .visible .entry a()\n{\n\tret;\n}\n\
                           .visible .entry b()\n{\n\tret;\n}\n";
        let ctx = CudaContext::new(0).unwrap();
        let kernels = load_ptx_many(&ctx, PTX, &["b", "a"]).unwrap();
        assert_eq!(kernels.len(), 2);
        assert!(matches!(
            load_ptx_many(&ctx, PTX, &["a", "missing"]),
            Err(OxideError::Load(_))
        ));
        assert!(matches!(
            load_ptx_many(&ctx, "not ptx", &["a"]),
            Err(OxideError::Load(_))
        ));
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
