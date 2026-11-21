//! Loader for this crate's PTX (`cuda` feature).
//!
//! This is a kernel source, not an op API: each op keeps its public launcher
//! and fetches its kernel with [`kernel`]. All entries live in one module,
//! loaded once per device; the PTX targets `sm_75`, so the driver JITs it for
//! any Turing-or-newer GPU and [`kernel`] refuses older ones (callers fall
//! back to NVRTC).
//!
//! The PTX is real only when some crate in the build enables
//! `kornia-oxide-kernels/ptx` (kornia-imgproc's `cuda` does); otherwise
//! [`kernel`] returns [`OxideError::Stub`].
//!
//! Both functions are reachable only through the hidden `__private` path:
//! [`CudaKernel`]'s launch is safe but takes unchecked arguments, so these
//! are for kornia's own launchers, which pass the right ones.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use cudarc::driver::{CudaContext, CudaStream};
use kornia_tensor::CudaKernel;

use crate::{ENTRIES, IS_STUB, PTX};

/// Oldest compute capability the PTX's `.target` runs on.
const MIN_SM: (i32, i32) = (7, 5);

/// One slot per device ordinal: a `CUfunction` belongs to the context that
/// loaded it, so a module cached for device 0 cannot launch on device 1. Only
/// a loaded module is cached: a failed load is retried on the next call.
/// After the first load a lookup is two atomic loads, no lock and no refcount.
#[derive(Default)]
struct Slot {
    module: OnceLock<Box<[CudaKernel]>>,
    lock: Mutex<()>,
}
static SLOTS: OnceLock<Box<[Slot]>> = OnceLock::new();
static SLOTS_LOCK: Mutex<()> = Mutex::new(());

/// `cell`'s value, or else `init`'s, stored only on success so an error (e.g.
/// a transient driver failure) is retried by the next call. `lock` makes
/// concurrent first calls wait for a single `init`.
fn get_or_try_init<'a, T, E>(
    cell: &'a OnceLock<T>,
    lock: &Mutex<()>,
    init: impl FnOnce() -> Result<T, E>,
) -> Result<&'a T, E> {
    if let Some(v) = cell.get() {
        return Ok(v);
    }
    let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(v) = cell.get() {
        return Ok(v);
    }
    let v = init()?;
    Ok(cell.get_or_init(|| v))
}

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
/// use kornia_oxide_kernels::{__private::load_ptx_many, ENTRIES, PTX};
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
/// the module on first use there. Only success is cached: after a failure the
/// next call tries again.
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
/// [`OxideError::Stub`] without real PTX; [`OxideError::Load`] if the device
/// is older than `sm_75`, the driver rejects the module or `name` is not an
/// entry.
///
/// # Example
///
/// ```no_run
/// use cudarc::driver::CudaContext;
/// use kornia_oxide_kernels::{__private::kernel, entry};
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
    let slots = get_or_try_init(&SLOTS, &SLOTS_LOCK, || {
        let n = CudaContext::device_count().map_err(|e| OxideError::Load(e.to_string()))?;
        Ok::<_, OxideError>((0..n.max(0)).map(|_| Slot::default()).collect())
    })?;
    let slot = slots
        .get(ctx.ordinal())
        .ok_or_else(|| OxideError::Load(format!("device {} not enumerated", ctx.ordinal())))?;
    // The load never re-enters `kernel`, so the slot lock cannot deadlock.
    let module = get_or_try_init(&slot.module, &slot.lock, || {
        let sm = ctx
            .compute_capability()
            .map_err(|e| OxideError::Load(e.to_string()))?;
        if sm < MIN_SM {
            return Err(OxideError::Load(format!(
                "sm_{}{} is older than the PTX target sm_{}{}",
                sm.0, sm.1, MIN_SM.0, MIN_SM.1
            )));
        }
        load_ptx_many(ctx, PTX, ENTRIES).map(|kernels| {
            kernels
                .into_iter()
                .inspect(|k| {
                    // Same carveout as kornia's NVRTC kernels; best effort.
                    let _ = k.prefer_l1_cache();
                })
                .collect()
        })
    })?;
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

    /// An error is returned but not stored, so the next call retries; a
    /// success is stored and later calls do not run `init`.
    #[test]
    fn get_or_try_init_caches_only_success() {
        let cell = OnceLock::new();
        let lock = Mutex::new(());
        let calls = std::cell::Cell::new(0);
        let init = |ok: bool| {
            calls.set(calls.get() + 1);
            if ok {
                Ok(7)
            } else {
                Err("transient")
            }
        };
        assert_eq!(
            get_or_try_init(&cell, &lock, || init(false)),
            Err("transient")
        );
        assert!(cell.get().is_none());
        assert_eq!(get_or_try_init(&cell, &lock, || init(true)), Ok(&7));
        assert_eq!(get_or_try_init(&cell, &lock, || init(false)), Ok(&7));
        assert_eq!(calls.get(), 2);
    }

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
