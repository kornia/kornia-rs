//! CUDA kernels authored in Rust and compiled to PTX by
//! [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
//!
//! Enabled by the `cuda-oxide` feature. The kernel source lives in the
//! standalone nightly crate `kernels-oxide/`; `build.rs` compiles it to PTX in
//! `OUT_DIR` and gates it (sm_75 target, no `fma`, no generic loads/stores).
//! kornia itself stays on stable Rust; see `docs/cuda-oxide.md`.
//!
//! The PTX targets `sm_75`, so the driver JITs it for any Turing-or-newer GPU.
//! If the module fails to load, each op's default launcher falls back to its
//! NVRTC kernel; [`ensure_loaded`] reports which path is live.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::color::{launch_map, CudaColorError, KernelSuiteCell, PxPerThread};

/// The single cuda-oxide PTX module, holding every entry in [`ENTRIES`].
pub(crate) static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// `.entry` names in [`PTX`]; indices below address this list.
pub(crate) const ENTRIES: &[&str] = &["gray_from_rgb_u8"];
pub(crate) const GRAY_FROM_RGB_U8: usize = 0;

static MODULE: KernelSuiteCell = KernelSuiteCell::new();

/// The oxide kernel at `index`, loading the module on first use. A load
/// failure is cached like an NVRTC compile failure.
pub(crate) fn kernel(
    stream: &Arc<CudaStream>,
    index: usize,
) -> Result<&'static CudaKernel, CudaColorError> {
    let suite = MODULE
        .get_or_init(|| {
            CudaKernel::load_ptx_many(stream.context(), PTX, ENTRIES)
                .inspect(|ks| {
                    // Same carveout as the NVRTC kernels, so an A/B differs only in codegen.
                    for k in ks {
                        let _ = k.prefer_l1_cache();
                    }
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| CudaColorError::Cuda(format!("cuda-oxide module load: {e}")))?;
    suite
        .get(index)
        .ok_or_else(|| CudaColorError::Cuda(format!("cuda-oxide entry {index} out of range")))
}

/// Load the cuda-oxide module on `stream`'s device.
///
/// The default launchers fall back to NVRTC silently when this fails, so call
/// it to find out which kernels are actually running.
///
/// # Arguments
///
/// * `stream` - Stream whose device the module is loaded on.
///
/// # Returns
///
/// `Ok(())` if the oxide kernels are in use.
///
/// # Errors
///
/// Returns [`CudaColorError::Cuda`] with the driver's message if the PTX does
/// not load (e.g. a pre-Turing GPU or a driver older than PTX ISA 6.3).
pub fn ensure_loaded(stream: &Arc<CudaStream>) -> Result<(), CudaColorError> {
    kernel(stream, GRAY_FROM_RGB_U8).map(|_| ())
}

/// RGB8 -> Gray8 through the cuda-oxide kernel only, with no NVRTC fallback.
///
/// Same contract and launch geometry as
/// [`launch_gray_from_rgb_u8`](super::color::gray::launch_gray_from_rgb_u8);
/// exists for parity tests and A/B benchmarks.
///
/// # Arguments
///
/// * `stream` - Stream to launch on.
/// * `src` - Interleaved RGB bytes, at least `3 * npixels` long.
/// * `dst` - Gray output, at least `npixels` long.
/// * `npixels` - Pixel count.
///
/// # Returns
///
/// `Ok(())` once the kernel is enqueued.
///
/// # Errors
///
/// Returns [`CudaColorError::SliceTooSmall`] if a buffer is short, or
/// [`CudaColorError::Cuda`] if the module fails to load or the launch fails.
pub fn launch_gray_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let k = kernel(stream, GRAY_FROM_RGB_U8)?;
    launch_map(k, stream, src, dst, npixels, 3, 1, PxPerThread::Four)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupt_ptx_errors_instead_of_panicking() {
        // The NVRTC fallback relies on a bad module surfacing as an Err.
        let stream = crate::cuda::color::test_utils::default_stream();
        let bad = PTX.replace(".target sm_75", ".target sm_999");
        assert!(CudaKernel::load_ptx_many(stream.context(), &bad, ENTRIES).is_err());
    }
}
