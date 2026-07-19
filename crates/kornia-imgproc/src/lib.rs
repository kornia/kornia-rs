#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
/// One-line residency arm: run `$body(stream)` on the device when the pair
/// is device-resident, else fall through to the CPU path below the call.
/// Defined at crate root (not inside the cfg-gated `cuda` module) so
/// non-CUDA builds can still see the macro — its expansion is what gates.
#[macro_export]
#[doc(hidden)]
macro_rules! __try_device {
    ($src:expr, $dst:expr, $body:expr) => {
        #[cfg(feature = "cuda")]
        if let $crate::cuda::dispatch::Residency::Device(exec) =
            $crate::cuda::dispatch::pair_residency($src, $dst)?
        {
            return exec.run($body);
        }
    };
}
/// One-line residency arm (see [`__try_device`]).
pub(crate) use crate::__try_device as try_device;

/// image undistortion module.
pub mod calibration;

/// contrast-limited adaptive histogram equalization (CLAHE) module.
pub mod clahe;

/// color transformations module.
pub mod color;

/// image basic operations module.
pub mod core;

/// image cropping module.
pub mod crop;

/// image padding module.
pub mod padding;

// NOTE: not ready yet
// pub mod distance_transform;

/// utilities to draw on images.
pub mod draw;

/// image enhancement module.
pub mod enhance;

/// feature detection module.
pub mod features;

/// image filtering module.
pub mod filter;

/// image morphology module.
pub mod morphology;

/// image flipping module.
pub mod flip;

/// compute image histogram module.
pub mod histogram;

/// utilities for interpolation.
pub mod interpolation;

/// module containing parallelization utilities.
pub mod parallel;

/// runtime CPU feature probe shared by SIMD-dispatching kernels.
pub mod simd;

/// image processing metrics module.
pub mod metrics;

/// operations to normalize images.
pub mod normalize;

/// utility functions for resizing images.
pub mod resize;

/// Image → model-input preprocessing (resize + pad + normalize to CHW f32).
///
/// CPU by default; the `cudarc` feature adds a fused GPU kernel path (build with
/// [`preprocess::PreprocessorBuilder::build_cuda`]). See [`preprocess::Preprocessor`].
pub mod preprocess;

/// operations to threshold images.
pub mod threshold;

/// image geometric transformations module.
pub mod warp;

/// Pyramid operations
pub mod pyramid;

/// distance transform
pub mod distance_transform;

/// optical flow module
pub mod optical_flow_pyr_lk;

/// contours
pub mod contours;

/// CUDA-accelerated image processing kernels (native NVRTC path).
///
/// Enabled by the `cuda` feature.
#[cfg(feature = "cuda")]
pub mod cuda;
