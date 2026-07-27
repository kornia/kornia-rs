//! CPU SIFT: NEON scale-space, held to the same bit-exactness contract as the
//! CUDA path.
//!
//! [`params`] is shared between the two backends, so a Gaussian coefficient can
//! never drift between them — a one-ULP difference in a single tap shifts every
//! layer of the pyramid and therefore every keypoint.

// The `pub(crate)` re-exports below exist so `cuda::sift` can name the shared
// numerics rather than re-declare the literals, which is how the layer sigmas
// drifted once before (see `SiftCudaConfig::shared_config`). Nothing else in the
// crate reads them, so they are gated to the build that has that consumer.
mod params;
pub(crate) use params::refl101;
#[cfg(feature = "cuda")]
pub(crate) use params::{gaussian_kernel_f32, gaussian_ksize};
pub use params::{SiftConfig, SiftConfigError};

mod pipeline;
#[cfg(feature = "cuda")]
pub(crate) use pipeline::final_order as sift_final_order;
pub use pipeline::{
    detect_and_compute as sift_detect_and_compute, detect_and_compute_with as sift_detect_with,
    FirstOctave, SiftFeatures, SiftKeypoint, SiftWorkspace,
};

mod descriptor;
pub use descriptor::DESCR_LEN;
#[cfg(feature = "cuda")]
pub(crate) use descriptor::{
    DESCR_HIST_BINS, DESCR_MAG_THR, DESCR_SCL_FCTR, DESCR_WIDTH, INT_DESCR_FCTR,
};

mod matcher;
pub use matcher::{
    l2_sq, l2_sq_scalar, match_descriptors as sift_match_descriptors,
    match_descriptors_scalar as sift_match_descriptors_scalar,
};

mod hal;

mod detect;
pub use detect::{IMG_BORDER as SIFT_IMG_BORDER, MAX_INTERP_STEPS as SIFT_MAX_INTERP_STEPS};

mod orient;
pub use orient::ORI_HIST_BINS;
#[cfg(feature = "cuda")]
pub(crate) use orient::{ORI_PEAK_RATIO, ORI_RADIUS, ORI_SIG_FCTR};

mod scalespace;
