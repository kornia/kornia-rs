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
pub(crate) mod params;
pub(crate) use params::refl101;
pub use params::{SiftConfig, SiftConfigError};

pub(crate) mod pipeline;
pub use pipeline::{
    detect_and_compute as sift_detect_and_compute, FirstOctave, SiftFeatures, SiftKeypoint,
    SiftWorkspace,
};

pub(crate) mod descriptor;
pub use descriptor::DESCR_LEN;

mod matcher;
pub use matcher::{
    l2_sq, l2_sq_scalar, match_descriptors as sift_match_descriptors,
    match_descriptors_scalar as sift_match_descriptors_scalar,
};

mod hal;

mod detect;
pub use detect::{IMG_BORDER as SIFT_IMG_BORDER, MAX_INTERP_STEPS as SIFT_MAX_INTERP_STEPS};

pub(crate) mod orient;
pub use orient::ORI_HIST_BINS;

mod scalespace;
