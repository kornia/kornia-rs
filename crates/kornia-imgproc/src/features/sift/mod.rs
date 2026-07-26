//! CPU SIFT: NEON scale-space, held to the same bit-exactness contract as the
//! CUDA path.
//!
//! [`params`] is shared between the two backends, so a Gaussian coefficient can
//! never drift between them — a one-ULP difference in a single tap shifts every
//! layer of the pyramid and therefore every keypoint.

mod params;
pub(crate) use params::refl101;
pub use params::{gaussian_kernel_f32, gaussian_ksize, SiftConfig};

mod pipeline;
pub use pipeline::{
    detect_and_compute as sift_detect_and_compute, detect_and_compute_with as sift_detect_with,
    FirstOctave, SiftFeatures, SiftKeypoint, SiftWorkspace,
};

mod descriptor;
pub use descriptor::{
    compute_descriptor, compute_descriptor_fast, descriptor_inputs, DESCR_LEN, FAST_SAMP,
};

mod matcher;
pub use matcher::{
    l2_sq, l2_sq_scalar, match_descriptors as sift_match_descriptors,
    match_descriptors_scalar as sift_match_descriptors_scalar,
};

mod hal;
pub use hal::{atan2_deg, exp as sift_exp, magnitude};

mod detect;
pub use detect::{
    extrema_threshold, find_extrema, RawKeypoint, IMG_BORDER as SIFT_IMG_BORDER,
    MAX_INTERP_STEPS as SIFT_MAX_INTERP_STEPS,
};

mod orient;
pub use orient::{assign_orientations, OrientedKeypoint, ORI_HIST_BINS};

mod scalespace;
pub use scalespace::{blur_h_f32, blur_h_f32_mode, blur_v_f32};
