//! CPU SIFT: NEON scale-space, held to the same bit-exactness contract as the
//! CUDA path.
//!
//! [`params`] is shared between the two backends, so a Gaussian coefficient can
//! never drift between them — a one-ULP difference in a single tap shifts every
//! layer of the pyramid and therefore every keypoint.

mod params;
pub(crate) use params::refl101;
pub use params::{gaussian_kernel_f32, gaussian_ksize, SiftConfig};

mod descriptor;
pub use descriptor::{compute_descriptor, descriptor_inputs, DESCR_LEN};

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
pub use scalespace::{blur_h_f32, blur_v_f32};
