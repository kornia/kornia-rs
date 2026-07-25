//! CPU SIFT: NEON scale-space, held to the same bit-exactness contract as the
//! CUDA path.
//!
//! [`params`] is shared between the two backends, so a Gaussian coefficient can
//! never drift between them — a one-ULP difference in a single tap shifts every
//! layer of the pyramid and therefore every keypoint.

mod params;
pub(crate) use params::refl101;
pub use params::{gaussian_kernel_f32, gaussian_ksize, SiftConfig};

mod scalespace;
pub use scalespace::{blur_h_f32, blur_v_f32};
