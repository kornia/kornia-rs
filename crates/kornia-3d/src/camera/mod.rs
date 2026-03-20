//! Camera models for projection and unprojection.
//!
//! This module provides:
//! - [`PinholeCamera`] — pinhole model with Brown-Conrady distortion.
//! - [`FisheyeCamera`] — Kannala-Brandt equidistant fisheye model.

mod fisheye;
mod pinhole;
mod traits;

pub use fisheye::FisheyeCamera;
pub use pinhole::PinholeCamera;
pub use traits::{project_point, CameraModel, ProjectionReject};
