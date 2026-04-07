//! Camera models for projection and unprojection.
//!
//! This module provides:
//! - [`PinholeCamera`] — pinhole model with Brown-Conrady distortion.
//! - [`FisheyeCamera`] — Kannala-Brandt equidistant fisheye model.

mod fisheye;
mod pinhole;

pub use fisheye::FisheyeCamera;
pub use pinhole::{PinholeCamera, ProjectionReject};

pub use fisheye::project_point_fisheye;
pub use pinhole::project_point;
