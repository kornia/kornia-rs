//! Camera models for projection and unprojection.
//!
//! This module provides:
//! - [`PinholeCamera`] — pinhole model with Brown-Conrady distortion.
//! - [`KannalaBrandtCamera`] — Kannala-Brandt equidistant fisheye model.

mod kannala_brandt;
mod pinhole;

pub use kannala_brandt::KannalaBrandtCamera;
pub use pinhole::{PinholeCamera, ProjectionReject};

pub use kannala_brandt::project_point_kb;
pub use pinhole::project_point;
