//! Camera models for projection and unprojection.
//!
//! This module provides:
//! - [`CameraModel`] — the projection interface shared by all models.
//! - [`PinholeCamera`] — pinhole model with Brown-Conrady distortion.
//! - [`FisheyeCamera`] — Kannala-Brandt equidistant fisheye model.
//! - [`Camera`] — either model, chosen at runtime.

mod fisheye;
mod model;
mod pinhole;

pub use fisheye::FisheyeCamera;
pub use model::{Camera, CameraModel};
pub use pinhole::{PinholeCamera, ProjectionReject};

pub use fisheye::project_point_fisheye;
pub use pinhole::project_point;
