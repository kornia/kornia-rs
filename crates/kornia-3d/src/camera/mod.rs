//! Camera models for projection and unprojection.
//!
//! This module provides:
//! - [`CameraModel`] — the projection interface shared by the central models (pinhole, fisheye).
//! - [`AffineCamera`] — float64 affine world-to-image projection and Jacobians.
//! - [`PinholeCamera`] — pinhole model with Brown-Conrady distortion.
//! - [`FisheyeCamera`] — Kannala-Brandt equidistant fisheye model.
//! - [`Fisheye624Camera`](crate::camera::Fisheye624Camera) — Aria radial-tangential thin-prism fisheye model.
//! - [`Camera`] — one of the central models, chosen at runtime.
//! - [`Ray`](crate::camera::Ray) — an origin and a unit bearing, e.g. a pixel's ray in a rig frame.

mod affine;
mod fisheye;
mod fisheye624;
mod model;
mod pinhole;

pub use affine::AffineCamera;
pub use fisheye::FisheyeCamera;
pub use fisheye624::Fisheye624Camera;
pub use model::{Camera, CameraModel, Ray};
pub use pinhole::{PinholeCamera, ProjectionReject};

pub use fisheye::project_point_fisheye;
pub use pinhole::project_point;
