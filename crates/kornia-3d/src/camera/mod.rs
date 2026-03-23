//! # Camera Models
//!
//! This module provides camera models for projecting and unprojecting points.

pub mod fisheye;
pub mod pinhole;

pub use fisheye::FisheyeCamera;
pub use pinhole::PinholeCamera;

use kornia_algebra::Vec3F64;

use serde::{Deserialize, Serialize};

/// Unified camera model enum supporting multiple projection models.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnyCamera {
    /// Pinhole camera model.
    Pinhole(PinholeCamera),
    /// Fisheye camera model (Kannala-Brandt).
    Fisheye(FisheyeCamera),
}

impl AnyCamera {
    /// Project a 3D point to pixel coordinates.
    pub fn project(&self, point: &Vec3F64) -> Option<[f64; 2]> {
        match self {
            AnyCamera::Pinhole(cam) => cam.project(point),
            AnyCamera::Fisheye(cam) => cam.project(point),
        }
    }

    /// Unproject a pixel to a unit bearing ray in camera frame.
    pub fn unproject(&self, pixel: [f64; 2]) -> Vec3F64 {
        match self {
            AnyCamera::Pinhole(cam) => cam.unproject(pixel),
            AnyCamera::Fisheye(cam) => cam.unproject(pixel),
        }
    }
}
