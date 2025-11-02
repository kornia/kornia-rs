//! Geometric image transformations using affine and perspective warps.
//!
//! This module provides functions for applying 2D transformations to images:
//!
//! - Affine transformations (rotation, translation, scaling, shearing)
//! - Perspective transformations (homographies)
//! - Rotation matrix generation
//! - Affine transform inversion
//!
//! # Examples
//!
//! Rotating an image by 45 degrees:
//!
//! ```no_run
//! use kornia_imgproc::warp::get_rotation_matrix2d;
//!
//! let rotation_matrix = get_rotation_matrix2d([128.0, 128.0], 45.0, 1.0);
//! // Use with warp_affine to rotate the image
//! ```

mod affine;
mod perspective;

pub use affine::{get_rotation_matrix2d, invert_affine_transform, warp_affine};
pub use perspective::warp_perspective;
