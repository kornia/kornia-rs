//! Geometric image transformation operations for warping and alignment.
//!
//! This module provides functions for applying geometric transformations to images,
//! essential for tasks like image registration, alignment, augmentation, and
//! perspective correction.
//!
//! # Supported Transformations
//!
//! * **Affine Transformations** ([`warp_affine`]) - Rotation, translation, scaling, shearing
//! * **Perspective Transformations** ([`warp_perspective`]) - Full projective transforms
//!
//! # Transformation Matrices
//!
//! ## Affine (2×3 matrix)
//!
//! ```text
//! [a b tx]     [x]     [x']
//! [c d ty]  ×  [y]  =  [y']
//!              [1]
//! ```
//!
//! Preserves parallel lines. Suitable for rotation, scaling, translation, shearing.
//!
//! ## Perspective (3×3 matrix)
//!
//! ```text
//! [h11 h12 h13]     [x]     [wx']
//! [h21 h22 h23]  ×  [y]  =  [wy']
//! [h31 h32 h33]     [1]     [w]
//! ```
//!
//! Full projective mapping. Allows perspective distortion (e.g., correcting camera tilt).
//!
//! # Example: Rotate Image
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::warp::{get_rotation_matrix2d, warp_affine};
//! use kornia_imgproc::interpolation::InterpolationMode;
//!
//! let src = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 100, height: 100 },
//!     0.5,
//! ).unwrap();
//!
//! let mut dst = Image::<f32, 3>::from_size_val(src.size(), 0.0).unwrap();
//!
//! // Rotate 45 degrees around center
//! let center = (50.0, 50.0);
//! let rotation_matrix = get_rotation_matrix2d(center, 45.0, 1.0);
//!
//! warp_affine(&src, &mut dst, &rotation_matrix, InterpolationMode::Bilinear).unwrap();
//! ```
//!
//! # See also
//!
//! * [`crate::interpolation`] for pixel interpolation methods
//! * Use these transforms for image augmentation in machine learning pipelines

mod affine;
mod perspective;

pub use affine::{get_rotation_matrix2d, invert_affine_transform, warp_affine};
pub use perspective::warp_perspective;
