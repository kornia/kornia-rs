//! Pixel interpolation methods for image transformations.
//!
//! This module provides various interpolation algorithms used when resampling
//! images during geometric transformations like resizing, warping, or remapping.
//!
//! # Interpolation Modes
//!
//! - **Nearest**: Fastest, uses nearest pixel value (no interpolation)
//! - **Bilinear**: Smooth linear interpolation between adjacent pixels
//!
//! # Common Use Cases
//!
//! - Image resizing with `crate::resize`
//! - Geometric warping with `crate::warp`
//! - Custom remapping operations

mod bilinear;

/// Grid generation and coordinate mapping utilities.
///
/// Functions for generating coordinate meshgrids used in image warping
/// and transformation operations.
pub mod grid;

pub(crate) mod interpolate;
mod nearest;
mod remap;

pub use interpolate::InterpolationMode;
pub use remap::remap;

pub use interpolate::interpolate_pixel;
