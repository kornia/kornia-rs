#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
/// image undistortion module.
pub mod calibration;

/// color transformations module.
pub mod color;

/// image basic operations module.
pub mod core;

/// image cropping module.
pub mod crop;

// NOTE: not ready yet
// pub mod distance_transform;

/// utilities to draw on images.
pub mod draw;

/// image enhancement module.
pub mod enhance;

/// feature detection module.
pub mod features;

/// image filtering module.
pub mod filter;

/// image flipping module.
pub mod flip;

/// compute image histogram module.
pub mod histogram;

/// utilities for interpolation.
pub mod interpolation;

/// module containing parallization utilities.
pub mod parallel;

/// image processing metrics module.
pub mod metrics;

/// operations to normalize images.
pub mod normalize;

/// utility functions for resizing images.
pub mod resize;

/// operations to threshold images.
pub mod threshold;

/// image geometric transformations module.
pub mod warp;

/// Pyramid operations
pub mod pyramid;
