//! Image processing operations.
#![deny(missing_docs)]

/// image undistortion module.
pub mod calibration;

/// color transformations module.
pub mod color;

/// image basic operations module.
pub mod core;
// NOTE: not ready yet
// pub mod distance_transform;

/// image enhancement module.
pub mod enhance;

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
