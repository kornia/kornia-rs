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

/// image padding module.
pub mod padding;

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

/// image morphology module.
pub mod morphology;

/// image flipping module.
pub mod flip;

/// compute image histogram module.
pub mod histogram;

/// utilities for interpolation.
pub mod interpolation;

/// module containing parallelization utilities.
pub mod parallel;

/// image processing metrics module.
pub mod metrics;

/// operations to normalize images.
pub mod normalize;

/// utility functions for resizing images.
pub mod resize;

/// operations to threshold images.
pub mod threshold;

/// SIMD-accelerated image processing backend (proof-of-concept).
///
/// Gated by the default-on `simd` cargo feature. Provides hand-vectorized
/// variants of selected scalar ops (`color::gray_from_rgb_u8`,
/// `color::gray_from_rgb`, `threshold::threshold_binary`) built on the `wide`
/// crate for portable SIMD. See `simd::gray_from_rgb_u8` etc. for entry points.
#[cfg(feature = "simd")]
pub mod simd;

/// image geometric transformations module.
pub mod warp;

/// Pyramid operations
pub mod pyramid;

/// distance transform
pub mod distance_transform;

/// contours
pub mod contours;
