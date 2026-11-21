//! Morphological operations
//!
//! This module provides morphological operations for image processing.

/// Morphological kernels
pub mod kernels;
pub use kernels::*;

/// Morphological operations
mod ops;
pub use ops::*;

/// Label-boundary erosion
mod label_boundaries;
pub use label_boundaries::{erode_label_boundaries, LabelPixel};

#[cfg(feature = "cuda")]
mod cuda;
