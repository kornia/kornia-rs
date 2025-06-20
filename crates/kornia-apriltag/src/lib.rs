#![deny(missing_docs)]
//! # Kornia AprilTag

/// Error types for AprilTag detection.
pub mod errors;

/// Utility functions for AprilTag detection.
pub mod utils;

/// Thresholding utilities for AprilTag detection.
pub mod threshold;

/// image iteration utilities module.
pub(crate) mod iter;

/// Segmentation utilities for AprilTag detection.
pub mod segmentation;

/// TODO
pub mod union_find;
