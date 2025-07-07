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

/// Union-find utilities for AprilTag detection.
pub mod union_find;

/// AprilTag family definitions and utilities.
pub mod family;

/// Quad detection utilities for AprilTag detection.
pub mod quad;

/// Decoding utilities for AprilTag detection.
pub mod decoder;
