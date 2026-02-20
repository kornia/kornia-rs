//! Morphological operations
//!
//! This module provides morphological operations for image processing.

/// Morphological kernels
pub mod kernels;
pub use kernels::*;

/// Morphological operations
mod ops;
pub use ops::*;
