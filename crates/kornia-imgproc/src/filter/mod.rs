//! Filter operations
//!
//! This module provides filter operations for image processing.

/// Filter kernels
pub mod kernels;

#[cfg(feature = "cuda")]
mod cuda;

/// Filter operations
mod ops;
pub use ops::*;

/// Separable filter operations
mod separable_filter;
pub use separable_filter::*;

/// Bilateral filter (cv2-byte-exact).
pub mod bilateral;
pub use bilateral::bilateral_filter;

/// Median blur (cv2/VPI-byte-exact).
pub mod median;
pub use median::median_blur;
