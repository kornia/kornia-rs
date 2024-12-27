//! Filter operations
//!
//! This module provides filter operations for image processing.
//!
//! # Example
//!
//! ```
//! use kornia_rs::imgproc::filter::sobel;
//!
//! let image = Image::<f32, 3>::from_size_val([12, 12].into(), 0.0)?;
//! let mut dst = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
//! sobel(&image, &mut dst, 3)?;
//! ```

/// Filter kernels
pub mod kernels;

/// Filter operations
mod ops;
pub use ops::*;

/// Separable filter operations
mod separable_filter;
pub use separable_filter::*;
