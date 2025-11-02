//! Image filtering and convolution operations.
//!
//! This module provides various filtering operations commonly used in image processing:
//!
//! - **Gaussian blur**: Smoothing and noise reduction
//! - **Sobel filters**: Edge detection
//! - **Box filters**: Fast averaging
//! - **Custom kernels**: Apply arbitrary convolution kernels
//! - **Separable filters**: Efficient 2D filtering via 1D separable kernels
//!
//! # Performance
//!
//! Separable filters (like Gaussian) are optimized by decomposing 2D convolutions
//! into sequential 1D operations, significantly improving performance.
//!
//! # Examples
//!
//! Applying Gaussian blur:
//!
//! ```no_run
//! use kornia_image::{Image, ImageSize};
//! use kornia_image::allocator::CpuAllocator;
//! use kornia_imgproc::filter::gaussian_blur2d;
//!
//! let image = Image::<f32, 3, _>::from_size_val(
//!     ImageSize { width: 256, height: 256 },
//!     0.5,
//!     CpuAllocator
//! ).unwrap();
//! let mut blurred = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
//! let kernel_size = (5, 5);
//! let sigma = (1.5, 1.5);
//! gaussian_blur2d(&image, &mut blurred, kernel_size, sigma).unwrap();
//! ```

/// Filter kernels and kernel generation functions.
pub mod kernels;

/// Core filter operations including convolution and common filters.
mod ops;
pub use ops::*;

/// Separable filter implementations for efficient 2D filtering.
mod separable_filter;
pub use separable_filter::*;
