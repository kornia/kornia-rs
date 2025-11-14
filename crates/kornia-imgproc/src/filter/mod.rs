//! Image filtering operations for smoothing, edge detection, and feature extraction.
//!
//! This module provides efficient implementations of common image filters used in
//! computer vision and image processing pipelines. All filters support multi-channel
//! images and are optimized with separable filtering where applicable.
//!
//! # Available Filters
//!
//! * **Box Blur** ([`box_blur`]) - Fast averaging filter, useful for quick smoothing
//! * **Gaussian Blur** ([`gaussian_blur`]) - Smooth blurring with adjustable sigma
//! * **Sobel Filter** ([`sobel`]) - Edge detection and gradient computation
//!
//! # Separable Filtering
//!
//! Many 2D filters can be decomposed into two 1D operations (horizontal and vertical),
//! significantly reducing computational complexity from O(k²) to O(k) per pixel.
//! This module automatically uses separable filtering when available.
//!
//! # Example: Gaussian Blur
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::filter::gaussian_blur;
//!
//! let src = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 640, height: 480 },
//!     0.5,
//! ).unwrap();
//!
//! let mut dst = Image::<f32, 3>::from_size_val(src.size(), 0.0).unwrap();
//!
//! // Apply Gaussian blur with automatic kernel size
//! gaussian_blur(&src, &mut dst, (0, 0), (1.5, 1.5)).unwrap();
//! ```
//!
//! # Performance Tips
//!
//! * Use box blur for quick, approximate smoothing
//! * Gaussian blur is more expensive but produces higher quality results
//! * Larger kernel sizes increase computational cost linearly
//! * Separable filters are automatically parallelized across image rows

/// Filter kernels
pub mod kernels;

/// Filter operations
mod ops;
pub use ops::*;

/// Separable filter operations
mod separable_filter;
pub use separable_filter::*;
