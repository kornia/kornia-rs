#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Image Processing
//!
//! Comprehensive image processing operations for computer vision applications.
//!
//! ## Key Features
//!
//! - **Color Conversions**: RGB ↔ Grayscale, HSV, YUV
//! - **Geometric Transforms**: Resize, crop, flip, warp, affine transforms
//! - **Filtering**: Gaussian, box blur, median, Sobel, and custom kernels
//! - **Feature Detection**: FAST corners, Harris, gradient-based features
//! - **Enhancement**: Histogram equalization, normalization, sharpening
//! - **Metrics**: MSE, L1, Huber loss for image comparison
//!
//! ## Example: Basic Image Processing
//!
//! ```rust
//! use kornia_image::{Image, ImageSize, CpuAllocator};
//! use kornia_imgproc::color::gray_from_rgb;
//! use kornia_imgproc::filter::gaussian_blur;
//!
//! // Create an RGB image
//! let rgb_img = Image::<u8, 3, _>::from_size_val(
//!     ImageSize { width: 640, height: 480 },
//!     128,
//!     CpuAllocator
//! )?;
//!
//! // Convert to grayscale
//! let gray_img = gray_from_rgb(&rgb_img)?;
//!
//! // Apply Gaussian blur
//! let blurred = gaussian_blur(&gray_img, [5, 5], [1.0, 1.0])?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: Feature Detection
//!
//! ```rust
//! use kornia_image::{Image, CpuAllocator};
//! use kornia_imgproc::features::fast::fast_score;
//!
//! // Detect FAST corners in a grayscale image
//! let img = Image::<u8, 1, _>::from_size_val([480, 640].into(), 0, CpuAllocator)?;
//! let corners = fast_score(&img, 9, 20)?;
//! println!("Detected {} corners", corners.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Camera calibration and lens distortion correction.
///
/// Supports polynomial distortion models for image undistortion.
pub mod calibration;

/// Color space transformations.
///
/// Convert between RGB, grayscale, HSV, YUV, and other color spaces.
pub mod color;

/// Core image operations.
///
/// Basic image manipulation functions and utilities.
pub mod core;

/// Image cropping operations.
///
/// Extract regions of interest from images.
pub mod crop;

// NOTE: not ready yet
// pub mod distance_transform;

/// Drawing utilities for visualization.
///
/// Draw shapes, text, and markers on images.
pub mod draw;

/// Image enhancement operations.
///
/// Histogram equalization, contrast adjustment, and sharpening.
pub mod enhance;

/// Feature detection algorithms.
///
/// FAST, Harris corners, and gradient-based feature detectors.
pub mod features;

/// Image filtering operations.
///
/// Convolution, Gaussian blur, median filter, and edge detection.
pub mod filter;

/// Image flipping operations.
///
/// Horizontal, vertical, and diagonal flips.
pub mod flip;

/// Histogram computation and analysis.
///
/// Calculate image histograms for visualization and equalization.
pub mod histogram;

/// Interpolation utilities.
///
/// Bilinear, nearest neighbor, and other interpolation methods.
pub mod interpolation;

/// Parallelization utilities for multi-threaded processing.
///
/// Utilities to leverage multi-core CPUs for faster image operations.
pub mod parallel;

/// Image quality metrics.
///
/// MSE, L1, Huber, and other similarity/distance metrics.
pub mod metrics;

/// Image normalization operations.
///
/// Rescale pixel values to specific ranges or distributions.
pub mod normalize;

/// Image resizing operations.
///
/// Scale images with various interpolation methods.
pub mod resize;

/// Thresholding operations.
///
/// Binary, adaptive, and Otsu thresholding methods.
pub mod threshold;

/// Geometric transformation operations.
///
/// Affine, perspective, and homography warping.
pub mod warp;

/// Image pyramid operations.
///
/// Gaussian and Laplacian pyramids for multi-scale processing.
pub mod pyramid;
