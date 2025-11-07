#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Camera calibration and lens distortion correction.
///
/// Provides undistortion models and calibration utilities for removing lens distortion
/// from camera images. See [`calibration`] module for distortion models.
pub mod calibration;

/// Color space conversions and transformations.
///
/// Convert between color spaces (RGB, grayscale, HSV, YUV) and perform color-based
/// operations. See [`color`] module for available conversions.
pub mod color;

/// Core image processing operations.
///
/// Fundamental operations including pixel access, data type conversions, and basic
/// arithmetic operations on images.
pub mod core;

/// Image cropping operations.
///
/// Extract rectangular regions from images with bounds checking.
pub mod crop;

// NOTE: not ready yet
// pub mod distance_transform;

/// Drawing utilities for visualization.
///
/// Draw shapes, text, and annotations on images for debugging and visualization.
pub mod draw;

/// Image enhancement operations.
///
/// Adjust contrast, brightness, and other visual properties to improve image quality.
pub mod enhance;

/// Feature detection algorithms.
///
/// Detect keypoints and features in images. See [`features`] module for available
/// detectors including FAST corner detection.
pub mod features;

/// Image filtering operations.
///
/// Apply convolution filters including Gaussian blur, Sobel edges, and custom kernels.
/// See [`filter`] module for available filters.
pub mod filter;

/// Image flipping operations.
///
/// Flip images horizontally or vertically. See [`flip`] module for details.
pub mod flip;

/// Histogram computation.
///
/// Compute intensity histograms for analysis and visualization.
pub mod histogram;

/// Interpolation utilities.
///
/// Pixel interpolation methods (nearest, bilinear, bicubic) used by geometric
/// transformations. See [`interpolation`] module for available modes.
pub mod interpolation;

/// Parallel processing utilities.
///
/// Internal utilities for efficient multi-threaded image processing using Rayon.
pub mod parallel;

/// Image quality metrics.
///
/// Compute metrics like MSE, L1 loss, and Huber loss for comparing images.
/// See [`metrics`] module for available metrics.
pub mod metrics;

/// Image normalization operations.
///
/// Normalize images using mean and standard deviation. Common preprocessing step
/// for neural networks. See [`normalize`] module for normalization functions.
pub mod normalize;

/// Image resizing operations.
///
/// Resize images with various interpolation modes. See [`resize`] module for
/// SIMD-accelerated resizing functions.
pub mod resize;

/// Thresholding operations.
///
/// Apply threshold operations to convert images to binary or multi-level representations.
pub mod threshold;

/// Geometric transformations.
///
/// Apply affine and perspective warps to images. See [`warp`] module for details.
pub mod warp;

/// Image pyramid operations.
///
/// Build and manipulate multi-scale image pyramids for coarse-to-fine processing.
pub mod pyramid;
