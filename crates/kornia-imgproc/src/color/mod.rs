//! Color space conversions for image processing.
//!
//! This module provides efficient color space conversion functions for the Kornia library.
//! All conversions are parallelized using Rayon for optimal performance.
//!
//! # Available Color Spaces
//!
//! - **RGB**: Red, Green, Blue - Standard color space with values in [0, 255]
//! - **BGR**: Blue, Green, Red - OpenCV-compatible color space with values in [0, 255]
//! - **Grayscale**: Single channel luminance with values in [0, 255] or [0, 1]
//! - **HSV**: Hue, Saturation, Value - Cylindrical color space
//!   - H: [0, 255] representing [0°, 360°)
//!   - S: [0, 255] representing [0%, 100%]
//!   - V: [0, 255] representing [0%, 100%]
//! - **YUV**: Y'UV color space for video - supports YUYV format with BT.601/BT.709 standards
//!
//! # Supported Conversions
//!
//! ## RGB ↔ Grayscale
//! - [`gray_from_rgb`] - Convert RGB to grayscale using standard luminance weights
//! - [`gray_from_rgb_u8`] - Optimized u8 version using fixed-point arithmetic
//! - [`rgb_from_gray`] - Convert grayscale to RGB by replicating the value
//!
//! ## BGR ↔ RGB
//! - [`bgr_from_rgb`] - Swap red and blue channels
//! - [`gray_from_bgr`] - Convert BGR to grayscale
//!
//! ## RGB ↔ HSV
//! - [`hsv_from_rgb`] - Convert RGB to HSV color space
//! - [`rgb_from_hsv`] - Convert HSV back to RGB color space
//!
//! ## RGBA/BGRA → RGB
//! - [`rgb_from_rgba`] - Remove alpha channel with optional background blending
//! - [`rgb_from_bgra`] - Remove alpha channel from BGRA with optional background blending
//!
//! ## YUV → RGB
//! - [`convert_yuyv_to_rgb_u8`] - Convert YUYV (YUY2) format to RGB
//!   - Supports BT.601 Full Range (JPEG standard)
//!   - Supports BT.709 Full Range (HDTV standard)
//!   - Supports BT.601 Limited Range (Broadcast standard)
//!
//! # Performance Characteristics
//!
//! - All conversions use parallel processing via Rayon
//! - Integer arithmetic variants (e.g., `gray_from_rgb_u8`) use fixed-point math for speed
//! - SIMD optimizations are used where applicable through dependencies
//! - Zero-copy operations where possible
//!
//! # Examples
//!
//! ```rust
//! use kornia_image::{Image, ImageSize};
//! use kornia_image::allocator::CpuAllocator;
//! use kornia_imgproc::color::{gray_from_rgb, hsv_from_rgb, rgb_from_hsv};
//!
//! // Create an RGB image
//! let image = Image::<f32, 3, _>::new(
//!     ImageSize { width: 100, height: 100 },
//!     vec![128.0; 100 * 100 * 3],
//!     CpuAllocator
//! ).unwrap();
//!
//! // Convert to grayscale
//! let mut gray = Image::<f32, 1, _>::from_size_val(
//!     image.size(),
//!     0.0,
//!     CpuAllocator
//! ).unwrap();
//! gray_from_rgb(&image, &mut gray).unwrap();
//!
//! // Convert to HSV and back
//! let mut hsv = Image::<f32, 3, _>::from_size_val(
//!     image.size(),
//!     0.0,
//!     CpuAllocator
//! ).unwrap();
//! hsv_from_rgb(&image, &mut hsv).unwrap();
//!
//! let mut rgb = Image::<f32, 3, _>::from_size_val(
//!     image.size(),
//!     0.0,
//!     CpuAllocator
//! ).unwrap();
//! rgb_from_hsv(&hsv, &mut rgb).unwrap();
//! ```
//!
//! # Color Space Standards
//!
//! ## Grayscale Conversion
//! Uses ITU-R BT.601 luminance coefficients:
//! - Y = 0.299 * R + 0.587 * G + 0.114 * B
//!
//! ## YUV Conversions
//! Supports multiple ITU-R standards:
//! - **BT.601**: Standard Definition Television (SDTV)
//! - **BT.709**: High Definition Television (HDTV)
//! - **BT.2020**: Ultra High Definition Television (UHDTV)
//!
//! See [`YuvToRgbMode`] for more details on YUV conversion modes.

mod gray;
mod hsv;
mod rgb;
mod yuv;

pub use gray::{bgr_from_rgb, gray_from_bgr, gray_from_rgb, gray_from_rgb_u8, rgb_from_gray};
pub use hsv::{hsv_from_rgb, rgb_from_hsv};
pub use rgb::{rgb_from_bgra, rgb_from_rgba};
pub use yuv::{convert_yuyv_to_rgb_u8, YuvToRgbMode};
