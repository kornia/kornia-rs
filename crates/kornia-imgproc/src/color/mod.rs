//! Color space conversions and channel manipulations.
//!
//! This module provides functions for converting between different color spaces
//! commonly used in computer vision:
//!
//! - **RGB ↔ Grayscale**: RGB to grayscale and vice versa
//! - **RGB ↔ BGR**: Channel reordering
//! - **RGB ↔ HSV**: Hue-Saturation-Value color space
//! - **RGBA → RGB**: Alpha channel removal
//! - **YUV → RGB**: YUV format conversions
//!
//! # Examples
//!
//! Converting an RGB image to grayscale:
//!
//! ```no_run
//! use kornia_image::{Image, ImageSize};
//! use kornia_image::allocator::CpuAllocator;
//! use kornia_imgproc::color::gray_from_rgb;
//!
//! let rgb = Image::<f32, 3, _>::from_size_val(
//!     ImageSize { width: 100, height: 100 },
//!     0.5,
//!     CpuAllocator
//! ).unwrap();
//! let mut gray = Image::<f32, 1, _>::from_size_val(rgb.size(), 0.0, CpuAllocator).unwrap();
//! gray_from_rgb(&rgb, &mut gray).unwrap();
//! ```

mod gray;
mod hsv;
mod rgb;
mod yuv;

pub use gray::{bgr_from_rgb, gray_from_rgb, gray_from_rgb_u8, rgb_from_gray};
pub use hsv::hsv_from_rgb;
pub use rgb::rgb_from_rgba;
pub use yuv::{convert_yuyv_to_rgb_u8, YuvToRgbMode};
