//! Color space conversion utilities for image processing.
//!
//! This module provides functions for converting images between different color spaces
//! commonly used in computer vision and image processing applications.
//!
//! # Supported Color Spaces
//!
//! * **Grayscale** - Single-channel luminance representation
//! * **RGB/BGR** - Standard red-green-blue color representation
//! * **RGBA/BGRA** - RGB with alpha (transparency) channel
//! * **HSV** - Hue-saturation-value cylindrical color space
//! * **YUV** - Luminance-chrominance color space (YUYV format)
//!
//! # Example: RGB to Grayscale
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::color::gray_from_rgb;
//!
//! let rgb_image = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 100, height: 100 },
//!     0.5,
//! ).unwrap();
//!
//! let mut gray_image = Image::<f32, 1>::from_size_val(
//!     rgb_image.size(),
//!     0.0,
//! ).unwrap();
//!
//! gray_from_rgb(&rgb_image, &mut gray_image).unwrap();
//! ```
//!
//! # See also
//!
//! * ITU-R Recommendation BT.601 for RGB to grayscale conversion weights

mod gray;
mod hsv;
mod rgb;
mod yuv;

pub use gray::{bgr_from_rgb, gray_from_rgb, gray_from_rgb_u8, rgb_from_gray};
pub use hsv::hsv_from_rgb;
pub use rgb::{rgb_from_bgra, rgb_from_rgba};
pub use yuv::{convert_yuyv_to_rgb_u8, YuvToRgbMode};
