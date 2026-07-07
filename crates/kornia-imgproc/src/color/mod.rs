//! Color space conversion utilities
//!
//! This module provides both type-safe and traditional APIs for color space conversions.
//!
//! # Type-Safe API (Recommended)
//!
//! Use explicit color space types for compile-time type safety:
//!
//! ```
//! use kornia_image::ImageSize;
//! use kornia_imgproc::color::{Rgb8, Gray8, ConvertColor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create RGB image directly
//! let rgb = Rgb8::from_size_vec(
//!     ImageSize { width: 4, height: 5 },
//!     vec![128u8; 4 * 5 * 3],
//! )?;
//!
//! let mut gray = Gray8::from_size_val(rgb.size(), 0)?;
//!
//! // Type-safe conversion
//! rgb.convert(&mut gray)?;
//! # Ok(())
//! # }
//! ```
//!
//! Available explicit types: `Rgb8`, `Rgb16`, `Rgbf32`, `Bgr8`, `Gray8`, `Grayf32`,
//! `Rgba8`, `Bgra8`, etc.
//!
//! # Traditional API (Backward Compatible)
//!
//! Direct function calls are still available:
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::color::gray_from_rgb;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let rgb = Image::<f32, 3>::new(
//!     ImageSize { width: 4, height: 5 },
//!     vec![0.5f32; 4 * 5 * 3],
//! )?;
//!
//! let mut gray = Image::<f32, 1>::from_size_val(rgb.size(), 0.0)?;
//! gray_from_rgb(&rgb, &mut gray)?;
//! # Ok(())
//! # }
//! ```

// Re-export color spaces from kornia-image
pub use kornia_image::color_spaces::{
    Bgr16, Bgr8, Bgra16, Bgra8, Bgraf32, Bgraf64, Bgrf32, Bgrf64, Gray16, Gray8, Grayf32, Grayf64,
    Hlsf32, Hlsf64, Hsvf32, Hsvf64, Labf32, Labf64, LinearRgbf32, LinearRgbf64, Luvf32, Luvf64,
    Nv12, Nv21, Rgb16, Rgb8, Rgba16, Rgba8, Rgbaf32, Rgbaf64, Rgbf32, Rgbf64, Uyvy8, Xyzf32,
    Xyzf64, YCbCr8, YCbCrf32, YCbCrf64, Yuv8, Yuvf32, Yuvf64, Yuyv8, Yv12, Yvyu8, I420,
};

mod convert;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_dispatch;
mod kernel_common;

pub(crate) mod bayer;
pub(crate) mod cie;
/// Colormap application (LUT-based, NEON-accelerated on aarch64).
pub mod colormap;
pub(crate) mod gray;
pub(crate) mod hls;
pub(crate) mod hsv;
// Shared generic 3×3 affine kernel; the CIE pipelines fuse the matrix in-register
// instead of calling it, but it stays available for the YUV/YCbCr family.
mod matrix;
pub(crate) mod rgb;
pub(crate) mod sepia;
pub(crate) mod yuv;

// Export traits for type-safe conversions
pub use convert::{
    ConvertColor, ConvertColorExt, ConvertColorWithBackground, NewColorImage, SrcSize, Tagged,
};

pub use colormap::{apply_colormap, ColormapType};
// Re-export Bayer mosaic types from kornia-image alongside the demosaic fns.
pub use kornia_image::color_spaces::{Bayer8, BayerPattern};
// Keep old functions available for backward compatibility
pub use bayer::*;
pub use cie::*;
pub use gray::*;
pub use hls::*;
pub use hsv::*;
pub use rgb::*;
pub use sepia::*;
pub use yuv::*;
