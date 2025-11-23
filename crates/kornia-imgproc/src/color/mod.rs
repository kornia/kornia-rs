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
//! use kornia_image::allocator::CpuAllocator;
//! use kornia_imgproc::color::{Rgb8, Gray8, ConvertColor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create RGB image directly
//! let rgb = Rgb8::from_size_vec(
//!     ImageSize { width: 4, height: 5 },
//!     vec![128u8; 4 * 5 * 3],
//!     CpuAllocator
//! )?;
//!
//! let mut gray = Gray8::from_size_val(rgb.size(), 0, CpuAllocator)?;
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
//! use kornia_image::allocator::CpuAllocator;
//! use kornia_imgproc::color::gray_from_rgb;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let rgb = Image::<f32, 3, _>::new(
//!     ImageSize { width: 4, height: 5 },
//!     vec![0.5f32; 4 * 5 * 3],
//!     CpuAllocator
//! )?;
//!
//! let mut gray = Image::<f32, 1, _>::from_size_val(rgb.size(), 0.0, CpuAllocator)?;
//! gray_from_rgb(&rgb, &mut gray)?;
//! # Ok(())
//! # }
//! ```

// Re-export color spaces from kornia-image
pub use kornia_image::color_spaces::{
    Bgr16, Bgr8, Bgra16, Bgra8, Bgraf32, Bgraf64, Bgrf32, Bgrf64, Gray16, Gray8, Grayf32, Grayf64,
    Hsvf32, Hsvf64, Rgb16, Rgb8, Rgba16, Rgba8, Rgbaf32, Rgbaf64, Rgbf32, Rgbf64,
};

mod convert;

mod gray;
mod hsv;
mod rgb;
mod yuv;

// Export traits for type-safe conversions
pub use convert::{ConvertColor, ConvertColorWithBackground};

// Keep old functions available for backward compatibility
pub use gray::*;
pub use hsv::*;
pub use rgb::*;
pub use yuv::*;
