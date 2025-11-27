#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// allocator module containing the memory management utilities.
pub mod allocator;

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

/// module containing ops implementations.
pub mod ops;

/// Typed color space wrappers for compile-time type safety.
pub mod color_spaces;

pub use crate::error::ImageError;
pub use crate::image::{Image, ImageLayout, PixelFormat, ImageSize};

/// Arrow integration for converting images to Arrow format
#[cfg(feature = "arrow")]
pub mod arrow;
