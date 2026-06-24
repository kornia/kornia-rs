#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// allocator module containing the memory management utilities.
pub mod allocator;

/// Runtime-typed image buffer with dynamic pixel format and color space.
pub mod dyn_image_buf;

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

/// module containing ops implementations.
pub mod ops;

/// Typed color space wrappers for compile-time type safety.
pub mod color_spaces;

/// Runtime color-space vocabulary shared by Rust and Python.
pub mod color_space;

pub use crate::color_space::{ColorSpace, DynImage};
pub use crate::dyn_image_buf::DynImageBuf;
pub use crate::error::ImageError;
pub use crate::image::{Image, ImageLayout, ImageSize, InterpolationMode, PixelFormat};

/// Arrow integration for converting images to Arrow format
#[cfg(feature = "arrow")]
pub mod arrow;

/// DLPack interoperability for [`DynImageBuf`].
#[cfg(feature = "dlpack")]
pub mod dlpack;
