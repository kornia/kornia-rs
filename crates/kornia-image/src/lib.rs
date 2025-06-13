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

pub use crate::error::ImageError;
pub use crate::image::{Image, ImageSize};
