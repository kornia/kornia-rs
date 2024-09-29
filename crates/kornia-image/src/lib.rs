#![deny(missing_docs)]
//! Image types and traits for generating and manipulating images

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

/// module containing ops implementations.
pub mod ops;

/// re-export the safe tensor types so that is easy to use them
pub use kornia_core::SafeTensorType;

pub use crate::error::ImageError;
pub use crate::image::{Image, ImageSize};
