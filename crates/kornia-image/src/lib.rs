#![deny(missing_docs)]
//! Image types and traits for generating and manipulating images

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

pub use crate::error::ImageError;
pub use crate::image::cast_and_scale;
pub use crate::image::{Image, ImageDtype, ImageSize};
