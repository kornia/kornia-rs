#![deny(missing_docs)]
//! Image types and traits for generating and manipulating images

/// image representation for computer vision purposes.
pub mod image;

pub use crate::image::{Image, ImageDtype, ImageSize};
