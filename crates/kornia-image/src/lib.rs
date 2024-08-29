#![deny(missing_docs)]
//! Image types and traits for generating and manipulating images

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

pub use crate::error::ImageError;
pub use crate::image::cast_and_scale;
pub use crate::image::{Image, ImageDtype, ImageSize};

impl<T, const CHANNELS: usize> From<ndarray::Array3<T>> for Image<T, CHANNELS>
where
    T: kornia_core::SafeTensorType,
{
    fn from(array: ndarray::Array3<T>) -> Image<T, CHANNELS> {
        let (height, width, channels) = array.dim();
        let data = array.into_raw_vec();
        assert_eq!(
            channels, CHANNELS,
            "Number of channels does not match the const generic parameter."
        );

        Self::new(ImageSize { height, width }, data).unwrap()
    }
}
