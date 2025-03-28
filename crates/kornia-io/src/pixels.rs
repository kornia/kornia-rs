use image::DynamicImage;
use kornia_image::{Image, ImageSize}; // Assuming these are defined in your crate
use crate::error::IoError;

/// A trait for converting a `DynamicImage` into an `Image<T, 3>`.
pub trait PixelType: Sized {
    /// Convert a `DynamicImage` into an `Image<T, 3>`.
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<Self, 3>, IoError>;
}

// Implementation for 8-bit pixels.
impl PixelType for u8 {
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<u8, 3>, IoError> {
        // If the decoded image is already in 8-bit RGB, use it directly.
        // Otherwise, force conversion to 8-bit.
        let rgb_img = match img {
            DynamicImage::ImageRgb8(rgb) => rgb,
            other => other.into_rgb8(),
        };
        Image::new(
            ImageSize {
                width: rgb_img.width() as usize,
                height: rgb_img.height() as usize,
            },
            rgb_img.to_vec(),
        )
        .map_err(|e| IoError::ImageConversionError(format!("{}", e)))
    }
}

// Implementation for 32-bit floating point pixels.
impl PixelType for f32 {
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<f32, 3>, IoError> {
        // If the image is already in 32-bit float RGB, use it;
        // otherwise, convert to RGB32F (this scales u8 images into [0.0,1.0]).
        let rgb_img = match img {
            DynamicImage::ImageRgb32F(rgb) => rgb,
            other => other.into_rgb32f(),
        };
        Image::new(
            ImageSize {
                width: rgb_img.width() as usize,
                height: rgb_img.height() as usize,
            },
            rgb_img.to_vec(),
        )
        .map_err(|e| IoError::ImageConversionError(format!("{}", e)))
    }
}

// Implementation for 16-bit pixels.
impl PixelType for u16 {
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<u16, 3>, IoError> {
        // If the image is already in 16-bit RGB, use it directly.
        // Otherwise, convert to RGB8 and then upconvert each pixel.
        match img {
            DynamicImage::ImageRgb16(rgb) => Image::new(
                ImageSize {
                    width: rgb.width() as usize,
                    height: rgb.height() as usize,
                },
                rgb.to_vec(),
            )
            .map_err(|e| IoError::ImageConversionError(format!("{}", e))),
            other => {
                let rgb8 = other.into_rgb8();
                let data: Vec<u16> = rgb8.to_vec().iter().map(|&x| x as u16).collect();
                Image::new(
                    ImageSize {
                        width: rgb8.width() as usize,
                        height: rgb8.height() as usize,
                    },
                    data,
                )
                .map_err(|e| IoError::ImageConversionError(format!("{}", e)))
            }
        }
    }
}
