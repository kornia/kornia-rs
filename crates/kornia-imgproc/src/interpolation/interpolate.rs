use super::bilinear::bilinear_interpolation;
use super::nearest::nearest_neighbor_interpolation;
use kornia_image::allocator::ImageAllocator;
use kornia_image::{Image, ImageError};

/// Interpolation mode for the resize operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    /// Bilinear interpolation
    Bilinear,
    /// Nearest neighbor interpolation
    Nearest,
    /// Lanczos interpolation
    Lanczos,
    /// Bicubic interpolation
    Bicubic,
}

/// Validate that the given interpolation mode is supported by `interpolate_pixel`.
///
/// Returns `Ok(())` if the mode is supported, or `Err` with a descriptive message.
/// Call this before entering parallel dispatch loops to catch unsupported modes early.
pub fn validate_interpolation(interpolation: InterpolationMode) -> Result<(), ImageError> {
    match interpolation {
        InterpolationMode::Bilinear | InterpolationMode::Nearest => Ok(()),
        mode => Err(ImageError::UnsupportedInterpolation(format!("{mode:?}"))),
    }
}

/// Kernel for interpolating a pixel value
///
/// # Arguments
///
/// * `image` - The input image container with shape (height, width, C).
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The interpolated pixel value, or an error if the interpolation mode is not supported.
pub fn interpolate_pixel<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> Result<f32, kornia_image::ImageError> {
    match interpolation {
        InterpolationMode::Bilinear => Ok(bilinear_interpolation(image, u, v, c)),
        InterpolationMode::Nearest => Ok(nearest_neighbor_interpolation(image, u, v, c)),
        InterpolationMode::Lanczos => Err(kornia_image::ImageError::UnsupportedInterpolation(
            "Lanczos".to_string(),
        )),
        InterpolationMode::Bicubic => Err(kornia_image::ImageError::UnsupportedInterpolation(
            "Bicubic".to_string(),
        )),
    }
}
