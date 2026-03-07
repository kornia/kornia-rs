use super::bilinear::bilinear_interpolation;
use super::nearest::nearest_neighbor_interpolation;
use kornia_image::allocator::ImageAllocator;
use kornia_image::{Image, ImageError};

pub use kornia_image::InterpolationMode;

/// Validate that the given interpolation mode is supported by `interpolate_pixel`.
///
/// Returns `Ok(())` if the mode is supported, or `Err` with a descriptive message.
/// Call this before entering parallel dispatch loops to catch unsupported modes early.
pub fn validate_interpolation(interpolation: InterpolationMode) -> Result<(), ImageError> {
    match interpolation {
        InterpolationMode::Bilinear | InterpolationMode::Nearest => Ok(()),
        mode => Err(ImageError::UnsupportedInterpolation(mode)),
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
/// The interpolated pixel value. Panics if an unsupported mode is passed (must be validated first).
pub fn interpolate_pixel<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> f32 {
    match interpolation {
        InterpolationMode::Bilinear => bilinear_interpolation(image, u, v, c),
        InterpolationMode::Nearest => nearest_neighbor_interpolation(image, u, v, c),
        _ => 0.0,
    }
}
