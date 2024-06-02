use super::bilinear::bilinear_interpolation;
use super::nearest::nearest_neighbor_interpolation;
use crate::image::ImageDtype;
use ndarray::Array3;

/// Interpolation mode for the resize operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    Bilinear,
    Nearest,
}

/// Kernel for interpolating a pixel value
///
/// # Arguments
///
/// * `image` - The input image container with shape (height, width, channels).
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The interpolated pixel value.
pub(crate) fn interpolate_pixel<T: ImageDtype>(
    image: &Array3<T>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> T {
    match interpolation {
        InterpolationMode::Bilinear => bilinear_interpolation(image, u, v, c),
        InterpolationMode::Nearest => nearest_neighbor_interpolation(image, u, v, c),
    }
}
