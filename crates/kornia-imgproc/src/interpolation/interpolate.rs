use super::bilinear::bilinear_interpolation;
use super::nearest::nearest_neighbor_interpolation;
use kornia_image::Image;

/// Interpolation mode for the resize operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    /// Bilinear interpolation
    Bilinear,
    /// Nearest neighbor interpolation
    Nearest,
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
/// The interpolated pixel value.
pub fn interpolate_pixel<const C: usize>(
    //image: &ArrayView3<T>,
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> f32 {
    match interpolation {
        InterpolationMode::Bilinear => bilinear_interpolation(image, u, v, c),
        InterpolationMode::Nearest => nearest_neighbor_interpolation(image, u, v, c),
    }
}
