use super::bicubic::bicubic_sample;
use super::bilinear::bilinear_interpolation;
use super::lanczos::lanczos_sample;
use super::nearest::nearest_neighbor_interpolation;
use kornia_image::{Image, ImageError};

pub use kornia_image::InterpolationMode;

/// Validate that the given interpolation mode is supported by `interpolate_pixel`.
///
/// Returns `Ok(())` if the mode is supported, or `Err` with a descriptive message.
/// Call this before entering parallel dispatch loops to catch unsupported modes early.
pub fn validate_interpolation(interpolation: InterpolationMode) -> Result<(), ImageError> {
    match interpolation {
        InterpolationMode::Bilinear
        | InterpolationMode::Nearest
        | InterpolationMode::Bicubic
        | InterpolationMode::Lanczos => Ok(()),
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
/// The interpolated pixel value, or an error if the interpolation mode is unsupported.
pub fn interpolate_pixel<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> Result<f32, ImageError> {
    validate_interpolation(interpolation)?;
    Ok(interpolate_pixel_fast(image, u, v, c, interpolation))
}

/// Fallible-free internal kernel for fast pixel interpolation (must be validated first)
///
/// Prefer hoisting the mode dispatch OUT of per-pixel loops (see `resize` /
/// `warp_perspective`): a call site that keeps the runtime `interpolation`
/// branch inside its hot loop pays for all four sampler bodies. This function
/// stays for per-point callers (optical flow's border path, the public
/// `interpolate_pixel`).
#[inline]
pub(crate) fn interpolate_pixel_fast<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> f32 {
    match interpolation {
        InterpolationMode::Bilinear => bilinear_interpolation(image, u, v, c),
        InterpolationMode::Nearest => nearest_neighbor_interpolation(image, u, v, c),
        InterpolationMode::Bicubic => bicubic_sample(image, u, v, c),
        InterpolationMode::Lanczos => lanczos_sample(image, u, v, c),
    }
}
