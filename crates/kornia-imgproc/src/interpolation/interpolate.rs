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
///
/// Coordinates outside the image are clamped to the border (replicate).
///
/// # Errors
///
/// Returns [`ImageError::ChannelIndexOutOfBounds`] if `c >= C`, and
/// [`ImageError::InvalidImageSize`] if the image is empty.
pub fn interpolate_pixel<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
    interpolation: InterpolationMode,
) -> Result<f32, ImageError> {
    validate_interpolation(interpolation)?;
    if c >= C {
        return Err(ImageError::ChannelIndexOutOfBounds(c, C));
    }
    if image.rows() == 0 || image.cols() == 0 {
        return Err(ImageError::InvalidImageSize(
            image.cols(),
            image.rows(),
            1,
            1,
        ));
    }
    // Clamp into `[0, cols] x [0, rows]`: a no-op for in-range coordinates and,
    // past the far edge, every sampler already replicates the last pixel. Below
    // zero the samplers would otherwise blend/extrapolate (bilinear truncates
    // toward zero, so `u = -0.5` returned `1.5 * p0 - 0.5 * p1`), and an infinite
    // coordinate overflowed the bicubic/lanczos tap arithmetic. NaN passes through.
    let u = u.clamp(0.0, image.cols() as f32);
    let v = v.clamp(0.0, image.rows() as f32);
    Ok(interpolate_pixel_fast(image, u, v, c, interpolation))
}

/// Fallible-free internal kernel for fast pixel interpolation (must be validated first)
///
/// Preconditions: `image` is non-empty and `c < C`. Any coordinate is memory
/// safe — the samplers clamp their taps and use bounds-checked reads.
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

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    /// Regression: the channel index and coordinates were passed unchecked to
    /// `Tensor::get_unchecked`. Out-of-range channels are now an error and
    /// out-of-range coordinates clamp to the border.
    #[test]
    fn interpolate_pixel_rejects_bad_channel_and_clamps_coords() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..16).map(|v| v as f32).collect(),
        )?;
        for mode in [
            InterpolationMode::Nearest,
            InterpolationMode::Bilinear,
            InterpolationMode::Bicubic,
            InterpolationMode::Lanczos,
        ] {
            assert!(matches!(
                interpolate_pixel(&src, 0.0, 0.0, 5, mode),
                Err(ImageError::ChannelIndexOutOfBounds(5, 1))
            ));
        }
        // Far right of row 0 -> last pixel of row 0.
        assert_eq!(
            interpolate_pixel(&src, 1.0e9, 0.0, 0, InterpolationMode::Bilinear)?,
            3.0
        );
        assert_eq!(
            interpolate_pixel(&src, 1.0e9, 1.0e9, 0, InterpolationMode::Nearest)?,
            15.0
        );
        // Negative / NaN coordinates stay in bounds (no panic, finite or NaN).
        let _ = interpolate_pixel(&src, -1.0e9, f32::NAN, 0, InterpolationMode::Bilinear)?;
        let _ = interpolate_pixel(&src, f32::NAN, -3.0, 0, InterpolationMode::Nearest)?;
        // Every mode replicates the border: slightly left of column 0 is pixel 0
        // (bilinear used to extrapolate to -0.5 here), and +inf is the last
        // column (bicubic/lanczos used to overflow their tap arithmetic).
        for mode in [
            InterpolationMode::Nearest,
            InterpolationMode::Bilinear,
            InterpolationMode::Bicubic,
            InterpolationMode::Lanczos,
        ] {
            let left = interpolate_pixel(&src, -0.5, 0.0, 0, mode)?;
            assert!(left.abs() < 1e-4, "{mode:?}: u=-0.5 gave {left}");
            let right = interpolate_pixel(&src, f32::INFINITY, 0.0, 0, mode)?;
            assert!((right - 3.0).abs() < 1e-4, "{mode:?}: u=inf gave {right}");
        }

        let empty = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 0,
                height: 0,
            },
            0.0,
        )?;
        assert!(interpolate_pixel(&empty, 0.0, 0.0, 0, InterpolationMode::Bilinear).is_err());
        Ok(())
    }
}
