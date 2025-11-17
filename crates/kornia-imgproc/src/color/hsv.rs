use crate::parallel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Convert an RGB image to an HSV image.
///
/// The input image is assumed to have 3 channels in the order R, G, B.
///
/// # Arguments
///
/// * `src` - The input RGB image assumed to have 3 channels.
/// * `dst` - The output HSV image.
///
/// # Returns
///
/// The HSV image with the following channels:
///
/// * H: The hue channel in the range [0, 255] (0-360 degrees).
/// * S: The saturation channel in the range [0, 255].
/// * V: The value channel in the range [0, 255].
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::hsv_from_rgb;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///        width: 4,
///        height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut hsv = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// hsv_from_rgb(&image, &mut hsv).unwrap();
///
/// assert_eq!(hsv.num_channels(), 3);
/// assert_eq!(hsv.size().width, 4);
/// assert_eq!(hsv.size().height, 5);
/// ```
pub fn hsv_from_rgb<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, 3, A1>,
    dst: &mut Image<f32, 3, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // compute the HSV values
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        // Normalize the input to the range [0, 1]
        let r = src_pixel[0] / 255.;
        let g = src_pixel[1] / 255.;
        let b = src_pixel[2] / 255.;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let h = if delta == 0.0 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };

        // Ensure h is in the range [0, 360)

        let h = if h < 0.0 { h + 360.0 } else { h };

        // scale h to [0, 255]

        let h = (h / 360.0) * 255.0;

        let s = if max == 0.0 {
            0.0
        } else {
            (delta / max) * 255.0
        };

        let v = max * 255.0;

        dst_pixel[0] = h;
        dst_pixel[1] = s;
        dst_pixel[2] = v;
    });

    Ok(())
}

/// Convert an HSV image to an RGB image.
///
/// The input image is assumed to have 3 channels in the order H, S, V.
///
/// # Arguments
///
/// * `src` - The input HSV image assumed to have 3 channels.
/// * `dst` - The output RGB image.
///
/// # Returns
///
/// The RGB image with the following channels:
///
/// * R: The red channel in the range [0, 255].
/// * G: The green channel in the range [0, 255].
/// * B: The blue channel in the range [0, 255].
///
/// # Input Format
///
/// * H: The hue channel in the range [0, 255] (representing 0-360 degrees).
/// * S: The saturation channel in the range [0, 255].
/// * V: The value channel in the range [0, 255].
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::{hsv_from_rgb, rgb_from_hsv};
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///        width: 2,
///        height: 2,
///     },
///     vec![255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut hsv = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
/// hsv_from_rgb(&image, &mut hsv).unwrap();
///
/// let mut rgb = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
/// rgb_from_hsv(&hsv, &mut rgb).unwrap();
///
/// assert_eq!(rgb.num_channels(), 3);
/// assert_eq!(rgb.size().width, 2);
/// assert_eq!(rgb.size().height, 2);
/// ```
pub fn rgb_from_hsv<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, 3, A1>,
    dst: &mut Image<f32, 3, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // compute the RGB values
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        // De-normalize from [0, 255] to [0, 1] and [0, 360) for hue
        let h = (src_pixel[0] / 255.0) * 360.0;
        let s = src_pixel[1] / 255.0;
        let v = src_pixel[2] / 255.0;

        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        // Scale back to [0, 255]
        dst_pixel[0] = (r_prime + m) * 255.0;
        dst_pixel[1] = (g_prime + m) * 255.0;
        dst_pixel[2] = (b_prime + m) * 255.0;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;
    use num_traits::Pow;

    #[test]
    fn hsv_from_rgb() -> Result<(), ImageError> {
        let image = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 128.0, 255.0, 255.0, 128.0, 0.0, 128.0, 255.0, 0.0, 255.0, 0.0, 128.0, 0.0,
                128.0, 255.0, 255.0, 128.0, 0.0,
            ],
            CpuAllocator,
        )?;

        let expected = [
            148.66667, 255.0, 255.0, 21.333334, 255.0, 255.0, 63.666668, 255.0, 255.0, 233.66667,
            255.0, 255.0, 148.66667, 255.0, 255.0, 21.333334, 255.0, 255.0,
        ];

        let mut hsv = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::hsv_from_rgb(&image, &mut hsv)?;

        assert_eq!(hsv.num_channels(), 3);
        assert_eq!(hsv.size().width, 2);
        assert_eq!(hsv.size().height, 3);

        for (a, b) in hsv.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).pow(2) < 1e-6f32);
        }

        Ok(())
    }

    #[test]
    fn rgb_from_hsv() -> Result<(), ImageError> {
        // Test with known HSV values
        let hsv = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                // Pure red (H=0, S=100%, V=100%)
                0.0, 255.0, 255.0, // Pure green (H=120, S=100%, V=100%)
                85.0, 255.0, 255.0, // Pure blue (H=240, S=100%, V=100%)
                170.0, 255.0, 255.0, // Gray (H=0, S=0%, V=50%)
                0.0, 0.0, 128.0, // White (H=0, S=0%, V=100%)
                0.0, 0.0, 255.0, // Black (H=0, S=0%, V=0%)
                0.0, 0.0, 0.0,
            ],
            CpuAllocator,
        )?;

        let mut rgb = Image::<f32, 3, _>::from_size_val(hsv.size(), 0.0, CpuAllocator)?;

        super::rgb_from_hsv(&hsv, &mut rgb)?;

        assert_eq!(rgb.num_channels(), 3);
        assert_eq!(rgb.size().width, 2);
        assert_eq!(rgb.size().height, 3);

        // Verify pure red
        assert!((rgb.get_unchecked([0, 0, 0]) - 255.0).abs() < 1.0);
        assert!(rgb.get_unchecked([0, 0, 1]).abs() < 1.0);
        assert!(rgb.get_unchecked([0, 0, 2]).abs() < 1.0);

        // Verify pure green
        assert!(rgb.get_unchecked([0, 1, 0]).abs() < 1.0);
        assert!((rgb.get_unchecked([0, 1, 1]) - 255.0).abs() < 1.0);
        assert!(rgb.get_unchecked([0, 1, 2]).abs() < 1.0);

        // Verify pure blue
        assert!(rgb.get_unchecked([1, 0, 0]).abs() < 1.0);
        assert!(rgb.get_unchecked([1, 0, 1]).abs() < 1.0);
        assert!((rgb.get_unchecked([1, 0, 2]) - 255.0).abs() < 1.0);

        // Verify gray
        assert!((rgb.get_unchecked([1, 1, 0]) - 128.0).abs() < 1.0);
        assert!((rgb.get_unchecked([1, 1, 1]) - 128.0).abs() < 1.0);
        assert!((rgb.get_unchecked([1, 1, 2]) - 128.0).abs() < 1.0);

        // Verify white
        assert!((rgb.get_unchecked([2, 0, 0]) - 255.0).abs() < 1.0);
        assert!((rgb.get_unchecked([2, 0, 1]) - 255.0).abs() < 1.0);
        assert!((rgb.get_unchecked([2, 0, 2]) - 255.0).abs() < 1.0);

        // Verify black
        assert!(rgb.get_unchecked([2, 1, 0]).abs() < 1.0);
        assert!(rgb.get_unchecked([2, 1, 1]).abs() < 1.0);
        assert!(rgb.get_unchecked([2, 1, 2]).abs() < 1.0);

        Ok(())
    }

    #[test]
    fn rgb_hsv_roundtrip() -> Result<(), ImageError> {
        // Test that converting RGB -> HSV -> RGB produces the original image
        let original = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![
                255.0, 0.0, 0.0, // Red
                0.0, 255.0, 0.0, // Green
                0.0, 0.0, 255.0, // Blue
                128.0, 128.0, 128.0, // Gray
            ],
            CpuAllocator,
        )?;

        let mut hsv = Image::<f32, 3, _>::from_size_val(original.size(), 0.0, CpuAllocator)?;
        super::hsv_from_rgb(&original, &mut hsv)?;

        let mut rgb = Image::<f32, 3, _>::from_size_val(original.size(), 0.0, CpuAllocator)?;
        super::rgb_from_hsv(&hsv, &mut rgb)?;

        // Check roundtrip accuracy (allow small numerical errors)
        for (a, b) in rgb.as_slice().iter().zip(original.as_slice().iter()) {
            assert!((a - b).abs() < 1.0, "Expected {}, got {}", b, a);
        }

        Ok(())
    }
}
