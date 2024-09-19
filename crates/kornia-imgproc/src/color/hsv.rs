use crate::parallel;
use kornia_image::{Image, ImageError};

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
/// use kornia_imgproc::color::hsv_from_rgb;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///        width: 4,
///        height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let mut hsv = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// hsv_from_rgb(&image, &mut hsv).unwrap();
///
/// assert_eq!(hsv.num_channels(), 3);
/// assert_eq!(hsv.size().width, 4);
/// assert_eq!(hsv.size().height, 5);
/// ```
pub fn hsv_from_rgb(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
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

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use num_traits::Pow;

    #[test]
    fn hsv_from_rgb() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 128.0, 255.0, 255.0, 128.0, 0.0, 128.0, 255.0, 0.0, 255.0, 0.0, 128.0, 0.0,
                128.0, 255.0, 255.0, 128.0, 0.0,
            ],
        )?;

        let expected = [
            148.66667, 255.0, 255.0, 21.333334, 255.0, 255.0, 63.666668, 255.0, 255.0, 233.66667,
            255.0, 255.0, 148.66667, 255.0, 255.0, 21.333334, 255.0, 255.0,
        ];

        let mut hsv = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

        super::hsv_from_rgb(&image, &mut hsv)?;

        assert_eq!(hsv.num_channels(), 3);
        assert_eq!(hsv.size().width, 2);
        assert_eq!(hsv.size().height, 3);

        for (a, b) in hsv.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).pow(2) < 1e-6f32);
        }

        Ok(())
    }
}
