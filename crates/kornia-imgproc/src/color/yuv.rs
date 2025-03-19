use crate::parallel;
use kornia_image::{Image, ImageError};

/// Convert an RGB image to an YUV image.
///
/// The input image is assumed to have 3 channels in the order R, G, B. in the range [0, 255].
///
/// # Arguments
///
/// * `src` - The input RGB image assumed to have 3 channels.
/// * `dst` - The output YUV image.
///
/// # Returns
///
/// The YUV image with the following channels:
///
/// * Y: The luminance channel in the range [0, 1].
/// * U: The chrominance-blue channel in the range [-0.436, +0.436].
/// * V: The chrominance-red channel in the range [-0.615, +0.615].
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::yuv_from_rgb;
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
/// let mut yuv = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// yuv_from_rgb(&image, &mut yuv).unwrap();
///
/// assert_eq!(yuv.num_channels(), 3);
/// assert_eq!(yuv.size().width, 4);
/// assert_eq!(yuv.size().height, 5);
/// ```
pub fn yuv_from_rgb(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
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

        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let u = -0.147 * r - 0.289 * g + 0.436 * b;
        let v = 0.615 * r - 0.515 * g - 0.100 * b;

        dst_pixel[0] = y;
        dst_pixel[1] = u;
        dst_pixel[2] = v;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use num_traits::Pow;

    #[test]
    fn yuv_from_rgb() -> Result<(), ImageError> {
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
            0.4087, 0.2909, -0.3585, 0.5937, -0.2921, 0.3565, 0.7371, -0.3628, -0.2063, 0.3562,
            0.0719, 0.5648, 0.4087, 0.2909, -0.3585, 0.5937, -0.2921, 0.3565,
        ];

        let mut yuv = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

        super::yuv_from_rgb(&image, &mut yuv)?;

        assert_eq!(yuv.num_channels(), 3);
        assert_eq!(yuv.size(), image.size());

        for (a, b) in yuv.as_slice().iter().zip(expected.iter()) {
            println!("{:?} {:?}", a, b);
            assert!((a - b).pow(2) < 1e-6f32);
        }
        Ok(())
    }
}
