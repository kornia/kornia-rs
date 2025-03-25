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
/// let mut yuv = Image::from_size_val(image.size(), 0.0).unwrap();
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

    // compute the YUV values
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

/// Convert a YUV image to an RGB image.
///
/// The input image is assumed to have 3 channels in the order Y, U, V. Where Y is in range[0, 1].
/// U is in range [-0.436, +0.436] and V is in range [-0.615, +0.615].
///
/// # Arguments
///
/// * `src` - The input YUV image.
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
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_yuv;
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
/// let mut rgb = Image::from_size_val(image.size(), 0.0).unwrap();
///
/// rgb_from_yuv(&image, &mut rgb).unwrap();
///
/// assert_eq!(rgb.num_channels(), 3);
/// assert_eq!(rgb.size().width, 4);
/// assert_eq!(rgb.size().height, 5);
/// ```
pub fn rgb_from_yuv(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
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
        // Transform the input to the range [0, 255]
        let y = src_pixel[0] * 255.;
        let u = src_pixel[1] * 255.;
        let v = src_pixel[2] * 255.;

        let r = y + 1.140 * v;
        let g = y - 0.396 * u - 0.581 * v;
        let b = y + 2.029 * u;

        dst_pixel[0] = r;
        dst_pixel[1] = g;
        dst_pixel[2] = b;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use num_traits::Pow;
    const RGB_TEST_DATA: [f32; 18] = [
        0.0, 128.0, 255.0, 255.0, 128.0, 0.0, 128.0, 255.0, 0.0, 255.0, 0.0, 128.0, 0.0, 128.0,
        255.0, 255.0, 128.0, 0.0,
    ];
    // corresponding YUV values to the RGB_TEST_DATA
    const YUV_TEST_DATA: [f32; 18] = [
        0.4087, 0.2909, -0.3585, 0.5937, -0.2921, 0.3565, 0.7371, -0.3628, -0.2063, 0.3562, 0.0719,
        0.5648, 0.4087, 0.2909, -0.3585, 0.5937, -0.2921, 0.3565,
    ];
    const RGB_FROM_YUV_DATA: [f32; 18] = [
        0.002548, 127.956985, 254.7287, 255.02805, 128.0725, 0.262405, 127.98908, 255.16042,
        0.249588, 255.01837, -0.107399, 128.03171, 0.002548, 127.956985, 254.7287, 255.02805,
        128.0725, 0.262405,
    ];
    const WIDTH: usize = 2;
    const HEIGHT: usize = 3;

    #[test]
    fn yuv_from_rgb() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            RGB_TEST_DATA.to_vec(),
        )?;
        let expected = YUV_TEST_DATA;

        let mut yuv = Image::from_size_val(image.size(), 0.0)?;

        super::yuv_from_rgb(&image, &mut yuv)?;

        assert_eq!(yuv.num_channels(), 3);
        assert_eq!(yuv.size(), image.size());

        for (a, b) in yuv.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).pow(2) < 1e-6f32);
        }
        Ok(())
    }

    #[test]
    fn rgb_from_yuv() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            YUV_TEST_DATA.to_vec(),
        )?;
        let expected = RGB_FROM_YUV_DATA;

        let mut rgb = Image::from_size_val(image.size(), 0.0)?;

        super::rgb_from_yuv(&image, &mut rgb)?;

        assert_eq!(rgb.num_channels(), 3);
        assert_eq!(rgb.size(), image.size());

        for (a, b) in rgb.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).pow(2) < 1e-6f32);
        }
        Ok(())
    }

    #[test]
    fn yuv_inverse_relation_test() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            RGB_TEST_DATA.to_vec(),
        )?;

        let mut yuv = Image::from_size_val(image.size(), 0.0)?;
        let mut rgb = Image::from_size_val(image.size(), 0.0)?;

        super::yuv_from_rgb(&image, &mut yuv)?;
        super::rgb_from_yuv(&yuv, &mut rgb)?;

        for (a, b) in rgb.as_slice().iter().zip(image.as_slice().iter()) {
            assert!((a - b).pow(2) < 1e-1f32);
        }
        Ok(())
    }

    #[test]
    fn yuv_utils_rs_comparison() -> Result<(), ImageError> {
        use yuvutils_rs::{
            rgb_to_yuv444, YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange,
            YuvStandardMatrix,
        };
        let rgb_image = Image::<f32, 3>::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            RGB_TEST_DATA.to_vec(),
        )?;
        let mut yuv = Image::from_size_val(rgb_image.size(), 0.0)?;
        super::yuv_from_rgb(&rgb_image, &mut yuv)?;
        super::parallel::par_iter_rows(&yuv.clone(), &mut yuv, |src_p, dst_p| {
            dst_p[0] = src_p[0] * 255.0;
            dst_p[1] = (src_p[1] + 0.436) * ((255.0) / (0.436 * 2.0));
            dst_p[2] = (src_p[2] + 0.615) * ((255.0) / (0.615 * 2.0));
        });
        // Create an RGB buffer for yuv
        let rgb_data = RGB_TEST_DATA.iter().map(|x| *x as u8).collect::<Vec<u8>>();
        let mut planar_image_444 = YuvPlanarImageMut::<u8>::alloc(
            WIDTH as u32,
            HEIGHT as u32,
            YuvChromaSubsampling::Yuv444,
        );
        rgb_to_yuv444(
            &mut planar_image_444,
            &rgb_data.as_slice(),
            WIDTH as u32 * 3,
            YuvRange::Full,
            YuvStandardMatrix::Bt601,
            YuvConversionMode::Balanced,
        )
        .unwrap();
        // construct valid representation with y,u,v planes
        let yuv_utils_rs_data = planar_image_444
            .y_plane
            .borrow()
            .iter()
            .zip(planar_image_444.u_plane.borrow().iter())
            .zip(planar_image_444.v_plane.borrow().iter())
            .flat_map(|((y, u), v)| vec![*y, *u, *v])
            .collect::<Vec<u8>>();
        for (&a, &b) in yuv.as_slice().iter().zip(yuv_utils_rs_data.iter()) {
            let b = b as f32;
            assert!((a - b).pow(2) <= 4e-1f32);
        }
        Ok(())
    }
}
