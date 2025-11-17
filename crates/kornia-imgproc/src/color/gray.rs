use crate::parallel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Define the RGB weights for the grayscale conversion.
const RW: f64 = 0.299;
const GW: f64 = 0.587;
const BW: f64 = 0.114;

/// Convert an RGB image to grayscale using the formula:
///
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::gray_from_rgb;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut gray = Image::<f32, 1, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// gray_from_rgb(&image, &mut gray).unwrap();
/// assert_eq!(gray.num_channels(), 1);
/// assert_eq!(gray.size().width, 4);
/// assert_eq!(gray.size().height, 5);
/// ```
pub fn gray_from_rgb<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 3, A1>,
    dst: &mut Image<T, 1, A2>,
) -> Result<(), ImageError>
where
    T: Send + Sync + num_traits::Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let rw = T::from(RW).ok_or(ImageError::CastError)?;
    let gw = T::from(GW).ok_or(ImageError::CastError)?;
    let bw = T::from(BW).ok_or(ImageError::CastError)?;

    // parallelize the grayscale conversion by rows
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let r = src_pixel[0];
        let g = src_pixel[1];
        let b = src_pixel[2];
        dst_pixel[0] = rw * r + gw * g + bw * b;
    });

    Ok(())
}

/// Convert an RGB8 image to grayscale using the formula:
///
/// Y = 77 * R + 150 * G + 29 * B
///
/// # Arguments
///
/// * `src` - The input RGB8 image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
pub fn gray_from_rgb_u8<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 1, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let r = src_pixel[0] as u16;
        let g = src_pixel[1] as u16;
        let b = src_pixel[2] as u16;
        dst_pixel[0] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
    });

    Ok(())
}

/// Convert a grayscale image to an RGB image by replicating the grayscale value across all three channels.
///
/// # Arguments
///
/// * `src` - The input grayscale image.
/// * `dst` - The output RGB image.
///
/// Precondition: the input image must have 1 channel.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::rgb_from_gray;
///
/// let image = Image::<f32, 1, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 1],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut rgb = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// rgb_from_gray(&image, &mut rgb).unwrap();
/// ```
pub fn rgb_from_gray<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 1, A1>,
    dst: &mut Image<T, 3, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // parallelize the grayscale conversion by rows
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[0];
        dst_pixel[2] = src_pixel[0];
    });

    Ok(())
}

/// Convert an RGB image to BGR by swapping the red and blue channels.
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output BGR image.
///
/// Precondition: the input and output images must have the same size.
pub fn bgr_from_rgb<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 3, A1>,
    dst: &mut Image<T, 3, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        dst_pixel
            .iter_mut()
            .zip(src_pixel.iter().rev())
            .for_each(|(d, s)| {
                *d = *s;
            });
    });

    Ok(())
}

/// Convert a BGR image to grayscale using the formula:
///
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// Note: The input is in BGR order, but the formula is applied to RGB values.
///
/// # Arguments
///
/// * `src` - The input BGR image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::gray_from_bgr;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut gray = Image::<f32, 1, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// gray_from_bgr(&image, &mut gray).unwrap();
/// assert_eq!(gray.num_channels(), 1);
/// assert_eq!(gray.size().width, 4);
/// assert_eq!(gray.size().height, 5);
/// ```
pub fn gray_from_bgr<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 3, A1>,
    dst: &mut Image<T, 1, A2>,
) -> Result<(), ImageError>
where
    T: Send + Sync + num_traits::Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let rw = T::from(RW).ok_or(ImageError::CastError)?;
    let gw = T::from(GW).ok_or(ImageError::CastError)?;
    let bw = T::from(BW).ok_or(ImageError::CastError)?;

    // parallelize the grayscale conversion by rows
    // Note: BGR order means src_pixel[2] is Red, src_pixel[1] is Green, src_pixel[0] is Blue
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let b = src_pixel[0];
        let g = src_pixel[1];
        let r = src_pixel[2];
        dst_pixel[0] = rw * r + gw * g + bw * b;
    });

    Ok(())
}

/// Convert a BGR8 image to grayscale using the formula:
///
/// Y = 77 * R + 150 * G + 29 * B
///
/// Note: The input is in BGR order, but the formula is applied to RGB values.
///
/// # Arguments
///
/// * `src` - The input BGR8 image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::gray_from_bgr_u8;
///
/// let image = Image::<u8, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0, 128, 255, 128, 0, 128, 0, 0, 0, 255, 255, 255],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut gray = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// gray_from_bgr_u8(&image, &mut gray).unwrap();
/// ```
pub fn gray_from_bgr_u8<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 1, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let b = src_pixel[0] as u16;
        let g = src_pixel[1] as u16;
        let r = src_pixel[2] as u16;
        dst_pixel[0] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{ops, Image, ImageSize};
    use kornia_io::functional as F;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn gray_from_rgb() -> Result<(), Box<dyn std::error::Error>> {
        let image = F::read_image_any_rgb8("../../tests/data/dog.jpeg")?;

        let mut image_norm = Image::from_size_val(image.size(), 0.0, CpuAllocator)?;
        ops::cast_and_scale(&image, &mut image_norm, 1. / 255.0)?;

        let mut gray = Image::<f32, 1, _>::from_size_val(image_norm.size(), 0.0, CpuAllocator)?;
        super::gray_from_rgb(&image_norm, &mut gray)?;

        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.cols(), 258);
        assert_eq!(gray.rows(), 195);

        Ok(())
    }

    #[test]
    fn gray_from_rgb_regression() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let image = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut gray = Image::<f32, 1, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::gray_from_rgb(&image, &mut gray)?;

        let expected: Image<f32, 1, _> = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
            CpuAllocator,
        )?;

        for (a, b) in gray.as_slice().iter().zip(expected.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn rgb_from_grayscale() -> Result<(), Box<dyn std::error::Error>> {
        let image = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let mut rgb = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::rgb_from_gray(&image, &mut rgb)?;

        #[rustfmt::skip]
        let expected: Image<f32, 3, _> = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 0.0, 0.0,
                1.0, 1.0, 1.0,
                2.0, 2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0,
                5.0, 5.0, 5.0,
            ],
            CpuAllocator
        )?;

        assert_eq!(rgb.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn bgr_from_rgb() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 3,
            },
            vec![
                0.0, 1.0, 2.0,
                3.0, 4.0, 5.0,
                6.0, 7.0, 8.0,
            ],
            CpuAllocator
        )?;

        let mut bgr = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::bgr_from_rgb(&image, &mut bgr)?;

        #[rustfmt::skip]
        let expected: Image<f32, 3, _> = Image::new(
            ImageSize {
                width: 1,
                height: 3,
            },
            vec![
                2.0, 1.0, 0.0,
                5.0, 4.0, 3.0,
                8.0, 7.0, 6.0,
            ],
            CpuAllocator
        )?;

        assert_eq!(bgr.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn gray_from_rgb_u8() -> Result<(), Box<dyn std::error::Error>> {
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![0, 128, 255, 128, 0, 128],
            CpuAllocator,
        )?;

        let mut gray = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::gray_from_rgb_u8(&image, &mut gray)?;

        assert_eq!(gray.as_slice(), &[103, 53]);

        Ok(())
    }

    #[test]
    fn gray_from_bgr_regression() -> Result<(), Box<dyn std::error::Error>> {
        // Create a BGR image (note the channel order is reversed compared to RGB)
        #[rustfmt::skip]
        let image = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 0.0, 1.0,  // Blue=0, Green=0, Red=1
                0.0, 1.0, 0.0,  // Blue=0, Green=1, Red=0
                1.0, 0.0, 0.0,  // Blue=1, Green=0, Red=0
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut gray = Image::<f32, 1, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::gray_from_bgr(&image, &mut gray)?;

        let expected: Image<f32, 1, _> = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
            CpuAllocator,
        )?;

        for (a, b) in gray.as_slice().iter().zip(expected.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn gray_from_rgb_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        // Test with pure white
        let white = Image::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![1.0, 1.0, 1.0],
            CpuAllocator,
        )?;
        let mut gray = Image::<f32, 1, _>::from_size_val(white.size(), 0.0, CpuAllocator)?;
        super::gray_from_rgb(&white, &mut gray)?;
        assert!((gray.as_slice()[0] - 1.0).abs() < 1e-6);

        // Test with pure black
        let black = Image::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![0.0, 0.0, 0.0],
            CpuAllocator,
        )?;
        let mut gray = Image::<f32, 1, _>::from_size_val(black.size(), 0.0, CpuAllocator)?;
        super::gray_from_rgb(&black, &mut gray)?;
        assert!(gray.as_slice()[0].abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn gray_from_rgb_u8_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        // Test with max values
        let white = Image::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255u8, 255, 255],
            CpuAllocator,
        )?;
        let mut gray = Image::<u8, 1, _>::from_size_val(white.size(), 0, CpuAllocator)?;
        super::gray_from_rgb_u8(&white, &mut gray)?;
        assert_eq!(gray.as_slice()[0], 255);

        // Test with min values
        let black = Image::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![0u8, 0, 0],
            CpuAllocator,
        )?;
        let mut gray = Image::<u8, 1, _>::from_size_val(black.size(), 0, CpuAllocator)?;
        super::gray_from_rgb_u8(&black, &mut gray)?;
        assert_eq!(gray.as_slice()[0], 0);

        Ok(())
    }

    #[test]
    fn gray_from_bgr_u8() -> Result<(), Box<dyn std::error::Error>> {
        // Test with BGR image (channels are swapped compared to RGB)
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![255, 128, 0, 128, 0, 128], // BGR order
            CpuAllocator,
        )?;

        let mut gray = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::gray_from_bgr_u8(&image, &mut gray)?;

        // Verify the values (same as gray_from_rgb_u8 test but with BGR input)
        assert_eq!(gray.as_slice(), &[103, 53]);

        Ok(())
    }
}
