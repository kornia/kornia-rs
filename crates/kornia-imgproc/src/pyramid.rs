use crate::filter::separable_filter;
use crate::interpolation::InterpolationMode;
use crate::resize::resize_native;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use kornia_tensor::CpuAllocator;

fn get_pyramid_gaussian_kernel() -> (Vec<f32>, Vec<f32>) {
    // The 2D kernel is:
    // [
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [6.0, 24.0, 36.0, 24.0, 6.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    // ] / 256.0

    let kernel_x = [1.0, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|&x| x / 16.0)
        .collect();
    let kernel_y = [1.0, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|&x| x / 16.0)
        .collect();

    (kernel_x, kernel_y)
}

/// Upsample an image and then blur it.
///
/// This function doubles the size of the input image using bilinear interpolation
/// and then applies a Gaussian blur to smooth the result.
///
/// # Arguments
///
/// * `src` - The source image to be upsampled.
/// * `dst` - The destination image to store the result.
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::pyramid::pyrup;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
///     CpuAllocator
/// ).unwrap();
///
/// let mut upsampled = Image::<f32, 3, _>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     0.0,
///     CpuAllocator
/// ).unwrap();
///
/// pyrup(&image, &mut upsampled).unwrap();
/// ```
pub fn pyrup<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
) -> Result<(), ImageError> {
    let expected_width = src.width() * 2;
    let expected_height = src.height() * 2;

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    let mut upsampled = Image::<f32, C, _>::from_size_val(dst.size(), 0.0, CpuAllocator)?;

    resize_native(src, &mut upsampled, InterpolationMode::Bilinear)?;

    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    separable_filter(&upsampled, dst, &kernel_x, &kernel_y)?;

    Ok(())
}

/// Downsample an image by applying Gaussian blur and then subsampling.
///
/// This function halves the size of the input image by first applying a Gaussian blur
/// and then subsampling every other pixel. This is the inverse operation of [`pyrup`].
///
/// # Arguments
///
/// * `src` - The source image to be downsampled.
/// * `dst` - The destination image to store the result (should be half the size of src).
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::pyramid::pyrdown;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
///          12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
///          24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
///          36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
/// ).unwrap();
///
/// let mut downsampled = Image::<f32, 3>::from_size_val(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     0.0,
/// ).unwrap();
///
/// pyrdown(&image, &mut downsampled).unwrap();
/// ```
pub fn pyrdown<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    let expected_width = (src.width() + 1) / 2;
    let expected_height = (src.height() + 1) / 2;

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    // First apply Gaussian blur
    let mut blurred = Image::<f32, C>::from_size_val(src.size(), 0.0)?;
    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    separable_filter(src, &mut blurred, &kernel_x, &kernel_y)?;

    // Then downsample using bilinear interpolation
    resize_native(&blurred, dst, InterpolationMode::Bilinear)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_pyrup() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 1.0, 2.0, 3.0],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrup(&src, &mut dst)?;

        assert_eq!(dst.width(), 4);
        assert_eq!(dst.height(), 4);

        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0,
            ],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;

        pyrdown(&src, &mut dst)?;

        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);

        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_3c() -> Result<(), ImageError> {
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..48).map(|x| x as f32).collect(),
        )?;

        let mut dst = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;

        pyrdown(&src, &mut dst)?;

        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);

        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_invalid_size() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0.0; 16],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0.0,
        )?;

        let result = pyrdown(&src, &mut dst);
        assert!(result.is_err());

        Ok(())
    }
}
