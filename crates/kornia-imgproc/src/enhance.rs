use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use num_traits::Float;

use crate::parallel;

/// Performs weighted addition of two images `src1` and `src2` with weights `alpha`
/// and `beta`, and an optional scalar `gamma`. The formula used is:
///
/// dst(x,y,c) = (src1(x,y,c) * alpha + src2(x,y,c) * beta + gamma)
///
/// # Arguments
///
/// * `src1` - The first input image.
/// * `alpha` - Weight of the first image elements to be multiplied.
/// * `src2` - The second input image.
/// * `beta` - Weight of the second image elements to be multiplied.
/// * `gamma` - Scalar added to each sum.
///
/// # Returns
///
/// Returns a new `Image` where each element is computed as described above.
///
/// # Errors
///
/// Returns an error if the sizes of `src1` and `src2` do not match.
/// Returns an error if the size of `dst` does not match the size of `src1` or `src2`.
pub fn add_weighted<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator, A3: ImageAllocator>(
    src1: &Image<T, C, A1>,
    alpha: T,
    src2: &Image<T, C, A2>,
    beta: T,
    gamma: T,
    dst: &mut Image<T, C, A3>,
) -> Result<(), ImageError>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    if src1.size() != src2.size() {
        return Err(ImageError::InvalidImageSize(
            src1.cols(),
            src1.rows(),
            src2.cols(),
            src2.rows(),
        ));
    }

    if src1.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src1.width(),
            src1.height(),
            dst.width(),
            dst.height(),
        ));
    }

    // compute the weighted sum
    parallel::par_iter_rows_val_two(src1, src2, dst, |&src1_pixel, &src2_pixel, dst_pixel| {
        *dst_pixel = (src1_pixel * alpha) + (src2_pixel * beta) + gamma;
    });

    Ok(())
}

/// Adjust the brightness of an image.
///
/// dst(x,y,c) = src(x,y,c) + factor
///
/// The result is clamped to the range [0.0, 1.0] if clip_output is true.
///
/// # Arguments
///
/// * `src` - The input image.
/// * `dst` - The output image to store the result.
/// * `factor` - The brightness factor to add. Can be negative (decrease) or positive (increase).
/// * `clip_output` - Whether to clamp the output to the [0.0, 1.0] range.
///
/// # Returns
///
/// Returns Ok(()) if the operation is successful.
///
/// # Errors
///
/// Returns an [ImageError::InvalidImageSize] if the sizes of `src` and `dst` do not match.
pub fn adjust_brightness<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    factor: T,
    clip_output: bool,
) -> Result<(), ImageError>
where
    // Removed FromPrimitive, as it's not needed
    T: Float + std::fmt::Debug + Send + Sync + Copy,
{
    if src.size() != dst.size() {
        // FIX: Use cols() and rows() for consistency
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // FIX: Moved the `if clip_output` check *outside* the loop for performance.
    if clip_output {
        let zero = T::zero();
        let one = T::one();
        parallel::par_iter_rows_val(src, dst, |&src_pixel, dst_pixel| {
            let val = src_pixel + factor;
            // Manual clamp logic
            *dst_pixel = if val < zero {
                zero
            } else if val > one {
                one
            } else {
                val
            };
        });
    } else {
        parallel::par_iter_rows_val(src, dst, |&src_pixel, dst_pixel| {
            *dst_pixel = src_pixel + factor;
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // FIX: Add missing imports for tests
    use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};

    #[test]
    fn test_add_weighted() -> Result<(), ImageError> {
        let src1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let src1 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src1_data,
            CpuAllocator,
        )?;
        let src2_data = vec![4.0f32, 5.0, 6.0, 7.0];
        let src2 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src2_data,
            CpuAllocator,
        )?;
        let alpha = 2.0f32;
        let beta = 2.0f32;
        let gamma = 1.0f32;
        let expected = [11.0, 15.0, 19.0, 23.0];

        let mut weighted = Image::<f32, 1, _>::from_size_val(src1.size(), 0.0, CpuAllocator)?;

        super::add_weighted(&src1, alpha, &src2, beta, gamma, &mut weighted)?;

        weighted
            .as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }

    // Helper function to create a base image for tests
    fn create_test_image(
    ) -> Result<(Image<f32, 1, CpuAllocator>, Image<f32, 1, CpuAllocator>), ImageError> {
        let src_data = vec![0.5f32, 0.5];
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            src_data,
            CpuAllocator,
        )?;
        let dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        Ok((src, dst))
    }

    #[test]
    fn test_adjust_brightness_normal() -> Result<(), ImageError> {
        let (src, mut dst) = create_test_image()?;
        let factor = 0.2f32;
        let expected = [0.7f32, 0.7];
        super::adjust_brightness(&src, &mut dst, factor, true)?;

        dst.as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
        Ok(())
    }

    #[test]
    fn test_adjust_brightness_clamping() -> Result<(), ImageError> {
        let (src, mut dst) = create_test_image()?;
        let factor = 0.8f32;
        let expected = [1.0f32, 1.0];
        super::adjust_brightness(&src, &mut dst, factor, true)?;

        dst.as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
        Ok(())
    }

    #[test]
    fn test_adjust_brightness_no_clip() -> Result<(), ImageError> {
        let (src, mut dst) = create_test_image()?;
        let factor = 0.8f32;
        let expected = [1.3f32, 1.3];
        super::adjust_brightness(&src, &mut dst, factor, false)?;

        dst.as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
        Ok(())
    }

    #[test]
    fn test_adjust_brightness_negative_factor() -> Result<(), ImageError> {
        let (src, mut dst) = create_test_image()?;
        let factor = -0.3f32;
        let expected = [0.2f32, 0.2];
        super::adjust_brightness(&src, &mut dst, factor, true)?;

        dst.as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
        Ok(())
    }
}
