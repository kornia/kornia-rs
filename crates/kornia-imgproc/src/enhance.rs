use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use num_traits::{Float, FromPrimitive};

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
/// This function follows the logic from the Python kornia implementation:
/// dst(x,y,c) = src(x,y,c) + factor
///
/// The result is clamped to the range [0.0, 1.0] if clip_output is true.
///
/// # Arguments
///
/// * `src` - The first input image.
/// * `factor` - The brightness factor to add to each pixel.
/// * `dst` - The output image to store the result.
/// * `clip_output` - Whether to clamp the output to the [0.0, 1.0] range.
///
/// # Returns
///
/// Returns Ok(()) if the operation is successful.
///
/// # Errors
///
/// Returns an error if the sizes of `src` and `dst` do not match.
pub fn adjust_brightness<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    factor: T,
    dst: &mut Image<T, C, A2>,
    clip_output: bool,
) -> Result<(), ImageError>
where
    T: Float + FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.width(),
            src.height(),
            dst.width(),
            dst.height(),
        ));
    }

    // This is the Rust translation of the logic:
    // img_adjust = image + factor
    // if clip_output:
    //     img_adjust = img_adjust.clamp(min=0.0, max=1.0)

    parallel::par_iter_rows_val(src, dst, |&src_pixel, dst_pixel| {
        let val = src_pixel + factor;
        if clip_output {
            // T::zero() is 0.0, T::one() is 1.0
            *dst_pixel = val.clamp(T::zero(), T::one());
        } else {
            *dst_pixel = val;
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

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

    #[test]
    fn test_adjust_brightness() -> Result<(), ImageError> {
        // Create a 2x1 image with 0.5f32 values
        let src_data = vec![0.5f32, 0.5];
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            src_data,
            CpuAllocator,
        )?;

        // Create an empty destination image
        let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;

        // 1. Test normal brightening (0.5 + 0.2 = 0.7)
        let factor1 = 0.2f32;
        let expected1 = [0.7f32, 0.7];
        super::adjust_brightness(&src, factor1, &mut dst, true)?;

        dst.as_slice()
            .iter()
            .zip(expected1.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        // 2. Test clamping (0.5 + 0.8 = 1.3, should clamp to 1.0)
        let factor2 = 0.8f32;
        let expected2 = [1.0f32, 1.0];
        super::adjust_brightness(&src, factor2, &mut dst, true)?;

        dst.as_slice()
            .iter()
            .zip(expected2.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        // 3. Test no-clip (0.5 + 0.8 = 1.3, should be 1.3)
        let expected3 = [1.3f32, 1.3];
        super::adjust_brightness(&src, factor2, &mut dst, false)?; // Same factor, clip_output=false

        dst.as_slice()
            .iter()
            .zip(expected3.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }
}
