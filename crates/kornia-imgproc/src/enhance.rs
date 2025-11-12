use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use num_traits::Float;

use crate::parallel;

/// Adjust the brightness of an image.
///
/// dst(x,y,c) = src(x,y,c) + factor
///
/// The result is clamped to the range [0.0, 1.0] if clip_output is true.
///
/// # Arguments
///
/// * `src` - The first input image.
/// * `dst` - The output image to store the result.
/// * `factor` - The brightness factor to add to each pixel.
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
    T: Float + std::fmt::Debug + Send + Sync + Copy,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.width(),
            src.height(),
            dst.width(),
            dst.height(),
        ));
    }

    // Manual clamping to support generic T: Float (no inherent clamp method on T)
    parallel::par_iter_rows_val(src, dst, |&src_pixel, dst_pixel| {
        let val = src_pixel + factor;
        if clip_output {
            let zero = T::zero();
            let one = T::one();
            let clamped = if val < zero {
                zero
            } else if val > one {
                one
            } else {
                val
            };
            *dst_pixel = clamped;
        } else {
            *dst_pixel = val;
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};

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
}
