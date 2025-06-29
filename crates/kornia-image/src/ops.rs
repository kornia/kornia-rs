use crate::{allocator::ImageAllocator, Image, ImageError};

/// Cast the pixel data of an image to a different type.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image.
/// * `scale` - The scale to multiply the pixel data with.
///
/// Example:
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_image::ops::cast_and_scale;
///
/// let image = Image::<u8, 1, _>::new(
///  ImageSize {
///   width: 2,
///  height: 1,
/// },
/// vec![0u8, 255],
/// CpuAllocator
/// ).unwrap();
///
/// let mut image_f32 = Image::from_size_val(image.size(), 0.0f32, CpuAllocator).unwrap();
///
/// cast_and_scale(&image, &mut image_f32, 1. / 255.0).unwrap();
///
/// assert_eq!(image_f32.get_pixel(0, 0, 0).unwrap(), &0.0f32);
/// assert_eq!(image_f32.get_pixel(1, 0, 0).unwrap(), &1.0f32);
/// ```
pub fn cast_and_scale<T, U, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<U, C, A2>,
    scale: U,
) -> Result<(), ImageError>
where
    T: Copy + num_traits::NumCast,
    U: Copy + num_traits::NumCast + std::ops::Mul<U, Output = U>,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.width(),
            src.height(),
            dst.width(),
            dst.height(),
        ));
    }

    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice().iter())
        .try_for_each(|(out, &inp)| {
            let x = U::from(inp).ok_or(ImageError::CastError(
                std::any::type_name::<U>().to_string(),
            ))?;
            *out = x * scale;
            Ok::<(), ImageError>(())
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageSize;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_cast_and_scale() -> Result<(), ImageError> {
        let image = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0u8, 0, 255, 0, 0, 255],
            CpuAllocator,
        )?;

        let mut image_f64: Image<f64, 3, CpuAllocator> =
            Image::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::cast_and_scale(&image, &mut image_f64, 1. / 255.0)?;

        let expected = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];

        assert_eq!(image_f64.as_slice(), expected);

        Ok(())
    }
}
