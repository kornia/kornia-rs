use crate::{Image, ImageError};

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
/// use kornia_image::ops::cast_and_scale;
///
/// let image = Image::<u8, 1>::new(
///  ImageSize {
///   width: 2,
///  height: 1,
/// },
/// vec![0u8, 255],
/// ).unwrap();
///
/// let mut image_f32 = Image::from_size_val(image.size(), 0.0f32).unwrap();
///
/// cast_and_scale(&image, &mut image_f32, 1. / 255.0).unwrap();
///
/// assert_eq!(image_f32.get_pixel(0, 0, 0).unwrap(), &0.0f32);
/// assert_eq!(image_f32.get_pixel(1, 0, 0).unwrap(), &1.0f32);
/// ```
pub fn cast_and_scale<T, U, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<U, C>,
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
            let x = U::from(inp).ok_or(ImageError::CastError)?;
            *out = x * scale;
            Ok::<(), ImageError>(())
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageSize;

    #[test]
    fn test_cast_and_scale() -> Result<(), ImageError> {
        let image = Image::<u8, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0u8, 0, 255, 0, 0, 255],
        )?;

        let mut image_f64: Image<f64, 3> = Image::from_size_val(image.size(), 0.0)?;

        super::cast_and_scale(&image, &mut image_f64, 1. / 255.0)?;

        let expected = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];

        assert_eq!(image_f64.as_slice(), expected);

        Ok(())
    }
}
