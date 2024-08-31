use crate::{
    interpolation::{interpolate_pixel, meshgrid_image, InterpolationMode},
    parallel,
};
use fast_image_resize::{self as fr};
use kornia_image::{Image, ImageError};
use std::num::NonZeroU32;

/// Resize an image to a new size.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports any number of C and data types.
///
/// # Arguments
///
/// * `src` - The input image container.
/// * `dst` - The output image container.
/// * `optional_args` - Optional arguments for the resize operation.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::resize::resize_native;
/// use kornia::imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///     width: 2,
///     height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0.0).unwrap();
///
/// resize_native(
///     &image,
///     &mut image_resized,
///     InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_native<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError>
where
{
    // check if the input and output images have the same size
    // and copy the input image to the output image if they have the same size
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    //// create a grid of x and y coordinates for the output image
    //// and interpolate the values from the input image.
    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());
    let (map_x, map_y) = meshgrid_image(dst_rows, src.rows(), dst_cols, src.cols())?;

    // iterate over the output image and interpolate the pixel values
    parallel::par_iter_rows_resample(dst, &map_x, &map_y, |&x, &y, dst_pixel| {
        // interpolate the pixel values for each channel
        dst_pixel.iter_mut().enumerate().for_each(|(k, pixel)| {
            *pixel = interpolate_pixel(src, x, y, k, interpolation);
        });
    });

    Ok(())
}

/// Resize an image to a new size using the [fast_image_resize](https://crates.io/crates/fast_image_resize) crate.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports only 3-channel images and u8 data type.
///
/// # Arguments
///
/// * `image` - The input image container with 3 C.
/// * `new_size` - The new size of the image.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::resize::resize_fast;
/// use kornia::imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3>::new(
///    ImageSize {
///       width: 4,
///      height: 5,
/// },
/// vec![0u8; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///   width: 2,
///   height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0).unwrap();
///
/// resize_fast(
///   &image,
///   &mut image_resized,
///   InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
///
/// # Errors
///
/// The function returns an error if the image cannot be resized.
pub fn resize_fast(
    src: &Image<u8, 3>,
    dst: &mut Image<u8, 3>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if dst.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    // prepare the input image for the fast_image_resize crate
    let src_width = NonZeroU32::new(src.width() as u32).ok_or(ImageError::CastError)?;
    let src_height = NonZeroU32::new(src.height() as u32).ok_or(ImageError::CastError)?;

    let src_data_len = src_width.get() as usize * src_height.get() as usize * 3;

    let src_image = fr::ImageView::<fast_image_resize::pixels::U8x3>::from_buffer(
        src_width,
        src_height,
        src.as_slice(),
    )
    .map_err(|_| ImageError::InvalidChape(src_data_len, src.width() * src.height() * 3))?;

    // prepare the output image for the fast_image_resize crate
    let dst_width = NonZeroU32::new(dst.width() as u32).ok_or(ImageError::CastError)?;
    let dst_height = NonZeroU32::new(dst.height() as u32).ok_or(ImageError::CastError)?;

    let dst_data_len = dst_width.get() as usize * dst_height.get() as usize * 3;

    let mut dst_image = fr::Image::from_slice_u8(
        dst_width,
        dst_height,
        dst.as_slice_mut(),
        fr::PixelType::U8x3,
    )
    .map_err(|_| ImageError::InvalidChape(dst_data_len, dst_data_len))?;

    let mut resizer = {
        match interpolation {
            InterpolationMode::Bilinear => {
                fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear))
            }
            InterpolationMode::Nearest => fr::Resizer::new(fr::ResizeAlg::Nearest),
        }
    };

    resizer
        .resize(&src_image.into(), &mut dst_image.view_mut())
        .map_err(|_| ImageError::IncompatiblePixelTypes)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_core::TensorError;
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn resize_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            (0..3 * 4 * 3).map(|x| x as f32).collect::<Vec<f32>>(),
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0.0)?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        assert_eq!(
            image_resized.as_slice(),
            [
                0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5, 27.0, 28.0, 29.0,
                33.0, 34.0, 35.0
            ]
        );

        Ok(())
    }

    #[test]
    fn resize_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 1>::from_size_val(new_size, 0.0f32)?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 1);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        assert_eq!(image_resized.as_slice(), image_resized.as_slice());

        Ok(())
    }

    #[test]
    fn meshgrid() -> Result<(), TensorError> {
        let (map_x, map_y) = crate::interpolation::meshgrid(3, 2)?;

        assert_eq!(map_x.shape, [3, 2]);
        assert_eq!(map_y.shape, [3, 2]);
        assert_eq!(map_x.get([0, 0]), Some(&0.0));
        assert_eq!(map_x.get([0, 1]), Some(&1.0));
        assert_eq!(map_y.get([0, 0]), Some(&0.0));
        assert_eq!(map_y.get([2, 0]), Some(&2.0));

        Ok(())
    }

    #[test]
    fn resize_fast() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0u8; 4 * 5 * 3],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0)?;

        super::resize_fast(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }
}
