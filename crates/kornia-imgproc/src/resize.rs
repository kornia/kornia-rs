use crate::interpolation::{interpolate_pixel, meshgrid, InterpolationMode};
use fast_image_resize as fr;
use kornia_image::{Image, ImageDtype, ImageError, ImageSize};
use ndarray::stack;
use std::num::NonZeroU32;

/// Resize an image to a new size.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports any number of channels and data types.
///
/// # Arguments
///
/// * `src` - The input image container.
/// * `dst` - The output image container.
/// * `new_size` - The new size of the image.
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
///     new_size,
///     InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_native<T: ImageDtype, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if dst.size() != new_size {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let x = ndarray::Array::linspace(0., (src.width() - 1) as f32, new_size.width)
        .insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::linspace(0., (src.height() - 1) as f32, new_size.height)
        .insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = meshgrid(&x, &y);

    // TODO: benchmark this
    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = stack![ndarray::Axis(2), xx, yy];

    // iterate over the output image and interpolate the pixel values

    ndarray::Zip::from(xy.rows())
        .and(dst.data.rows_mut())
        .par_for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // compute the pixel values for each channel
            let pixels =
                (0..CHANNELS).map(|k| interpolate_pixel(&src.data, u, v, k, interpolation));

            // write the pixel values to the output image
            for (k, pixel) in pixels.enumerate() {
                out[k] = pixel;
            }
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
/// * `image` - The input image container with 3 channels.
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
///   new_size,
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
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if dst.size() != new_size {
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

    let src_image_data = src.data.to_owned().into_raw_vec();

    let src_image =
        fr::Image::from_vec_u8(src_width, src_height, src_image_data, fr::PixelType::U8x3)
            .map_err(|_| {
                ImageError::InvalidChannelShape(src.data.len(), src.width() * src.height() * 3)
            })?;

    // prepare the output image for the fast_image_resize crate
    let dst_width = NonZeroU32::new(new_size.width as u32).ok_or(ImageError::CastError)?;
    let dst_height = NonZeroU32::new(new_size.height as u32).ok_or(ImageError::CastError)?;

    let dst_data_len = dst_width.get() as usize * dst_height.get() as usize * 3;

    let mut dst_image = fr::Image::from_slice_u8(
        dst_width,
        dst_height,
        dst.data.as_slice_mut().expect("Failed to get image data"),
        fr::PixelType::U8x3,
    )
    .map_err(|_| {
        ImageError::InvalidChannelShape(dst_data_len, new_size.width * new_size.height * 3)
    })?;

    let mut resizer = {
        match interpolation {
            InterpolationMode::Bilinear => {
                fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear))
            }
            InterpolationMode::Nearest => fr::Resizer::new(fr::ResizeAlg::Nearest),
        }
    };

    resizer
        .resize(&src_image.view(), &mut dst_image.view_mut())
        .map_err(|_| ImageError::IncompatiblePixelTypes)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn resize_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5 * 3],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0.0)?;

        super::resize_native(
            &image,
            &mut image_resized,
            new_size,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }

    #[test]
    fn resize_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0; 4 * 5],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 1>::from_size_val(new_size, 0)?;

        super::resize_native(
            &image,
            &mut image_resized,
            new_size,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 1);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }

    #[test]
    fn meshgrid() {
        let x = ndarray::Array::linspace(0., 4., 5).insert_axis(ndarray::Axis(0));
        let y = ndarray::Array::linspace(0., 3., 4).insert_axis(ndarray::Axis(0));
        let (xx, yy) = super::meshgrid(&x, &y);
        assert_eq!(xx.shape(), &[4, 5]);
        assert_eq!(yy.shape(), &[4, 5]);
        assert_eq!(xx[[0, 0]], 0.);
        assert_eq!(xx[[0, 4]], 4.);
        assert_eq!(yy[[0, 0]], 0.);
        assert_eq!(yy[[3, 0]], 3.);
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
            new_size,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }
}
