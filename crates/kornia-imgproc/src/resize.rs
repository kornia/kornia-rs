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
/// * `image` - The input image container.
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
/// let image_resized: Image<f32, 3> = resize_native(
///     &image,
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_native<T: ImageDtype, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<Image<T, CHANNELS>, ImageError> {
    // create the output image
    let mut output = Image::from_size_val(new_size, T::default())?;

    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let x = ndarray::Array::linspace(0., (image.width() - 1) as f32, new_size.width)
        .insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::linspace(0., (image.height() - 1) as f32, new_size.height)
        .insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = meshgrid(&x, &y);

    // TODO: benchmark this
    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = stack![ndarray::Axis(2), xx, yy];

    // iterate over the output image and interpolate the pixel values

    ndarray::Zip::from(xy.rows())
        .and(output.data.rows_mut())
        .par_for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // compute the pixel values for each channel
            let pixels = (0..image.num_channels())
                .map(|k| interpolate_pixel(&image.data, u, v, k, interpolation));

            // write the pixel values to the output image
            for (k, pixel) in pixels.enumerate() {
                out[k] = pixel;
            }
        });

    Ok(output)
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
/// let image_resized: Image<u8, 3> = resize_fast(
///   &image,
///   ImageSize {
///     width: 2,
///     height: 3,
///   },
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
    image: &Image<u8, 3>,
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<Image<u8, 3>, ImageError> {
    // cast the image dimensions to u32
    let src_width = NonZeroU32::new(image.width() as u32).ok_or(ImageError::CastError)?;
    let src_height = NonZeroU32::new(image.height() as u32).ok_or(ImageError::CastError)?;

    let image_data = image.data.to_owned().into_raw_vec();
    let image_data_len = image_data.len();

    let src_image = fr::Image::from_vec_u8(src_width, src_height, image_data, fr::PixelType::U8x3)
        .map_err(|_| {
            ImageError::InvalidChannelShape(image_data_len, image.width() * image.height() * 3)
        })?;

    let dst_width = NonZeroU32::new(new_size.width as u32).ok_or(ImageError::CastError)?;
    let dst_height = NonZeroU32::new(new_size.height as u32).ok_or(ImageError::CastError)?;

    let mut dst_image = fr::Image::new(dst_width, dst_height, src_image.pixel_type());
    let mut dst_view = dst_image.view_mut();

    let mut resizer = {
        match interpolation {
            InterpolationMode::Bilinear => {
                fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear))
            }
            InterpolationMode::Nearest => fr::Resizer::new(fr::ResizeAlg::Nearest),
        }
    };

    resizer
        .resize(&src_image.view(), &mut dst_view)
        .map_err(|_| ImageError::IncompatiblePixelTypes)?;

    // TODO: create a new image from the buffer directly from a slice
    Ok(Image::new(new_size, dst_image.buffer().to_vec()).expect("Failed to create image"))
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
        let image_resized = super::resize_native(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
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
        let image_resized = super::resize_native(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
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
        let image_resized = super::resize_fast(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Nearest,
        )?;
        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }
}
