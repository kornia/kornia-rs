use crate::image::{Image, ImageSize};
use anyhow::Result;
use fast_image_resize as fr;
use ndarray::{stack, Array2, Array3};
use std::num::NonZeroU32;

/// Create a meshgrid of x and y coordinates
///
/// # Arguments
///
/// * `x` - A 1D array of x coordinates
/// * `y` - A 1D array of y coordinates
///
/// # Returns
///
/// A tuple of 2D arrays of shape (height, width) containing the x and y coordinates
///
/// # Example
///
/// ```
/// let x = ndarray::Array::linspace(0., 4., 5).insert_axis(ndarray::Axis(0));
/// let y = ndarray::Array::linspace(0., 3., 4).insert_axis(ndarray::Axis(0));
/// let (xx, yy) = kornia_rs::resize::meshgrid(&x, &y);
/// assert_eq!(xx.shape(), &[4, 5]);
/// assert_eq!(yy.shape(), &[4, 5]);
/// assert_eq!(xx[[0, 0]], 0.);
/// assert_eq!(xx[[0, 4]], 4.);
/// ```
pub fn meshgrid(x: &Array2<f32>, y: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    // create the meshgrid of x and y coordinates
    let nx = x.len_of(ndarray::Axis(1));
    let ny = y.len_of(ndarray::Axis(1));

    // broadcast the x and y coordinates to create a 2D grid, and then transpose the y coordinates
    // to create the meshgrid of x and y coordinates of shape (height, width)
    let xx = x.broadcast((ny, nx)).unwrap().to_owned();
    let yy = y.broadcast((nx, ny)).unwrap().t().to_owned();

    (xx, yy)
}

/// Kernel for bilinear interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel value.
// TODO: add support for other data types. Maybe use a trait? or template?
fn bilinear_interpolation(image: &Array3<f32>, u: f32, v: f32, c: usize) -> f32 {
    let (height, width, _) = image.dim();
    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let frac_u = u.fract();
    let frac_v = v.fract();
    let val00 = image[[iv, iu, c]];
    let val01 = if iu + 1 < width {
        image[[iv, iu + 1, c]]
    } else {
        val00
    };
    let val10 = if iv + 1 < height {
        image[[iv + 1, iu, c]]
    } else {
        val00
    };
    let val11 = if iu + 1 < width && iv + 1 < height {
        image[[iv + 1, iu + 1, c]]
    } else {
        val00
    };

    let frac_uu = 1. - frac_u;
    let frac_vv = 1. - frac_v;

    val00 * frac_uu * frac_vv
        + val01 * frac_u * frac_vv
        + val10 * frac_uu * frac_v
        + val11 * frac_u * frac_v
}

/// Kernel for nearest neighbor interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel value.
fn nearest_neighbor_interpolation(image: &Array3<f32>, u: f32, v: f32, c: usize) -> f32 {
    let (height, width, _) = image.dim();

    let iu = u.round() as usize;
    let iv = v.round() as usize;

    let iu = iu.clamp(0, width - 1);
    let iv = iv.clamp(0, height - 1);

    image[[iv, iu, c]]
}

/// Interpolation mode for the resize operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    Bilinear,
    Nearest,
}

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
/// use kornia_rs::image::{Image, ImageSize};
/// let image = Image::<_, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
/// let image_resized: Image<f32, 3> = kornia_rs::resize::resize_native(
///     &image,
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     kornia_rs::resize::InterpolationMode::Nearest,
/// )
/// .unwrap();
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_native<const CHANNELS: usize>(
    image: &Image<f32, CHANNELS>,
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<Image<f32, CHANNELS>> {
    // create the output image
    let mut output = Image::from_size_val(new_size, 0.0)?;

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
            let pixels = (0..image.num_channels()).map(|k| match interpolation {
                InterpolationMode::Bilinear => bilinear_interpolation(&image.data, u, v, k),
                InterpolationMode::Nearest => nearest_neighbor_interpolation(&image.data, u, v, k),
            });

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
/// use kornia_rs::image::{Image, ImageSize};
/// let image = Image::<_, 3>::new(
///    ImageSize {
///       width: 4,
///      height: 5,
/// },
/// vec![0u8; 4 * 5 * 3],
/// )
/// .unwrap();
/// let image_resized: Image<u8, 3> = kornia_rs::resize::resize_fast(
///   &image,
///  ImageSize {
///    width: 2,
///   height: 3,
/// },
/// kornia_rs::resize::InterpolationMode::Nearest,
/// )
/// .unwrap();
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
) -> Result<Image<u8, 3>> {
    let src_width = NonZeroU32::new(image.width() as u32).unwrap();
    let src_height = NonZeroU32::new(image.height() as u32).unwrap();

    // TODO: pass as slice
    let src_image = fr::Image::from_vec_u8(
        src_width,
        src_height,
        image.data.as_slice().unwrap().to_vec(),
        fr::PixelType::U8x3,
    )?;

    let dst_width = NonZeroU32::new(new_size.width as u32).unwrap();
    let dst_height = NonZeroU32::new(new_size.height as u32).unwrap();

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
    resizer.resize(&src_image.view(), &mut dst_view)?;

    // TODO: create a new image from the buffer directly from a slice
    Image::new(new_size, dst_image.buffer().to_vec())
}

#[cfg(test)]
mod tests {

    #[test]
    fn resize_smoke_ch3() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5 * 3],
        )
        .unwrap();
        let image_resized = super::resize_native(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Bilinear,
        )
        .unwrap();
        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
    }

    #[test]
    fn resize_smoke_ch1() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5],
        )
        .unwrap();
        let image_resized = super::resize_native(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Nearest,
        )
        .unwrap();
        assert_eq!(image_resized.num_channels(), 1);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
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
    fn resize_fast() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0u8; 4 * 5 * 3],
        )
        .unwrap();
        let image_resized = super::resize_fast(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Nearest,
        )
        .unwrap();
        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
    }
}
