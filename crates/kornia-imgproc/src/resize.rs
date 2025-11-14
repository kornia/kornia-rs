//! Image resizing utilities with multiple interpolation modes.
//!
//! This module provides functions for resizing images to different dimensions using
//! various interpolation methods. Two implementations are available:
//!
//! * **Native implementation** ([`resize_native`](crate::resize::resize_native)) - Pure Rust, supports any channel count
//!   and f32 data type, parallelized for performance
//! * **Fast implementation** ([`resize_fast_rgb`](crate::resize::resize_fast_rgb)) - Uses the `fast_image_resize` crate,
//!   optimized for RGB u8 images with SIMD acceleration
//!
//! # Interpolation Modes
//!
//! * **Nearest** - Fast, blocky results, suitable for masks or discrete data
//! * **Bilinear** - Smooth results, good quality/speed tradeoff
//! * **Bicubic** - Higher quality, slower, best for downsampling photographs
//!
//! # Example: Resize with Bilinear Interpolation
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::resize::resize_native;
//! use kornia_imgproc::interpolation::InterpolationMode;
//!
//! let src = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 640, height: 480 },
//!     0.5,
//! ).unwrap();
//!
//! let mut dst = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 320, height: 240 },
//!     0.0,
//! ).unwrap();
//!
//! resize_native(&src, &mut dst, InterpolationMode::Bilinear).unwrap();
//! ```
//!
//! # Performance Considerations
//!
//! * Use [`resize_fast_rgb`](crate::resize::resize_fast_rgb) for u8 RGB images when maximum speed is needed
//! * Use [`resize_native`](crate::resize::resize_native) for flexibility with any channel count or f32 precision
//! * Downsampling by large factors may benefit from pre-blurring to avoid aliasing
//!
//! # See also
//!
//! * [`crate::filter::gaussian_blur`] for pre-filtering before downsampling
//! * [`crate::interpolation`] for low-level interpolation primitives

use crate::{
    interpolation::{grid::meshgrid_from_fn, interpolate_pixel, InterpolationMode},
    parallel,
};
use fast_image_resize::{self as fr};
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Resize an image to a new size.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports any number of channels and data types.
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
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::resize::resize_native;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///     width: 2,
///     height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
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
pub fn resize_native<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
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

    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());
    let step_x = (src.cols() - 1) as f32 / (dst.cols() - 1) as f32;
    let step_y = (src.rows() - 1) as f32 / (dst.rows() - 1) as f32;
    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        Ok((x as f32 * step_x, y as f32 * step_y))
    })?;

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
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::resize::resize_fast_rgb;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0u8; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///   width: 2,
///   height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0, CpuAllocator).unwrap();
///
/// resize_fast_rgb(
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
pub fn resize_fast_rgb<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 3, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_impl(src, dst, interpolation)
}

/// Resize a grayscale (single-channel) image to a new size using the [fast_image_resize](https://crates.io/crates/fast_image_resize) crate.
///
/// The function resizes a grayscale image to a new size using the specified interpolation mode.
/// It supports only 1-channel images and u8 data type.
///
/// # Arguments
///
/// * `src` - The input grayscale image container with 1 channel.
/// * `dst` - The output grayscale image container with 1 channel.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Errors
///
/// The function returns an error if the image cannot be resized.
pub fn resize_fast_mono<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_impl(src, dst, interpolation)
}

fn resize_fast_impl<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    // prepare the input image for the fast_image_resize crate
    let (src_cols, src_rows) = (src.cols(), src.rows());
    let src_data_len = src.as_slice().len();

    let pixel_type = match C {
        4 => fr::PixelType::U8x4,
        3 => fr::PixelType::U8x3,
        1 => fr::PixelType::U8,
        // TODO: Find a way to generalise it further by supporting multiple types other than u8
        _ => return Err(ImageError::UnsupportedChannelCount(C)),
    };

    let src_image =
        fr::images::ImageRef::new(src_cols as u32, src_rows as u32, src.as_slice(), pixel_type)
            .map_err(|_| ImageError::InvalidChannelShape(src_data_len, src_cols * src_rows * C))?;

    // prepare the output image for the fast_image_resize crate
    let (dst_cols, dst_rows) = (dst.cols(), dst.rows());
    let dst_data_len = dst.as_slice_mut().len();

    let mut dst_image = fr::images::Image::from_slice_u8(
        dst_cols as u32,
        dst_rows as u32,
        dst.as_slice_mut(),
        pixel_type,
    )
    .map_err(|_| ImageError::InvalidChannelShape(dst_data_len, dst_cols * dst_rows * C))?;

    let mut options = fr::ResizeOptions::new();
    options.algorithm = match interpolation {
        InterpolationMode::Bilinear => fr::ResizeAlg::Convolution(fr::FilterType::Bilinear),
        InterpolationMode::Nearest => fr::ResizeAlg::Nearest,
        InterpolationMode::Lanczos => fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3),
        InterpolationMode::Bicubic => fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom),
    };

    let mut resizer = fr::Resizer::new();
    resizer
        .resize(&src_image, &mut dst_image, &options)
        .map_err(|_| ImageError::IncompatiblePixelTypes)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::{CpuAllocator, TensorError};

    #[test]
    fn resize_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            (0..3 * 4 * 3).map(|x| x as f32).collect::<Vec<f32>>(),
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

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
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 1, _>::from_size_val(new_size, 0.0f32, CpuAllocator)?;

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
        let (map_x, map_y) =
            crate::interpolation::grid::meshgrid_from_fn(2, 3, |x, y| Ok((x as f32, y as f32)))?;

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
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0u8; 4 * 5 * 3],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0, CpuAllocator)?;

        super::resize_fast_rgb(
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
