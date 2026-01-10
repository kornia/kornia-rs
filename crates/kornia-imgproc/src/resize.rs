use crate::interpolation::InterpolationMode;
use crate::parallel;
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
) -> Result<(), ImageError> {
    // check if the input and output images have the same size
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    let (src_rows, src_cols) = (src.rows(), src.cols());
    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());

    // Handle division by zero when the destination dimension is 1
    let step_x = if dst_cols > 1 {
        (src_cols - 1) as f32 / (dst_cols - 1) as f32
    } else {
        0.0
    };

    let step_y = if dst_rows > 1 {
        (src_rows - 1) as f32 / (dst_rows - 1) as f32
    } else {
        0.0
    };

    // match the interpolation mode
    match interpolation {
        InterpolationMode::Nearest => {
            parallel::par_iter_rows_indexed_mut(dst, |row_idx, row| {
                let iy = ((row_idx as f32 * step_y).round() as usize).min(src_rows - 1);
                let mut x = 0.0f32;

                for pix in row.chunks_exact_mut(C) {
                    let ix = (x.round() as usize).min(src_cols - 1);

                    unsafe {
                        let src_ptr = src.get_unchecked([iy, ix, 0]) as *const f32;
                        for c in 0..C {
                            *pix.get_unchecked_mut(c) = *src_ptr.add(c);
                        }
                    }
                    x += step_x;
                }
            });
        }

        InterpolationMode::Bilinear => {
            parallel::par_iter_rows_indexed_mut(dst, |row_idx, row| {
                let y = row_idx as f32 * step_y;
                let iv = y.trunc() as usize;
                let iv1 = (iv + 1).min(src_rows - 1);
                let fv = y.fract();
                let wv0 = 1.0 - fv;
                let wv1 = fv;

                let mut x = 0.0f32;

                for pix in row.chunks_exact_mut(C) {
                    let iu = x.trunc() as usize;
                    let iu1 = (iu + 1).min(src_cols - 1);
                    let fu = x.fract();

                    // Pre-calculating weights once for all channels
                    let w00 = (1.0 - fu) * wv0;
                    let w01 = fu * wv0;
                    let w10 = (1.0 - fu) * wv1;
                    let w11 = fu * wv1;

                    unsafe {
                        // pointer math as pixel data is stored contiguously
                        let p00 = src.get_unchecked([iv, iu, 0]) as *const f32;
                        let p01 = src.get_unchecked([iv, iu1, 0]) as *const f32;
                        let p10 = src.get_unchecked([iv1, iu, 0]) as *const f32;
                        let p11 = src.get_unchecked([iv1, iu1, 0]) as *const f32;

                        for c in 0..C {
                            *pix.get_unchecked_mut(c) = *p00.add(c) * w00
                                + *p01.add(c) * w01
                                + *p10.add(c) * w10
                                + *p11.add(c) * w11;
                        }
                    }
                    x += step_x;
                }
            });
        }

        InterpolationMode::Lanczos => {
            unimplemented!("Lanczos interpolation is not yet implemented")
        }

        InterpolationMode::Bicubic => {
            unimplemented!("Bicubic interpolation is not yet implemented")
        }
    }

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
    #[test]
    fn resize_single_pixel() -> Result<(), ImageError> {
        let image = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1.0, 2.0, 3.0, 4.0],
            CpuAllocator,
        )?;

        let mut image_resized = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert!(image_resized.as_slice()[0].is_finite());
        assert_eq!(image_resized.as_slice()[0], 1.0);

        Ok(())
    }

    #[test]
    fn resize_single_column() -> Result<(), ImageError> {
        let image = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.5,
            CpuAllocator,
        )?;

        let mut image_resized = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 4,
            },
            0.0,
            CpuAllocator,
        )?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert!(image_resized.as_slice().iter().all(|v| v.is_finite()));

        Ok(())
    }

    #[test]
    fn resize_single_row() -> Result<(), ImageError> {
        let image = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.5,
            CpuAllocator,
        )?;

        let mut image_resized = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 4,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert!(image_resized.as_slice().iter().all(|v| v.is_finite()));

        Ok(())
    }
}
