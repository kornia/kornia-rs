use kornia_image::{Image, ImageError, ImageSize};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use super::{fast_horizontal_filter, kernels, separable_filter};

/// Blur an image using a box blur filter
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
///
/// PRECONDITION: `src` and `dst` must have the same shape.
pub fn box_blur<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    kernel_size: (usize, usize),
) -> Result<(), ImageError> {
    let kernel_x = kernels::box_blur_kernel_1d(kernel_size.0);
    let kernel_y = kernels::box_blur_kernel_1d(kernel_size.1);
    separable_filter(src, dst, &kernel_x, &kernel_y)?;
    Ok(())
}

/// Blur an image using a gaussian blur filter
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
/// * `sigma` - The sigma of the gaussian kernel.
///
/// PRECONDITION: `src` and `dst` must have the same shape.
/// NOTE: This function uses a constant border type.
pub fn gaussian_blur<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> Result<(), ImageError> {
    let kernel_x = kernels::gaussian_kernel_1d(kernel_size.0, sigma.0);
    let kernel_y = kernels::gaussian_kernel_1d(kernel_size.1, sigma.1);
    separable_filter(src, dst, &kernel_x, &kernel_y)?;
    Ok(())
}

/// Computer sobel filter
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
///
/// PRECONDITION: `src` and `dst` must have the same shape.
pub fn sobel<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    kernel_size: usize,
) -> Result<(), ImageError> {
    // get the sobel kernels
    let (kernel_x, kernel_y) = kernels::sobel_kernel_1d(kernel_size);

    // apply the sobel filter using separable filter
    let mut gx = Image::<f32, C>::from_size_val(src.size(), 0.0)?;
    separable_filter(src, &mut gx, &kernel_x, &kernel_y)?;

    let mut gy = Image::<f32, C>::from_size_val(src.size(), 0.0)?;
    separable_filter(src, &mut gy, &kernel_y, &kernel_x)?;

    // compute the magnitude in parallel by rows
    dst.as_slice_mut()
        .iter_mut()
        .zip(gx.as_slice().iter())
        .zip(gy.as_slice().iter())
        .for_each(|((dst, &gx), &gy)| {
            *dst = (gx * gx + gy * gy).sqrt();
        });

    Ok(())
}

/// Blur an image using a box blur filter multiple times to achieve a near gaussian blur
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
/// * `sigma` - The sigma of the gaussian kernel, xy-ordered.
///
/// PRECONDITION: `src` and `dst` must have the same shape.
pub fn box_blur_fast<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    sigma: (f32, f32),
) -> Result<(), ImageError> {
    let half_kernel_x_sizes = kernels::box_blur_fast_kernels_1d(sigma.0, 3);
    let half_kernel_y_sizes = kernels::box_blur_fast_kernels_1d(sigma.1, 3);

    let transposed_size = ImageSize {
        width: src.size().height,
        height: src.size().width,
    };

    let mut input_img = src.clone();
    let mut transposed = Image::<f32, C>::from_size_val(transposed_size, 0.0)?;

    for (half_kernel_x_size, half_kernel_y_size) in
        half_kernel_x_sizes.iter().zip(half_kernel_y_sizes.iter())
    {
        fast_horizontal_filter(&input_img, &mut transposed, *half_kernel_x_size)?;
        fast_horizontal_filter(&transposed, dst, *half_kernel_y_size)?;

        input_img = dst.clone();
    }

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float<const C: usize>(
    src: &Image<f32, C>,
    dx: &mut Image<f32, C>,
    dy: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .chunks_mut(cols * C)
        .zip(dy.as_slice_mut().chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .chunks_mut(C)
                .zip(dy_row.chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    let mut sum_x = [0.0; C];
                    let mut sum_y = [0.0; C];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let row = (r + dy).min(src.rows()).max(1) - 1;
                            let col = (c + dx).min(src.cols()).max(1) - 1;
                            for ch in 0..C {
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x[ch] += val * sobel_x[dy][dx];
                                sum_y[ch] += val * sobel_y[dy][dx];
                            }
                        }
                    }
                    dx_c.copy_from_slice(&sum_x);
                    dy_c.copy_from_slice(&sum_y);
                });
        });

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
/// Parallel by row.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_parallel_row<const C: usize>(
    src: &Image<f32, C>,
    dx: &mut Image<f32, C>,
    dy: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .par_chunks_mut(cols * C)
        .zip(dy.as_slice_mut().par_chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .chunks_mut(C)
                .zip(dy_row.chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    let mut sum_x = [0.0; C];
                    let mut sum_y = [0.0; C];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let row = (r + dy).min(src.rows()).max(1) - 1;
                            let col = (c + dx).min(src.cols()).max(1) - 1;
                            for ch in 0..C {
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x[ch] += val * sobel_x[dy][dx];
                                sum_y[ch] += val * sobel_y[dy][dx];
                            }
                        }
                    }
                    dx_c.copy_from_slice(&sum_x);
                    dy_c.copy_from_slice(&sum_y);
                });
        });

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
/// Parallel both by row and col.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_parallel<const C: usize>(
    src: &Image<f32, C>,
    dx: &mut Image<f32, C>,
    dy: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .par_chunks_mut(cols * C)
        .zip(dy.as_slice_mut().par_chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .par_chunks_mut(C)
                .zip(dy_row.par_chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    let mut sum_x = [0.0; C];
                    let mut sum_y = [0.0; C];
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let row = (r + dy).min(src.rows()).max(1) - 1;
                            let col = (c + dx).min(src.cols()).max(1) - 1;
                            for ch in 0..C {
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x[ch] += val * sobel_x[dy][dx];
                                sum_y[ch] += val * sobel_y[dy][dx];
                            }
                        }
                    }
                    dx_c.copy_from_slice(&sum_x);
                    dy_c.copy_from_slice(&sum_y);
                });
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_blur_fast() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            (0..25).map(|x| x as f32).collect(),
        )?;
        let mut dst = Image::<_, 1>::from_size_val(size, 0.0)?;

        box_blur_fast(&img, &mut dst, (0.5, 0.5))?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                4.444444, 4.9259257, 5.7037034, 6.4814816, 6.962963,
                6.851851, 7.3333335, 8.111111, 8.888889, 9.370372,
                10.740741, 11.222222, 12.0, 12.777779, 13.259262,
                14.629628, 15.111112, 15.888888, 16.666666, 17.14815,
                17.037035, 17.518518, 18.296295, 19.074074, 19.555555,
            ],
        );

        Ok(())
    }

    #[test]
    fn test_gaussian_blur() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            (0..25).map(|x| x as f32).collect(),
        )?;

        let mut dst = Image::<_, 1>::from_size_val(size, 0.0)?;

        gaussian_blur(&img, &mut dst, (3, 3), (0.5, 0.5))?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.57097936, 1.4260278, 2.3195207, 3.213014, 3.5739717,
                4.5739717, 5.999999, 7.0, 7.999999, 7.9349294,
                9.041435, 10.999999, 12.0, 12.999998, 12.402394,
                13.5089, 15.999998, 17.0, 17.999996, 16.86986,
                15.58594, 18.230816, 19.124311, 20.017801, 18.588936,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_spatial_gradient() -> Result<(), ImageError> {
        // First, define a type alias for the function signature
        type FilterFunction =
            fn(&Image<f32, 2>, &mut Image<f32, 2>, &mut Image<f32, 2>) -> Result<(), ImageError>;

        // Then, define a type for the test tuple
        type TestCase = (FilterFunction, &'static str);

        // Now use these types in the static array
        static TEST_FUNCTIONS: &[TestCase] = &[
            (spatial_gradient_float, "spatial_gradient_float"),
            (
                spatial_gradient_float_parallel_row,
                "spatial_gradient_float_parallel_row",
            ),
            (
                spatial_gradient_float_parallel,
                "spatial_gradient_float_parallel",
            ),
        ];

        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::<f32, 2>::new(
            size,
            (0..25).flat_map(|x| [x as f32, x as f32 + 25.0]).collect(),
        )?;
        for (test_fn, fn_name) in TEST_FUNCTIONS {
            let mut dx = Image::<_, 2>::from_size_val(size, 0.0)?;
            let mut dy = Image::<_, 2>::from_size_val(size, 0.0)?;

            test_fn(&img, &mut dx, &mut dy)?;

            #[rustfmt::skip]
            assert_eq!(
                dx.channel(0)?.as_slice(),
                &[
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000
                ],
                "{} dx channel(0)",
                fn_name
            );

            #[rustfmt::skip]
            assert_eq!(
                dx.channel(1)?.as_slice(),
                &[
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
                    0.5000, 1.0000, 1.0000, 1.0000, 0.5000
                ],
                "{} dx channel(1)",
                fn_name
            );

            #[rustfmt::skip]
            assert_eq!(
                dy.channel(0)?.as_slice(),
                &[
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000
                ],
                "{} dy channel(0)",
                fn_name
            );

            #[rustfmt::skip]
            assert_eq!(
                dy.channel(1)?.as_slice(),
                &[
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    5.0000, 5.0000, 5.0000, 5.0000, 5.0000,
                    2.5000, 2.5000, 2.5000, 2.5000, 2.5000
                ],
                "{} dy channel(1)",
                fn_name
            );
        }

        Ok(())
    }
}
