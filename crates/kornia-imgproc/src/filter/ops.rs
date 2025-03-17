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

/// Compute one pixel data with 3x3 kernel
///
/// # Argument
///
/// * `src_data` - The source image with full data
/// * `src_cols` - Cols of source image
/// * `kernel_x` - 3x3 kernel for compute dx_dst
/// * `kernel_y` - 3x3 kernel for compute dy_dst
/// * `dx_dst` - The slice for the current pixel; in dx output, length should be C
/// * `dy_dst` - The slice for the current pixel; in dy output, length should be C
/// * `row` - current row idx in source image
/// * `row_pos_bias` - A function for input row index and kernel y index and output a real row index for access source image data
/// * `col` - current col idx in source image
/// * `col_pos_bias` - A function for input col index and kernel x index and output a real col index for access source image data
fn filter_kernel3_pix_calc<
    const C: usize,
    FR: Fn(usize, usize) -> usize,
    FC: Fn(usize, usize) -> usize,
>(
    src_data: &[f32],
    src_cols: usize,
    kernel_x: &[[f32; 3]; 3],
    kernel_y: &[[f32; 3]; 3],
    dx_dst: &mut [f32],
    dy_dst: &mut [f32],
    row: usize,
    col: usize,
    row_pos_bias: FR,
    col_pos_bias: FC,
) {
    let mut sum_x = [0.0; C];
    let mut sum_y = [0.0; C];
    for dy in 0..3 {
        for dx in 0..3 {
            let row = row_pos_bias(row, dy);
            let col = col_pos_bias(col, dx);
            for ch in 0..C {
                let src_pix_offset = (row * src_cols + col) * C + ch;
                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                sum_x[ch] += val * kernel_x[dy][dx];
                sum_y[ch] += val * kernel_y[dy][dx];
            }
        }
    }
    unsafe { &mut *(dx_dst.as_mut_ptr() as *mut [f32; C]) }.copy_from_slice(&sum_x);
    unsafe { &mut *(dy_dst.as_mut_ptr() as *mut [f32; C]) }.copy_from_slice(&sum_y);
}

/// Compute one row data with 3x3 kernel
///
/// # Argument
///
/// * `src_data` - The source image with full data
/// * `src_cols` - Cols of source image
/// * `kernel_x` - 3x3 kernel for compute dx_dst
/// * `kernel_y` - 3x3 kernel for compute dy_dst
/// * `dx_dst` - The slice for the current row in dx output, length should be src_cols * C
/// * `dy_dst` - The slice for the current row in dy output, length should be src_cols * C
/// * `row` - current row idx in source image
/// * `row_pos_bias` - A function for input row index and kernel index and output a real row index for access source image data
fn filter_kernel3_row_calc<const C: usize, FR: ?Sized + Fn(usize, usize) -> usize>(
    src_data: &[f32],
    src_cols: usize,
    kernel_x: &[[f32; 3]; 3],
    kernel_y: &[[f32; 3]; 3],
    dx_dst: &mut [f32],
    dy_dst: &mut [f32],
    row: usize,
    row_pos_bias: &FR,
) {
    {
        filter_kernel3_pix_calc::<C, _, _>(
            src_data,
            src_cols,
            kernel_x,
            kernel_y,
            &mut dx_dst[0..C],
            &mut dy_dst[0..C],
            row,
            0,
            row_pos_bias,
            |_, dx| dx.max(1) - 1,
        );
    }
    for c in 1..src_cols - 1 {
        let col_offset = c * C;
        filter_kernel3_pix_calc::<C, _, _>(
            src_data,
            src_cols,
            kernel_x,
            kernel_y,
            &mut dx_dst[col_offset..col_offset + C],
            &mut dy_dst[col_offset..col_offset + C],
            row,
            c,
            row_pos_bias,
            |c, dx| (c + dx) - 1,
        );
    }
    {
        let c = src_cols - 1;
        let col_offset = c * C;
        filter_kernel3_pix_calc::<C, _, _>(
            src_data,
            src_cols,
            kernel_x,
            kernel_y,
            &mut dx_dst[col_offset..col_offset + C],
            &mut dy_dst[col_offset..col_offset + C],
            row,
            c,
            row_pos_bias,
            |c, dx| (c + dx).min(src_cols) - 1,
        );
    }
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

    let src_data = src.as_slice();
    let dx_data = dx.as_slice_mut();
    let dy_data = dy.as_slice_mut();
    let full_len = src.rows() * src.cols() * C;
    let row_len = src.cols() * C;

    {
        filter_kernel3_row_calc::<C, _>(
            src_data,
            src.cols(),
            &sobel_x,
            &sobel_y,
            &mut dx_data[0..row_len],
            &mut dy_data[0..row_len],
            0,
            &|_, dy| dy.max(1) - 1,
        );
    }
    for r in 1..src.rows() - 1 {
        let row_offset = r * row_len;
        filter_kernel3_row_calc::<C, _>(
            src_data,
            src.cols(),
            &sobel_x,
            &sobel_y,
            &mut dx_data[row_offset..row_offset + row_len],
            &mut dy_data[row_offset..row_offset + row_len],
            r,
            &|r, dy| r + dy - 1,
        );
    }
    {
        let r = src.rows() - 1;
        filter_kernel3_row_calc::<C, _>(
            src_data,
            src.cols(),
            &sobel_x,
            &sobel_y,
            &mut dx_data[full_len - row_len..full_len],
            &mut dy_data[full_len - row_len..full_len],
            r,
            &|r, dy| (r + dy).min(src.rows()) - 1,
        );
    }

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
/// Implement within [separable_filter]
///
/// # NOTICE
///
/// This only used for benchmark beacuse
/// `spatial_gradient` require replicate padding, but `separable_filter` use zero padding.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_by_separable_filter<const C: usize>(
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

    let dx_kernel_y = [0.35355339059, 0.70710678118, 0.35355339059]; // 1/sqrt(8), 2/sqrt(8), 1/sqrt(8)
    let dx_kernel_x = [-0.35355339059, 0.0, 0.35355339059]; // -1/sqrt(8), 0, 1/sqrt(8)

    let dy_kernel_y = [-0.35355339059, 0.0, 0.35355339059]; // -1/sqrt(8), 0, 1/sqrt(8)
    let dy_kernel_x = [0.35355339059, 0.70710678118, 0.35355339059]; // 1/sqrt(8), 2/sqrt(8), 1/sqrt(8)

    separable_filter(src, dx, &dx_kernel_x, &dx_kernel_y)?;
    separable_filter(src, dy, &dy_kernel_x, &dy_kernel_y)?;

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
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
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_rayon_parallel<const C: usize>(
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
                    for ch in 0..C {
                        let mut sum_x = 0.0;
                        let mut sum_y = 0.0;
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let row = (r + dy).min(src.rows()).max(1) - 1;
                                let col = (c + dx).min(src.cols()).max(1) - 1;
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x += val * sobel_x[dy][dx];
                                sum_y += val * sobel_y[dy][dx];
                            }
                        }
                        dx_c[ch] = sum_x;
                        dy_c[ch] = sum_y;
                    }
                });
        });

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_rayon_row_parallel<const C: usize>(
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
            if r == 0 {
                filter_kernel3_row_calc::<C, _>(
                    src_data,
                    src.cols(),
                    &sobel_x,
                    &sobel_y,
                    dx_row,
                    dy_row,
                    0,
                    &|_, dy| dy.max(1) - 1,
                );
            } else if r == src.rows() - 1 {
                filter_kernel3_row_calc::<C, _>(
                    src_data,
                    src.cols(),
                    &sobel_x,
                    &sobel_y,
                    dx_row,
                    dy_row,
                    r,
                    &|r, dy| (r + dy).min(src.rows()) - 1,
                );
            } else {
                filter_kernel3_row_calc::<C, _>(
                    src_data,
                    src.cols(),
                    &sobel_x,
                    &sobel_y,
                    dx_row,
                    dy_row,
                    r,
                    &|r, dy| r + dy - 1,
                );
            }
        });
    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_row_parallel<const C: usize>(
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
            if r == 0 {
                filter_kernel3_row_calc::<C, _>(
                    src_data,
                    src.cols(),
                    &sobel_x,
                    &sobel_y,
                    dx_row,
                    dy_row,
                    0,
                    &|_, dy| dy.max(1) - 1,
                );
            } else if r == src.rows() - 1 {
                filter_kernel3_row_calc::<C, _>(
                    src_data,
                    src.cols(),
                    &sobel_x,
                    &sobel_y,
                    dx_row,
                    dy_row,
                    r,
                    &|r, dy| (r + dy).min(src.rows()) - 1,
                );
            } else {
                filter_kernel3_row_calc::<C, _>(
                    src_data,
                    src.cols(),
                    &sobel_x,
                    &sobel_y,
                    dx_row,
                    dy_row,
                    r,
                    &|r, dy| r + dy - 1,
                );
            }
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
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::<f32, 2>::new(
            size,
            (0..25).into_iter().flat_map(|x| [x as f32, x as f32 + 25.0]).collect(),
        )?;
        static TEST_FUNCTIONS: &'static [(
            fn(&Image<f32, 2>, &mut Image<f32, 2>, &mut Image<f32, 2>) -> Result<(), ImageError>,
            &'static str,
        )] = &[
            (spatial_gradient_float, "spatial_gradient_float"),
            (
                spatial_gradient_float_row_parallel,
                "spatial_gradient_float_row_parallel",
            ),
            (
                spatial_gradient_float_parallel,
                "spatial_gradient_float_parallel",
            ),
            (
                spatial_gradient_float_rayon_parallel,
                "spatial_gradient_float_rayon_parallel",
            ),
            (
                spatial_gradient_float_rayon_row_parallel,
                "spatial_gradient_float_rayon_row_parallel",
            ),
        ];
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
