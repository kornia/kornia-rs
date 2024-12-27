use kornia_image::{Image, ImageError};

use super::{kernels, separable_filter};

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
