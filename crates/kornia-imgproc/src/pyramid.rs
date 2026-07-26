use kornia_image::{Image, ImageError, ImageSize};
use rayon::prelude::*;

// Gaussian kernel weights

const SCALE_EVEN: f32 = 0.125;
const SCALE_ODD: f32 = 0.5;
const W_CENTER: f32 = 6.0;
const W_NEIGHBOR: f32 = 1.0;
const W_BORDER: f32 = 7.0;

// Helper functions for horizontal and vertical passes

/// Performs the horizontal pass of Gaussian pyramid upsampling.
///
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination buffer for the horizontal pass.
/// * `dst_width` - The width of the destination image.
fn pyrup_horizontal_pass_f32<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut [f32],
    dst_width: usize,
) {
    let src_width = src.width();
    let src_data = src.as_slice();
    let dst_stride = dst_width * C;

    dst.par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let src_row_offset = y * src_width * C;

            // Special case of single column image
            if src_width == 1 {
                for k in 0..C {
                    let val = src_data[src_row_offset + k];
                    dst_row[k] = val;
                    if k + C < dst_stride {
                        dst_row[k + C] = val;
                    }
                }
                return;
            }

            for k in 0..C {
                let pixel_left = src_data[src_row_offset + k];
                let pixel_right = src_data[src_row_offset + C + k];

                dst_row[k] = (W_CENTER * pixel_left + 2.0 * pixel_right) * SCALE_EVEN;
                dst_row[C + k] = (pixel_left + pixel_right) * SCALE_ODD;
            }

            for x in 1..(src_width - 1) {
                let off = src_row_offset + x * C;
                let dst_off = 2 * x * C;

                for k in 0..C {
                    let pixel_prev = src_data[off - C + k];
                    let pixel_curr = src_data[off + k];
                    let pixel_next = src_data[off + C + k];

                    dst_row[dst_off + k] =
                        (W_NEIGHBOR * pixel_prev + W_CENTER * pixel_curr + W_NEIGHBOR * pixel_next)
                            * SCALE_EVEN;

                    dst_row[dst_off + C + k] = (pixel_curr + pixel_next) * SCALE_ODD;
                }
            }

            let last_x = src_width - 1;
            let off_last = src_row_offset + last_x * C;
            let dst_off_last = 2 * last_x * C;

            for k in 0..C {
                let pixel_prev = src_data[off_last - C + k];
                let pixel_curr = src_data[off_last + k];

                dst_row[dst_off_last + k] =
                    (W_NEIGHBOR * pixel_prev + W_BORDER * pixel_curr) * SCALE_EVEN;
                dst_row[dst_off_last + C + k] = pixel_curr;
            }
        });
}

/// Performs the vertical pass of Gaussian pyramid upsampling.
///
///
/// # Arguments
///
/// * `src_buffer` - The intermediate buffer from the horizontal pass.
/// * `src_buffer_width` - The width of the intermediate buffer.
/// * `src_height` - The height of the original source image.
/// * `dst_data` - The final destination buffer.
/// * `dst_width` - The width of the destination image.
fn pyrup_vertical_pass_f32<const C: usize>(
    src_buffer: &[f32],
    src_buffer_width: usize,
    src_height: usize,
    dst_data: &mut [f32],
    dst_width: usize,
) {
    let stride = src_buffer_width * C;
    let dst_stride = dst_width * C;

    dst_data
        .par_chunks_mut(2 * dst_stride)
        .enumerate()
        .for_each(|(y, dst_rows_chunk)| {
            if y >= src_height {
                return;
            }

            let (row_top, row_center, row_bottom) = if src_height == 1 {
                (0, 0, 0)
            } else if y == 0 {
                (0, 0, 1)
            } else if y == src_height - 1 {
                (src_height - 2, src_height - 1, src_height - 1)
            } else {
                (y - 1, y, y + 1)
            };

            let offset_top = row_top * stride;
            let offset_center = row_center * stride;
            let offset_bottom = row_bottom * stride;

            let (row_even, row_odd) = dst_rows_chunk.split_at_mut(dst_stride);

            if y == 0 {
                // Top border
                for i in 0..stride {
                    row_even[i] = (W_CENTER * src_buffer[offset_center + i]
                        + 2.0 * src_buffer[offset_bottom + i])
                        * SCALE_EVEN;
                    row_odd[i] =
                        (src_buffer[offset_center + i] + src_buffer[offset_bottom + i]) * SCALE_ODD;
                }
            } else if y == src_height - 1 {
                // Bottom border
                for i in 0..stride {
                    row_even[i] = (W_NEIGHBOR * src_buffer[offset_top + i]
                        + W_BORDER * src_buffer[offset_center + i])
                        * SCALE_EVEN;
                    row_odd[i] = src_buffer[offset_center + i];
                }
            } else {
                for i in 0..stride {
                    row_even[i] = (W_NEIGHBOR * src_buffer[offset_top + i]
                        + W_CENTER * src_buffer[offset_center + i]
                        + W_NEIGHBOR * src_buffer[offset_bottom + i])
                        * SCALE_EVEN;

                    row_odd[i] =
                        (src_buffer[offset_center + i] + src_buffer[offset_bottom + i]) * SCALE_ODD;
                }
            }
        });
}

// Main pyrup function

/// Upsample an image by a factor of 2.
///
/// This function doubles the size of the input image by injecting zero rows
/// and columns and then convolving with a 5x5 Gaussian kernel. The implementation
/// performs this as a single efficient polyphase operation.
///
/// # Arguments
///
/// * `src` - The source image to be upsampled.
/// * `dst` - The destination image to store the result. Must have dimensions
///   exactly `(2 * width, 2 * height)` of the source.
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Errors
///
/// Returns `ImageError::InvalidImageSize` if the destination image dimensions
/// do not match the expected 2x upsampling size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::pyramid::pyrup_f32;
///
/// let image = Image::<f32, 1>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0.0, 1.0, 2.0, 3.0],
/// ).unwrap();
///
/// let mut upsampled = Image::<f32, 1>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     0.0,
/// ).unwrap();
///
/// pyrup_f32(&image, &mut upsampled).unwrap();
/// ```
pub fn pyrup_f32<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    let expected_width = src.width() * 2;
    let expected_height = src.height() * 2;

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::pyrup_f32_cuda(
            src, dst, stream
        ));
    }

    // Intermediate buffer for horizontal pass, pyrup_horizontal writes here
    let mut buffer = vec![0.0f32; dst.width() * src.height() * C];
    pyrup_horizontal_pass_f32::<C>(src, &mut buffer, dst.width());

    // Vertical pass reads from buffer and finally writes to dst
    let dst_width = dst.width();
    pyrup_vertical_pass_f32::<C>(
        &buffer,
        dst_width,
        src.height(),
        dst.as_slice_mut(),
        dst_width,
    );

    Ok(())
}

#[inline]
fn reflect_101(mut p: i32, len: i32) -> i32 {
    if len == 1 {
        return 0;
    }

    if p < 0 {
        p = -p;
    }

    let period = 2 * (len - 1);
    p %= period;

    if p >= len {
        p = period - p;
    }

    p
}

/// Downsample an image by applying Gaussian blur and then subsampling.
///
/// This function halves the size of the input image by first applying a Gaussian blur
/// and then subsampling every other pixel. This is the inverse operation of [`pyrup_f32`].
///
/// Uses BORDER_REFLECT_101 border mode (same as OpenCV default) for handling pixels
/// near image boundaries.
///
/// # Arguments
///
/// * `src` - The source image to be downsampled.
/// * `dst` - The destination image to store the result (should be half the size of src).
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::pyramid::pyrdown_f32;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     (0..48).map(|x| x as f32).collect(),
/// ).unwrap();
///
/// let mut downsampled = Image::<f32, 3>::from_size_val(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     0.0,
/// ).unwrap();
///
/// pyrdown_f32(&image, &mut downsampled).unwrap();
/// ```
pub fn pyrdown_f32<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    let expected_width = src.width().div_ceil(2);
    let expected_height = src.height().div_ceil(2);

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::pyrdown_f32_cuda(
            src, dst, stream
        ));
    }

    let src_width = src.width();
    let src_height = src.height();
    let dst_width = dst.width();
    let dst_height = dst.height();

    // Fused Gaussian blur + downsample in a single pass.
    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    // Standard values for the 5x5 Gaussian kernel
    let kernel_x = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let kernel_y = [0.0625, 0.25, 0.375, 0.25, 0.0625];

    // Precompute flattened kernel
    let mut kernel_weights = [0.0f32; 25];
    let mut idx = 0;
    for &ky_weight in kernel_y.iter() {
        for &kx_weight in kernel_x.iter() {
            kernel_weights[idx] = ky_weight * kx_weight;
            idx += 1;
        }
    }

    let center_y_start = 1;
    let center_y_end = (dst_height - 1).min(dst_height);
    let center_x_start = 1;
    let center_x_end = (dst_width - 1).min(dst_width);

    for dst_y in center_y_start..center_y_end {
        let src_center_y = dst_y * 2;

        for dst_x in center_x_start..center_x_end {
            let src_center_x = dst_x * 2;

            for c in 0..C {
                let mut sum = 0.0f32;
                let mut k_idx = 0;

                for ky in 0..5 {
                    let src_y = src_center_y + ky - 2;
                    for kx in 0..5 {
                        let src_x = src_center_x + kx - 2;
                        let src_idx = (src_y * src_width + src_x) * C + c;
                        sum += src_data[src_idx] * kernel_weights[k_idx];
                        k_idx += 1;
                    }
                }

                let dst_idx = (dst_y * dst_width + dst_x) * C + c;
                dst_data[dst_idx] = sum;
            }
        }
    }

    for dst_y in 0..dst_height {
        let src_center_y = (dst_y * 2) as i32;
        let is_border_y = dst_y < center_y_start || dst_y >= center_y_end;

        for dst_x in 0..dst_width {
            let src_center_x = (dst_x * 2) as i32;
            let is_border_x = dst_x < center_x_start || dst_x >= center_x_end;

            if !is_border_y && !is_border_x {
                continue;
            }

            for c in 0..C {
                let mut sum = 0.0f32;
                let mut k_idx = 0;

                for ky in 0..5 {
                    let src_y = reflect_101(src_center_y + ky - 2, src_height as i32) as usize;
                    for kx in 0..5 {
                        let src_x = reflect_101(src_center_x + kx - 2, src_width as i32) as usize;
                        let src_idx = (src_y * src_width + src_x) * C + c;
                        sum += src_data[src_idx] * kernel_weights[k_idx];
                        k_idx += 1;
                    }
                }

                let dst_idx = (dst_y * dst_width + dst_x) * C + c;
                dst_data[dst_idx] = sum;
            }
        }
    }

    Ok(())
}

/// Build a Gaussian pyramid from an image.
/// # Arguments
/// * `src` - The source image with shape (H, W, C).
/// * `max_level` - The maximum level of the pyramid (0 means just the original image).
/// # Returns
/// A vector of images representing the Gaussian pyramid.
pub fn build_pyramid<const C: usize>(
    src: &Image<f32, C>,
    max_level: usize,
) -> Result<Vec<Image<f32, C>>, ImageError> {
    let mut pyramid = Vec::with_capacity(max_level + 1);
    pyramid.push(src.clone());

    for l in 0..max_level {
        let current_img = &pyramid[l];
        let new_size = ImageSize {
            width: current_img.cols().div_ceil(2),
            height: current_img.rows().div_ceil(2),
        };
        let mut downsampled =
            Image::from_size_val_in(new_size, 0.0, current_img.storage.alloc().clone())?;
        pyrdown_f32(current_img, &mut downsampled)?;
        pyramid.push(downsampled);
    }

    Ok(pyramid)
}

/// Downsample a u8 image by applying Gaussian blur and then subsampling.
///
/// This function halves the size of the input image by first applying a Gaussian blur
/// and then subsampling every other pixel. This is the inverse operation of [`pyrup_u8`].
///
/// # Arguments
///
/// * `src` - The source image to be downsampled.
/// * `dst` - The destination image to store the result (should be half the size of src).
///
/// Uses BORDER_REFLECT_101 border mode (same as OpenCV default) for handling pixels
/// near image boundaries.
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
pub fn pyrdown_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
) -> Result<(), ImageError> {
    let expected_width = src.width().div_ceil(2);
    let expected_height = src.height().div_ceil(2);

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::pyrdown_u8_cuda(
            src, dst, stream
        ));
    }

    let buffer_width = dst.width();
    let buffer_height = src.height();
    let mut buffer = vec![0u16; buffer_width * buffer_height * C];

    // Horizontal pass: 1D convolution + downsample
    pyrdown_horizontal_pass_u8::<C>(src, &mut buffer, buffer_width);

    // Vertical pass
    let dst_width = dst.width();
    pyrdown_vertical_pass_u8::<C>(
        &buffer,
        buffer_width,
        src.height(),
        dst.as_slice_mut(),
        dst_width,
    );

    Ok(())
}

fn pyrdown_horizontal_pass_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut [u16],
    dst_width: usize,
) {
    let src_width = src.width();
    let src_data = src.as_slice();
    let dst_stride = dst_width * C;

    // Kernel: [1, 4, 6, 4, 1]

    dst.par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let src_row_offset = y * src_width * C;

            // Safe range for the central loop where the kernel fits completely
            let safe_start = 1;
            let safe_end = if src_width > 2 {
                (src_width - 1) / 2
            } else {
                safe_start
            };

            // Left border
            for dst_x in 0..safe_start.min(dst_width) {
                process_pixel_pyrdown_checked::<C>(
                    src_data,
                    dst_row,
                    src_row_offset,
                    dst_x,
                    src_width,
                );
            }

            // Ensure we don't exceed dst_width
            let loop_end = safe_end.min(dst_width);
            if loop_end > safe_start {
                for dst_x in safe_start..loop_end {
                    let src_center_x = dst_x * 2;
                    let base_idx = src_row_offset + src_center_x * C;

                    for k in 0..C {
                        // Direct access without boundary checks
                        let v_m2 = src_data[base_idx - 2 * C + k] as u16;
                        let v_m1 = src_data[base_idx - C + k] as u16;
                        let v_0 = src_data[base_idx + k] as u16;
                        let v_p1 = src_data[base_idx + C + k] as u16;
                        let v_p2 = src_data[base_idx + 2 * C + k] as u16;

                        let sum = v_m2 + 4 * v_m1 + 6 * v_0 + 4 * v_p1 + v_p2;
                        dst_row[dst_x * C + k] = sum;
                    }
                }
            }

            // Right border
            for dst_x in loop_end..dst_width {
                process_pixel_pyrdown_checked::<C>(
                    src_data,
                    dst_row,
                    src_row_offset,
                    dst_x,
                    src_width,
                );
            }
        });
}

#[inline(always)]
fn process_pixel_pyrdown_checked<const C: usize>(
    src_data: &[u8],
    dst_row: &mut [u16],
    src_row_offset: usize,
    dst_x: usize,
    src_width: usize,
) {
    let src_center_x = (dst_x * 2) as i32;
    // Loop unrolled for kernel size 5
    let idx_m2 = (reflect_101(src_center_x - 2, src_width as i32) as usize) * C;
    let idx_m1 = (reflect_101(src_center_x - 1, src_width as i32) as usize) * C;
    let idx_0 = (reflect_101(src_center_x, src_width as i32) as usize) * C;
    let idx_p1 = (reflect_101(src_center_x + 1, src_width as i32) as usize) * C;
    let idx_p2 = (reflect_101(src_center_x + 2, src_width as i32) as usize) * C;

    for k in 0..C {
        let v_m2 = src_data[src_row_offset + idx_m2 + k] as u16;
        let v_m1 = src_data[src_row_offset + idx_m1 + k] as u16;
        let v_0 = src_data[src_row_offset + idx_0 + k] as u16;
        let v_p1 = src_data[src_row_offset + idx_p1 + k] as u16;
        let v_p2 = src_data[src_row_offset + idx_p2 + k] as u16;

        let sum = v_m2 + 4 * v_m1 + 6 * v_0 + 4 * v_p1 + v_p2;
        dst_row[dst_x * C + k] = sum;
    }
}

fn pyrdown_vertical_pass_u8<const C: usize>(
    src_buffer: &[u16],
    src_buffer_width: usize,
    src_height: usize,
    dst_data: &mut [u8],
    dst_width: usize,
) {
    let stride = src_buffer_width * C;
    let dst_stride = dst_width * C;

    dst_data
        .par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let src_center_y = (dst_y * 2) as i32;

            // Reflect 101 for y coordinates
            let y_m2 = reflect_101(src_center_y - 2, src_height as i32) as usize;
            let y_m1 = reflect_101(src_center_y - 1, src_height as i32) as usize;
            let y_0 = reflect_101(src_center_y, src_height as i32) as usize;
            let y_p1 = reflect_101(src_center_y + 1, src_height as i32) as usize;
            let y_p2 = reflect_101(src_center_y + 2, src_height as i32) as usize;

            let off_m2 = y_m2 * stride;
            let off_m1 = y_m1 * stride;
            let off_0 = y_0 * stride;
            let off_p1 = y_p1 * stride;
            let off_p2 = y_p2 * stride;

            for i in 0..dst_stride {
                let v_m2 = src_buffer[off_m2 + i] as u32;
                let v_m1 = src_buffer[off_m1 + i] as u32;
                let v_0 = src_buffer[off_0 + i] as u32;
                let v_p1 = src_buffer[off_p1 + i] as u32;
                let v_p2 = src_buffer[off_p2 + i] as u32;

                // Sum weights: 1, 4, 6, 4, 1. Total 16.
                let sum = v_m2 + 4 * v_m1 + 6 * v_0 + 4 * v_p1 + v_p2;

                // Rounding: (sum + 128) >> 8
                let val = (sum + 128) >> 8;
                dst_row[i] = val.min(255) as u8;
            }
        });
}

fn pyrup_horizontal_pass_u8<const C: usize>(src: &Image<u8, C>, dst: &mut [u8], dst_width: usize) {
    let src_width = src.width();
    let src_data = src.as_slice();
    let dst_stride = dst_width * C;

    //Process borders separately to keep the inner loop fast
    dst.par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let src_row_offset = y * src_width * C;

            // Left border
            if src_width > 0 {
                process_pixel_pyrup_checked::<C>(
                    src_data,
                    dst_row,
                    src_row_offset,
                    0,
                    src_width,
                    dst_stride,
                );
            }

            // Fast path without boundary checks
            if src_width > 2 {
                for x in 1..src_width - 1 {
                    let idx_base = src_row_offset + x * C;

                    for k in 0..C {
                        let p_prev = src_data[idx_base - C + k] as u16;
                        let p_curr = src_data[idx_base + k] as u16;
                        let p_next = src_data[idx_base + C + k] as u16;

                        // Even pixel: (p_prev + 6*p_curr + p_next + 4) / 8
                        let val_even = (p_prev + 6 * p_curr + p_next + 4) >> 3;
                        dst_row[2 * x * C + k] = val_even as u8;

                        // Odd pixel: (p_curr + p_next + 1) / 2
                        if (2 * x + 1) * C < dst_stride {
                            let val_odd = (p_curr + p_next + 1) >> 1;
                            dst_row[(2 * x + 1) * C + k] = val_odd as u8;
                        }
                    }
                }
            }

            // Right border
            if src_width > 1 {
                process_pixel_pyrup_checked::<C>(
                    src_data,
                    dst_row,
                    src_row_offset,
                    src_width - 1,
                    src_width,
                    dst_stride,
                );
            }
        });
}

#[inline(always)]
fn process_pixel_pyrup_checked<const C: usize>(
    src_data: &[u8],
    dst_row: &mut [u8],
    src_row_offset: usize,
    x: usize,
    src_width: usize,
    dst_stride: usize,
) {
    let src_x = x as i32;
    let idx_base = src_row_offset + x * C;

    let idx_prev = src_row_offset + (reflect_101(src_x - 1, src_width as i32) as usize) * C;
    let idx_next = src_row_offset + (reflect_101(src_x + 1, src_width as i32) as usize) * C;

    for k in 0..C {
        let p_curr = src_data[idx_base + k] as u16;
        let p_prev = src_data[idx_prev + k] as u16;
        let p_next = src_data[idx_next + k] as u16;

        let val_even = (p_prev + 6 * p_curr + p_next + 4) >> 3;
        dst_row[2 * x * C + k] = val_even as u8;

        if (2 * x + 1) * C < dst_stride {
            let val_odd = (p_curr + p_next + 1) >> 1;
            dst_row[(2 * x + 1) * C + k] = val_odd as u8;
        }
    }
}

fn pyrup_vertical_pass_u8<const C: usize>(
    src_buffer: &[u8],
    src_buffer_width: usize,
    src_height: usize,
    dst_data: &mut [u8],
    dst_width: usize,
) {
    let stride = src_buffer_width * C;
    let dst_stride = dst_width * C;

    dst_data
        .par_chunks_mut(2 * dst_stride)
        .enumerate()
        .for_each(|(y, dst_rows_chunk)| {
            if y >= src_height {
                return;
            }

            let src_y = y as i32;
            let (row_even, row_odd) = dst_rows_chunk.split_at_mut(dst_stride);

            // Determine source row indices
            let y_prev = reflect_101(src_y - 1, src_height as i32) as usize;
            let y_curr = y;
            let y_next = reflect_101(src_y + 1, src_height as i32) as usize;

            let off_prev = y_prev * stride;
            let off_curr = y_curr * stride;
            let off_next = y_next * stride;

            for i in 0..stride {
                let p_curr = src_buffer[off_curr + i] as u16;
                let p_prev = src_buffer[off_prev + i] as u16;
                let p_next = src_buffer[off_next + i] as u16;

                // Even row
                row_even[i] = ((p_prev + 6 * p_curr + p_next + 4) >> 3) as u8;

                // Odd row
                row_odd[i] = ((p_curr + p_next + 1) >> 1) as u8;
            }
        });
}

/// Upsample a u8 image by a factor of 2.
///
/// This function doubles the size of the input image by injecting zero rows
/// and columns and then convolving with a 5x5 Gaussian kernel.
///
/// # Arguments
///
/// * `src` - The source image to be upsampled.
/// * `dst` - The destination image to store the result. Must have dimensions
///   exactly `(2 * width, 2 * height)` of the source.
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
pub fn pyrup_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
) -> Result<(), ImageError> {
    let expected_width = src.width() * 2;
    let expected_height = src.height() * 2;

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    // Intermediate buffer for horizontal pass
    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::pyrup_u8_cuda(
            src, dst, stream
        ));
    }

    let mut buffer = vec![0u8; dst.width() * src.height() * C];
    pyrup_horizontal_pass_u8::<C>(src, &mut buffer, dst.width());

    // Vertical pass
    let dst_width = dst.width();
    pyrup_vertical_pass_u8::<C>(
        &buffer,
        dst_width,
        src.height(),
        dst.as_slice_mut(),
        dst_width,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn test_build_pyramid_levels_and_sizes_odd_dimensions() {
        // Create a 7x5 image (height x width) so that div_ceil(2) behavior is exercised.
        let size = ImageSize {
            width: 5,
            height: 7,
        };
        let src: Image<f32, 1> = Image::from_size_val(size, 1.0_f32).unwrap();

        let max_level = 3;
        let pyramid = build_pyramid(&src, max_level).unwrap();

        // Check the number of levels: original + max_level downsamples.
        assert_eq!(pyramid.len(), max_level + 1);

        // Expected sizes after repeatedly applying div_ceil(2).
        let expected_sizes = [
            (7_usize, 5_usize), // level 0
            (4_usize, 3_usize), // level 1
            (2_usize, 2_usize), // level 2
            (1_usize, 1_usize), // level 3
        ];

        for (level, (expected_rows, expected_cols)) in expected_sizes.iter().enumerate() {
            let img = &pyramid[level];
            assert_eq!(
                (img.rows(), img.cols()),
                (*expected_rows, *expected_cols),
                "Mismatch at pyramid level {}",
                level
            );
        }
    }

    #[test]
    fn test_pyrup() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 1.0, 2.0, 3.0],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.0,
        )?;

        pyrup_f32(&src, &mut dst)?;

        assert_eq!(dst.width(), 4);
        assert_eq!(dst.height(), 4);

        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;

        pyrdown_f32(&src, &mut dst)?;

        let expected = [3.75, 4.875, 8.25, 9.375];

        let actual = dst.as_slice();
        for (idx, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (act - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                idx,
                exp,
                act
            );
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_3c() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..48).map(|x| x as f32).collect(),
        )?;

        let mut dst = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;

        pyrdown_f32(&src, &mut dst)?;

        let expected = [
            11.25, 12.25, 13.25, // pixel (0,0)
            14.625, 15.625, 16.625, // pixel (0,1)
            24.75, 25.75, 26.75, // pixel (1,0)
            28.125, 29.125, 30.125, // pixel (1,1)
        ];

        let actual = dst.as_slice();
        for (idx, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (act - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                idx,
                exp,
                act
            );
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_invalid_size() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0.0; 16],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0.0,
        )?;

        let result = pyrdown_f32(&src, &mut dst);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pyrdown_odd_dims() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 5,
                height: 7,
            },
            (0..(5 * 7)).map(|x| x as f32).collect(),
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0.0,
        )?;

        pyrdown_f32(&src, &mut dst)?;

        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 4);
        for &v in dst.as_slice() {
            assert!(v.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_min_sizes() -> Result<(), ImageError> {
        let src1 = Image::<f32, 1>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![42.0],
        )?;
        let mut dst1 = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
        )?;
        pyrdown_f32(&src1, &mut dst1)?;
        assert_eq!(dst1.width(), 1);
        assert_eq!(dst1.height(), 1);
        assert!(dst1.as_slice().iter().all(|v| v.is_finite()));

        let src2 = Image::<f32, 1>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![1.0, 2.0],
        )?;
        let mut dst2 = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
        )?;
        pyrdown_f32(&src2, &mut dst2)?;
        assert_eq!(dst2.width(), 1);
        assert_eq!(dst2.height(), 1);
        assert!(dst2.as_slice().iter().all(|v| v.is_finite()));

        let src3 = Image::<f32, 1>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1.0, 2.0],
        )?;
        let mut dst3 = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
        )?;
        pyrdown_f32(&src3, &mut dst3)?;
        assert_eq!(dst3.width(), 1);
        assert_eq!(dst3.height(), 1);
        eprintln!("dst3 values: {:?}", dst3.as_slice());
        assert!(dst3.as_slice().iter().all(|v| v.is_finite()));

        Ok(())
    }

    #[test]
    fn test_pyrdown_numeric_extremes() -> Result<(), ImageError> {
        let large_val = 1e9_f32;
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![large_val; 16],
        )?;
        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;
        pyrdown_f32(&src, &mut dst)?;
        for &v in dst.as_slice() {
            assert!(v.is_finite());
            assert!(v.abs() <= large_val);
        }

        Ok(())
    }
    #[test]
    fn test_pyrdown_u8_smoke() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0,
        )?;

        pyrdown_u8(&src, &mut dst)?;

        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);

        for &val in dst.as_slice() {
            assert!(val < 255); // Should be within range
        }

        Ok(())
    }

    #[test]
    fn test_pyrup_u8_smoke() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0, 10, 20, 30],
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )?;

        pyrup_u8(&src, &mut dst)?;

        assert_eq!(dst.width(), 4);
        assert_eq!(dst.height(), 4);

        Ok(())
    }

    #[test]
    fn test_pyrup_u8_invalid_size() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0; 4],
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3, // Expected 4
                height: 4,
            },
            0,
        )?;

        let result = pyrup_u8(&src, &mut dst);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pyrup_u8_min_sizes() -> Result<(), ImageError> {
        let src1 = Image::<u8, 1>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![100],
        )?;
        let mut dst1 = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0,
        )?;

        pyrup_u8(&src1, &mut dst1)?;
        assert_eq!(dst1.width(), 2);
        assert_eq!(dst1.height(), 2);

        for &val in dst1.as_slice() {
            assert_eq!(val, 100);
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_u8_flat() -> Result<(), ImageError> {
        let val = 100u8;
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 16,
                height: 16,
            },
            vec![val; 16 * 16],
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 8,
                height: 8,
            },
            0,
        )?;

        pyrdown_u8(&src, &mut dst)?;

        for &v in dst.as_slice() {
            // Gaussian kernel is normalized, so constant input should result in same constant output
            // potentially +/- 1 due to integer rounding
            assert!(
                (v as i32 - val as i32).abs() <= 1,
                "Expected {}, got {}",
                val,
                v
            );
        }

        Ok(())
    }
    #[test]
    fn test_pyrdown_u8_invalid_size() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0; 16],
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0,
        )?;

        let result = pyrdown_u8(&src, &mut dst);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pyrdown_u8_odd_dims() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 5,
                height: 7,
            },
            (0..(5 * 7)).map(|x| x as u8).collect(),
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0,
        )?;

        pyrdown_u8(&src, &mut dst)?;

        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 4);

        Ok(())
    }

    #[test]
    fn test_pyrdown_u8_min_sizes() -> Result<(), ImageError> {
        let src1 = Image::<u8, 1>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![42],
        )?;
        let mut dst1 = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0,
        )?;
        pyrdown_u8(&src1, &mut dst1)?;
        assert_eq!(dst1.width(), 1);
        assert_eq!(dst1.height(), 1);
        assert_eq!(dst1.get_pixel(0, 0, 0).unwrap(), &42); // Should be preserved (or close)

        let src2 = Image::<u8, 1>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![10, 20],
        )?;
        let mut dst2 = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0,
        )?;
        pyrdown_u8(&src2, &mut dst2)?;
        assert_eq!(dst2.width(), 1);
        assert_eq!(dst2.height(), 1);

        Ok(())
    }

    #[test]
    fn test_pyrdown_u8_3c() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        // src = np.arange(4*4*3, dtype=np.uint8).reshape(4,4,3)
        // dst = cv2.pyrDown(src)
        let src = Image::<u8, 3>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..48).map(|x| x as u8).collect(),
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0,
        )?;

        pyrdown_u8(&src, &mut dst)?;

        let expected = [
            11, 12, 13, // pixel (0,0)
            15, 16, 17, // pixel (0,1)
            25, 26, 27, // pixel (1,0)
            28, 29, 30, // pixel (1,1)
        ];

        let actual = dst.as_slice();
        for (idx, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
            // Allow small difference due to integer arithmetic differences (if any)
            // But for small values and standard implementation, it should match exactly or be very close.
            let diff = (*act as i32 - *exp).abs();
            assert!(
                diff <= 1,
                "Mismatch at index {}: expected {}, got {}, diff {}",
                idx,
                exp,
                act,
                diff
            );
        }

        Ok(())
    }

    #[test]
    fn test_pyrup_u8_3c() -> Result<(), ImageError> {
        let src = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![
                10, 20, 30, // (0,0)
                40, 50, 60, // (0,1)
                70, 80, 90, // (1,0)
                100, 110, 120, // (1,1)
            ],
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )?;

        pyrup_u8(&src, &mut dst)?;

        assert_eq!(dst.width(), 4);
        assert_eq!(dst.height(), 4);

        for &val in dst.as_slice() {
            assert!(val < 255);
        }

        Ok(())
    }
}

// ── CUDA adapters + PyramidPlan ──────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::dispatch::{device_slices, dims_u32, untyped_device_err};
    use crate::cuda::pyramid::{
        launch_pyrdown_f32, launch_pyrdown_u8, launch_pyrup_f32, launch_pyrup_u8,
    };
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn pyrdown_f32_cuda<const C: usize>(
        src: &Image<f32, C>,
        dst: &mut Image<f32, C>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let (sw, sh) = dims_u32(src)?;
        let (dw, dh) = dims_u32(dst)?;
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_pyrdown_f32(ctx, stream, s, d, sw, sh, dw, dh, C as u32).map_err(err)
    }

    pub(super) fn pyrup_f32_cuda<const C: usize>(
        src: &Image<f32, C>,
        dst: &mut Image<f32, C>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let (sw, sh) = dims_u32(src)?;
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        let n = (sw as usize * 2) * sh as usize * C;
        let mut scratch = unsafe { stream.alloc::<f32>(n) }.map_err(err)?;
        launch_pyrup_f32(ctx, stream, s, d, &mut scratch, sw, sh, C as u32).map_err(err)
    }

    pub(super) fn pyrdown_u8_cuda<const C: usize>(
        src: &Image<u8, C>,
        dst: &mut Image<u8, C>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let (sw, sh) = dims_u32(src)?;
        let (dw, dh) = dims_u32(dst)?;
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        let n = dw as usize * sh as usize * C;
        let mut scratch = unsafe { stream.alloc::<u16>(n) }.map_err(err)?;
        launch_pyrdown_u8(ctx, stream, s, d, &mut scratch, sw, sh, dw, dh, C as u32).map_err(err)
    }

    pub(super) fn pyrup_u8_cuda<const C: usize>(
        src: &Image<u8, C>,
        dst: &mut Image<u8, C>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let (sw, sh) = dims_u32(src)?;
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        let n = (sw as usize * 2) * sh as usize * C;
        let mut scratch = unsafe { stream.alloc::<u8>(n) }.map_err(err)?;
        launch_pyrup_u8(ctx, stream, s, d, &mut scratch, sw, sh, C as u32).map_err(err)
    }
}

/// Device Gaussian pyramid with every buffer (levels AND pass scratch)
/// preallocated — the steady state allocates nothing, so `run` is a pure
/// kernel-launch sequence (CUDA-Graph-capturable).
#[cfg(feature = "cuda")]
pub struct PyramidPlan<const C: usize> {
    levels: Vec<Image<f32, C>>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl<const C: usize> PyramidPlan<C> {
    /// Allocate a plan producing `max_level` downsampled levels of `size`
    /// (level 0 buffer holds a copy of the input; levels shrink by 2×).
    pub fn new(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        size: ImageSize,
        max_level: usize,
    ) -> Result<Self, ImageError> {
        let mut levels = Vec::with_capacity(max_level + 1);
        let (mut w, mut h) = (size.width, size.height);
        for _ in 0..=max_level {
            levels.push(Image::<f32, C>::zeros_cuda(
                ImageSize {
                    width: w,
                    height: h,
                },
                stream,
            )?);
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        Ok(Self {
            levels,
            stream: stream.clone(),
        })
    }

    /// Run the pyramid: `src` must match level 0's size and residency.
    /// After return, [`Self::levels`] holds the Gaussian pyramid.
    pub fn run(&mut self, src: &Image<f32, C>) -> Result<(), ImageError> {
        if src.size() != self.levels[0].size() {
            return Err(ImageError::InvalidImageSize(
                self.levels[0].width(),
                self.levels[0].height(),
                src.width(),
                src.height(),
            ));
        }
        // Level 0 = copy of src (device-to-device, stream-ordered).
        {
            let s0 = src
                .0
                .as_cudaslice()
                .ok_or_else(|| ImageError::Cuda("plan src not device-resident".into()))?;
            let d0 = self.levels[0]
                .0
                .as_cudaslice_mut()
                .ok_or_else(|| ImageError::Cuda("plan level not device-resident".into()))?;
            self.stream
                .memcpy_dtod(s0, d0)
                .map_err(|e| ImageError::Cuda(e.to_string()))?;
        }
        // Levels are device-resident by construction, so call the adapter
        // directly (no per-level residency re-check) — one shared path to
        // the launcher instead of open-coding slice extraction here.
        for i in 1..self.levels.len() {
            let (prev, rest) = self.levels.split_at_mut(i);
            cuda_adapters::pyrdown_f32_cuda(&prev[i - 1], &mut rest[0], &self.stream)?;
        }
        Ok(())
    }

    /// The pyramid levels (level 0 = full resolution).
    pub fn levels(&self) -> &[Image<f32, C>] {
        &self.levels
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32, pattern_u8};

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// f32 pyrdown/pyrup device output must be BIT-identical to the CPU
    /// (fused 5x5 weights, polyphase expressions, fmad-free), including the
    /// reflect_101 border ring and degenerate 1-wide/1-tall sources.
    #[test]
    fn f32_pyramid_device_equals_host_bitexact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (5, 4), (1, 7), (8, 1)] {
            let src = Image::<f32, 3>::new(sz(w, h), pattern_f32(w * h * 3)).unwrap();
            let d_src = src.to_cuda(&stream).unwrap();

            // pyrdown
            let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
            let mut cpu = Image::<f32, 3>::from_size_val(sz(dw, dh), 0.0).unwrap();
            pyrdown_f32(&src, &mut cpu).unwrap();
            let mut d_dst = Image::<f32, 3>::zeros_cuda(sz(dw, dh), &stream).unwrap();
            pyrdown_f32(&d_src, &mut d_dst).unwrap();
            let back = d_dst.to_host_owned().unwrap();
            for (i, (a, b)) in back.as_slice().iter().zip(cpu.as_slice()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "pyrdown {w}x{h} elem {i}: {a} vs {b}"
                );
            }

            // pyrup
            let (uw, uh) = (w * 2, h * 2);
            let mut cpu = Image::<f32, 3>::from_size_val(sz(uw, uh), 0.0).unwrap();
            pyrup_f32(&src, &mut cpu).unwrap();
            let mut d_dst = Image::<f32, 3>::zeros_cuda(sz(uw, uh), &stream).unwrap();
            pyrup_f32(&d_src, &mut d_dst).unwrap();
            let back = d_dst.to_host_owned().unwrap();
            for (i, (a, b)) in back.as_slice().iter().zip(cpu.as_slice()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "pyrup {w}x{h} elem {i}: {a} vs {b}"
                );
            }
        }
    }

    /// u8 pyrdown/pyrup must be byte-exact vs the CPU separable paths.
    #[test]
    fn u8_pyramid_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (5, 4), (1, 7), (8, 1)] {
            let src = Image::<u8, 3>::new(sz(w, h), pattern_u8(w * h * 3)).unwrap();
            let d_src = src.to_cuda(&stream).unwrap();

            let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
            let mut cpu = Image::<u8, 3>::from_size_val(sz(dw, dh), 0).unwrap();
            pyrdown_u8(&src, &mut cpu).unwrap();
            let mut d_dst = Image::<u8, 3>::zeros_cuda(sz(dw, dh), &stream).unwrap();
            pyrdown_u8(&d_src, &mut d_dst).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "pyrdown_u8 {w}x{h}"
            );

            let (uw, uh) = (w * 2, h * 2);
            let mut cpu = Image::<u8, 3>::from_size_val(sz(uw, uh), 0).unwrap();
            pyrup_u8(&src, &mut cpu).unwrap();
            let mut d_dst = Image::<u8, 3>::zeros_cuda(sz(uw, uh), &stream).unwrap();
            pyrup_u8(&d_src, &mut d_dst).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "pyrup_u8 {w}x{h}"
            );
        }
    }

    /// PyramidPlan levels must be bit-identical to the CPU build_pyramid.
    #[test]
    fn pyramid_plan_matches_cpu_build_pyramid() {
        let stream = default_stream();
        let (w, h) = (97usize, 61usize);
        let src = Image::<f32, 1>::new(sz(w, h), pattern_f32(w * h)).unwrap();
        let cpu_levels = build_pyramid(&src, 3).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut plan = PyramidPlan::<1>::new(&stream, sz(w, h), 3).unwrap();
        plan.run(&d_src).unwrap();
        // Reuse: second run must produce the same result from the same input.
        plan.run(&d_src).unwrap();

        assert_eq!(plan.levels().len(), cpu_levels.len());
        for (li, (gpu_lvl, cpu_lvl)) in plan.levels().iter().zip(&cpu_levels).enumerate() {
            let back = gpu_lvl.to_host_owned().unwrap();
            for (i, (a, b)) in back.as_slice().iter().zip(cpu_lvl.as_slice()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "plan level {li} elem {i}: {a} vs {b}"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_probe {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32, pattern_u8};

    /// Sustained 1080p probes + single launches for ncu (ignored).
    #[test]
    #[ignore]
    fn probe_pyramid_1080p() {
        let stream = default_stream();
        let sz = ImageSize {
            width: 1920,
            height: 1080,
        };
        let dsz = ImageSize {
            width: 960,
            height: 540,
        };
        let f = Image::<f32, 1>::new(sz, pattern_f32(1920 * 1080)).unwrap();
        let u = Image::<u8, 3>::new(sz, pattern_u8(1920 * 1080 * 3)).unwrap();
        let df = f.to_cuda(&stream).unwrap();
        let du = u.to_cuda(&stream).unwrap();
        let mut df_d = Image::<f32, 1>::zeros_cuda(dsz, &stream).unwrap();
        let mut du_d = Image::<u8, 3>::zeros_cuda(dsz, &stream).unwrap();

        let bench_min = |name: &str, mut op: Box<dyn FnMut() + '_>| {
            for _ in 0..30 {
                op();
            }
            stream.synchronize().unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                for _ in 0..100 {
                    op();
                }
                stream.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / 100.0);
            }
            println!("{name}: {best:.3} ms (min of 5x100)");
        };

        bench_min(
            "pyrdown_f32 C1",
            Box::new(|| pyrdown_f32(&df, &mut df_d).unwrap()),
        );
        bench_min(
            "pyrdown_u8 C3",
            Box::new(|| pyrdown_u8(&du, &mut du_d).unwrap()),
        );
        let mut plan = PyramidPlan::<1>::new(&stream, sz, 3).unwrap();
        bench_min(
            "PyramidPlan 4-level f32 C1 1080p",
            Box::new(|| plan.run(&df).unwrap()),
        );
    }

    /// One launch per kernel for ncu (ignored).
    #[test]
    #[ignore]
    fn ncu_single_launches() {
        let stream = default_stream();
        let sz = ImageSize {
            width: 1920,
            height: 1080,
        };
        let dsz = ImageSize {
            width: 960,
            height: 540,
        };
        let usz = ImageSize {
            width: 3840,
            height: 2160,
        };
        let f = Image::<f32, 1>::new(sz, pattern_f32(1920 * 1080)).unwrap();
        let u = Image::<u8, 3>::new(sz, pattern_u8(1920 * 1080 * 3)).unwrap();
        let df = f.to_cuda(&stream).unwrap();
        let du = u.to_cuda(&stream).unwrap();
        let mut d1 = Image::<f32, 1>::zeros_cuda(dsz, &stream).unwrap();
        let mut d2 = Image::<u8, 3>::zeros_cuda(dsz, &stream).unwrap();
        let mut u1 = Image::<f32, 1>::zeros_cuda(usz, &stream).unwrap();
        let mut u2 = Image::<u8, 3>::zeros_cuda(usz, &stream).unwrap();
        pyrdown_f32(&df, &mut d1).unwrap();
        pyrdown_u8(&du, &mut d2).unwrap();
        pyrup_f32(&df, &mut u1).unwrap();
        pyrup_u8(&du, &mut u2).unwrap();
        stream.synchronize().unwrap();
    }
}
