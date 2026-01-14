use kornia_image::{allocator::ImageAllocator, Image, ImageError};
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
fn pyrup_horizontal_pass_par<const C: usize, A>(
    src: &Image<f32, C, A>,
    dst: &mut [f32],
    dst_width: usize,
) where
    A: ImageAllocator,
{
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
fn pyrup_vertical_pass_par<const C: usize>(
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
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::pyramid::pyrup;
///
/// let image = Image::<f32, 1, _>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0.0, 1.0, 2.0, 3.0],
///     CpuAllocator
/// ).unwrap();
///
/// let mut upsampled = Image::<f32, 1, _>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     0.0,
///     CpuAllocator
/// ).unwrap();
///
/// pyrup(&image, &mut upsampled).unwrap();
/// ```
pub fn pyrup<const C: usize, A1, A2>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
) -> Result<(), ImageError>
where
    A1: ImageAllocator,
    A2: ImageAllocator,
{
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

    // Intermediate buffer for horizontal pass, pyrup_horizontal writes here
    let mut buffer = vec![0.0f32; dst.width() * src.height() * C];
    pyrup_horizontal_pass_par::<C, _>(src, &mut buffer, dst.width());

    // Vertical pass reads from buffer and finally writes to dst
    let dst_width = dst.width();
    pyrup_vertical_pass_par::<C>(
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
/// and then subsampling every other pixel. This is the inverse operation of [`pyrup`].
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
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::pyramid::pyrdown;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     (0..48).map(|x| x as f32).collect(),
///     CpuAllocator,
/// ).unwrap();
///
/// let mut downsampled = Image::<f32, 3, _>::from_size_val(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     0.0,
///     CpuAllocator,
/// ).unwrap();
///
/// pyrdown(&image, &mut downsampled).unwrap();
/// ```
pub fn pyrdown<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
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

    let src_width = src.width();
    let src_height = src.height();
    let dst_width = dst.width();
    let dst_height = dst.height();

    // Fused Gaussian blur + downsample in a single pass.
    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();
    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();

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

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    #[test]
    fn test_pyrup() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 1.0, 2.0, 3.0],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrup(&src, &mut dst)?;

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
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrdown(&src, &mut dst)?;

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
        let src = Image::<f32, 3, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..48).map(|x| x as f32).collect(),
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 3, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrdown(&src, &mut dst)?;

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
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0.0; 16],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0.0,
            CpuAllocator,
        )?;

        let result = pyrdown(&src, &mut dst);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pyrdown_odd_dims() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 5,
                height: 7,
            },
            (0..(5 * 7)).map(|x| x as f32).collect(),
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrdown(&src, &mut dst)?;

        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 4);
        for &v in dst.as_slice() {
            assert!(v.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_min_sizes() -> Result<(), ImageError> {
        let src1 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![42.0],
            CpuAllocator,
        )?;
        let mut dst1 = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src1, &mut dst1)?;
        assert_eq!(dst1.width(), 1);
        assert_eq!(dst1.height(), 1);
        assert!(dst1.as_slice().iter().all(|v| v.is_finite()));

        let src2 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![1.0, 2.0],
            CpuAllocator,
        )?;
        let mut dst2 = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src2, &mut dst2)?;
        assert_eq!(dst2.width(), 1);
        assert_eq!(dst2.height(), 1);
        assert!(dst2.as_slice().iter().all(|v| v.is_finite()));

        let src3 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1.0, 2.0],
            CpuAllocator,
        )?;
        let mut dst3 = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src3, &mut dst3)?;
        assert_eq!(dst3.width(), 1);
        assert_eq!(dst3.height(), 1);
        eprintln!("dst3 values: {:?}", dst3.as_slice());
        assert!(dst3.as_slice().iter().all(|v| v.is_finite()));

        Ok(())
    }

    #[test]
    fn test_pyrdown_numeric_extremes() -> Result<(), ImageError> {
        let large_val = 1e9_f32;
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![large_val; 16],
            CpuAllocator,
        )?;
        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src, &mut dst)?;
        for &v in dst.as_slice() {
            assert!(v.is_finite());
            assert!(v.abs() <= large_val);
        }

        Ok(())
    }
}
