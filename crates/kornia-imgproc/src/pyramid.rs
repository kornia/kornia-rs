use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;

// Helper functions

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

            if src_width == 1 {
                for k in 0..C {
                    let val = src_data[src_row_offset + k];
                    dst_row[k] = val;
                    if C < dst_stride {
                        dst_row[k + C] = val;
                    }
                }
                return;
            }

            for k in 0..C {
                let v0 = src_data[src_row_offset + k];
                let v1 = src_data[src_row_offset + C + k];
                // Even
                dst_row[k] = (6.0 * v0 + 2.0 * v1) * 0.125;
                // Odd
                dst_row[C + k] = (v0 + v1) * 0.5;
            }

            for x in 1..(src_width - 1) {
                let off = src_row_offset + x * C;
                let dst_off = 2 * x * C;

                for k in 0..C {
                    let v_prev = src_data[off - C + k];
                    let v_curr = src_data[off + k];
                    let v_next = src_data[off + C + k];

                    // Even
                    dst_row[dst_off + k] = (v_prev + 6.0 * v_curr + v_next) * 0.125;
                    // Odd
                    dst_row[dst_off + C + k] = (v_curr + v_next) * 0.5;
                }
            }

            let last_x = src_width - 1;
            let off_last = src_row_offset + last_x * C;
            let dst_off_last = 2 * last_x * C;

            for k in 0..C {
                let v_prev = src_data[off_last - C + k];
                let v_curr = src_data[off_last + k];

                // Even
                dst_row[dst_off_last + k] = (v_prev + 7.0 * v_curr) * 0.125;
                // Odd
                dst_row[dst_off_last + C + k] = v_curr;

                if dst_off_last + 2 * C < dst_stride {
                    dst_row[dst_off_last + 2 * C + k] = v_curr;
                }
            }
        });
}

fn pyrup_vertical_pass_par<const C: usize>(
    src_buffer: &[f32],
    src_buffer_width: usize,
    src_height: usize,
    dst_data: &mut [f32],
    dst_width: usize,
) {
    let stride = src_buffer_width * C;
    let dst_stride = dst_width * C;

    // Process source rows in parallel
    dst_data
        .par_chunks_mut(2 * dst_stride)
        .enumerate()
        .for_each(|(y, dst_rows_chunk)| {
            if y >= src_height {
                return;
            }

            let (v_t, v_c, v_b);

            if src_height == 1 {
                v_t = 0;
                v_c = 0;
                v_b = 0;
            } else if y == 0 {
                v_t = 0;
                v_c = 0;
                v_b = 1;
            } else if y == src_height - 1 {
                v_t = src_height - 2;
                v_c = src_height - 1;
                v_b = src_height - 1;
            } else {
                v_t = y - 1;
                v_c = y;
                v_b = y + 1;
            }

            let off_t = v_t * stride;
            let off_c = v_c * stride;
            let off_b = v_b * stride;

            let (row_even, row_odd) = dst_rows_chunk.split_at_mut(dst_stride);

            if y == 0 {
                for i in 0..stride {
                    row_even[i] =
                        (6.0 * src_buffer[off_c + i] + 2.0 * src_buffer[off_b + i]) * 0.125;
                    if !row_odd.is_empty() {
                        row_odd[i] = (src_buffer[off_c + i] + src_buffer[off_b + i]) * 0.5;
                    }
                }
            } else if y == src_height - 1 {
                for i in 0..stride {
                    row_even[i] = (src_buffer[off_t + i] + 7.0 * src_buffer[off_c + i]) * 0.125;
                    if !row_odd.is_empty() {
                        row_odd[i] = src_buffer[off_c + i];
                    }
                }
            } else {
                for i in 0..stride {
                    row_even[i] = (src_buffer[off_t + i]
                        + 6.0 * src_buffer[off_c + i]
                        + src_buffer[off_b + i])
                        * 0.125;
                    if !row_odd.is_empty() {
                        row_odd[i] = (src_buffer[off_c + i] + src_buffer[off_b + i]) * 0.5;
                    }
                }
            }
        });
}
// --- Main Function ---

/// This function upsamples an image by a factor of 2. It's efficient
/// by combining upsampling and blurring in a single polyphase operation.
///
/// # Arguments
///
/// * `src` - The source image to be upsampled.
/// * `dst` - The destination image to store the result. Must have dimensions twice the width and height of the source image.
///
/// # Returns
///
/// * `Result<(), ImageError>` - `Ok(())` if successful, `Err(ImageError)` otherwise.
///
/// # Errors
///
/// Returns `ImageError::InvalidImageSize` if the destination image does not
/// have the expected dimensions (2x width, 2x height of the source).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_tensor::CpuAllocator;
/// use kornia_imgproc::pyramid::pyrup;
///
/// let image = Image::<f32, 1, _>::new(
///     ImageSize { width: 2, height: 2 },
///     vec![0.0, 1.0, 2.0, 3.0],
///     CpuAllocator,
/// ).unwrap();
///
/// let mut upsampled = Image::<f32, 1, _>::from_size_val(
///     ImageSize { width: 4, height: 4 },
///     0.0,
///     CpuAllocator,
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

    // Check dimensions
    if (dst.width() as i32 - expected_width as i32).abs() > 1
        || (dst.height() as i32 - expected_height as i32).abs() > 1
    {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    let mut buffer = vec![0.0f32; dst.width() * src.height() * C];
    pyrup_horizontal_pass_par::<C, _>(src, &mut buffer, dst.width());

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
}
