use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

/// Flip the input image horizontally.
///
/// # Arguments
///
/// * `src` - The input image with shape (H, W, C).
/// * `dst` - The output image with shape (H, W, C).
///
/// Precondition: the input and output images must have the same size.
///
/// # Errors
///
/// Returns an error if the sizes of `src` and `dst` do not match.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::flip::horizontal_flip;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0f32; 2 * 3 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut flipped = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// horizontal_flip(&image, &mut flipped).unwrap();
/// ```
pub fn horizontal_flip<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + 'static,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::any::TypeId;
        if C == 3 && TypeId::of::<T>() == TypeId::of::<u8>() {
            let cols = src.cols();
            let row_bytes = cols * 3;
            // Safety: byte view of u8 C=3 data.
            let src_bytes = unsafe {
                std::slice::from_raw_parts(
                    src.as_slice().as_ptr() as *const u8,
                    std::mem::size_of_val(src.as_slice()),
                )
            };
            let dst_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    dst.as_slice_mut().as_mut_ptr() as *mut u8,
                    std::mem::size_of_val(dst.as_slice()),
                )
            };
            const ROWS_PER_TASK: usize = 16;
            dst_bytes
                .par_chunks_mut(ROWS_PER_TASK * row_bytes)
                .zip_eq(src_bytes.par_chunks(ROWS_PER_TASK * row_bytes))
                .for_each(|(dst_big, src_big)| {
                    dst_big
                        .chunks_exact_mut(row_bytes)
                        .zip(src_big.chunks_exact(row_bytes))
                        .for_each(|(dst_row, src_row)| {
                            hflip_rgb_u8_neon(src_row, dst_row, cols);
                        });
                });
            return Ok(());
        }
    }

    const ROWS_PER_TASK: usize = 16;
    let row_len = src.cols() * C;
    dst.as_slice_mut()
        .par_chunks_mut(ROWS_PER_TASK * row_len)
        .zip_eq(src.as_slice().par_chunks(ROWS_PER_TASK * row_len))
        .for_each(|(dst_big, src_big)| {
            dst_big
                .chunks_exact_mut(row_len)
                .zip(src_big.chunks_exact(row_len))
                .for_each(|(dst_row, src_row)| {
                    dst_row
                        .chunks_exact_mut(C)
                        .zip(src_row.chunks_exact(C).rev())
                        .for_each(|(dst_pixel, src_pixel)| {
                            dst_pixel.copy_from_slice(src_pixel);
                        });
                });
        });

    Ok(())
}

/// NEON RGB u8 horizontal flip. Reads 16 src pixels via vld3q_u8, reverses
/// each channel register in place (vrev64q_u8 + vextq_u8), then writes them
/// to the mirrored destination position via vst3q_u8.
///
/// The head (last) tail of dst is written first since we process src left→right,
/// with the store position moving right→left. Any pixels that don't fit a
/// 16-pixel batch are handled scalar.
#[cfg(target_arch = "aarch64")]
#[inline]
fn hflip_rgb_u8_neon(src: &[u8], dst: &mut [u8], cols: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        // Process in batches of 16 pixels from the left.
        let bulk = cols & !15;
        let mut x = 0usize;
        while x < bulk {
            let v = vld3q_u8(sp.add(x * 3));
            // Reverse 16 lanes: vrev64q swaps within each 64-bit half,
            // then vextq_u8(hi, lo, 8) produces the fully reversed 128-bit vector.
            let r_rev = vextq_u8(vrev64q_u8(v.0), vrev64q_u8(v.0), 8);
            let g_rev = vextq_u8(vrev64q_u8(v.1), vrev64q_u8(v.1), 8);
            let b_rev = vextq_u8(vrev64q_u8(v.2), vrev64q_u8(v.2), 8);
            let out = uint8x16x3_t(r_rev, g_rev, b_rev);
            // Destination position: src pixel x..x+16 maps to dst cols-x-16..cols-x.
            vst3q_u8(dp.add((cols - x - 16) * 3), out);
            x += 16;
        }
        // Scalar tail: whatever remains (last cols % 16 pixels of src → first pixels of dst).
        while x < cols {
            let s = sp.add(x * 3);
            let d = dp.add((cols - 1 - x) * 3);
            *d = *s;
            *d.add(1) = *s.add(1);
            *d.add(2) = *s.add(2);
            x += 1;
        }
    }
}

/// Flip the input image vertically.
///
/// # Arguments
///
/// * `src` - The input image with shape (H, W, C).
/// * `dst` - The output image with shape (H, W, C).
///
/// Precondition: the input and output images must have the same size.
///
/// # Errors
///
/// Returns an error if the sizes of `src` and `dst` do not match.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::flip::vertical_flip;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0f32; 2 * 3 * 3],
///     CpuAllocator,
/// )
/// .unwrap();
///
/// let mut flipped = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// vertical_flip(&image, &mut flipped).unwrap();
///
/// ```
pub fn vertical_flip<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let row_len = src.cols() * C;
    let rows = src.rows();

    // Group rows into coarse chunks so rayon task count stays ~O(cores×16),
    // not O(rows). Per-row parallelism buries memcpy in spawn overhead at any
    // resolution up to 8K. Within a chunk we use raw-ptr copy_nonoverlapping
    // to skip per-row slice-bounds checks (visible at 1080p: ~100 μs of 930).
    const ROWS_PER_TASK: usize = 16;
    let chunk_elems = ROWS_PER_TASK * row_len;

    // Pass src as usize address; raw pointers aren't Send. Each rayon task
    // reconstructs the pointer and reads its own disjoint row ranges.
    let src_addr = src.as_slice().as_ptr() as usize;
    let dst_slice = dst.as_slice_mut();

    dst_slice
        .par_chunks_mut(chunk_elems)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let n_rows_in_chunk = dst_chunk.len() / row_len;
            let dst_row_base = chunk_idx * ROWS_PER_TASK;
            let dst_ptr = dst_chunk.as_mut_ptr();
            let src_ptr = src_addr as *const T;
            // SAFETY: src.size() == dst.size() was validated at entry; each
            // rayon task owns a disjoint dst range, and src reads are also
            // disjoint (one-to-one row mirror). No aliasing possible.
            unsafe {
                for r in 0..n_rows_in_chunk {
                    let src_row_idx = rows - 1 - (dst_row_base + r);
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(src_row_idx * row_len),
                        dst_ptr.add(r * row_len),
                        row_len,
                    );
                }
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_hflip() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };
        let image = Image::<_, 3, _>::new(
            image_size,
            vec![
                0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            ],
            CpuAllocator,
        )?;
        let data_expected = vec![
            3u8, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14,
        ];
        let mut flipped = Image::<_, 3, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::horizontal_flip(&image, &mut flipped)?;
        assert_eq!(flipped.as_slice(), &data_expected);
        Ok(())
    }

    #[test]
    fn test_vflip() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };
        let image = Image::<_, 3, _>::new(
            image_size,
            vec![
                0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            ],
            CpuAllocator,
        )?;
        let data_expected = vec![
            12u8, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
        ];
        let mut flipped = Image::<_, 3, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::vertical_flip(&image, &mut flipped)?;
        assert_eq!(flipped.as_slice(), &data_expected);
        Ok(())
    }

    #[test]
    fn test_horizontal_flip_single_row() -> Result<(), ImageError> {
        // A single-row (rows=1) image flipped horizontally must reverse the pixel order.
        // Image layout: 1 row × 4 columns × 1 channel.
        let image_size = ImageSize {
            width: 4,
            height: 1,
        };
        let image = Image::<_, 1, _>::new(image_size, vec![10u8, 20, 30, 40], CpuAllocator)?;
        let mut flipped = Image::<_, 1, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::horizontal_flip(&image, &mut flipped)?;
        // Columns reversed: [40, 30, 20, 10]
        assert_eq!(flipped.as_slice(), &[40u8, 30, 20, 10]);
        Ok(())
    }

    #[test]
    fn test_vertical_flip_single_column() -> Result<(), ImageError> {
        // A single-column (cols=1) image flipped vertically must reverse the row order.
        // Image layout: 4 rows × 1 column × 1 channel.
        let image_size = ImageSize {
            width: 1,
            height: 4,
        };
        let image = Image::<_, 1, _>::new(image_size, vec![10u8, 20, 30, 40], CpuAllocator)?;
        let mut flipped = Image::<_, 1, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::vertical_flip(&image, &mut flipped)?;
        // Rows reversed: [40, 30, 20, 10]
        assert_eq!(flipped.as_slice(), &[40u8, 30, 20, 10]);
        Ok(())
    }
}
