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

    {
        use std::any::TypeId;
        if C == 3 && TypeId::of::<T>() == TypeId::of::<u8>() {
            // Pick the per-row kernel by architecture once; the rayon
            // dispatcher below stays generic over the kernel choice.
            #[cfg(target_arch = "aarch64")]
            let kernel: fn(&[u8], &mut [u8], usize) = hflip_rgb_u8_neon;
            #[cfg(target_arch = "x86_64")]
            let kernel: fn(&[u8], &mut [u8], usize) = {
                let cpu = crate::simd::cpu_features();
                if cpu.has_avx2 {
                    hflip_rgb_u8_avx2_dispatch
                } else {
                    hflip_rgb_u8_scalar
                }
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let kernel: fn(&[u8], &mut [u8], usize) = hflip_rgb_u8_scalar;

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
                            kernel(src_row, dst_row, cols);
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

/// Portable scalar RGB u8 horizontal flip — fallback when SIMD isn't available.
#[allow(dead_code)]
fn hflip_rgb_u8_scalar(src: &[u8], dst: &mut [u8], cols: usize) {
    for x in 0..cols {
        let s = x * 3;
        let d = (cols - 1 - x) * 3;
        dst[d] = src[s];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s + 2];
    }
}

/// AVX2 wrapper that re-marks the body with `target_feature` so callers can
/// take its address as a `fn` pointer (since `target_feature`-gated unsafe
/// fns can't be coerced directly).
#[cfg(target_arch = "x86_64")]
fn hflip_rgb_u8_avx2_dispatch(src: &[u8], dst: &mut [u8], cols: usize) {
    // SAFETY: caller path verified `cpu.has_avx2` before installing this fn ptr.
    unsafe { hflip_rgb_u8_avx2(src, dst, cols) }
}

/// AVX2 RGB u8 horizontal flip. Processes 5 pixels (15 bytes) per iteration
/// via SSSE3 `_mm_shuffle_epi8` with a constant lane-reverse mask, going
/// right-to-left in src so each batch's lane-15 don't-care byte is overwritten
/// by the next-lower batch (or the scalar tail).
///
/// Two end-of-row hazards to manage:
/// * **Source over-read on the rightmost batch.** `_mm_loadu_si128` always
///   reads 16 bytes; at `x = cols-5` the 16th byte sits one past the row
///   (`row_bytes = cols*3`). On numpy buffers without trailing slack this
///   SIGSEGVs, so the rightmost 5 pixels are handled scalar.
/// * **Destination over-write on the leftmost batch.** When `cols % 5 == 0`
///   the leftmost SIMD batch sits at the row end with no successor to clobber
///   its don't-care byte — handle it with safe 8+4+2+1 partial stores.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hflip_rgb_u8_avx2(src: &[u8], dst: &mut [u8], cols: usize) {
    use std::arch::x86_64::*;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    // Reverses 5 RGB pixels packed as 15 bytes within a 16-byte register.
    let mask = _mm_setr_epi8(12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1);
    let bulk_start = cols % 5;

    // Rightmost 5 src pixels → leftmost 5 dst pixels: scalar to avoid the
    // 1-byte source over-read that an aligned-by-3 16-byte SIMD load would
    // commit at the very end of the row.
    if cols >= 5 {
        for k in 0..5 {
            let xs = cols - 5 + k;
            let s = sp.add(xs * 3);
            let d = dp.add((cols - 1 - xs) * 3);
            *d = *s;
            *d.add(1) = *s.add(1);
            *d.add(2) = *s.add(2);
        }
    }

    // SIMD bulk iterates src x = cols-10, cols-15, ..., bulk_start.
    // Needs cols >= 10 since the rightmost batch (x=cols-5) is handled above.
    if cols >= 10 {
        let mut x = cols - 10;
        loop {
            let v = _mm_loadu_si128(sp.add(x * 3) as *const __m128i);
            let rev = _mm_shuffle_epi8(v, mask);
            let dst_pix = cols - x - 5;
            let dp0 = dp.add(dst_pix * 3);

            // Only the leftmost iter (x == bulk_start) AND no scalar tail
            // (bulk_start == 0) hits the buffer end. All other iters write
            // 16 bytes safely — their lane-15 don't-care is overwritten.
            if x == bulk_start && bulk_start == 0 {
                *(dp0 as *mut u64) = _mm_cvtsi128_si64(rev) as u64;
                *(dp0.add(8) as *mut u32) = _mm_extract_epi32::<2>(rev) as u32;
                *(dp0.add(12) as *mut u16) = _mm_extract_epi16::<6>(rev) as u16;
                *dp0.add(14) = _mm_extract_epi8::<14>(rev) as u8;
                break;
            }
            _mm_storeu_si128(dp0 as *mut __m128i, rev);
            if x == bulk_start {
                break;
            }
            x -= 5;
        }
    }

    // Scalar tail: leftmost (cols % 5) src pixels → rightmost dst pixels.
    for x in 0..bulk_start {
        let s = sp.add(x * 3);
        let d = dp.add((cols - 1 - x) * 3);
        *d = *s;
        *d.add(1) = *s.add(1);
        *d.add(2) = *s.add(2);
    }
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
    fn test_horizontal_flip_rgb_u8_cols_multiple_of_five() -> Result<(), ImageError> {
        // cols=80 (cols % 5 == 0) is the case that historically tripped the
        // AVX2 path: the rightmost 5-pixel SIMD batch's 16-byte load read 1
        // byte past the row end, and the leftmost batch's 16-byte store
        // wrote 1 byte past dst end. Bit-identical scalar comparison.
        let image_size = ImageSize {
            width: 80,
            height: 4,
        };
        let pixels: Vec<u8> = (0..(80 * 4 * 3)).map(|i| (i & 0xff) as u8).collect();
        let image = Image::<u8, 3, _>::new(image_size, pixels.clone(), CpuAllocator)?;
        let mut flipped = Image::<u8, 3, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::horizontal_flip(&image, &mut flipped)?;

        let mut expected = vec![0u8; 80 * 4 * 3];
        for r in 0..4 {
            for c in 0..80 {
                let s = (r * 80 + c) * 3;
                let d = (r * 80 + (79 - c)) * 3;
                expected[d..d + 3].copy_from_slice(&pixels[s..s + 3]);
            }
        }
        assert_eq!(flipped.as_slice(), &expected[..]);
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
