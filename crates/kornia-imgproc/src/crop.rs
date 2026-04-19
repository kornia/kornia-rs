use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

/// Hand-rolled per-row copy optimized for strided source reads on ARM64.
///
/// Issues an L1 streaming prefetch before copying and uses a 32-byte-per-iter
/// NEON loop (two 128-bit vld1/vst1) to hide the stride latency of jumping
/// between non-contiguous source rows. Tail is handled with 16-byte and
/// byte-wise remainders.
///
/// # Safety
/// - `src` must be valid for reads of `n` bytes.
/// - `dst` must be valid for writes of `n` bytes.
/// - `src` and `dst` must not overlap.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn copy_row_neon(src: *const u8, dst: *mut u8, n: usize) {
    use std::arch::aarch64::{vld1q_u8, vst1q_u8};
    use std::arch::asm;

    // Prefetch ~2KB ahead of the current row to hide L2 fetch latency on the
    // next iteration (rows are spaced by src stride bytes; prefetcher won't
    // see the pattern across rows, so we hint manually).
    asm!(
        "prfm pldl1strm, [{0}, #2048]",
        in(reg) src,
        options(nostack, preserves_flags, readonly)
    );

    let mut i: usize = 0;
    // Main 32-byte loop: two 128-bit loads/stores per iter.
    while i + 32 <= n {
        let s0 = src.add(i);
        let s1 = src.add(i + 16);
        let d0 = dst.add(i);
        let d1 = dst.add(i + 16);
        let v0 = vld1q_u8(s0);
        let v1 = vld1q_u8(s1);
        vst1q_u8(d0, v0);
        vst1q_u8(d1, v1);
        i += 32;
    }
    // 16-byte remainder.
    if i + 16 <= n {
        let v = vld1q_u8(src.add(i));
        vst1q_u8(dst.add(i), v);
        i += 16;
    }
    // Byte tail.
    if i < n {
        std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), n - i);
    }
}

/// Copy one destination row from the source slice.
///
/// On aarch64 with `T` = 1 byte and a row of at least 128 bytes, dispatches to
/// the NEON + prefetch path. Otherwise falls back to `copy_from_slice`.
#[inline(always)]
fn copy_row<T: Copy>(src_row: &[T], dst_row: &mut [T]) {
    debug_assert_eq!(src_row.len(), dst_row.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::mem::size_of::<T>() == 1 && dst_row.len() >= 128 {
            // SAFETY: src/dst are distinct slices of equal length; T is 1 byte.
            unsafe {
                copy_row_neon(
                    src_row.as_ptr() as *const u8,
                    dst_row.as_mut_ptr() as *mut u8,
                    dst_row.len(),
                );
            }
            return;
        }
    }

    dst_row.copy_from_slice(src_row);
}

/// Crop an image to a specified region.
///
/// # Arguments
///
/// * `src` - The source image to crop.
/// * `dst` - The destination image to store the cropped image.
/// * `x` - The x-coordinate of the top-left corner of the region to crop.
/// * `y` - The y-coordinate of the top-left corner of the region to crop.
///
/// # Examples
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::crop::crop_image;
///
/// let image = Image::<_, 1, _>::new(ImageSize { width: 4, height: 4 }, vec![
///     0u8, 1, 2, 3,
///     4u8, 5, 6, 7,
///     8u8, 9, 10, 11,
///     12u8, 13, 14, 15
/// ], CpuAllocator).unwrap();
///
/// let mut cropped = Image::<_, 1, _>::from_size_val(ImageSize { width: 2, height: 2 }, 0u8, CpuAllocator).unwrap();
///
/// crop_image(&image, &mut cropped, 1, 1).unwrap();
///
/// assert_eq!(cropped.as_slice(), &[5u8, 6, 9, 10]);
/// ```
pub fn crop_image<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    x: usize,
    y: usize,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    let dst_cols = dst.cols();
    let src_cols = src.cols();
    let row_bytes = dst_cols * C * std::mem::size_of::<T>();
    // Below ~1 MB rayon overhead dominates a plain memcpy loop.
    let total_bytes = dst.rows() * row_bytes;
    let dst_row_len = dst_cols * C;
    let src_slice = src.as_slice();

    if total_bytes < 1 << 20 {
        for (i, dst_row) in dst.as_slice_mut().chunks_exact_mut(dst_row_len).enumerate() {
            let offset = (y + i) * src_cols * C + x * C;
            copy_row(&src_slice[offset..offset + dst_row_len], dst_row);
        }
    } else {
        dst.as_slice_mut()
            .par_chunks_exact_mut(dst_row_len)
            .enumerate()
            .for_each(|(i, dst_row)| {
                let offset = (y + i) * src_cols * C + x * C;
                copy_row(&src_slice[offset..offset + dst_row_len], dst_row);
            });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_crop() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };

        #[rustfmt::skip]
        let image = Image::<_, 3, _>::new(
            image_size,
            vec![
                0u8, 1, 2, 3, 4, 5,
                6u8, 7, 8, 9, 10, 11,
                12u8, 13, 14, 15, 16, 17,
            ],
            CpuAllocator
        )?;

        let data_expected = vec![9u8, 10, 11, 15, 16, 17];

        let crop_size = ImageSize {
            width: 1,
            height: 2,
        };

        let mut cropped = Image::<_, 3, _>::from_size_val(crop_size, 0u8, CpuAllocator)?;

        super::crop_image(&image, &mut cropped, 1, 1)?;

        assert_eq!(cropped.as_slice(), &data_expected);

        Ok(())
    }

    #[test]
    fn test_crop_full_image() -> Result<(), ImageError> {
        // Cropping the entire image (x=0, y=0, dst size == src size) must yield
        // pixel-identical data to the original.
        let image_size = ImageSize {
            width: 3,
            height: 2,
        };

        #[rustfmt::skip]
        let data = vec![
            10u8, 20, 30, 40, 50, 60,
            70u8, 80, 90, 100, 110, 120,
        ];

        let image = Image::<_, 2, _>::new(image_size, data.clone(), CpuAllocator)?;
        let mut dst = Image::<_, 2, _>::from_size_val(image_size, 0u8, CpuAllocator)?;

        super::crop_image(&image, &mut dst, 0, 0)?;

        assert_eq!(dst.as_slice(), data.as_slice());

        Ok(())
    }

    #[test]
    fn test_crop_single_pixel() -> Result<(), ImageError> {
        // A 1x1 crop from a known location must return exactly that pixel's channels.
        let image_size = ImageSize {
            width: 4,
            height: 4,
        };

        #[rustfmt::skip]
        let image = Image::<_, 1, _>::new(
            image_size,
            vec![
                 0u8,  1,  2,  3,
                 4u8,  5,  6,  7,
                 8u8,  9, 10, 11,
                12u8, 13, 14, 15,
            ],
            CpuAllocator,
        )?;

        let crop_size = ImageSize {
            width: 1,
            height: 1,
        };
        let mut dst = Image::<_, 1, _>::from_size_val(crop_size, 0u8, CpuAllocator)?;

        // Pixel at column 2, row 3 is value 14.
        super::crop_image(&image, &mut dst, 2, 3)?;

        assert_eq!(dst.as_slice(), &[14u8]);

        Ok(())
    }

    #[test]
    fn test_crop_neon_large_row() -> Result<(), ImageError> {
        // 224x224 RGB crop from 1920x1080 image exercises the NEON path
        // (row_bytes = 672 > 128, 32-byte main loop + 16-byte tail).
        let src_w = 1920;
        let src_h = 1080;
        let mut src_data = vec![0u8; src_w * src_h * 3];
        for (i, v) in src_data.iter_mut().enumerate() {
            *v = (i & 0xFF) as u8;
        }
        let src = Image::<_, 3, _>::new(
            ImageSize {
                width: src_w,
                height: src_h,
            },
            src_data.clone(),
            CpuAllocator,
        )?;
        let crop_w = 224;
        let crop_h = 224;
        let x0 = 848;
        let y0 = 428;
        let mut dst = Image::<_, 3, _>::from_size_val(
            ImageSize {
                width: crop_w,
                height: crop_h,
            },
            0u8,
            CpuAllocator,
        )?;
        super::crop_image(&src, &mut dst, x0, y0)?;

        // Verify every row matches a scalar crop.
        let dst_slice = dst.as_slice();
        for r in 0..crop_h {
            let dst_row = &dst_slice[r * crop_w * 3..(r + 1) * crop_w * 3];
            let src_off = (y0 + r) * src_w * 3 + x0 * 3;
            let src_row = &src_data[src_off..src_off + crop_w * 3];
            assert_eq!(dst_row, src_row, "row {} mismatch", r);
        }
        Ok(())
    }

    #[test]
    fn test_crop_odd_row_width() -> Result<(), ImageError> {
        // Exercise tail: 17 bytes (not multiple of 16) per row * 3 channels = 51 bytes.
        // Use width so row_bytes > 128 to hit the NEON path with all tail branches.
        let src_w = 64;
        let src_h = 8;
        let mut src_data = vec![0u8; src_w * src_h * 3];
        for (i, v) in src_data.iter_mut().enumerate() {
            *v = ((i * 7) & 0xFF) as u8;
        }
        let src = Image::<_, 3, _>::new(
            ImageSize {
                width: src_w,
                height: src_h,
            },
            src_data.clone(),
            CpuAllocator,
        )?;
        // 43 columns * 3 chans = 129 bytes row (hits main + 16-byte tail + 1-byte tail).
        let crop_w = 43;
        let crop_h = 4;
        let x0 = 5;
        let y0 = 2;
        let mut dst = Image::<_, 3, _>::from_size_val(
            ImageSize {
                width: crop_w,
                height: crop_h,
            },
            0u8,
            CpuAllocator,
        )?;
        super::crop_image(&src, &mut dst, x0, y0)?;
        let dst_slice = dst.as_slice();
        for r in 0..crop_h {
            let dst_row = &dst_slice[r * crop_w * 3..(r + 1) * crop_w * 3];
            let src_off = (y0 + r) * src_w * 3 + x0 * 3;
            let src_row = &src_data[src_off..src_off + crop_w * 3];
            assert_eq!(dst_row, src_row, "row {} mismatch", r);
        }
        Ok(())
    }
}
