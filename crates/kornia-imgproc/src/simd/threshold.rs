//! SIMD binary threshold.

use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;
use wide::CmpGt;

/// SIMD binary threshold for `u8` images: `out = src > threshold ? max_value : 0`.
///
/// API matches [`crate::threshold::threshold_binary`] when `T = u8`.
pub fn threshold_binary<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    threshold: u8,
    max_value: u8,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // One contiguous logical buffer. Parallelize over row-aligned chunks so
    // the SIMD lanes inside each chunk stay cache-friendly.
    let row_len = C * src.cols();
    src.as_slice()
        .par_chunks_exact(row_len)
        .zip(dst.as_slice_mut().par_chunks_exact_mut(row_len))
        .for_each(|(s, d)| threshold_row_u8(s, d, threshold, max_value));

    Ok(())
}

#[inline]
fn threshold_row_u8(src: &[u8], dst: &mut [u8], threshold: u8, max_value: u8) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                super::threshold_x86::threshold_row_u8_avx2(src, dst, threshold, max_value);
            }
            return;
        }
    }
    threshold_row_u8_wide(src, dst, threshold, max_value);
}

#[inline]
fn threshold_row_u8_wide(src: &[u8], dst: &mut [u8], threshold: u8, max_value: u8) {
    use wide::i8x16;
    const LANES: usize = 16;

    // `wide::u8x16` doesn't expose an unsigned compare; use i8x16 with an
    // XOR-0x80 offset so a signed compare yields unsigned ordering.
    let bias = i8x16::splat(-128);
    let thr_biased = i8x16::splat((threshold ^ 0x80) as i8);
    let max_v = i8x16::splat(max_value as i8);
    let zero_v = i8x16::splat(0);

    let mut i = 0;
    let n = src.len();
    while i + LANES <= n {
        let chunk: [u8; LANES] = src[i..i + LANES].try_into().unwrap();
        let v_u: i8x16 = unsafe { core::mem::transmute(chunk) };
        let v = v_u ^ bias;
        let mask = v.simd_gt(thr_biased);
        let out = mask.blend(max_v, zero_v);
        let out_bytes: [u8; LANES] = unsafe { core::mem::transmute(out) };
        dst[i..i + LANES].copy_from_slice(&out_bytes);
        i += LANES;
    }

    while i < n {
        dst[i] = if src[i] > threshold { max_value } else { 0 };
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threshold::threshold_binary as scalar_threshold_binary;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn mk(w: usize, h: usize) -> Image<u8, 1, CpuAllocator> {
        let data: Vec<u8> = (0..w * h).map(|i| (i * 37 % 256) as u8).collect();
        Image::new(ImageSize { width: w, height: h }, data, CpuAllocator).unwrap()
    }

    #[test]
    fn simd_matches_scalar_aligned() {
        let img = mk(64, 5);
        let mut scalar_out =
            Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        let mut simd_out =
            Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        scalar_threshold_binary(&img, &mut scalar_out, 100u8, 255u8).unwrap();
        threshold_binary(&img, &mut simd_out, 100u8, 255u8).unwrap();
        assert_eq!(scalar_out.as_slice(), simd_out.as_slice());
    }

    #[test]
    fn simd_matches_scalar_tail() {
        // 19 pixels per row — not a multiple of the 16-lane block.
        let img = mk(19, 7);
        let mut scalar_out =
            Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        let mut simd_out =
            Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        scalar_threshold_binary(&img, &mut scalar_out, 77u8, 200u8).unwrap();
        threshold_binary(&img, &mut simd_out, 77u8, 200u8).unwrap();
        assert_eq!(scalar_out.as_slice(), simd_out.as_slice());
    }

    #[test]
    fn simd_boundary_semantics() {
        // Verify strict `>` semantics (matches the scalar impl).
        let data = vec![99u8, 100, 101, 0, 255, 100];
        let img = Image::<u8, 1, _>::new(
            ImageSize { width: 6, height: 1 },
            data,
            CpuAllocator,
        )
        .unwrap();
        let mut dst =
            Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        threshold_binary(&img, &mut dst, 100u8, 255u8).unwrap();
        assert_eq!(dst.as_slice(), &[0, 0, 255, 0, 255, 0]);
    }
}
