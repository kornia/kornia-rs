//! SIMD grayscale conversion.
//!
//! Currently provides a hand-vectorized `gray_from_rgb_u8` that beats the
//! scalar (auto-vectorized) baseline on x86_64 AVX2 at small/medium image
//! sizes. A float variant was evaluated and dropped from the PoC — LLVM's
//! autovec of the `chunks_exact(3).zip` pattern already emits full AVX2 +
//! FMA for f32, and a hand `vgatherdps`-based path was ~2× slower on our
//! test hardware (gather is micro-coded on Zen/Alder Lake). A shuffle-based
//! f32 deinterleave is future work.

use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;
use wide::u16x8;

/// SIMD RGB8 → Gray8 using the integer formula `Y = (77·R + 150·G + 29·B) >> 8`.
///
/// API mirrors [`crate::color::gray_from_rgb_u8`]. Dispatches to AVX2 on
/// x86_64 where available (32 pixels/iter via `pshufb`-based deinterleave),
/// falling back to an SSSE3 path (16 px/iter) and finally to a portable
/// `wide`-crate path.
pub fn gray_from_rgb_u8<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 1, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let cols = src.cols();

    src.as_slice()
        .par_chunks_exact(3 * cols)
        .zip(dst.as_slice_mut().par_chunks_exact_mut(cols))
        .for_each(|(src_row, dst_row)| gray_row_u8(src_row, dst_row));

    Ok(())
}

/// Per-row dispatch to the best available u8 gray kernel.
#[inline]
fn gray_row_u8(src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                super::color_x86::gray_row_u8_avx2(src, dst);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            unsafe {
                super::color_x86::gray_row_u8_ssse3(src, dst);
            }
            return;
        }
    }
    gray_row_u8_wide(src, dst);
}

/// Portable fallback using the `wide` crate.
#[inline]
fn gray_row_u8_wide(src: &[u8], dst: &mut [u8]) {
    const LANES: usize = 8;
    let rw = u16x8::splat(77);
    let gw = u16x8::splat(150);
    let bw = u16x8::splat(29);

    let mut i = 0;
    let n = dst.len();

    while i + LANES <= n {
        let base = i * 3;
        let mut r = [0u16; LANES];
        let mut g = [0u16; LANES];
        let mut b = [0u16; LANES];
        for k in 0..LANES {
            r[k] = src[base + 3 * k] as u16;
            g[k] = src[base + 3 * k + 1] as u16;
            b[k] = src[base + 3 * k + 2] as u16;
        }
        let rv = u16x8::from(r);
        let gv = u16x8::from(g);
        let bv = u16x8::from(b);
        let y: u16x8 = (rv * rw + gv * gw + bv * bw) >> 8u16;
        let out = y.to_array();
        for k in 0..LANES {
            dst[i + k] = out[k] as u8;
        }
        i += LANES;
    }

    while i < n {
        let base = i * 3;
        let r = src[base] as u16;
        let g = src[base + 1] as u16;
        let b = src[base + 2] as u16;
        dst[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn scalar_gray_u8(src: &[u8]) -> Vec<u8> {
        src.chunks_exact(3)
            .map(|p| {
                let (r, g, b) = (p[0] as u16, p[1] as u16, p[2] as u16);
                ((r * 77 + g * 150 + b * 29) >> 8) as u8
            })
            .collect()
    }

    fn mk_u8(w: usize, h: usize) -> Image<u8, 3, CpuAllocator> {
        let data: Vec<u8> = (0..w * h * 3).map(|i| (i * 31 % 251) as u8).collect();
        Image::new(ImageSize { width: w, height: h }, data, CpuAllocator).unwrap()
    }

    #[test]
    fn simd_matches_scalar_u8_aligned() {
        // width is a multiple of both 16 (SSSE3) and 32 (AVX2).
        let img = mk_u8(64, 17);
        let mut dst = Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        gray_from_rgb_u8(&img, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), scalar_gray_u8(img.as_slice()).as_slice());
    }

    #[test]
    fn simd_matches_scalar_u8_tail() {
        // 17 pixels per row exercises BOTH the 32-px AVX2 body and the
        // 16-px SSSE3 body fallthrough plus the scalar tail.
        let img = mk_u8(17, 13);
        let mut dst = Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        gray_from_rgb_u8(&img, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), scalar_gray_u8(img.as_slice()).as_slice());
    }

    #[test]
    fn simd_matches_scalar_u8_size_33() {
        // 33 pixels exercises "one AVX2 block + 1 scalar pixel".
        let img = mk_u8(33, 3);
        let mut dst = Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        gray_from_rgb_u8(&img, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), scalar_gray_u8(img.as_slice()).as_slice());
    }

    #[test]
    fn simd_u8_regression() {
        let img = Image::new(
            ImageSize { width: 1, height: 2 },
            vec![0, 128, 255, 128, 0, 128],
            CpuAllocator,
        )
        .unwrap();
        let mut dst = Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
        gray_from_rgb_u8(&img, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), &[103, 53]);
    }

    #[test]
    fn simd_size_mismatch_errors() {
        let img = mk_u8(8, 8);
        let mut dst = Image::<u8, 1, _>::from_size_val(
            ImageSize { width: 4, height: 8 },
            0,
            CpuAllocator,
        )
        .unwrap();
        assert!(gray_from_rgb_u8(&img, &mut dst).is_err());
    }
}
