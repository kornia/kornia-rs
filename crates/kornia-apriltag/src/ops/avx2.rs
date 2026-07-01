//! AVX2 (x86_64) kernel implementations, mirroring the NEON variants.

use super::scalar::tile_min_max;
use crate::utils::Pixel;
use std::arch::x86_64::*;

/// Classify a row, 32 pixels/iteration. AVX2 only has signed byte compares, so
/// both operands are biased by `0x80` to turn unsigned `>` into a signed `cmpgt`;
/// the `0xFF`/`0x00` result equals `Pixel::White` (255) / `Pixel::Black` (0).
///
/// # Safety
/// AVX2 available; `src.len() == dst.len()`; `Pixel` is `#[repr(u8)]`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn classify_row(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    let len = src.len();
    // SAFETY: Pixel is #[repr(u8)] so *mut Pixel == *mut u8 for layout purposes.
    let dst_u8 = core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, len);
    let bias = _mm256_set1_epi8(0x80u8 as i8);
    let tv = _mm256_xor_si256(_mm256_set1_epi8(thresh as i8), bias);
    let mut i = 0;
    while i + 32 <= len {
        let px = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let gt = _mm256_cmpgt_epi8(_mm256_xor_si256(px, bias), tv);
        _mm256_storeu_si256(dst_u8.as_mut_ptr().add(i) as *mut __m256i, gt);
        i += 32;
    }
    while i < len {
        dst_u8[i] = if src[i] > thresh { 255 } else { 0 };
        i += 1;
    }
}

/// Fill per-tile min/max, batching 8 tiles per iteration when `tile_size == 4`.
/// Each tile's 4 bytes lie in one 32-bit lane, so `srli_epi32` (which shifts
/// *within* each lane) reduces all 4 into the lane's byte 0.
///
/// # Safety
/// `tiles_y * tile_size * img_width + tiles_x * tile_size ≤ img_data.len()`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn fill_tile_stats(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_min: &mut [u8],
    tile_max: &mut [u8],
) {
    for tile_y in 0..tiles_y {
        let mut tile_x = 0usize;
        while tile_size == 4 && tile_x + 8 <= tiles_x {
            let mut vmin = _mm256_set1_epi8(0xFFu8 as i8);
            let mut vmax = _mm256_setzero_si256();
            for row in 0..tile_size {
                let offset = (tile_y * tile_size + row) * img_width + tile_x * tile_size;
                let v = _mm256_loadu_si256(img_data.as_ptr().add(offset) as *const __m256i);
                vmin = _mm256_min_epu8(vmin, v);
                vmax = _mm256_max_epu8(vmax, v);
            }
            let mn = _mm256_min_epu8(vmin, _mm256_srli_epi32::<16>(vmin));
            let mn = _mm256_min_epu8(mn, _mm256_srli_epi32::<8>(mn));
            let mx = _mm256_max_epu8(vmax, _mm256_srli_epi32::<16>(vmax));
            let mx = _mm256_max_epu8(mx, _mm256_srli_epi32::<8>(mx));
            let mut bmin = [0u8; 32];
            let mut bmax = [0u8; 32];
            _mm256_storeu_si256(bmin.as_mut_ptr() as *mut __m256i, mn);
            _mm256_storeu_si256(bmax.as_mut_ptr() as *mut __m256i, mx);
            let base = tile_y * tiles_x + tile_x;
            for g in 0..8 {
                tile_min[base + g] = bmin[g * 4];
                tile_max[base + g] = bmax[g * 4];
            }
            tile_x += 8;
        }
        while tile_x < tiles_x {
            let idx = tile_y * tiles_x + tile_x;
            let (lo, hi) = tile_min_max(img_data, img_width, tile_size, tile_x, tile_y);
            tile_min[idx] = lo;
            tile_max[idx] = hi;
            tile_x += 1;
        }
    }
}

/// AVX2+FMA interior Gaussian smooth (8 outputs/iteration).
///
/// # Safety
/// AVX2+FMA available; `half <= len`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn smooth_interior(
    errors: &[f32],
    kernel: &[f32],
    out: &mut [f32],
    half: usize,
    len: usize,
) {
    let flen = kernel.len();
    let interior_end = len - half;
    let mut iy = half;
    while iy + 8 <= interior_end {
        let mut acc = _mm256_setzero_ps();
        let base = errors.as_ptr().add(iy - half);
        for ki in 0..flen {
            let kv = _mm256_set1_ps(*kernel.get_unchecked(ki));
            let ev = _mm256_loadu_ps(base.add(ki));
            acc = _mm256_fmadd_ps(kv, ev, acc);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(iy), acc);
        iy += 8;
    }
    while iy < interior_end {
        let mut acc = 0.0f32;
        for ki in 0..flen {
            acc += *errors.get_unchecked(iy - half + ki) * *kernel.get_unchecked(ki);
        }
        *out.get_unchecked_mut(iy) = acc;
        iy += 1;
    }
}
