//! NEON (aarch64) kernel implementations. NEON is part of the ARMv8-A baseline.
#![cfg(target_arch = "aarch64")]

use super::scalar::tile_min_max;
use crate::utils::Pixel;
use std::arch::aarch64::*;

/// Classify a row: `pixel > thresh` → 255 (White) / 0 (Black) via `vcgtq_u8`.
///
/// # Safety
/// `src.len() == dst.len()`; `Pixel` is `#[repr(u8)]`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn classify_row(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    let thresh_v = vdupq_n_u8(thresh);
    let len = src.len();
    // SAFETY: Pixel is #[repr(u8)] so *mut Pixel == *mut u8 for layout purposes.
    let dst_u8 = core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, len);
    let mut i = 0;
    while i + 16 <= len {
        let px = vld1q_u8(src.as_ptr().add(i));
        vst1q_u8(dst_u8.as_mut_ptr().add(i), vcgtq_u8(px, thresh_v));
        i += 16;
    }
    while i < len {
        dst_u8[i] = if src[i] > thresh { 255 } else { 0 };
        i += 1;
    }
}

/// Fill per-tile min/max, batching 4 tiles per iteration when `tile_size == 4`:
/// 16 contiguous pixels are 4 side-by-side tiles, reduced via two `vpminq`/`vpmaxq`.
///
/// # Safety
/// `tiles_y * tile_size * img_width + tiles_x * tile_size ≤ img_data.len()`.
#[target_feature(enable = "neon")]
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
        while tile_size == 4 && tile_x + 4 <= tiles_x {
            let mut vmin = vdupq_n_u8(255);
            let mut vmax = vdupq_n_u8(0);
            for row in 0..tile_size {
                let offset = (tile_y * tile_size + row) * img_width + tile_x * tile_size;
                let v = vld1q_u8(img_data.as_ptr().add(offset));
                vmin = vminq_u8(vmin, v);
                vmax = vmaxq_u8(vmax, v);
            }
            vmin = vpminq_u8(vmin, vmin);
            vmin = vpminq_u8(vmin, vmin);
            vmax = vpmaxq_u8(vmax, vmax);
            vmax = vpmaxq_u8(vmax, vmax);
            let base = tile_y * tiles_x + tile_x;
            tile_min[base] = vgetq_lane_u8::<0>(vmin);
            tile_min[base + 1] = vgetq_lane_u8::<1>(vmin);
            tile_min[base + 2] = vgetq_lane_u8::<2>(vmin);
            tile_min[base + 3] = vgetq_lane_u8::<3>(vmin);
            tile_max[base] = vgetq_lane_u8::<0>(vmax);
            tile_max[base + 1] = vgetq_lane_u8::<1>(vmax);
            tile_max[base + 2] = vgetq_lane_u8::<2>(vmax);
            tile_max[base + 3] = vgetq_lane_u8::<3>(vmax);
            tile_x += 4;
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

/// NEON interior Gaussian smooth (4 outputs/iteration via `vfmaq_f32`).
///
/// # Safety
/// `half <= len`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn smooth_interior(errors: &[f32], kernel: &[f32], out: &mut [f32], half: usize, len: usize) {
    let flen = kernel.len();
    let interior_end = len - half;
    let mut iy = half;
    while iy + 4 <= interior_end {
        let mut acc = vdupq_n_f32(0.0);
        let base = errors.as_ptr().add(iy - half);
        for ki in 0..flen {
            let kv = vdupq_n_f32(*kernel.get_unchecked(ki));
            let ev = vld1q_f32(base.add(ki));
            acc = vfmaq_f32(acc, kv, ev);
        }
        vst1q_f32(out.as_mut_ptr().add(iy), acc);
        iy += 4;
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
