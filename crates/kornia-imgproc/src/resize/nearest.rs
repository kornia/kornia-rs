//! Nearest-neighbor resize. Works for any channel count and any ratio.
//!
//! Algorithm file: LUT precompute + row parallelism only — the per-row gather
//! lives in [`super::kernels::nearest_row_u8`] (the per-arch seam).

use rayon::prelude::*;

use super::kernels::nearest_row_u8;

/// Nearest source index for destination index `i` on an axis sampled at
/// `scale = src_len / dst_len`: `clamp(floor((i + 0.5) * scale))` in f64 —
/// pixel-center then floor, deliberately NO `- 0.5` (matches OpenCV
/// `INTER_NEAREST`'s effective u8 behavior).
///
/// BYTE-EXACT CONTRACT: single source of the u8 nearest coordinates — the
/// CPU LUTs and the CUDA `resize_u8` launcher both call it.
#[inline]
pub(super) fn nearest_index(i: usize, scale: f64, src_len: usize) -> usize {
    let v = ((i as f64 + 0.5) * scale).floor() as i64;
    v.clamp(0, src_len as i64 - 1) as usize
}

/// Per-axis nearest LUT as `i32` (the CUDA kernel's gather index type).
#[cfg(feature = "cuda")]
pub(super) fn nearest_axis_lut(src_len: usize, dst_len: usize) -> Vec<i32> {
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|i| nearest_index(i, scale, src_len) as i32)
        .collect()
}

/// Nearest-neighbor u8 resize. Precomputes an `x → src_x` LUT (branch-free
/// row/col index map) and parallelizes over destination rows.
pub(super) fn resize_nearest_u8<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    let src_stride = src_w * C;
    let dst_stride = dst_w * C;
    let sy = src_h as f64 / dst_h as f64;

    let sx = src_w as f64 / dst_w as f64;
    let xmap: Vec<usize> = (0..dst_w).map(|x| nearest_index(x, sx, src_w)).collect();

    const ROWS_PER_TASK: usize = 16;
    dst.par_chunks_mut(ROWS_PER_TASK * dst_stride)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let row_base = chunk_idx * ROWS_PER_TASK;
            dst_chunk
                .chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(dr, dst_row)| {
                    let y = row_base + dr;
                    let yi = nearest_index(y, sy, src_h);
                    let src_row = &src[yi * src_stride..(yi + 1) * src_stride];
                    nearest_row_u8::<C>(src_row, &xmap, dst_row);
                });
        });
}
