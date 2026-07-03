//! Nearest-neighbor resize. Works for any channel count and any ratio.
//!
//! Algorithm file: LUT precompute + row parallelism only — the per-row gather
//! lives in [`super::kernels::nearest_row_u8`] (the per-arch seam).

use rayon::prelude::*;

use super::kernels::nearest_row_u8;

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
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    let xmap: Vec<usize> = (0..dst_w)
        .map(|x| {
            let v = ((x as f64 + 0.5) * sx).floor() as i64;
            v.clamp(0, src_w as i64 - 1) as usize
        })
        .collect();

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
                    let yi = (((y as f64 + 0.5) * sy).floor() as i64).clamp(0, src_h as i64 - 1)
                        as usize;
                    let src_row = &src[yi * src_stride..(yi + 1) * src_stride];
                    nearest_row_u8::<C>(src_row, &xmap, dst_row);
                });
        });
}
