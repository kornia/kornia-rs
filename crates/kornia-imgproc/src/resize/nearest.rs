//! Nearest-neighbor resize. Works for any channel count and any ratio.

use rayon::prelude::*;

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
                    for (x, xi) in xmap.iter().enumerate() {
                        let so = xi * C;
                        let d_o = x * C;
                        dst_row[d_o..d_o + C].copy_from_slice(&src_row[so..so + C]);
                    }
                });
        });
}
