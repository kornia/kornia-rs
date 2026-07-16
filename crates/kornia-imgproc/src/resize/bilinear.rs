//! Generic N-channel bilinear u8 resize (Q14).
//!
//! Fallback for arbitrary `src×dst` bilinear when the exact-2× paths in
//! [`super::pyramid`] don't match. Supports `C ∈ {1, 3, 4}`.
//!
//! Algorithm file: LUT precompute + row parallelism only — the per-row inner
//! loop lives in [`super::kernels::bilinear_row_u8`] (the per-arch seam).

use rayon::prelude::*;

use super::kernels::{bilinear_row_u8, BilinearXTaps};

/// Q14 fixed-point scale shared by the bilinear taps.
pub(super) const Q14_SCALE: u32 = 1 << 14;

/// One Q14 bilinear tap for destination index `i` on an axis of length
/// `src_len` sampled at `scale = src_len / dst_len`: returns `(ofs, fq)`
/// where `ofs` is the left source index (clamped to `[0, src_len-2]`) and
/// `fq ∈ [0, 16384]` the Q14 fractional weight of the right neighbor.
///
/// BYTE-EXACT CONTRACT: this f64 half-pixel computation is the single source
/// of the u8 bilinear coordinates — the CPU LUTs and the CUDA `resize_u8`
/// launcher both call it, so the two backends cannot drift.
#[inline]
pub(super) fn bilinear_tap(i: usize, scale: f64, src_len: usize) -> (u32, u32) {
    let s = (i as f64 + 0.5) * scale - 0.5;
    let i0 = s.floor() as i64;
    let f = s - i0 as f64;
    let (i0, f) = if i0 < 0 {
        (0i64, 0.0)
    } else if i0 >= src_len as i64 - 1 {
        (src_len as i64 - 2, 1.0)
    } else {
        (i0, f)
    };
    let fq = ((f * Q14_SCALE as f64).round() as u32).min(Q14_SCALE);
    (i0 as u32, fq)
}

/// Per-axis Q14 bilinear LUT: `(ofs, fx, fx1)` with `fx1 = 16384 - fx`.
pub(super) fn bilinear_axis_lut(src_len: usize, dst_len: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let scale = src_len as f64 / dst_len as f64;
    let mut ofs = Vec::with_capacity(dst_len);
    let mut fx = Vec::with_capacity(dst_len);
    let mut fx1 = Vec::with_capacity(dst_len);
    for i in 0..dst_len {
        let (o, fq) = bilinear_tap(i, scale, src_len);
        ofs.push(o);
        fx.push(fq);
        fx1.push(Q14_SCALE - fq);
    }
    (ofs, fx, fx1)
}

/// Generic N-channel bilinear u8 resize with precomputed x tables.
pub(super) fn resize_bilinear_u8_nch<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    let scale_y = src_h as f64 / dst_h as f64;
    let (xofs, xfx, xfx1) = bilinear_axis_lut(src_w, dst_w);

    let src_stride = src_w * C;
    let dst_stride = dst_w * C;

    const ROWS_PER_TASK: usize = 16;
    dst.par_chunks_mut(ROWS_PER_TASK * dst_stride)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let row_base = chunk_idx * ROWS_PER_TASK;
            let x_taps = BilinearXTaps {
                ofs: &xofs,
                fx: &xfx,
                fx1: &xfx1,
            };
            dst_chunk
                .chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(dr, dst_row)| {
                    let y = row_base + dr;
                    let (yi, fy) = bilinear_tap(y, scale_y, src_h);
                    let fy1 = Q14_SCALE - fy;
                    let yi = yi as usize;

                    let row0 = &src[yi * src_stride..(yi + 1) * src_stride];
                    let row1 = &src[(yi + 1) * src_stride..(yi + 2) * src_stride];

                    bilinear_row_u8::<C>(row0, row1, &x_taps, fy, fy1, dst_row);
                });
        });
}
