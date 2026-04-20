//! Generic N-channel bilinear u8 resize (scalar, Q14).
//!
//! Fallback for arbitrary `src×dst` bilinear when the exact-2× paths in
//! [`super::pyramid`] don't match. Supports `C ∈ {1, 3, 4}`.

use rayon::prelude::*;

/// Generic N-channel bilinear u8 resize with precomputed x tables.
pub(super) fn resize_bilinear_u8_nch<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    const Q: u32 = 14;
    const SCALE: u32 = 1 << Q;
    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    let mut xofs = Vec::<u32>::with_capacity(dst_w);
    let mut xfx = Vec::<u32>::with_capacity(dst_w);
    let mut xfx1 = Vec::<u32>::with_capacity(dst_w);
    for x in 0..dst_w {
        let sx = (x as f64 + 0.5) * scale_x - 0.5;
        let xi = sx.floor() as i64;
        let f = sx - xi as f64;
        let (xi, f) = if xi < 0 {
            (0i64, 0.0)
        } else if xi >= src_w as i64 - 1 {
            (src_w as i64 - 2, 1.0)
        } else {
            (xi, f)
        };
        let fq = ((f * SCALE as f64).round() as u32).min(SCALE);
        xofs.push(xi as u32);
        xfx.push(fq);
        xfx1.push(SCALE - fq);
    }

    let src_stride = src_w * C;
    let dst_stride = dst_w * C;

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
                    let sy = (y as f64 + 0.5) * scale_y - 0.5;
                    let yi = sy.floor() as i64;
                    let f = sy - yi as f64;
                    let (yi, f) = if yi < 0 {
                        (0i64, 0.0)
                    } else if yi >= src_h as i64 - 1 {
                        (src_h as i64 - 2, 1.0)
                    } else {
                        (yi, f)
                    };
                    let fy = ((f * SCALE as f64).round() as u32).min(SCALE);
                    let fy1 = SCALE - fy;

                    let row0 =
                        &src[(yi as usize) * src_stride..(yi as usize + 1) * src_stride];
                    let row1 = &src
                        [(yi as usize + 1) * src_stride..(yi as usize + 2) * src_stride];

                    let round = 1u64 << 27;
                    for x in 0..dst_w {
                        let xi = xofs[x] as usize;
                        let fx = xfx[x] as u64;
                        let fx1 = xfx1[x] as u64;
                        let off = xi * C;
                        for ch in 0..C {
                            let p00 = row0[off + ch] as u64;
                            let p01 = row0[off + C + ch] as u64;
                            let p10 = row1[off + ch] as u64;
                            let p11 = row1[off + C + ch] as u64;
                            let top = p00 * fx1 + p01 * fx;
                            let bot = p10 * fx1 + p11 * fx;
                            let v = ((top * fy1 as u64 + bot * fy as u64 + round) >> 28)
                                as u8;
                            dst_row[x * C + ch] = v;
                        }
                    }
                });
        });
}
