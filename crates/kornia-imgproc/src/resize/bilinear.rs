//! Generic N-channel bilinear u8 resize (Q14).
//!
//! Fallback for arbitrary `src×dst` bilinear when the exact-2× paths in
//! [`super::pyramid`] don't match. Supports `C ∈ {1, 3, 4}`.
//!
//! Algorithm file: LUT precompute + row parallelism only — the per-row inner
//! loop lives in [`super::kernels::bilinear_row_u8`] (the per-arch seam).

use kornia_image::ImageError;
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

/// Generates a per-axis Q14 bilinear lookup table (LUT): `(ofs, fx, fx1)` with `fx1 = 16384 - fx`.
///
/// # Arguments
///
/// * `src_len` - Source dimension size.
/// * `dst_len` - Target dimension size.
///
/// # Returns
///
/// A tuple `(ofs, fx, fx1)` of vectors containing offset indices and fixed-point interpolation weights.
pub fn bilinear_axis_lut(src_len: usize, dst_len: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
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

/// Caller-owned x tables of [`resize_bilinear_u8_c1_serial`].
#[derive(Debug, Default, Clone)]
pub(crate) struct BilinearLutScratch {
    ofs: Vec<u32>,
    fx: Vec<u32>,
    fx1: Vec<u32>,
}

/// Single-threaded single-channel bilinear u8 resize with caller-owned x
/// tables: the same bytes as [`super::resize_fast_u8`] (`Bilinear`, `C = 1`),
/// without allocating once `lut` has grown to the widest destination.
///
/// An empty destination is a no-op. Errors, as `resize_fast_u8`:
/// [`ImageError::InvalidImageSize`] `(src_w, src_h, 2, 2)` for a source
/// narrower or shorter than 2 px (no second bilinear tap), and
/// `(len, 1, w, h)` for a buffer shorter than its `w x h` plane.
pub(crate) fn resize_bilinear_u8_c1_serial(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    lut: &mut BilinearLutScratch,
) -> Result<(), ImageError> {
    if dst_w == 0 || dst_h == 0 {
        return Ok(());
    }
    if src_w < 2 || src_h < 2 {
        return Err(ImageError::InvalidImageSize(src_w, src_h, 2, 2));
    }
    if src.len() < src_w * src_h {
        return Err(ImageError::InvalidImageSize(src.len(), 1, src_w, src_h));
    }
    if dst.len() < dst_w * dst_h {
        return Err(ImageError::InvalidImageSize(dst.len(), 1, dst_w, dst_h));
    }
    let scale_x = src_w as f64 / dst_w as f64;
    lut.ofs.clear();
    lut.fx.clear();
    lut.fx1.clear();
    for i in 0..dst_w {
        let (o, fq) = bilinear_tap(i, scale_x, src_w);
        lut.ofs.push(o);
        lut.fx.push(fq);
        lut.fx1.push(Q14_SCALE - fq);
    }
    let x_taps = BilinearXTaps {
        ofs: &lut.ofs,
        fx: &lut.fx,
        fx1: &lut.fx1,
    };
    let scale_y = src_h as f64 / dst_h as f64;
    for (y, dst_row) in dst.chunks_exact_mut(dst_w).take(dst_h).enumerate() {
        let (yi, fy) = bilinear_tap(y, scale_y, src_h);
        let yi = yi as usize;
        let row0 = &src[yi * src_w..(yi + 1) * src_w];
        let row1 = &src[(yi + 1) * src_w..(yi + 2) * src_w];
        bilinear_row_u8::<1>(row0, row1, &x_taps, fy, Q14_SCALE - fy, dst_row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_c1_rejects_sizes_without_two_taps_or_short_buffers() {
        let mut lut = BilinearLutScratch::default();
        let src = [7u8; 12];
        let mut dst = [0u8; 4];
        for (w, h) in [(1, 1), (1, 6), (6, 1), (0, 3), (3, 0)] {
            assert!(matches!(
                resize_bilinear_u8_c1_serial(&src, w, h, &mut dst, 2, 2, &mut lut),
                Err(ImageError::InvalidImageSize(a, b, 2, 2)) if (a, b) == (w, h)
            ));
        }
        assert!(resize_bilinear_u8_c1_serial(&src[..5], 3, 2, &mut dst, 2, 2, &mut lut).is_err());
        assert!(resize_bilinear_u8_c1_serial(&src, 4, 3, &mut dst[..3], 2, 2, &mut lut).is_err());
        // Empty destination: nothing to write, even from a degenerate source.
        resize_bilinear_u8_c1_serial(&src, 1, 1, &mut dst, 0, 2, &mut lut).unwrap();
        resize_bilinear_u8_c1_serial(&src, 4, 3, &mut dst, 2, 2, &mut lut).unwrap();
        assert_eq!(dst, [7; 4]);
    }
}
