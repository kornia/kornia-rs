//! Shared helpers for the `*_u8` warp kernels.
//!
//! Both `warp_affine_u8` and `warp_perspective_u8` sample the same way once a
//! source coordinate has been produced — only how that coordinate is generated
//! differs (linear vs rational). This module extracts the common per-pixel
//! bilinear sample.

/// Bilinear sample of a u8 source image at `(xf, yf)` into `dst_pixel`, using
/// Q10 fixed-point weights. Writes zeros on out-of-bounds. Kept tiny and
/// `#[inline(always)]` so the hot warp loops get the same codegen as before.
///
/// `src_stride` is `src_cols * C` (row stride in bytes for u8).
#[inline(always)]
pub(super) fn bilinear_sample_u8<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    xf: f32,
    yf: f32,
    dst_pixel: &mut [u8],
) {
    let xi = xf.floor() as i32;
    let yi = yf.floor() as i32;

    if xi < 0 || xi >= src_w || yi < 0 || yi >= src_h {
        for ch in 0..C {
            dst_pixel[ch] = 0;
        }
        return;
    }

    let xi1 = if xi + 1 < src_w { xi + 1 } else { xi };
    let yi1 = if yi + 1 < src_h { yi + 1 } else { yi };

    let fx = ((xf - xi as f32) * 1024.0) as u32;
    let fy = ((yf - yi as f32) * 1024.0) as u32;
    let fx1 = 1024 - fx;
    let fy1 = 1024 - fy;

    let row0 = (yi as usize) * src_stride;
    let row1 = (yi1 as usize) * src_stride;
    let xoff0 = (xi as usize) * C;
    let xoff1 = (xi1 as usize) * C;
    let off00 = row0 + xoff0;
    let off01 = row0 + xoff1;
    let off10 = row1 + xoff0;
    let off11 = row1 + xoff1;

    for ch in 0..C {
        let p00 = src[off00 + ch] as u32;
        let p01 = src[off01 + ch] as u32;
        let p10 = src[off10 + ch] as u32;
        let p11 = src[off11 + ch] as u32;
        let top = p00 * fx1 + p01 * fx;
        let bot = p10 * fx1 + p11 * fx;
        let v = (top * fy1 + bot * fy + (1 << 19)) >> 20;
        dst_pixel[ch] = v as u8;
    }
}

/// Sample at a pre-bounds-checked coord with Q10 fractional weights.
///
/// Assumes `xi ∈ [0, src_w - 1]` and `yi ∈ [0, src_h - 1]`. When `xi` (or
/// `yi`) is at the extreme edge, the neighbor index is clamped to the
/// same pixel — equivalent to `BORDER_REPLICATE` for the interior of a
/// single-pixel border. At an exact-integer source coord on the edge
/// (fx_q10=0 or fy_q10=0) the clamped neighbor is weighted to zero, so
/// the output is exactly `src[yi, xi]` — matching the f32 reference
/// kernel's identity-preservation.
///
/// `fx_q10`/`fy_q10` must be in `[0, 1024]`.
#[inline(always)]
pub(super) fn bilinear_sample_u8_valid<const C: usize>(
    src: &[u8],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    xi: i32,
    yi: i32,
    fx_q10: u32,
    fy_q10: u32,
    dst_pixel: &mut [u8],
) {
    let fx1 = 1024 - fx_q10;
    let fy1 = 1024 - fy_q10;

    let xi1 = if xi + 1 < src_w { xi + 1 } else { xi };
    let yi1 = if yi + 1 < src_h { yi + 1 } else { yi };

    let row0 = (yi as usize) * src_stride;
    let row1 = (yi1 as usize) * src_stride;
    let xoff0 = (xi as usize) * C;
    let xoff1 = (xi1 as usize) * C;
    let off00 = row0 + xoff0;
    let off01 = row0 + xoff1;
    let off10 = row1 + xoff0;
    let off11 = row1 + xoff1;

    // Safety: caller guaranteed xi, yi are in-bounds, and xi1, yi1 are
    // clamped into range, so all four offsets + C <= src.len().
    unsafe {
        let p = src.as_ptr();
        for ch in 0..C {
            let p00 = *p.add(off00 + ch) as u32;
            let p01 = *p.add(off01 + ch) as u32;
            let p10 = *p.add(off10 + ch) as u32;
            let p11 = *p.add(off11 + ch) as u32;
            let top = p00 * fx1 + p01 * fx_q10;
            let bot = p10 * fx1 + p11 * fx_q10;
            let v = (top * fy1 + bot * fy_q10 + (1 << 19)) >> 20;
            *dst_pixel.as_mut_ptr().add(ch) = v as u8;
        }
    }
}
