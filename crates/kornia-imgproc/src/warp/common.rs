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

    if xi < 0 || xi + 1 >= src_w || yi < 0 || yi + 1 >= src_h {
        for ch in 0..C {
            dst_pixel[ch] = 0;
        }
        return;
    }

    let fx = ((xf - xi as f32) * 1024.0) as u32;
    let fy = ((yf - yi as f32) * 1024.0) as u32;
    let fx1 = 1024 - fx;
    let fy1 = 1024 - fy;

    let off00 = (yi as usize) * src_stride + (xi as usize) * C;
    let off01 = off00 + C;
    let off10 = off00 + src_stride;
    let off11 = off10 + C;

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
/// Identical math to `bilinear_sample_u8` but assumes:
///   - `xi ∈ [0, src_w - 2]` so `xi` and `xi+1` are both in range,
///   - `yi ∈ [0, src_h - 2]` so `yi` and `yi+1` are both in range.
///
/// Caller is responsible for the bounds check; this lets the caller hoist
/// the check out of the inner loop for the valid-region of dst pixels.
/// `fx_q10`/`fy_q10` must be in `[0, 1024]`.
#[inline(always)]
pub(super) fn bilinear_sample_u8_valid<const C: usize>(
    src: &[u8],
    src_stride: usize,
    xi: i32,
    yi: i32,
    fx_q10: u32,
    fy_q10: u32,
    dst_pixel: &mut [u8],
) {
    let fx1 = 1024 - fx_q10;
    let fy1 = 1024 - fy_q10;

    let off00 = (yi as usize) * src_stride + (xi as usize) * C;
    let off01 = off00 + C;
    let off10 = off00 + src_stride;
    let off11 = off10 + C;

    // Safety: caller guaranteed xi, yi are in-bounds, so off11 + C <= src.len().
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
