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
        for p in dst_pixel.iter_mut().take(C) {
            *p = 0;
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
#[allow(clippy::too_many_arguments)]
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
    // NEON fast path for C=3: holds RGB in lanes 0-2 of u32x4, does
    // Q10 bilinear math in SIMD. Safe whenever the 4-byte unaligned
    // read at (xi+1, yi+1) is in-bounds, i.e. xi < src_w-2 OR yi < src_h-2.
    #[cfg(target_arch = "aarch64")]
    {
        if C == 3 && (xi < src_w - 2 || yi < src_h - 2) {
            // Safety: bounds checked above; caller guarantees xi, yi valid.
            unsafe {
                bilinear_sample_u8_valid_c3_neon(
                    src.as_ptr(),
                    src_w,
                    src_h,
                    src_stride,
                    xi,
                    yi,
                    fx_q10,
                    fy_q10,
                    dst_pixel.as_mut_ptr(),
                );
            }
            return;
        }
    }

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

/// NEON u8 bilinear sampler, C=3 specialization.
///
/// Holds RGB in lanes 0-2 of a u32x4 vector (lane 3 dead), does Q10
/// fixed-point bilinear math in SIMD. Each corner is loaded via one
/// unaligned u32 read (3 bytes RGB + 1 byte slack), then widened
/// u8→u16→u32.
///
/// # Safety
/// - `src` must point to a valid u8 buffer of size ≥ `src_h * src_stride`.
/// - `src_stride == src_w * 3`.
/// - `xi ∈ [0, src_w-1]`, `yi ∈ [0, src_h-1]`.
/// - Must have `xi < src_w-2 OR yi < src_h-2`: ensures the 4-byte read
///   at the (xi+1, yi+1) corner is in-bounds. At the last pixel the
///   unaligned read would otherwise spill 1 byte past the buffer.
/// - `dst_pixel` must point to 3 writable bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn bilinear_sample_u8_valid_c3_neon(
    src: *const u8,
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    xi: i32,
    yi: i32,
    fx_q10: u32,
    fy_q10: u32,
    dst_pixel: *mut u8,
) {
    use std::arch::aarch64::*;

    let xi1 = if xi + 1 < src_w {
        (xi + 1) as usize
    } else {
        xi as usize
    };
    let yi1 = if yi + 1 < src_h {
        (yi + 1) as usize
    } else {
        yi as usize
    };

    let row0 = (yi as usize) * src_stride;
    let row1 = yi1 * src_stride;
    let xoff0 = (xi as usize) * 3;
    let xoff1 = xi1 * 3;

    // Load 4 bytes per corner via unaligned u32 read, widen u8x4 → u32x4.
    // Lane 3 is garbage (next pixel's R or padding) — ignored by the
    // narrow at the end since we only store 3 bytes.
    let load = |off: usize| -> uint32x4_t {
        let raw = core::ptr::read_unaligned(src.add(off) as *const u32);
        let u8_vec = vreinterpret_u8_u32(vcreate_u32(raw as u64));
        let u16_vec = vmovl_u8(u8_vec);
        vmovl_u16(vget_low_u16(u16_vec))
    };

    let p00 = load(row0 + xoff0);
    let p01 = load(row0 + xoff1);
    let p10 = load(row1 + xoff0);
    let p11 = load(row1 + xoff1);

    let fx_v = vdupq_n_u32(fx_q10);
    let fx1_v = vdupq_n_u32(1024 - fx_q10);
    let fy_v = vdupq_n_u32(fy_q10);
    let fy1_v = vdupq_n_u32(1024 - fy_q10);

    // top = p00 * fx1 + p01 * fx   (values ≤ 255 * 1024 * 2 = 522k, fits u32)
    // bot = p10 * fx1 + p11 * fx
    let top = vmlaq_u32(vmulq_u32(p00, fx1_v), p01, fx_v);
    let bot = vmlaq_u32(vmulq_u32(p10, fx1_v), p11, fx_v);
    // sum = top * fy1 + bot * fy + (1<<19)  (≤ 522k * 1024 + 522k * 1024 = 1.07e9, fits u32)
    let sum = vmlaq_u32(vmulq_u32(top, fy1_v), bot, fy_v);
    let sum = vaddq_u32(sum, vdupq_n_u32(1 << 19));
    let res = vshrq_n_u32::<20>(sum);

    // Narrow u32x4 → u8 (first 3 bytes are R, G, B).
    let u16x4 = vmovn_u32(res);
    let u8x8 = vmovn_u16(vcombine_u16(u16x4, u16x4));
    *dst_pixel.add(0) = vget_lane_u8::<0>(u8x8);
    *dst_pixel.add(1) = vget_lane_u8::<1>(u8x8);
    *dst_pixel.add(2) = vget_lane_u8::<2>(u8x8);
}
