//! YUV-family color conversion kernels (NEON / AVX2-scaffold / scalar).
//!
//! Two families live here:
//!
//! **Family A — RGB ↔ YCbCr / YUV (planar 3-channel).** Full-range OpenCV constants.
//! `YCbCr` (OpenCV `YCrCb`) stores channels `[Y, Cr, Cb]`; `YUV` stores `[Y, U=Cb, V=Cr]`.
//! The math is identical — only the chroma store order differs — so one kernel serves
//! both via a `ChromaOrder` flag. `u8` runs a Q14 fixed-point path; `f32` runs in `[0,1]`.
//!
//! **Family B — video decode (u8 only).** BT.601 *limited* range, `ITUR_BT_601` Q20
//! constants. A shared inner kernel `decode_chroma_pair` converts one (U,V) pair plus two
//! luma samples; the packed (YUYV/UYVY/YVYU) and planar (NV12/NV21/I420/YV12) entry points
//! differ only in how they load/deinterleave luma and chroma.
//!
//! House rules: NEON is the aarch64 baseline (no `#[target_feature]`); naming is
//! `<out>_from_<in>` at every level; `u8` paths use saturating narrows; the scalar paths
//! are the bit-exact oracles for the tests.

use super::super::kernel_common::par_strip_dispatch;

// ===== Family A: shared Q14 constants (full-range OpenCV) ============================

const Q14_SHIFT: i32 = 14;
const Q14_HALF: i32 = 1 << 13; // rounding bias

// Forward RGB -> YCbCr/YUV
const C_YR: i32 = 4899; // 0.299  * 2^14
const C_YG: i32 = 9617; // 0.587  * 2^14
const C_YB: i32 = 1868; // 0.114  * 2^14
const C_YCRI: i32 = 11682; // 0.713 * 2^14  ((R-Y) -> Cr)
const C_YCBI: i32 = 9241; //  0.564 * 2^14  ((B-Y) -> Cb)

// Inverse YCbCr/YUV -> RGB
const C_CR2R: i32 = 22987; //  1.403  * 2^14
const C_CR2G: i32 = -11698; // -0.714 * 2^14
const C_CB2G: i32 = -5636; //  -0.344 * 2^14
const C_CB2B: i32 = 29049; //  1.773  * 2^14

// f32 full-range coefficients (inputs in [0,1]).
const F_YR: f32 = 0.299;
const F_YG: f32 = 0.587;
const F_YB: f32 = 0.114;
const F_CR: f32 = 0.713; // (R-Y) * 0.713 + 0.5
const F_CB: f32 = 0.564; // (B-Y) * 0.564 + 0.5

/// Store order for the chroma channels of an RGB->{YCbCr,YUV} conversion.
///
/// `YCrCb` writes `[Y, Cr, Cb]` (OpenCV `YCrCb`); `YuvCbCr` writes `[Y, U=Cb, V=Cr]`
/// (planar YUV). Same math, different permutation of the two chroma results.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChromaOrder {
    /// `[Y, Cr, Cb]` — OpenCV `YCrCb`.
    YCrCb,
    /// `[Y, U=Cb, V=Cr]` — planar YUV.
    YuvCbCr,
}

// ===== Family A scalar oracles (u8 Q14) =============================================

/// Scalar RGB u8 -> (Y, Cr, Cb) u8, full-range Q14. Bit-exact oracle.
#[inline]
fn ycc_from_rgb_u8_px(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    let y = (C_YR * r + C_YG * g + C_YB * b + Q14_HALF) >> Q14_SHIFT;
    let cr = ((r - y) * C_YCRI + (128 << Q14_SHIFT) + Q14_HALF) >> Q14_SHIFT;
    let cb = ((b - y) * C_YCBI + (128 << Q14_SHIFT) + Q14_HALF) >> Q14_SHIFT;
    (
        y.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
    )
}

/// Scalar (Y, Cr, Cb) u8 -> RGB u8, full-range Q14. Bit-exact oracle.
#[inline]
fn rgb_from_ycc_u8_px(y: i32, cr: i32, cb: i32) -> (u8, u8, u8) {
    let cr = cr - 128;
    let cb = cb - 128;
    let r = y + ((C_CR2R * cr + Q14_HALF) >> Q14_SHIFT);
    let g = y + ((C_CR2G * cr + C_CB2G * cb + Q14_HALF) >> Q14_SHIFT);
    let b = y + ((C_CB2B * cb + Q14_HALF) >> Q14_SHIFT);
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

// ===== Family A: RGB u8 -> YCbCr/YUV u8 =============================================

/// Slice-level RGB u8 -> YCbCr/YUV u8 (full-range Q14). `order` picks the chroma store
/// permutation. Parallelized over row-strips for large images.
pub fn ycc_from_rgb_u8(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 16, move |s, d, n| {
        ycc_from_rgb_u8_kernel(s, d, n, order)
    });
}

#[inline]
fn ycc_from_rgb_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    #[cfg(target_arch = "aarch64")]
    {
        ycc_from_rgb_u8_neon(src, dst, npixels, order);
        return;
    }
    #[allow(unreachable_code)]
    ycc_from_rgb_u8_scalar(src, dst, npixels, order);
}

/// NEON RGB u8 -> YCbCr/YUV u8: 16 px/iter via `vld3q_u8`. Widen u8->s16, `vmull_n_s16`
/// into s32, round+shift, saturating-narrow back to u8 (`vqmovun`). Stores via `vst3q_u8`
/// with the chroma lanes permuted per `order`.
#[cfg(target_arch = "aarch64")]
fn ycc_from_rgb_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let half = vdupq_n_s32(Q14_HALF);
        let bias128 = vdupq_n_s32((128 << Q14_SHIFT) + Q14_HALF);

        // Process one 8-lane (s16) half: r,g,b are int16x8 in [0,255].
        let conv8 =
            |r: int16x8_t, g: int16x8_t, b: int16x8_t| -> (uint8x8_t, uint8x8_t, uint8x8_t) {
                let (rl, rh) = (vget_low_s16(r), vget_high_s16(r));
                let (gl, gh) = (vget_low_s16(g), vget_high_s16(g));
                let (bl, bh) = (vget_low_s16(b), vget_high_s16(b));

                // Y = (C_YR*R + C_YG*G + C_YB*B + half) >> 14
                let yl = vshrq_n_s32::<Q14_SHIFT>(vaddq_s32(
                    vmlal_n_s16(
                        vmlal_n_s16(vmull_n_s16(rl, C_YR as i16), gl, C_YG as i16),
                        bl,
                        C_YB as i16,
                    ),
                    half,
                ));
                let yh = vshrq_n_s32::<Q14_SHIFT>(vaddq_s32(
                    vmlal_n_s16(
                        vmlal_n_s16(vmull_n_s16(rh, C_YR as i16), gh, C_YG as i16),
                        bh,
                        C_YB as i16,
                    ),
                    half,
                ));
                let y32_lo = yl;
                let y32_hi = yh;
                // narrow Y (s32 -> s16) for the (R-Y),(B-Y) diffs
                let y16 = vcombine_s16(vmovn_s32(y32_lo), vmovn_s32(y32_hi));
                let (yl16, yh16) = (vget_low_s16(y16), vget_high_s16(y16));

                // Cr = ((R-Y)*C_YCRI + 128<<14 + half) >> 14
                let drl = vsub_s16(rl, yl16);
                let drh = vsub_s16(rh, yh16);
                let crl = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, drl, C_YCRI as i16));
                let crh = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, drh, C_YCRI as i16));

                // Cb = ((B-Y)*C_YCBI + 128<<14 + half) >> 14
                let dbl = vsub_s16(bl, yl16);
                let dbh = vsub_s16(bh, yh16);
                let cbl = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, dbl, C_YCBI as i16));
                let cbh = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, dbh, C_YCBI as i16));

                // saturating narrow s32 -> s16 -> u8
                let y8 = vqmovun_s16(vcombine_s16(vqmovn_s32(y32_lo), vqmovn_s32(y32_hi)));
                let cr8 = vqmovun_s16(vcombine_s16(vqmovn_s32(crl), vqmovn_s32(crh)));
                let cb8 = vqmovun_s16(vcombine_s16(vqmovn_s32(cbl), vqmovn_s32(cbh)));
                (y8, cr8, cb8)
            };

        let bulk16 = npixels & !15;
        let mut i = 0usize;
        while i < bulk16 {
            let rgb = vld3q_u8(sp.add(i * 3));
            let r16 = (
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.0))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgb.0))),
            );
            let g16 = (
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.1))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgb.1))),
            );
            let b16 = (
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.2))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgb.2))),
            );

            let (y_lo, cr_lo, cb_lo) = conv8(r16.0, g16.0, b16.0);
            let (y_hi, cr_hi, cb_hi) = conv8(r16.1, g16.1, b16.1);

            let y = vcombine_u8(y_lo, y_hi);
            let cr = vcombine_u8(cr_lo, cr_hi);
            let cb = vcombine_u8(cb_lo, cb_hi);

            let out = match order {
                ChromaOrder::YCrCb => uint8x16x3_t(y, cr, cb),
                ChromaOrder::YuvCbCr => uint8x16x3_t(y, cb, cr),
            };
            vst3q_u8(dp.add(i * 3), out);
            i += 16;
        }
        // scalar tail
        while i < npixels {
            let si = i * 3;
            let (y, cr, cb) = ycc_from_rgb_u8_px(
                *sp.add(si) as i32,
                *sp.add(si + 1) as i32,
                *sp.add(si + 2) as i32,
            );
            store_ycc(dp, si, y, cr, cb, order);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store_ycc(dp: *mut u8, si: usize, y: u8, cr: u8, cb: u8, order: ChromaOrder) {
    *dp.add(si) = y;
    match order {
        ChromaOrder::YCrCb => {
            *dp.add(si + 1) = cr;
            *dp.add(si + 2) = cb;
        }
        ChromaOrder::YuvCbCr => {
            *dp.add(si + 1) = cb;
            *dp.add(si + 2) = cr;
        }
    }
}

/// Portable scalar RGB u8 -> YCbCr/YUV u8. Oracle for the tests.
pub fn ycc_from_rgb_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (y, cr, cb) =
            ycc_from_rgb_u8_px(src[si] as i32, src[si + 1] as i32, src[si + 2] as i32);
        dst[si] = y;
        match order {
            ChromaOrder::YCrCb => {
                dst[si + 1] = cr;
                dst[si + 2] = cb;
            }
            ChromaOrder::YuvCbCr => {
                dst[si + 1] = cb;
                dst[si + 2] = cr;
            }
        }
    }
}

// ===== Family A: YCbCr/YUV u8 -> RGB u8 =============================================

/// Slice-level YCbCr/YUV u8 -> RGB u8 (full-range Q14). `order` picks the chroma load
/// permutation. Parallelized over row-strips for large images.
pub fn rgb_from_ycc_u8(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 16, move |s, d, n| {
        rgb_from_ycc_u8_kernel(s, d, n, order)
    });
}

#[inline]
fn rgb_from_ycc_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_ycc_u8_neon(src, dst, npixels, order);
        return;
    }
    #[allow(unreachable_code)]
    rgb_from_ycc_u8_scalar(src, dst, npixels, order);
}

/// NEON YCbCr/YUV u8 -> RGB u8: 16 px/iter via `vld3q_u8`, Q14 inverse matrix, saturating
/// narrow to u8.
#[cfg(target_arch = "aarch64")]
fn rgb_from_ycc_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let half = vdupq_n_s32(Q14_HALF);
        let c128 = vdup_n_s16(128);

        let conv8 =
            |y: int16x8_t, cr: int16x8_t, cb: int16x8_t| -> (uint8x8_t, uint8x8_t, uint8x8_t) {
                let (yl, yh) = (vget_low_s16(y), vget_high_s16(y));
                let crl = vsub_s16(vget_low_s16(cr), c128);
                let crh = vsub_s16(vget_high_s16(cr), c128);
                let cbl = vsub_s16(vget_low_s16(cb), c128);
                let cbh = vsub_s16(vget_high_s16(cb), c128);

                // y in s32 (no shift), as base for adds
                let yl32 = vmovl_s16(yl);
                let yh32 = vmovl_s16(yh);

                // R = Y + ((C_CR2R*cr + half) >> 14)
                let rl = vaddq_s32(
                    yl32,
                    vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(half, crl, C_CR2R as i16)),
                );
                let rh = vaddq_s32(
                    yh32,
                    vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(half, crh, C_CR2R as i16)),
                );
                // G = Y + ((C_CR2G*cr + C_CB2G*cb + half) >> 14)
                let gl = vaddq_s32(
                    yl32,
                    vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(
                        vmlal_n_s16(half, crl, C_CR2G as i16),
                        cbl,
                        C_CB2G as i16,
                    )),
                );
                let gh = vaddq_s32(
                    yh32,
                    vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(
                        vmlal_n_s16(half, crh, C_CR2G as i16),
                        cbh,
                        C_CB2G as i16,
                    )),
                );
                // B = Y + ((C_CB2B*cb + half) >> 14)
                let bl = vaddq_s32(
                    yl32,
                    vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(half, cbl, C_CB2B as i16)),
                );
                let bh = vaddq_s32(
                    yh32,
                    vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(half, cbh, C_CB2B as i16)),
                );

                let r8 = vqmovun_s16(vcombine_s16(vqmovn_s32(rl), vqmovn_s32(rh)));
                let g8 = vqmovun_s16(vcombine_s16(vqmovn_s32(gl), vqmovn_s32(gh)));
                let b8 = vqmovun_s16(vcombine_s16(vqmovn_s32(bl), vqmovn_s32(bh)));
                (r8, g8, b8)
            };

        let bulk16 = npixels & !15;
        let mut i = 0usize;
        while i < bulk16 {
            let p = vld3q_u8(sp.add(i * 3));
            // p.0 = Y; chroma channels depend on order.
            let (cr_ch, cb_ch) = match order {
                ChromaOrder::YCrCb => (p.1, p.2),
                ChromaOrder::YuvCbCr => (p.2, p.1),
            };
            let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p.0)));
            let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p.0)));
            let cr_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cr_ch)));
            let cr_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cr_ch)));
            let cb_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cb_ch)));
            let cb_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cb_ch)));

            let (r_lo, g_lo, b_lo) = conv8(y_lo, cr_lo, cb_lo);
            let (r_hi, g_hi, b_hi) = conv8(y_hi, cr_hi, cb_hi);

            vst3q_u8(
                dp.add(i * 3),
                uint8x16x3_t(
                    vcombine_u8(r_lo, r_hi),
                    vcombine_u8(g_lo, g_hi),
                    vcombine_u8(b_lo, b_hi),
                ),
            );
            i += 16;
        }
        while i < npixels {
            let si = i * 3;
            let (cr, cb) = match order {
                ChromaOrder::YCrCb => (*sp.add(si + 1) as i32, *sp.add(si + 2) as i32),
                ChromaOrder::YuvCbCr => (*sp.add(si + 2) as i32, *sp.add(si + 1) as i32),
            };
            let (r, g, b) = rgb_from_ycc_u8_px(*sp.add(si) as i32, cr, cb);
            *dp.add(si) = r;
            *dp.add(si + 1) = g;
            *dp.add(si + 2) = b;
            i += 1;
        }
    }
}

/// Portable scalar YCbCr/YUV u8 -> RGB u8. Oracle for the tests.
pub fn rgb_from_ycc_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (cr, cb) = match order {
            ChromaOrder::YCrCb => (src[si + 1] as i32, src[si + 2] as i32),
            ChromaOrder::YuvCbCr => (src[si + 2] as i32, src[si + 1] as i32),
        };
        let (r, g, b) = rgb_from_ycc_u8_px(src[si] as i32, cr, cb);
        dst[si] = r;
        dst[si + 1] = g;
        dst[si + 2] = b;
    }
}

// ===== Family A: f32 paths (in [0,1]) ==============================================

/// Slice-level RGB f32 -> YCbCr/YUV f32 (full-range, inputs in `[0,1]`).
pub fn ycc_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, move |s, d, n| {
        ycc_from_rgb_f32_kernel(s, d, n, order)
    });
}

#[inline]
fn ycc_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    #[cfg(target_arch = "aarch64")]
    {
        ycc_from_rgb_f32_neon(src, dst, npixels, order);
        return;
    }
    #[allow(unreachable_code)]
    ycc_from_rgb_f32_scalar(src, dst, npixels, order);
}

#[cfg(target_arch = "aarch64")]
fn ycc_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let yr = vdupq_n_f32(F_YR);
        let yg = vdupq_n_f32(F_YG);
        let yb = vdupq_n_f32(F_YB);
        let kcr = vdupq_n_f32(F_CR);
        let kcb = vdupq_n_f32(F_CB);
        let half = vdupq_n_f32(0.5);

        let transform = |p: float32x4x3_t| -> float32x4x3_t {
            let y = vfmaq_f32(vfmaq_f32(vmulq_f32(p.2, yb), p.1, yg), p.0, yr);
            let cr = vfmaq_f32(half, vsubq_f32(p.0, y), kcr);
            let cb = vfmaq_f32(half, vsubq_f32(p.2, y), kcb);
            match order {
                ChromaOrder::YCrCb => float32x4x3_t(y, cr, cb),
                ChromaOrder::YuvCbCr => float32x4x3_t(y, cb, cr),
            }
        };

        let bulk8 = npixels & !7;
        let mut i = 0usize;
        while i < bulk8 {
            let a = vld3q_f32(sp.add(i * 3));
            let c = vld3q_f32(sp.add((i + 4) * 3));
            vst3q_f32(dp.add(i * 3), transform(a));
            vst3q_f32(dp.add((i + 4) * 3), transform(c));
            i += 8;
        }
        if i + 4 <= npixels {
            let a = vld3q_f32(sp.add(i * 3));
            vst3q_f32(dp.add(i * 3), transform(a));
            i += 4;
        }
        while i < npixels {
            let si = i * 3;
            let (y, cr, cb) = ycc_from_rgb_f32_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
            store_ycc_f32(dp, si, y, cr, cb, order);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store_ycc_f32(dp: *mut f32, si: usize, y: f32, cr: f32, cb: f32, order: ChromaOrder) {
    *dp.add(si) = y;
    match order {
        ChromaOrder::YCrCb => {
            *dp.add(si + 1) = cr;
            *dp.add(si + 2) = cb;
        }
        ChromaOrder::YuvCbCr => {
            *dp.add(si + 1) = cb;
            *dp.add(si + 2) = cr;
        }
    }
}

#[inline]
fn ycc_from_rgb_f32_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = F_YR * r + F_YG * g + F_YB * b;
    let cr = (r - y) * F_CR + 0.5;
    let cb = (b - y) * F_CB + 0.5;
    (y, cr, cb)
}

/// Portable scalar RGB f32 -> YCbCr/YUV f32. Oracle for the tests.
pub fn ycc_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (y, cr, cb) = ycc_from_rgb_f32_px(src[si], src[si + 1], src[si + 2]);
        dst[si] = y;
        match order {
            ChromaOrder::YCrCb => {
                dst[si + 1] = cr;
                dst[si + 2] = cb;
            }
            ChromaOrder::YuvCbCr => {
                dst[si + 1] = cb;
                dst[si + 2] = cr;
            }
        }
    }
}

/// Slice-level YCbCr/YUV f32 -> RGB f32 (full-range, outputs in `[0,1]`).
pub fn rgb_from_ycc_f32(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, move |s, d, n| {
        rgb_from_ycc_f32_kernel(s, d, n, order)
    });
}

#[inline]
fn rgb_from_ycc_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_ycc_f32_neon(src, dst, npixels, order);
        return;
    }
    #[allow(unreachable_code)]
    rgb_from_ycc_f32_scalar(src, dst, npixels, order);
}

#[cfg(target_arch = "aarch64")]
fn rgb_from_ycc_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        // inverse f32 coefficients derived from the [0,1] forward defs:
        //   r = y + (cr-0.5)/F_CR
        //   b = y + (cb-0.5)/F_CB
        //   g = (y - F_YR*r - F_YB*b) / F_YG
        let inv_cr = vdupq_n_f32(1.0 / F_CR);
        let inv_cb = vdupq_n_f32(1.0 / F_CB);
        let inv_yg = vdupq_n_f32(1.0 / F_YG);
        let yr = vdupq_n_f32(F_YR);
        let yb = vdupq_n_f32(F_YB);
        let half = vdupq_n_f32(0.5);

        let transform = |p: float32x4x3_t| -> float32x4x3_t {
            let y = p.0;
            let (cr, cb) = match order {
                ChromaOrder::YCrCb => (p.1, p.2),
                ChromaOrder::YuvCbCr => (p.2, p.1),
            };
            let r = vfmaq_f32(y, vsubq_f32(cr, half), inv_cr);
            let b = vfmaq_f32(y, vsubq_f32(cb, half), inv_cb);
            // g = (y - yr*r - yb*b) * inv_yg
            let num = vsubq_f32(vsubq_f32(y, vmulq_f32(yr, r)), vmulq_f32(yb, b));
            let g = vmulq_f32(num, inv_yg);
            float32x4x3_t(r, g, b)
        };

        let bulk8 = npixels & !7;
        let mut i = 0usize;
        while i < bulk8 {
            let a = vld3q_f32(sp.add(i * 3));
            let c = vld3q_f32(sp.add((i + 4) * 3));
            vst3q_f32(dp.add(i * 3), transform(a));
            vst3q_f32(dp.add((i + 4) * 3), transform(c));
            i += 8;
        }
        if i + 4 <= npixels {
            let a = vld3q_f32(sp.add(i * 3));
            vst3q_f32(dp.add(i * 3), transform(a));
            i += 4;
        }
        while i < npixels {
            let si = i * 3;
            let (cr, cb) = match order {
                ChromaOrder::YCrCb => (*sp.add(si + 1), *sp.add(si + 2)),
                ChromaOrder::YuvCbCr => (*sp.add(si + 2), *sp.add(si + 1)),
            };
            let (r, g, b) = rgb_from_ycc_f32_px(*sp.add(si), cr, cb);
            *dp.add(si) = r;
            *dp.add(si + 1) = g;
            *dp.add(si + 2) = b;
            i += 1;
        }
    }
}

#[inline]
fn rgb_from_ycc_f32_px(y: f32, cr: f32, cb: f32) -> (f32, f32, f32) {
    let r = y + (cr - 0.5) / F_CR;
    let b = y + (cb - 0.5) / F_CB;
    let g = (y - F_YR * r - F_YB * b) / F_YG;
    (r, g, b)
}

/// Portable scalar YCbCr/YUV f32 -> RGB f32. Oracle for the tests.
pub fn rgb_from_ycc_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (cr, cb) = match order {
            ChromaOrder::YCrCb => (src[si + 1], src[si + 2]),
            ChromaOrder::YuvCbCr => (src[si + 2], src[si + 1]),
        };
        let (r, g, b) = rgb_from_ycc_f32_px(src[si], cr, cb);
        dst[si] = r;
        dst[si + 1] = g;
        dst[si + 2] = b;
    }
}

// ===== Family B: video decode (BT.601 limited, Q20) =================================

const ITUR_SHIFT: i32 = 20;
const ITUR_HALF: i32 = 1 << 19;
const CY: i32 = 1220542; // 1.164 * 2^20
const CUB: i32 = 2116026; // 2.018 * 2^20  (U -> B)
const CUG: i32 = -409993; // -0.391 * 2^20 (U -> G)
const CVG: i32 = -852492; // -0.813 * 2^20 (V -> G)
const CVR: i32 = 1673527; // 1.596 * 2^20  (V -> R)

/// Decode one luma sample with a shared (U,V) chroma pair to an RGB triple (BT.601
/// limited, Q20). `yy` is the pre-scaled luma term `max(0, Y-16) * CY`. Bit-exact oracle.
#[inline]
fn decode_px(yy: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let u = u - 128;
    let v = v - 128;
    let b = (yy + CUB * u + ITUR_HALF) >> ITUR_SHIFT;
    let g = (yy + CUG * u + CVG * v + ITUR_HALF) >> ITUR_SHIFT;
    let r = (yy + CVR * v + ITUR_HALF) >> ITUR_SHIFT;
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[inline]
fn yy_term(y: i32) -> i32 {
    (y - 16).max(0) * CY
}

// ---- 4:2:2 packed ----------------------------------------------------------------

/// Byte layout of a packed 4:2:2 stream within each 4-byte (2-pixel) group.
#[derive(Clone, Copy)]
pub enum Packed422 {
    /// `Y0 U Y1 V`
    Yuyv,
    /// `U Y0 V Y1`
    Uyvy,
    /// `Y0 V Y1 U`
    Yvyu,
}

impl Packed422 {
    /// Offsets `(y0, u, y1, v)` within a 4-byte group.
    #[inline]
    fn offsets(self) -> (usize, usize, usize, usize) {
        match self {
            Packed422::Yuyv => (0, 1, 2, 3),
            Packed422::Uyvy => (1, 0, 3, 2),
            Packed422::Yvyu => (0, 3, 2, 1),
        }
    }
}

/// Decode a packed 4:2:2 buffer (`src`, 2 bytes/px) to RGB (`dst`, 3 bytes/px).
///
/// `width` must be even. Rows are decoded independently and parallelized over strips for
/// large images. NEON deinterleaves 16 px/iter via `vld4q_u8`; the scalar tail/fallback is
/// the oracle.
pub fn rgb_from_packed422(src: &[u8], dst: &mut [u8], width: usize, height: usize, fmt: Packed422) {
    debug_assert!(width.is_multiple_of(2));
    debug_assert!(src.len() >= width * height * 2);
    debug_assert!(dst.len() >= width * height * 3);

    let src_row = width * 2;
    let dst_row = width * 3;
    use super::super::kernel_common::PAR_THRESHOLD;
    if width * height < PAR_THRESHOLD {
        for row in 0..height {
            rgb_from_packed422_row(
                &src[row * src_row..row * src_row + src_row],
                &mut dst[row * dst_row..row * dst_row + dst_row],
                width,
                fmt,
            );
        }
        return;
    }
    use rayon::prelude::*;
    dst.par_chunks_mut(dst_row)
        .enumerate()
        .for_each(|(row, drow)| {
            let srow = &src[row * src_row..row * src_row + src_row];
            rgb_from_packed422_row(srow, drow, width, fmt);
        });
}

#[inline]
fn rgb_from_packed422_row(src: &[u8], dst: &mut [u8], width: usize, fmt: Packed422) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_packed422_row_neon(src, dst, width, fmt);
        return;
    }
    #[allow(unreachable_code)]
    rgb_from_packed422_row_scalar(src, dst, width, fmt);
}

/// NEON packed-4:2:2 row decode: `vld4q_u8` deinterleaves 16 px (4×16 bytes) into the four
/// byte planes in one load. We remap those planes to (Y0,U,Y1,V) per `fmt`, decode the two
/// luma streams sharing chroma, and re-interleave with `vzipq_u8` for a `vst3q_u8` store.
#[cfg(target_arch = "aarch64")]
fn rgb_from_packed422_row_neon(src: &[u8], dst: &mut [u8], width: usize, fmt: Packed422) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let pairs = width / 2; // number of 2-px groups
                               // vld4q_u8 reads 16 elements per lane = 16 groups (64 bytes) = 32 px per iter.
        let bulk = pairs & !15;

        let (o_y0, o_u, o_y1, o_v) = fmt.offsets();
        let mut gi = 0usize; // group index
        while gi < bulk {
            let q = vld4q_u8(sp.add(gi * 4));
            let lanes = [q.0, q.1, q.2, q.3];
            let y0 = lanes[o_y0]; // 16 Y0 samples (even pixels)
            let y1 = lanes[o_y1]; // 16 Y1 samples (odd pixels)
            let u = lanes[o_u]; // 16 U samples (shared across the 2-px group)
            let v = lanes[o_v]; // 16 V samples (shared)

            // Pre-center chroma once (shared by both luma streams): U-128, V-128 as s16x16.
            let u_c = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u))),
                vdupq_n_s16(128),
            );
            let u_ch = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u))),
                vdupq_n_s16(128),
            );
            let v_c = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))),
                vdupq_n_s16(128),
            );
            let v_ch = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))),
                vdupq_n_s16(128),
            );

            // Chroma contributions (R/G/B) computed once per group, reused for Y0 and Y1.
            // Q20 coefficients exceed i16, so multiply against s32 lanes via vmlaq_n_s32.
            let chroma_rgb = |uc: int16x8_t, vc: int16x8_t| -> [(int32x4_t, int32x4_t); 3] {
                let (ul, uh) = (vmovl_s16(vget_low_s16(uc)), vmovl_s16(vget_high_s16(uc)));
                let (vl, vh) = (vmovl_s16(vget_low_s16(vc)), vmovl_s16(vget_high_s16(vc)));
                let half = vdupq_n_s32(ITUR_HALF);
                // R: + CVR*v ; G: + CUG*u + CVG*v ; B: + CUB*u
                let r_l = vmlaq_n_s32(half, vl, CVR);
                let r_h = vmlaq_n_s32(half, vh, CVR);
                let g_l = vmlaq_n_s32(vmlaq_n_s32(half, ul, CUG), vl, CVG);
                let g_h = vmlaq_n_s32(vmlaq_n_s32(half, uh, CUG), vh, CVG);
                let b_l = vmlaq_n_s32(half, ul, CUB);
                let b_h = vmlaq_n_s32(half, uh, CUB);
                [(r_l, r_h), (g_l, g_h), (b_l, b_h)]
            };
            let crgb_lo = chroma_rgb(u_c, v_c);
            let crgb_hi = chroma_rgb(u_ch, v_ch);

            // yy = max(0, Y-16) * CY (s32), for one u8x8 luma group.
            let yy_pair = |y8: uint8x8_t| -> (int32x4_t, int32x4_t) {
                let y16 = vreinterpretq_s16_u16(vmovl_u8(vqsub_u8(y8, vdup_n_u8(16))));
                (
                    vmulq_n_s32(vmovl_s16(vget_low_s16(y16)), CY),
                    vmulq_n_s32(vmovl_s16(vget_high_s16(y16)), CY),
                )
            };

            // combine yy + chroma contribution, >>shift, saturating narrow to u8x8
            let finish = |yy: (int32x4_t, int32x4_t), c: (int32x4_t, int32x4_t)| -> uint8x8_t {
                let lo = vshrq_n_s32::<ITUR_SHIFT>(vaddq_s32(yy.0, c.0));
                let hi = vshrq_n_s32::<ITUR_SHIFT>(vaddq_s32(yy.1, c.1));
                vqmovun_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)))
            };

            let decode8 = |y8: uint8x8_t,
                           crgb: &[(int32x4_t, int32x4_t); 3]|
             -> (uint8x8_t, uint8x8_t, uint8x8_t) {
                let yy = yy_pair(y8);
                (
                    finish(yy, crgb[0]),
                    finish(yy, crgb[1]),
                    finish(yy, crgb[2]),
                )
            };

            let (r0l, g0l, b0l) = decode8(vget_low_u8(y0), &crgb_lo);
            let (r1l, g1l, b1l) = decode8(vget_low_u8(y1), &crgb_lo);
            let (r0h, g0h, b0h) = decode8(vget_high_u8(y0), &crgb_hi);
            let (r1h, g1h, b1h) = decode8(vget_high_u8(y1), &crgb_hi);

            // Re-interleave even/odd luma pixels: output order y0[0],y1[0],y0[1],y1[1]...
            let r_lo = vzip_u8(r0l, r1l); // px 0..15 (groups 0..7)
            let g_lo = vzip_u8(g0l, g1l);
            let b_lo = vzip_u8(b0l, b1l);
            let r_hi = vzip_u8(r0h, r1h); // px 16..31 (groups 8..15)
            let g_hi = vzip_u8(g0h, g1h);
            let b_hi = vzip_u8(b0h, b1h);

            vst3q_u8(
                dp.add(gi * 2 * 3),
                uint8x16x3_t(
                    vcombine_u8(r_lo.0, r_lo.1),
                    vcombine_u8(g_lo.0, g_lo.1),
                    vcombine_u8(b_lo.0, b_lo.1),
                ),
            );
            vst3q_u8(
                dp.add((gi + 8) * 2 * 3),
                uint8x16x3_t(
                    vcombine_u8(r_hi.0, r_hi.1),
                    vcombine_u8(g_hi.0, g_hi.1),
                    vcombine_u8(b_hi.0, b_hi.1),
                ),
            );
            gi += 16;
        }
        // scalar tail over remaining groups
        let mut g = gi;
        while g < pairs {
            let base = g * 4;
            let y0 = *sp.add(base + o_y0) as i32;
            let u = *sp.add(base + o_u) as i32;
            let y1 = *sp.add(base + o_y1) as i32;
            let v = *sp.add(base + o_v) as i32;
            let (r0, g0, b0) = decode_px(yy_term(y0), u, v);
            let (r1, g1b, b1) = decode_px(yy_term(y1), u, v);
            let d = g * 2 * 3;
            *dp.add(d) = r0;
            *dp.add(d + 1) = g0;
            *dp.add(d + 2) = b0;
            *dp.add(d + 3) = r1;
            *dp.add(d + 4) = g1b;
            *dp.add(d + 5) = b1;
            g += 1;
        }
    }
}

/// Portable scalar packed-4:2:2 row decode. Oracle for the tests.
pub fn rgb_from_packed422_row_scalar(src: &[u8], dst: &mut [u8], width: usize, fmt: Packed422) {
    let (o_y0, o_u, o_y1, o_v) = fmt.offsets();
    for g in 0..width / 2 {
        let base = g * 4;
        let y0 = src[base + o_y0] as i32;
        let u = src[base + o_u] as i32;
        let y1 = src[base + o_y1] as i32;
        let v = src[base + o_v] as i32;
        let (r0, g0, b0) = decode_px(yy_term(y0), u, v);
        let (r1, g1b, b1) = decode_px(yy_term(y1), u, v);
        let d = g * 2 * 3;
        dst[d] = r0;
        dst[d + 1] = g0;
        dst[d + 2] = b0;
        dst[d + 3] = r1;
        dst[d + 4] = g1b;
        dst[d + 5] = b1;
    }
}

// ---- 4:2:0 planar ----------------------------------------------------------------

/// Chroma layout of a planar 4:2:0 stream after the Y plane.
#[derive(Clone, Copy)]
pub enum Planar420 {
    /// Interleaved `UV` (NV12).
    Nv12,
    /// Interleaved `VU` (NV21).
    Nv21,
    /// Separate U then V planes (I420).
    I420,
    /// Separate V then U planes (YV12).
    Yv12,
}

/// Decode a planar 4:2:0 buffer to RGB (`dst`, 3 bytes/px).
///
/// `y` is the full-res luma plane (`width*height` bytes); `c0`/`c1` are the two chroma
/// inputs whose meaning depends on `fmt` (for NV formats `c1` is empty and `c0` is the
/// interleaved plane). `width` and `height` must be even. Processes 2 luma rows per chroma
/// row; chroma is upsampled in-register so each (U,V) serves a 2×2 luma block.
pub fn rgb_from_planar420(
    y: &[u8],
    c0: &[u8],
    c1: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    fmt: Planar420,
) {
    debug_assert!(width.is_multiple_of(2) && height.is_multiple_of(2));
    let dst_row = width * 3;
    let cw = width / 2; // chroma width
    use super::super::kernel_common::PAR_THRESHOLD;

    let process_block = |cy: usize, dst: &mut [u8]| {
        // two luma rows for this chroma row
        let y_top = &y[(2 * cy) * width..(2 * cy) * width + width];
        let y_bot = &y[(2 * cy + 1) * width..(2 * cy + 1) * width + width];
        let (d_top, d_bot) = dst.split_at_mut(dst_row);
        rgb_from_planar420_block_scalar(y_top, y_bot, c0, c1, d_top, d_bot, width, cw, cy, fmt);
    };

    if width * height < PAR_THRESHOLD {
        for cy in 0..height / 2 {
            let two_rows = &mut dst[(2 * cy) * dst_row..(2 * cy) * dst_row + 2 * dst_row];
            process_block(cy, two_rows);
        }
        return;
    }
    use rayon::prelude::*;
    dst.par_chunks_mut(2 * dst_row)
        .enumerate()
        .for_each(|(cy, two_rows)| {
            process_block(cy, two_rows);
        });
}

/// Decode one 2-row block of a planar 4:2:0 image. Chroma is read once per 2×2 block and
/// applied to all four luma samples. Scalar (oracle) path; the NEON path upsamples chroma
/// in-register via `vzip` but the math is identical.
#[allow(clippy::too_many_arguments)]
fn rgb_from_planar420_block_scalar(
    y_top: &[u8],
    y_bot: &[u8],
    c0: &[u8],
    c1: &[u8],
    d_top: &mut [u8],
    d_bot: &mut [u8],
    width: usize,
    cw: usize,
    cy: usize,
    fmt: Planar420,
) {
    let _ = width;
    for cx in 0..cw {
        let (u, v) = chroma_at(c0, c1, cw, cy, cx, fmt);
        let x = cx * 2;
        // each (U,V) serves the 2×2 luma block at (x..x+2, top/bottom rows)
        for dx in 0..2 {
            let dt = (x + dx) * 3;
            let (r, g, b) = decode_px(yy_term(y_top[x + dx] as i32), u, v);
            d_top[dt] = r;
            d_top[dt + 1] = g;
            d_top[dt + 2] = b;
            let (r2, g2, b2) = decode_px(yy_term(y_bot[x + dx] as i32), u, v);
            d_bot[dt] = r2;
            d_bot[dt + 1] = g2;
            d_bot[dt + 2] = b2;
        }
    }
}

/// Read the (U, V) chroma sample for block `(cy, cx)` according to `fmt`.
#[inline]
fn chroma_at(c0: &[u8], c1: &[u8], cw: usize, cy: usize, cx: usize, fmt: Planar420) -> (i32, i32) {
    match fmt {
        Planar420::Nv12 => {
            let idx = cy * cw * 2 + cx * 2;
            (c0[idx] as i32, c0[idx + 1] as i32)
        }
        Planar420::Nv21 => {
            let idx = cy * cw * 2 + cx * 2;
            (c0[idx + 1] as i32, c0[idx] as i32)
        }
        Planar420::I420 => {
            let idx = cy * cw + cx;
            (c0[idx] as i32, c1[idx] as i32)
        }
        Planar420::Yv12 => {
            // c0 = V plane, c1 = U plane
            let idx = cy * cw + cx;
            (c1[idx] as i32, c0[idx] as i32)
        }
    }
}

// ===== Tests ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_u8(n: usize) -> Vec<u8> {
        (0..n).map(|v| (v * 7 + 3) as u8).collect()
    }

    // ----- Family A: u8 bit-exact vs scalar oracle -----
    #[test]
    fn ycc_from_rgb_u8_matches_scalar() {
        for order in [ChromaOrder::YCrCb, ChromaOrder::YuvCbCr] {
            let npix = 19; // exercises 16-px body + 3-px tail
            let src = ramp_u8(npix * 3);
            let mut simd = vec![0u8; npix * 3];
            let mut oracle = vec![0u8; npix * 3];
            ycc_from_rgb_u8(&src, &mut simd, npix, order);
            ycc_from_rgb_u8_scalar(&src, &mut oracle, npix, order);
            assert_eq!(simd, oracle, "order {:?}", order as u8);
        }
    }

    #[test]
    fn rgb_from_ycc_u8_matches_scalar() {
        for order in [ChromaOrder::YCrCb, ChromaOrder::YuvCbCr] {
            let npix = 19;
            let src = ramp_u8(npix * 3);
            let mut simd = vec![0u8; npix * 3];
            let mut oracle = vec![0u8; npix * 3];
            rgb_from_ycc_u8(&src, &mut simd, npix, order);
            rgb_from_ycc_u8_scalar(&src, &mut oracle, npix, order);
            assert_eq!(simd, oracle, "order {:?}", order as u8);
        }
    }

    #[test]
    fn ycc_u8_round_trip_close() {
        // u8 quantization makes this lossy; allow a small tolerance.
        let npix = 64;
        let rgb = ramp_u8(npix * 3);
        let mut ycc = vec![0u8; npix * 3];
        let mut back = vec![0u8; npix * 3];
        ycc_from_rgb_u8(&rgb, &mut ycc, npix, ChromaOrder::YCrCb);
        rgb_from_ycc_u8(&ycc, &mut back, npix, ChromaOrder::YCrCb);
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 3, "{a} vs {b}");
        }
    }

    #[test]
    fn ycc_u8_known_value_gray() {
        // A neutral gray (128,128,128): Y≈128, Cr=Cb=128.
        let src = vec![128u8, 128, 128];
        let mut dst = vec![0u8; 3];
        ycc_from_rgb_u8(&src, &mut dst, 1, ChromaOrder::YCrCb);
        assert_eq!(dst[0], 128);
        assert_eq!(dst[1], 128);
        assert_eq!(dst[2], 128);
    }

    // ----- Family A: f32 vs scalar -----
    #[test]
    fn ycc_from_rgb_f32_matches_scalar() {
        for order in [ChromaOrder::YCrCb, ChromaOrder::YuvCbCr] {
            let npix = 11;
            let src: Vec<f32> = (0..npix * 3).map(|v| (v % 17) as f32 / 16.0).collect();
            let mut simd = vec![0f32; npix * 3];
            let mut oracle = vec![0f32; npix * 3];
            ycc_from_rgb_f32(&src, &mut simd, npix, order);
            ycc_from_rgb_f32_scalar(&src, &mut oracle, npix, order);
            for (a, b) in simd.iter().zip(oracle.iter()) {
                assert!((a - b).abs() <= 1e-6, "{a} != {b}");
            }
        }
    }

    #[test]
    fn ycc_f32_round_trip() {
        let npix = 11;
        let rgb: Vec<f32> = (0..npix * 3).map(|v| (v % 17) as f32 / 16.0).collect();
        let mut ycc = vec![0f32; npix * 3];
        let mut back = vec![0f32; npix * 3];
        ycc_from_rgb_f32(&rgb, &mut ycc, npix, ChromaOrder::YuvCbCr);
        rgb_from_ycc_f32(&ycc, &mut back, npix, ChromaOrder::YuvCbCr);
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!((a - b).abs() <= 1e-5, "{a} != {b}");
        }
    }

    // ----- Family B: packed 4:2:2 -----
    #[test]
    fn packed422_matches_scalar() {
        for fmt in [Packed422::Yuyv, Packed422::Uyvy, Packed422::Yvyu] {
            let (w, h) = (18, 3); // 18 even; 9 groups/row -> 8-group body + 1 tail
            let src = ramp_u8(w * h * 2);
            let mut simd = vec![0u8; w * h * 3];
            let mut oracle = vec![0u8; w * h * 3];
            rgb_from_packed422(&src, &mut simd, w, h, fmt);
            for row in 0..h {
                rgb_from_packed422_row_scalar(
                    &src[row * w * 2..row * w * 2 + w * 2],
                    &mut oracle[row * w * 3..row * w * 3 + w * 3],
                    w,
                    fmt,
                );
            }
            assert_eq!(simd, oracle);
        }
    }

    #[test]
    fn packed422_known_gray() {
        // Y=16 (black floor), U=V=128 -> RGB ~ (0,0,0) limited range.
        let src = vec![16u8, 128, 16, 128]; // YUYV: Y0=16,U=128,Y1=16,V=128
        let mut dst = vec![0u8; 2 * 3];
        rgb_from_packed422(&src, &mut dst, 2, 1, Packed422::Yuyv);
        assert_eq!(&dst, &[0, 0, 0, 0, 0, 0]);
    }

    // ----- Family B: planar 4:2:0 -----
    #[test]
    fn planar420_nv12_matches_reference() {
        let (w, h) = (4, 4);
        let y: Vec<u8> = (0..w * h).map(|v| (v * 9 + 16) as u8).collect();
        let uv: Vec<u8> = (0..w * h / 2).map(|v| (v * 5 + 100) as u8).collect();
        let mut dst = vec![0u8; w * h * 3];
        rgb_from_planar420(&y, &uv, &[], &mut dst, w, h, Planar420::Nv12);

        // independent per-pixel reference
        let cw = w / 2;
        let mut expected = vec![0u8; w * h * 3];
        for row in 0..h {
            for col in 0..w {
                let cy = row / 2;
                let cx = col / 2;
                let idx = cy * cw * 2 + cx * 2;
                let u = uv[idx] as i32;
                let v = uv[idx + 1] as i32;
                let yy = (y[row * w + col] as i32 - 16).max(0) * CY;
                let (r, g, b) = decode_px(yy, u, v);
                let d = (row * w + col) * 3;
                expected[d] = r;
                expected[d + 1] = g;
                expected[d + 2] = b;
            }
        }
        assert_eq!(dst, expected);
    }

    #[test]
    fn planar420_i420_and_yv12_swap() {
        let (w, h) = (4, 2);
        let y: Vec<u8> = (0..w * h).map(|v| (v * 11 + 20) as u8).collect();
        let u: Vec<u8> = (0..w * h / 4).map(|v| (v * 3 + 110) as u8).collect();
        let v: Vec<u8> = (0..w * h / 4).map(|v| (v * 7 + 130) as u8).collect();

        let mut i420 = vec![0u8; w * h * 3];
        rgb_from_planar420(&y, &u, &v, &mut i420, w, h, Planar420::I420);

        // YV12 with planes swapped (V first, U second) must produce identical output.
        let mut yv12 = vec![0u8; w * h * 3];
        rgb_from_planar420(&y, &v, &u, &mut yv12, w, h, Planar420::Yv12);
        assert_eq!(i420, yv12);
    }
}
