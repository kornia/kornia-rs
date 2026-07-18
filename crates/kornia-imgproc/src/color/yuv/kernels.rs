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

// cv2 RGB2YUV / YUV2RGB (analog Y'UV) chroma constants — kornia's YUV order
// matches cv2.cvtColor byte-for-byte (and kornia-python's rgb_to_yuv).
const C_VI: i32 = 14369; //  0.877 * 2^14  ((R-Y) -> V)
const C_UI: i32 = 8061; //   0.492 * 2^14  ((B-Y) -> U)
const C_V2R: i32 = 18678; //  1.140 * 2^14
const C_U2G: i32 = -6472; // -0.395 * 2^14
const C_V2G: i32 = -9519; // -0.581 * 2^14
                          // 2.032 * 2^14 = 33292 overflows i16 for the NEON `vmlal_n_s16` path, so it
                          // is carried as 16646 and doubled in 32-bit — exact, since 33292 = 16646 << 1.
const C_U2B_HALF: i32 = 16646;

// f32 full-range coefficients (inputs in [0,1]).
const F_YR: f32 = 0.299;
const F_YG: f32 = 0.587;
const F_YB: f32 = 0.114;
const F_CR: f32 = 0.713; // (R-Y) * 0.713 + 0.5
const F_CB: f32 = 0.564; // (B-Y) * 0.564 + 0.5
                         // f32 Y'UV (cv2 RGB2YUV / YUV2RGB float constants).
const F_VI: f32 = 0.877;
const F_UI: f32 = 0.492;
const F_V2R: f32 = 1.140;
const F_U2G: f32 = -0.395;
const F_V2G: f32 = -0.581;
const F_U2B: f32 = 2.032;

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

/// Scalar RGB u8 -> (Y, chroma-R-slot, chroma-B-slot) u8, full-range Q14.
/// `YCrCb` produces OpenCV's (Cr, Cb); `YuvCbCr` produces cv2 RGB2YUV's
/// (V, U) — same shape, Y'UV chroma scales. Bit-exact oracle.
#[inline]
fn ycc_from_rgb_u8_px(r: i32, g: i32, b: i32, order: ChromaOrder) -> (u8, u8, u8) {
    let (c_rv, c_bu) = match order {
        ChromaOrder::YCrCb => (C_YCRI, C_YCBI),
        ChromaOrder::YuvCbCr => (C_VI, C_UI),
    };
    let y = (C_YR * r + C_YG * g + C_YB * b + Q14_HALF) >> Q14_SHIFT;
    let cr = ((r - y) * c_rv + (128 << Q14_SHIFT) + Q14_HALF) >> Q14_SHIFT;
    let cb = ((b - y) * c_bu + (128 << Q14_SHIFT) + Q14_HALF) >> Q14_SHIFT;
    (
        y.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
    )
}

/// Scalar (Y, Cr|V, Cb|U) u8 -> RGB u8, full-range Q14, per-order inverse
/// matrix (`YuvCbCr` = cv2 YUV2RGB constants). Bit-exact oracle.
#[inline]
fn rgb_from_ycc_u8_px(y: i32, cr: i32, cb: i32, order: ChromaOrder) -> (u8, u8, u8) {
    let cr = cr - 128;
    let cb = cb - 128;
    let (r, g, b) = match order {
        ChromaOrder::YCrCb => (
            y + ((C_CR2R * cr + Q14_HALF) >> Q14_SHIFT),
            y + ((C_CR2G * cr + C_CB2G * cb + Q14_HALF) >> Q14_SHIFT),
            y + ((C_CB2B * cb + Q14_HALF) >> Q14_SHIFT),
        ),
        ChromaOrder::YuvCbCr => (
            y + ((C_V2R * cr + Q14_HALF) >> Q14_SHIFT),
            y + ((C_V2G * cr + C_U2G * cb + Q14_HALF) >> Q14_SHIFT),
            y + (((C_U2B_HALF * cb) * 2 + Q14_HALF) >> Q14_SHIFT),
        ),
    };
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
        let (c_rv, c_bu) = match order {
            ChromaOrder::YCrCb => (C_YCRI as i16, C_YCBI as i16),
            ChromaOrder::YuvCbCr => (C_VI as i16, C_UI as i16),
        };

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
                let crl = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, drl, c_rv));
                let crh = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, drh, c_rv));

                // Cb = ((B-Y)*C_YCBI + 128<<14 + half) >> 14
                let dbl = vsub_s16(bl, yl16);
                let dbh = vsub_s16(bh, yh16);
                let cbl = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, dbl, c_bu));
                let cbh = vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(bias128, dbh, c_bu));

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
                order,
            );
            store_ycc(dp, si, y, cr, cb, order);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store_ycc(dp: *mut u8, si: usize, y: u8, cr: u8, cb: u8, order: ChromaOrder) {
    unsafe {
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
}

/// Portable scalar RGB u8 -> YCbCr/YUV u8. Oracle for the tests.
pub fn ycc_from_rgb_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (y, cr, cb) = ycc_from_rgb_u8_px(
            src[si] as i32,
            src[si + 1] as i32,
            src[si + 2] as i32,
            order,
        );
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
        let (c_cr2r, c_cr2g, c_cb2g) = match order {
            ChromaOrder::YCrCb => (C_CR2R as i16, C_CR2G as i16, C_CB2G as i16),
            ChromaOrder::YuvCbCr => (C_V2R as i16, C_V2G as i16, C_U2G as i16),
        };
        // The B coefficient may exceed i16 (cv2's 2.032*2^14 = 33292); carry
        // its half and double the 32-bit product — exact.
        let (c_cb2b_half, double_b) = match order {
            ChromaOrder::YCrCb => (C_CB2B as i16, false),
            ChromaOrder::YuvCbCr => (C_U2B_HALF as i16, true),
        };

        let conv8 = |y: int16x8_t,
                     cr: int16x8_t,
                     cb: int16x8_t|
         -> (uint8x8_t, uint8x8_t, uint8x8_t) {
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
                vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(half, crl, c_cr2r)),
            );
            let rh = vaddq_s32(
                yh32,
                vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(half, crh, c_cr2r)),
            );
            // G = Y + ((C_CR2G*cr + C_CB2G*cb + half) >> 14)
            let gl = vaddq_s32(
                yl32,
                vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(vmlal_n_s16(half, crl, c_cr2g), cbl, c_cb2g)),
            );
            let gh = vaddq_s32(
                yh32,
                vshrq_n_s32::<Q14_SHIFT>(vmlal_n_s16(vmlal_n_s16(half, crh, c_cr2g), cbh, c_cb2g)),
            );
            // B = Y + ((c_b*cb + half) >> 14), with the product doubled
            // when the coefficient was halved to fit i16.
            let mut bpl = vmull_n_s16(cbl, c_cb2b_half);
            let mut bph = vmull_n_s16(cbh, c_cb2b_half);
            if double_b {
                bpl = vaddq_s32(bpl, bpl);
                bph = vaddq_s32(bph, bph);
            }
            let bl = vaddq_s32(yl32, vshrq_n_s32::<Q14_SHIFT>(vaddq_s32(bpl, half)));
            let bh = vaddq_s32(yh32, vshrq_n_s32::<Q14_SHIFT>(vaddq_s32(bph, half)));

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
            let (r, g, b) = rgb_from_ycc_u8_px(*sp.add(si) as i32, cr, cb, order);
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
        let (r, g, b) = rgb_from_ycc_u8_px(src[si] as i32, cr, cb, order);
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
        let (k_rv, k_bu) = match order {
            ChromaOrder::YCrCb => (F_CR, F_CB),
            ChromaOrder::YuvCbCr => (F_VI, F_UI),
        };
        let kcr = vdupq_n_f32(k_rv);
        let kcb = vdupq_n_f32(k_bu);
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
            let (y, cr, cb) =
                ycc_from_rgb_f32_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2), order);
            store_ycc_f32(dp, si, y, cr, cb, order);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store_ycc_f32(dp: *mut f32, si: usize, y: f32, cr: f32, cb: f32, order: ChromaOrder) {
    unsafe {
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
}

#[inline]
fn ycc_from_rgb_f32_px(r: f32, g: f32, b: f32, order: ChromaOrder) -> (f32, f32, f32) {
    let (k_rv, k_bu) = match order {
        ChromaOrder::YCrCb => (F_CR, F_CB),
        ChromaOrder::YuvCbCr => (F_VI, F_UI),
    };
    let y = F_YR * r + F_YG * g + F_YB * b;
    let cr = (r - y) * k_rv + 0.5;
    let cb = (b - y) * k_bu + 0.5;
    (y, cr, cb)
}

/// Portable scalar RGB f32 -> YCbCr/YUV f32. Oracle for the tests.
pub fn ycc_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (y, cr, cb) = ycc_from_rgb_f32_px(src[si], src[si + 1], src[si + 2], order);
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
        // YCrCb inverts via the derived reciprocals; YuvCbCr uses cv2's
        // literal YUV2RGB float constants (1.140 / 2.032 / -0.395 / -0.581).
        let (kr, kb) = match order {
            ChromaOrder::YCrCb => (1.0 / F_CR, 1.0 / F_CB),
            ChromaOrder::YuvCbCr => (F_V2R, F_U2B),
        };
        let inv_cr = vdupq_n_f32(kr);
        let inv_cb = vdupq_n_f32(kb);
        let ku2g = vdupq_n_f32(F_U2G);
        let kv2g = vdupq_n_f32(F_V2G);
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
            let g = match order {
                ChromaOrder::YCrCb => {
                    // g = (y - yr*r - yb*b) * inv_yg
                    let num = vsubq_f32(vsubq_f32(y, vmulq_f32(yr, r)), vmulq_f32(yb, b));
                    vmulq_f32(num, inv_yg)
                }
                ChromaOrder::YuvCbCr => {
                    // g = y - 0.395*(u-0.5) - 0.581*(v-0.5)
                    vfmaq_f32(
                        vfmaq_f32(y, vsubq_f32(cb, half), ku2g),
                        vsubq_f32(cr, half),
                        kv2g,
                    )
                }
            };
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
            let (r, g, b) = rgb_from_ycc_f32_px(*sp.add(si), cr, cb, order);
            *dp.add(si) = r;
            *dp.add(si + 1) = g;
            *dp.add(si + 2) = b;
            i += 1;
        }
    }
}

#[inline]
fn rgb_from_ycc_f32_px(y: f32, cr: f32, cb: f32, order: ChromaOrder) -> (f32, f32, f32) {
    match order {
        ChromaOrder::YCrCb => {
            let r = y + (cr - 0.5) / F_CR;
            let b = y + (cb - 0.5) / F_CB;
            let g = (y - F_YR * r - F_YB * b) / F_YG;
            (r, g, b)
        }
        ChromaOrder::YuvCbCr => {
            let r = y + F_V2R * (cr - 0.5);
            let g = y + F_U2G * (cb - 0.5) + F_V2G * (cr - 0.5);
            let b = y + F_U2B * (cb - 0.5);
            (r, g, b)
        }
    }
}

/// Portable scalar YCbCr/YUV f32 -> RGB f32. Oracle for the tests.
pub fn rgb_from_ycc_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize, order: ChromaOrder) {
    for i in 0..npixels {
        let si = i * 3;
        let (cr, cb) = match order {
            ChromaOrder::YCrCb => (src[si + 1], src[si + 2]),
            ChromaOrder::YuvCbCr => (src[si + 2], src[si + 1]),
        };
        let (r, g, b) = rgb_from_ycc_f32_px(src[si], cr, cb, order);
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub(crate) fn offsets(self) -> (usize, usize, usize, usize) {
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

            // combine yy + chroma contribution>>shift, saturating narrow to u8x8
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on aarch64; row slices sized by the caller.
            unsafe {
                rgb_from_planar420_block_neon(y_top, y_bot, c0, c1, d_top, d_bot, cw, cy, fmt)
            };
            return;
        }
        #[allow(unreachable_code)]
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

/// NEON planar 4:2:0 two-row block decode: 32 luma cols (16 chroma) per iter. Chroma is
/// upsampled ×2 horizontally in-register via `vzipq_u8` (each (U,V) serves a 2×2 luma
/// block, reused across both rows). Q20 BT.601 math is identical to the scalar oracle;
/// a scalar tail covers `cw % 16`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn rgb_from_planar420_block_neon(
    y_top: &[u8],
    y_bot: &[u8],
    c0: &[u8],
    c1: &[u8],
    d_top: &mut [u8],
    d_bot: &mut [u8],
    cw: usize,
    cy: usize,
    fmt: Planar420,
) {
    unsafe {
        use std::arch::aarch64::*;
        let ytp = y_top.as_ptr();
        let ybp = y_bot.as_ptr();
        let dtp = d_top.as_mut_ptr();
        let dbp = d_bot.as_mut_ptr();

        // Load 16 (U,V) chroma samples starting at column `cx` per layout.
        let load_uv = |cx: usize| -> (uint8x16_t, uint8x16_t) {
            match fmt {
                Planar420::Nv12 => {
                    let q = vld2q_u8(c0.as_ptr().add(cy * cw * 2 + cx * 2));
                    (q.0, q.1)
                }
                Planar420::Nv21 => {
                    let q = vld2q_u8(c0.as_ptr().add(cy * cw * 2 + cx * 2));
                    (q.1, q.0)
                }
                Planar420::I420 => (
                    vld1q_u8(c0.as_ptr().add(cy * cw + cx)),
                    vld1q_u8(c1.as_ptr().add(cy * cw + cx)),
                ),
                Planar420::Yv12 => (
                    vld1q_u8(c1.as_ptr().add(cy * cw + cx)),
                    vld1q_u8(c0.as_ptr().add(cy * cw + cx)),
                ),
            }
        };

        // Chroma RGB contribution for a centred u8x8 (U,V) → [(r_lo,r_hi),(g..),(b..)] s32 pairs.
        let chroma_rgb = |u8x8: uint8x8_t, v8x8: uint8x8_t| -> [(int32x4_t, int32x4_t); 3] {
            let uc = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8x8)), vdupq_n_s16(128));
            let vc = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v8x8)), vdupq_n_s16(128));
            let (ul, uh) = (vmovl_s16(vget_low_s16(uc)), vmovl_s16(vget_high_s16(uc)));
            let (vl, vh) = (vmovl_s16(vget_low_s16(vc)), vmovl_s16(vget_high_s16(vc)));
            let half = vdupq_n_s32(ITUR_HALF);
            let r = (vmlaq_n_s32(half, vl, CVR), vmlaq_n_s32(half, vh, CVR));
            let g = (
                vmlaq_n_s32(vmlaq_n_s32(half, ul, CUG), vl, CVG),
                vmlaq_n_s32(vmlaq_n_s32(half, uh, CUG), vh, CVG),
            );
            let b = (vmlaq_n_s32(half, ul, CUB), vmlaq_n_s32(half, uh, CUB));
            [r, g, b]
        };
        let yy_pair = |y8: uint8x8_t| -> (int32x4_t, int32x4_t) {
            let y16 = vreinterpretq_s16_u16(vmovl_u8(vqsub_u8(y8, vdup_n_u8(16))));
            (
                vmulq_n_s32(vmovl_s16(vget_low_s16(y16)), CY),
                vmulq_n_s32(vmovl_s16(vget_high_s16(y16)), CY),
            )
        };
        let finish = |yy: (int32x4_t, int32x4_t), c: (int32x4_t, int32x4_t)| -> uint8x8_t {
            let lo = vshrq_n_s32::<ITUR_SHIFT>(vaddq_s32(yy.0, c.0));
            let hi = vshrq_n_s32::<ITUR_SHIFT>(vaddq_s32(yy.1, c.1));
            vqmovun_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)))
        };
        // Decode a 16-px luma run (`yp`+col) with chroma groups for px 0..8 (clo) / 8..16 (chi).
        let decode16_store = |yp: *const u8,
                              dp: *mut u8,
                              col: usize,
                              clo: &[(int32x4_t, int32x4_t); 3],
                              chi: &[(int32x4_t, int32x4_t); 3]| {
            let y = vld1q_u8(yp.add(col));
            let yl = yy_pair(vget_low_u8(y));
            let yh = yy_pair(vget_high_u8(y));
            let r = vcombine_u8(finish(yl, clo[0]), finish(yh, chi[0]));
            let g = vcombine_u8(finish(yl, clo[1]), finish(yh, chi[1]));
            let b = vcombine_u8(finish(yl, clo[2]), finish(yh, chi[2]));
            vst3q_u8(dp.add(col * 3), uint8x16x3_t(r, g, b));
        };

        let bulk = cw & !15;
        let mut cx = 0usize;
        while cx < bulk {
            let (u, v) = load_uv(cx);
            // ×2 horizontal upsample: lane i → luma cols 2i, 2i+1.
            let uz = vzipq_u8(u, u);
            let vz = vzipq_u8(v, v);
            let g0 = chroma_rgb(vget_low_u8(uz.0), vget_low_u8(vz.0)); // cols 0..8
            let g1 = chroma_rgb(vget_high_u8(uz.0), vget_high_u8(vz.0)); // cols 8..16
            let g2 = chroma_rgb(vget_low_u8(uz.1), vget_low_u8(vz.1)); // cols 16..24
            let g3 = chroma_rgb(vget_high_u8(uz.1), vget_high_u8(vz.1)); // cols 24..32
            let col = cx * 2;
            decode16_store(ytp, dtp, col, &g0, &g1);
            decode16_store(ytp, dtp, col + 16, &g2, &g3);
            decode16_store(ybp, dbp, col, &g0, &g1);
            decode16_store(ybp, dbp, col + 16, &g2, &g3);
            cx += 16;
        }
        // scalar tail (cw % 16)
        while cx < cw {
            let (u, v) = chroma_at(c0, c1, cw, cy, cx, fmt);
            let x = cx * 2;
            for dx in 0..2 {
                let dt = (x + dx) * 3;
                let (r, g, b) = decode_px(yy_term(*ytp.add(x + dx) as i32), u, v);
                *dtp.add(dt) = r;
                *dtp.add(dt + 1) = g;
                *dtp.add(dt + 2) = b;
                let (r2, g2, b2) = decode_px(yy_term(*ybp.add(x + dx) as i32), u, v);
                *dbp.add(dt) = r2;
                *dbp.add(dt + 1) = g2;
                *dbp.add(dt + 2) = b2;
            }
            cx += 1;
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

// ===== Family C: video encode (RGB -> YUV, BT.601 limited, Q8) ======================
//
// Forward libyuv BT.601 limited-range coefficients (Q8). These are the exact inverse
// of the Family-B decode constants (1.164 / 2.018 / -0.391 / -0.813 / 1.596), so an
// encode→decode round-trip is stable to within chroma-subsampling error.

const ENC_SHIFT: i32 = 8;
const ENC_HALF: i32 = 1 << 7;
const YR: i32 = 66; // R -> Y   (+16 offset)
const YG: i32 = 129; // G -> Y
const YB: i32 = 25; // B -> Y
const UR: i32 = -38; // R -> U   (+128 offset)
const UG: i32 = -74; // G -> U
const UB: i32 = 112; // B -> U
const VR: i32 = 112; // R -> V   (+128 offset)
const VG: i32 = -94; // G -> V
const VB: i32 = -18; // B -> V

/// Encode one RGB triple to a BT.601-limited luma byte. Bit-exact oracle.
#[inline]
fn encode_y(r: i32, g: i32, b: i32) -> u8 {
    (((YR * r + YG * g + YB * b + ENC_HALF) >> ENC_SHIFT) + 16).clamp(0, 255) as u8
}

/// Encode an (already-averaged) RGB triple to a BT.601-limited `(U, V)` chroma pair.
#[inline]
fn encode_uv(r: i32, g: i32, b: i32) -> (u8, u8) {
    let u = ((UR * r + UG * g + UB * b + ENC_HALF) >> ENC_SHIFT) + 128;
    let v = ((VR * r + VG * g + VB * b + ENC_HALF) >> ENC_SHIFT) + 128;
    (u.clamp(0, 255) as u8, v.clamp(0, 255) as u8)
}

// ---- 4:2:2 packed encode ---------------------------------------------------------

/// Encode RGB (`src`, 3 bytes/px) to packed 4:2:2 YUYV (`dst`, 2 bytes/px), BT.601
/// limited. Luma is per-pixel; the shared chroma of each horizontal pixel pair is the
/// rounded average of the two pixels. `width` must be even. Parallel over rows.
pub fn yuyv_from_rgb(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    debug_assert!(width.is_multiple_of(2));
    debug_assert!(src.len() >= width * height * 3);
    debug_assert!(dst.len() >= width * height * 2);
    let src_row = width * 3;
    let dst_row = width * 2;
    use super::super::kernel_common::PAR_THRESHOLD;
    if width * height < PAR_THRESHOLD {
        for row in 0..height {
            yuyv_from_rgb_row(
                &src[row * src_row..row * src_row + src_row],
                &mut dst[row * dst_row..row * dst_row + dst_row],
                width,
            );
        }
        return;
    }
    use rayon::prelude::*;
    dst.par_chunks_mut(dst_row)
        .enumerate()
        .for_each(|(row, d)| {
            yuyv_from_rgb_row(&src[row * src_row..row * src_row + src_row], d, width);
        });
}

#[inline]
fn yuyv_from_rgb_row(src: &[u8], dst: &mut [u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; slices sized by the caller.
    unsafe {
        yuyv_from_rgb_row_neon(src, dst, width);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        // SAFETY: avx2 verified above; slices sized by the caller.
        unsafe {
            yuyv_from_rgb_row_avx2(src, dst, width);
            return;
        }
    }
    #[allow(unreachable_code)]
    yuyv_from_rgb_row_scalar(src, dst, width);
}

/// Scalar YUYV row encode (oracle + SIMD tail). Each pair → `Y0 U Y1 V`.
#[inline]
fn yuyv_from_rgb_row_scalar(src: &[u8], dst: &mut [u8], width: usize) {
    for g in 0..width / 2 {
        let s = g * 6;
        let (r0, g0, b0) = (src[s] as i32, src[s + 1] as i32, src[s + 2] as i32);
        let (r1, g1, b1) = (src[s + 3] as i32, src[s + 4] as i32, src[s + 5] as i32);
        let y0 = encode_y(r0, g0, b0);
        let y1 = encode_y(r1, g1, b1);
        // Shared chroma: rounded average of the two source pixels.
        let ra = (r0 + r1 + 1) >> 1;
        let ga = (g0 + g1 + 1) >> 1;
        let ba = (b0 + b1 + 1) >> 1;
        let (u, v) = encode_uv(ra, ga, ba);
        let d = g * 4;
        dst[d] = y0; // Y0 U Y1 V
        dst[d + 1] = u;
        dst[d + 2] = y1;
        dst[d + 3] = v;
    }
}

/// NEON YUYV row encode, 16 px (8 pairs → 32 dst bytes) per iter. `vld3q_u8`
/// deinterleaves RGB; luma uses the Q8 (66,129,25) matrix; per-pair chroma is the
/// rounded pixel average (`vpaddlq_u8` + `vrshrn<1>`) put through the signed U/V
/// matrix. The interleaved `[U0,V0,U1,V1,…]` chroma vector zipped against luma by
/// `vst2q_u8` yields the `Y0 U Y1 V` stream directly. Bit-identical to scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn yuyv_from_rgb_row_neon(src: &[u8], dst: &mut [u8], width: usize) {
    unsafe {
        use std::arch::aarch64::*;
        let bulk = width & !15;
        let mut x = 0;
        while x < bulk {
            let rgb = vld3q_u8(src.as_ptr().add(x * 3));
            // Luma for 16 px.
            let yacc = |r: uint8x8_t, g: uint8x8_t, b: uint8x8_t| -> uint8x8_t {
                let mut a = vmull_u8(r, vdup_n_u8(YR as u8));
                a = vmlal_u8(a, g, vdup_n_u8(YG as u8));
                a = vmlal_u8(a, b, vdup_n_u8(YB as u8));
                vqadd_u8(vrshrn_n_u16::<8>(a), vdup_n_u8(16))
            };
            let y16 = vcombine_u8(
                yacc(vget_low_u8(rgb.0), vget_low_u8(rgb.1), vget_low_u8(rgb.2)),
                yacc(
                    vget_high_u8(rgb.0),
                    vget_high_u8(rgb.1),
                    vget_high_u8(rgb.2),
                ),
            );
            // Per-pair rounded chroma averages (8 pairs).
            let ravg = vrshrn_n_u16::<1>(vpaddlq_u8(rgb.0));
            let gavg = vrshrn_n_u16::<1>(vpaddlq_u8(rgb.1));
            let bavg = vrshrn_n_u16::<1>(vpaddlq_u8(rgb.2));
            let r16 = vreinterpretq_s16_u16(vmovl_u8(ravg));
            let g16 = vreinterpretq_s16_u16(vmovl_u8(gavg));
            let b16 = vreinterpretq_s16_u16(vmovl_u8(bavg));
            let chroma = |cr: i16, cg: i16, cb: i16| -> uint8x8_t {
                let mut a = vmulq_s16(r16, vdupq_n_s16(cr));
                a = vmlaq_s16(a, g16, vdupq_n_s16(cg));
                a = vmlaq_s16(a, b16, vdupq_n_s16(cb));
                // (a + 128) >> 8 (rounded, signed) + 128, saturate to u8.
                vqmovun_s16(vaddq_s16(vrshrq_n_s16::<8>(a), vdupq_n_s16(128)))
            };
            let u8v = chroma(UR as i16, UG as i16, UB as i16);
            let v8v = chroma(VR as i16, VG as i16, VB as i16);
            // chroma stream [U0,V0,U1,V1,…]; vst2 zips it against luma → Y0 U Y1 V.
            let uv = vzip_u8(u8v, v8v);
            let c16 = vcombine_u8(uv.0, uv.1);
            vst2q_u8(dst.as_mut_ptr().add(x * 2), uint8x16x2_t(y16, c16));
            x += 16;
        }
        if bulk < width {
            yuyv_from_rgb_row_scalar(&src[bulk * 3..], &mut dst[bulk * 2..], width - bulk);
        }
    }
}

/// AVX2 YUYV row encode, 16 px/iter. `pshufb` deinterleave (no native `vld3`); luma
/// via the Q8 matrix; per-pair chroma average via `maddubs` (pairwise sum) + rounded
/// halve, through the signed U/V matrix. `unpacklo/hi_epi8(Y, [U0,V0,U1,V1,…])`
/// builds the `Y0 U Y1 V` stream. Bit-identical to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn yuyv_from_rgb_row_avx2(src: &[u8], dst: &mut [u8], width: usize) {
    use std::arch::x86_64::*;
    let m_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let m_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let m_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    let c_yr = _mm_set1_epi16(YR as i16);
    let c_yg = _mm_set1_epi16(YG as i16);
    let c_yb = _mm_set1_epi16(YB as i16);
    let half = _mm_set1_epi16(ENC_HALF as i16);
    let off16 = _mm_set1_epi16(16);
    let off128 = _mm_set1_epi16(128);
    let ones = _mm_set1_epi8(1);
    let zero = _mm_setzero_si128();

    let yhalf = |r: __m128i, g: __m128i, b: __m128i| -> __m128i {
        let mut t = _mm_mullo_epi16(r, c_yr);
        t = _mm_add_epi16(t, _mm_mullo_epi16(g, c_yg));
        t = _mm_add_epi16(t, _mm_mullo_epi16(b, c_yb));
        t = _mm_srli_epi16(_mm_add_epi16(t, half), 8);
        _mm_add_epi16(t, off16)
    };
    // pairwise rounded average of a u8x16 channel → u16x8 in [0,255].
    let pair_avg = |c: __m128i| -> __m128i {
        let sum = _mm_maddubs_epi16(c, ones); // [c0+c1, c2+c3, …]
        _mm_srli_epi16(_mm_add_epi16(sum, _mm_set1_epi16(1)), 1)
    };

    let bulk = width & !15;
    let mut x = 0;
    while x < bulk {
        let p = src.as_ptr().add(x * 3);
        let s0 = _mm_loadu_si128(p as *const __m128i);
        let s1 = _mm_loadu_si128(p.add(16) as *const __m128i);
        let s2 = _mm_loadu_si128(p.add(32) as *const __m128i);
        let r = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(s0, m_r0), _mm_shuffle_epi8(s1, m_r1)),
            _mm_shuffle_epi8(s2, m_r2),
        );
        let g = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(s0, m_g0), _mm_shuffle_epi8(s1, m_g1)),
            _mm_shuffle_epi8(s2, m_g2),
        );
        let b = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(s0, m_b0), _mm_shuffle_epi8(s1, m_b1)),
            _mm_shuffle_epi8(s2, m_b2),
        );
        let y16 = _mm_packus_epi16(
            yhalf(
                _mm_unpacklo_epi8(r, zero),
                _mm_unpacklo_epi8(g, zero),
                _mm_unpacklo_epi8(b, zero),
            ),
            yhalf(
                _mm_unpackhi_epi8(r, zero),
                _mm_unpackhi_epi8(g, zero),
                _mm_unpackhi_epi8(b, zero),
            ),
        );
        let (ravg, gavg, bavg) = (pair_avg(r), pair_avg(g), pair_avg(b));
        let chroma = |cr: i16, cg: i16, cb: i16| -> __m128i {
            let mut a = _mm_mullo_epi16(ravg, _mm_set1_epi16(cr));
            a = _mm_add_epi16(a, _mm_mullo_epi16(gavg, _mm_set1_epi16(cg)));
            a = _mm_add_epi16(a, _mm_mullo_epi16(bavg, _mm_set1_epi16(cb)));
            a = _mm_srai_epi16(_mm_add_epi16(a, half), 8); // (a+128)>>8 signed
            _mm_add_epi16(a, off128)
        };
        let u = _mm_packus_epi16(chroma(UR as i16, UG as i16, UB as i16), zero);
        let v = _mm_packus_epi16(chroma(VR as i16, VG as i16, VB as i16), zero);
        let c16 = _mm_unpacklo_epi8(u, v); // [U0,V0,U1,V1,…,U7,V7]
        _mm_storeu_si128(
            dst.as_mut_ptr().add(x * 2) as *mut __m128i,
            _mm_unpacklo_epi8(y16, c16), // Y0 U Y1 V … (first 8 px)
        );
        _mm_storeu_si128(
            dst.as_mut_ptr().add(x * 2 + 16) as *mut __m128i,
            _mm_unpackhi_epi8(y16, c16), // … (next 8 px)
        );
        x += 16;
    }
    if bulk < width {
        yuyv_from_rgb_row_scalar(&src[bulk * 3..], &mut dst[bulk * 2..], width - bulk);
    }
}

// ---- 4:2:0 planar encode (NV12) --------------------------------------------------

/// Encode RGB (`src`, 3 bytes/px) to planar 4:2:0 NV12 (BT.601 limited): a full-res
/// luma plane `y_out` (`w*h`) followed by an interleaved `UV` plane `uv_out` (`w*h/2`).
/// Each chroma pair is the rounded average of its 2×2 luma block. `w`/`h` must be even.
pub fn nv12_from_rgb(src: &[u8], y_out: &mut [u8], uv_out: &mut [u8], width: usize, height: usize) {
    debug_assert!(width.is_multiple_of(2) && height.is_multiple_of(2));
    debug_assert!(src.len() >= width * height * 3);
    debug_assert!(y_out.len() >= width * height);
    debug_assert!(uv_out.len() >= width * height / 2);
    let src_row = width * 3;
    use super::super::kernel_common::PAR_THRESHOLD;

    // Fused: one task per chroma row handles its 2 luma rows + 1 UV row, so each
    // source row is read once (the Y and UV reads hit the same L1-hot rows) instead
    // of two separate full-image passes.
    let process = |cy: usize, y_block: &mut [u8], uv_row: &mut [u8]| {
        let top = &src[(2 * cy) * src_row..(2 * cy) * src_row + src_row];
        let bot = &src[(2 * cy + 1) * src_row..(2 * cy + 1) * src_row + src_row];
        let (y_top, y_bot) = y_block.split_at_mut(width);
        encode_y_row(top, y_top, width);
        encode_y_row(bot, y_bot, width);
        encode_uv_row(top, bot, uv_row, width);
    };

    if width * height < PAR_THRESHOLD {
        for cy in 0..height / 2 {
            let y_block = &mut y_out[(2 * cy) * width..(2 * cy) * width + 2 * width];
            let uv_row = &mut uv_out[cy * width..cy * width + width];
            process(cy, y_block, uv_row);
        }
        return;
    }
    use rayon::prelude::*;
    y_out
        .par_chunks_mut(2 * width)
        .zip(uv_out.par_chunks_mut(width))
        .enumerate()
        .for_each(|(cy, (y_block, uv_row))| process(cy, y_block, uv_row));
}

/// Luma-row encode dispatcher (NEON / AVX2 / scalar).
#[inline]
fn encode_y_row(src: &[u8], dst: &mut [u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; slices sized by the caller.
    unsafe {
        encode_y_row_neon(src, dst, width);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        // SAFETY: avx2 verified above; slices sized by the caller.
        unsafe {
            encode_y_row_avx2(src, dst, width);
            return;
        }
    }
    #[allow(unreachable_code)]
    for (x, yo) in dst.iter_mut().enumerate() {
        let s = x * 3;
        *yo = encode_y(src[s] as i32, src[s + 1] as i32, src[s + 2] as i32);
    }
}

/// Chroma-row encode dispatcher (NEON / AVX2 / scalar): 2×2 average → interleaved U,V.
#[inline]
fn encode_uv_row(top: &[u8], bot: &[u8], uv: &mut [u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; slices sized by the caller.
    unsafe {
        encode_uv_row_neon(top, bot, uv, width);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        // SAFETY: avx2 verified above; slices sized by the caller.
        unsafe {
            encode_uv_row_avx2(top, bot, uv, width);
            return;
        }
    }
    #[allow(unreachable_code)]
    encode_uv_row_scalar(top, bot, uv, width);
}

/// Scalar chroma-row encode (oracle + SIMD tail): each 2×2 RGB block → one (U,V).
#[inline]
fn encode_uv_row_scalar(top: &[u8], bot: &[u8], uv: &mut [u8], width: usize) {
    for cx in 0..width / 2 {
        let s = cx * 6;
        let r = top[s] as i32 + top[s + 3] as i32 + bot[s] as i32 + bot[s + 3] as i32;
        let g = top[s + 1] as i32 + top[s + 4] as i32 + bot[s + 1] as i32 + bot[s + 4] as i32;
        let b = top[s + 2] as i32 + top[s + 5] as i32 + bot[s + 2] as i32 + bot[s + 5] as i32;
        let (u, v) = encode_uv((r + 2) >> 2, (g + 2) >> 2, (b + 2) >> 2);
        uv[cx * 2] = u;
        uv[cx * 2 + 1] = v;
    }
}

/// NEON chroma-row encode, 16 chroma px (32 src px × 2 rows → 32 UV bytes) per iter.
/// `vpaddlq_u8` pairwise-sums each row, the two rows add to the 2×2 block sum, and
/// `vrshrn<2>` does `(sum+2)>>2`; the signed U/V matrix then runs on the averages,
/// zipped to the interleaved `[U0,V0,…]` UV plane. Bit-identical to scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_uv_row_neon(top: &[u8], bot: &[u8], uv: &mut [u8], width: usize) {
    unsafe {
        use std::arch::aarch64::*;
        let cw = width / 2;
        let mut cx = 0;
        while cx + 16 <= cw {
            let sp = cx * 2 * 3; // byte offset of src px 2*cx
            let t0 = vld3q_u8(top.as_ptr().add(sp));
            let t1 = vld3q_u8(top.as_ptr().add(sp + 48));
            let b0 = vld3q_u8(bot.as_ptr().add(sp));
            let b1 = vld3q_u8(bot.as_ptr().add(sp + 48));
            // 2×2 block average for one channel across the two 16-px chunks → u8x16.
            let avg = |tc0, tc1, bc0, bc1| -> uint8x16_t {
                let blk0 = vaddq_u16(vpaddlq_u8(tc0), vpaddlq_u8(bc0));
                let blk1 = vaddq_u16(vpaddlq_u8(tc1), vpaddlq_u8(bc1));
                vcombine_u8(vrshrn_n_u16::<2>(blk0), vrshrn_n_u16::<2>(blk1))
            };
            let ravg = avg(t0.0, t1.0, b0.0, b1.0);
            let gavg = avg(t0.1, t1.1, b0.1, b1.1);
            let bavg = avg(t0.2, t1.2, b0.2, b1.2);
            let chroma = |cr: i16, cg: i16, cb: i16| -> uint8x16_t {
                let half = |r: uint8x8_t, g: uint8x8_t, b: uint8x8_t| -> uint8x8_t {
                    let r = vreinterpretq_s16_u16(vmovl_u8(r));
                    let g = vreinterpretq_s16_u16(vmovl_u8(g));
                    let b = vreinterpretq_s16_u16(vmovl_u8(b));
                    let mut a = vmulq_s16(r, vdupq_n_s16(cr));
                    a = vmlaq_s16(a, g, vdupq_n_s16(cg));
                    a = vmlaq_s16(a, b, vdupq_n_s16(cb));
                    vqmovun_s16(vaddq_s16(vrshrq_n_s16::<8>(a), vdupq_n_s16(128)))
                };
                vcombine_u8(
                    half(vget_low_u8(ravg), vget_low_u8(gavg), vget_low_u8(bavg)),
                    half(vget_high_u8(ravg), vget_high_u8(gavg), vget_high_u8(bavg)),
                )
            };
            let u16v = chroma(UR as i16, UG as i16, UB as i16);
            let v16v = chroma(VR as i16, VG as i16, VB as i16);
            let uvz = vzipq_u8(u16v, v16v);
            vst1q_u8(uv.as_mut_ptr().add(cx * 2), uvz.0);
            vst1q_u8(uv.as_mut_ptr().add(cx * 2 + 16), uvz.1);
            cx += 16;
        }
        if cx < cw {
            encode_uv_row_scalar(
                &top[cx * 2 * 3..],
                &bot[cx * 2 * 3..],
                &mut uv[cx * 2..],
                (cw - cx) * 2,
            );
        }
    }
}

/// AVX2 chroma-row encode, 16 chroma px/iter. `pshufb` deinterleave; `maddubs`
/// pairwise-sums each row; the two rows add to the 2×2 block sum and `(sum+2)>>2`
/// averages; the signed U/V matrix runs in u16 lanes, then `packus` + `unpack` build
/// the interleaved `[U0,V0,…]` UV plane. Bit-identical to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_uv_row_avx2(top: &[u8], bot: &[u8], uv: &mut [u8], width: usize) {
    use std::arch::x86_64::*;
    let m_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let m_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let m_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
    let half = _mm_set1_epi16(ENC_HALF as i16);
    let off128 = _mm_set1_epi16(128);
    let two = _mm_set1_epi16(2);
    let ones = _mm_set1_epi8(1);

    // pshufb deinterleave of one 16-px (48-byte) chunk → (R, G, B).
    let deint = |p: *const u8| -> (__m128i, __m128i, __m128i) {
        let s0 = _mm_loadu_si128(p as *const __m128i);
        let s1 = _mm_loadu_si128(p.add(16) as *const __m128i);
        let s2 = _mm_loadu_si128(p.add(32) as *const __m128i);
        (
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(s0, m_r0), _mm_shuffle_epi8(s1, m_r1)),
                _mm_shuffle_epi8(s2, m_r2),
            ),
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(s0, m_g0), _mm_shuffle_epi8(s1, m_g1)),
                _mm_shuffle_epi8(s2, m_g2),
            ),
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(s0, m_b0), _mm_shuffle_epi8(s1, m_b1)),
                _mm_shuffle_epi8(s2, m_b2),
            ),
        )
    };
    // 2×2 average of one channel (a chunk from each row) → u16x8 in [0,255].
    let blk_avg = |tc: __m128i, bc: __m128i| -> __m128i {
        let sum = _mm_add_epi16(_mm_maddubs_epi16(tc, ones), _mm_maddubs_epi16(bc, ones));
        _mm_srli_epi16(_mm_add_epi16(sum, two), 2)
    };
    let chroma = |r: __m128i, g: __m128i, b: __m128i, cr: i16, cg: i16, cb: i16| -> __m128i {
        let mut a = _mm_mullo_epi16(r, _mm_set1_epi16(cr));
        a = _mm_add_epi16(a, _mm_mullo_epi16(g, _mm_set1_epi16(cg)));
        a = _mm_add_epi16(a, _mm_mullo_epi16(b, _mm_set1_epi16(cb)));
        a = _mm_srai_epi16(_mm_add_epi16(a, half), 8);
        _mm_add_epi16(a, off128)
    };

    let cw = width / 2;
    let mut cx = 0;
    while cx + 16 <= cw {
        let sp = cx * 2 * 3;
        let (tar, tag, tab) = deint(top.as_ptr().add(sp)); // chroma px cx..cx+7
        let (tbr, tbg, tbb) = deint(top.as_ptr().add(sp + 48)); // cx+8..cx+15
        let (bar, bag, bab) = deint(bot.as_ptr().add(sp));
        let (bbr, bbg, bbb) = deint(bot.as_ptr().add(sp + 48));
        let (ra, ga, ba) = (blk_avg(tar, bar), blk_avg(tag, bag), blk_avg(tab, bab));
        let (rb, gb, bb) = (blk_avg(tbr, bbr), blk_avg(tbg, bbg), blk_avg(tbb, bbb));
        let u = _mm_packus_epi16(
            chroma(ra, ga, ba, UR as i16, UG as i16, UB as i16),
            chroma(rb, gb, bb, UR as i16, UG as i16, UB as i16),
        );
        let v = _mm_packus_epi16(
            chroma(ra, ga, ba, VR as i16, VG as i16, VB as i16),
            chroma(rb, gb, bb, VR as i16, VG as i16, VB as i16),
        );
        _mm_storeu_si128(
            uv.as_mut_ptr().add(cx * 2) as *mut __m128i,
            _mm_unpacklo_epi8(u, v), // [U0,V0,…,U7,V7]
        );
        _mm_storeu_si128(
            uv.as_mut_ptr().add(cx * 2 + 16) as *mut __m128i,
            _mm_unpackhi_epi8(u, v), // [U8,V8,…,U15,V15]
        );
        cx += 16;
    }
    if cx < cw {
        encode_uv_row_scalar(
            &top[cx * 2 * 3..],
            &bot[cx * 2 * 3..],
            &mut uv[cx * 2..],
            (cw - cx) * 2,
        );
    }
}

/// NEON luma-row encode: `vld3q_u8` deinterleaves 16 px → R/G/B, widening MACs apply
/// the Q8 (66,129,25) matrix, then `+16` and a narrowing store. Scalar tail handles
/// the `width % 16` remainder. Bit-identical to the scalar `encode_y`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_y_row_neon(src: &[u8], dst: &mut [u8], width: usize) {
    unsafe {
        use std::arch::aarch64::*;
        let bulk = width & !15;
        let mut x = 0;
        while x < bulk {
            let rgb = vld3q_u8(src.as_ptr().add(x * 3));
            // Widen each channel to two u16x8 halves and accumulate Y in u16 (Q8 coeffs
            // fit: 66+129+25 = 220 < 256, so 220*255 = 56100 < 65535).
            let acc = |r: uint8x8_t, g: uint8x8_t, b: uint8x8_t| -> uint8x8_t {
                let mut a = vmull_u8(r, vdup_n_u8(YR as u8));
                a = vmlal_u8(a, g, vdup_n_u8(YG as u8));
                a = vmlal_u8(a, b, vdup_n_u8(YB as u8));
                // (a + 128) >> 8, then +16, saturating to u8.
                let y = vrshrn_n_u16::<8>(a); // rounded narrow ≈ (a+128)>>8
                vqadd_u8(y, vdup_n_u8(16))
            };
            let lo = acc(vget_low_u8(rgb.0), vget_low_u8(rgb.1), vget_low_u8(rgb.2));
            let hi = acc(
                vget_high_u8(rgb.0),
                vget_high_u8(rgb.1),
                vget_high_u8(rgb.2),
            );
            vst1q_u8(dst.as_mut_ptr().add(x), vcombine_u8(lo, hi));
            x += 16;
        }
        for (x, d) in dst.iter_mut().enumerate().skip(bulk) {
            let s = x * 3;
            *d = encode_y(src[s] as i32, src[s + 1] as i32, src[s + 2] as i32);
        }
    }
}

/// AVX2 luma-row encode (16 px/iter). AVX2 has no 3-way deinterleave, so it extracts
/// R/G/B as u8x16 via 9 `pshufb` and 6 `por` (same trick as the fused-resize loader);
/// the Q8 (66,129,25) matrix is then applied in u16 lanes (`(acc+128)>>8 + 16`), then
/// `packus` to u8. Bit-identical to the scalar `encode_y`. Scalar tail for `% 16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_y_row_avx2(src: &[u8], dst: &mut [u8], width: usize) {
    use std::arch::x86_64::*;
    // R/G/B deinterleave masks for one 16-px (48-byte) chunk across 3 loads.
    let m_r0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_r1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let m_r2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let m_g0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_g1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let m_g2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let m_b0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m_b1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let m_b2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    let c_yr = _mm_set1_epi16(YR as i16);
    let c_yg = _mm_set1_epi16(YG as i16);
    let c_yb = _mm_set1_epi16(YB as i16);
    let half = _mm_set1_epi16(ENC_HALF as i16);
    let off16 = _mm_set1_epi16(16);
    let zero = _mm_setzero_si128();

    // Y for one 8-px u16 half: (66R + 129G + 25B + 128) >> 8 + 16. Coeff sum
    // 220*255 = 56100 < 65535, so u16 mullo/add never overflow.
    let yhalf = |r: __m128i, g: __m128i, b: __m128i| -> __m128i {
        let mut t = _mm_mullo_epi16(r, c_yr);
        t = _mm_add_epi16(t, _mm_mullo_epi16(g, c_yg));
        t = _mm_add_epi16(t, _mm_mullo_epi16(b, c_yb));
        t = _mm_add_epi16(t, half);
        t = _mm_srli_epi16(t, 8);
        _mm_add_epi16(t, off16)
    };

    let bulk = width & !15;
    let mut x = 0;
    while x < bulk {
        let p = src.as_ptr().add(x * 3);
        let s0 = _mm_loadu_si128(p as *const __m128i);
        let s1 = _mm_loadu_si128(p.add(16) as *const __m128i);
        let s2 = _mm_loadu_si128(p.add(32) as *const __m128i);
        let r = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(s0, m_r0), _mm_shuffle_epi8(s1, m_r1)),
            _mm_shuffle_epi8(s2, m_r2),
        );
        let g = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(s0, m_g0), _mm_shuffle_epi8(s1, m_g1)),
            _mm_shuffle_epi8(s2, m_g2),
        );
        let b = _mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(s0, m_b0), _mm_shuffle_epi8(s1, m_b1)),
            _mm_shuffle_epi8(s2, m_b2),
        );
        // Widen low/high 8 px to u16 and apply the matrix; packus back to u8x16.
        let lo = yhalf(
            _mm_unpacklo_epi8(r, zero),
            _mm_unpacklo_epi8(g, zero),
            _mm_unpacklo_epi8(b, zero),
        );
        let hi = yhalf(
            _mm_unpackhi_epi8(r, zero),
            _mm_unpackhi_epi8(g, zero),
            _mm_unpackhi_epi8(b, zero),
        );
        _mm_storeu_si128(
            dst.as_mut_ptr().add(x) as *mut __m128i,
            _mm_packus_epi16(lo, hi),
        );
        x += 16;
    }
    for (x, d) in dst.iter_mut().enumerate().skip(bulk) {
        let s = x * 3;
        *d = encode_y(src[s] as i32, src[s + 1] as i32, src[s + 2] as i32);
    }
}

// ===== Tests ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_u8(n: usize) -> Vec<u8> {
        (0..n).map(|v| (v * 7 + 3) as u8).collect()
    }

    // ----- Family C: video encode -----

    /// The NEON luma-row encode must be bit-identical to the scalar `encode_y`.
    #[test]
    fn encode_y_row_matches_scalar() {
        let width = 35; // 16-px body twice + 3-px tail
        let src = ramp_u8(width * 3);
        let mut neon = vec![0u8; width];
        let mut scalar = vec![0u8; width];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            super::encode_y_row_neon(&src, &mut neon, width);
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (x, out) in neon.iter_mut().enumerate().take(width) {
            let s = x * 3;
            *out = encode_y(src[s] as i32, src[s + 1] as i32, src[s + 2] as i32);
        }
        for (x, out) in scalar.iter_mut().enumerate().take(width) {
            let s = x * 3;
            *out = encode_y(src[s] as i32, src[s + 1] as i32, src[s + 2] as i32);
        }
        assert_eq!(neon, scalar);

        // On x86 with AVX2, the AVX2 luma path must also match the scalar oracle.
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_avx2 {
            let mut avx = vec![0u8; width];
            unsafe { super::encode_y_row_avx2(&src, &mut avx, width) };
            assert_eq!(avx, scalar);
        }
    }

    /// A constant-color image survives encode→decode exactly (no subsampling error
    /// when every pixel is identical), for both YUYV and NV12.
    #[test]
    fn encode_decode_constant_is_exact() {
        let (w, h) = (8, 6);
        for &(r, g, b) in &[(200u8, 50, 25), (0, 0, 0), (255, 255, 255), (17, 200, 99)] {
            let rgb: Vec<u8> = (0..w * h).flat_map(|_| [r, g, b]).collect();

            // YUYV round-trip.
            let mut yuyv = vec![0u8; w * h * 2];
            yuyv_from_rgb(&rgb, &mut yuyv, w, h);
            let mut back = vec![0u8; w * h * 3];
            rgb_from_packed422(&yuyv, &mut back, w, h, Packed422::Yuyv);
            for i in 0..w * h {
                assert!((back[i * 3] as i32 - r as i32).abs() <= 2, "yuyv R");
                assert!((back[i * 3 + 1] as i32 - g as i32).abs() <= 2, "yuyv G");
                assert!((back[i * 3 + 2] as i32 - b as i32).abs() <= 2, "yuyv B");
            }

            // NV12 round-trip.
            let mut nv12 = vec![0u8; w * h * 3 / 2];
            let (y, uv) = nv12.split_at_mut(w * h);
            nv12_from_rgb(&rgb, y, uv, w, h);
            let mut back2 = vec![0u8; w * h * 3];
            rgb_from_planar420(y, uv, &[], &mut back2, w, h, Planar420::Nv12);
            for i in 0..w * h {
                assert!((back2[i * 3] as i32 - r as i32).abs() <= 2, "nv12 R");
                assert!((back2[i * 3 + 1] as i32 - g as i32).abs() <= 2, "nv12 G");
                assert!((back2[i * 3 + 2] as i32 - b as i32).abs() <= 2, "nv12 B");
            }
        }
    }

    /// YUYV byte layout: `Y0 U Y1 V` per pair, luma per-pixel, chroma the pair average.
    #[test]
    fn yuyv_layout_and_values() {
        // two pixels: (255,0,0) and (0,0,255)
        let rgb = [255u8, 0, 0, 0, 0, 255];
        let mut out = [0u8; 4];
        yuyv_from_rgb(&rgb, &mut out, 2, 1);
        let y0 = encode_y(255, 0, 0);
        let y1 = encode_y(0, 0, 255);
        // `(r1 + r2 + 1) >> 1` average formula; the 0s are explicit channel values.
        #[allow(clippy::identity_op)]
        let (u, v) = encode_uv((255 + 0 + 1) >> 1, 0, (0 + 255 + 1) >> 1);
        assert_eq!(out, [y0, u, y1, v]);
    }

    /// The SIMD chroma row (NEON/AVX2) must be bit-identical to the scalar oracle.
    #[test]
    fn nv12_uv_row_simd_matches_scalar() {
        let width = 70; // 16-chroma-px bulk twice (64 src px) + 6-px (3-chroma) tail
        let top = ramp_u8(width * 3);
        let bot: Vec<u8> = (0..width * 3).map(|v| (v * 5 + 11) as u8).collect();
        let mut simd = vec![0u8; width]; // uv row = width bytes
        let mut scalar = vec![0u8; width];
        encode_uv_row(&top, &bot, &mut simd, width);
        encode_uv_row_scalar(&top, &bot, &mut scalar, width);
        assert_eq!(simd, scalar);
    }

    /// The SIMD YUYV row (NEON on aarch64, AVX2 on x86) must be bit-identical to the
    /// scalar oracle, across the vector bulk + the scalar tail.
    #[test]
    fn yuyv_row_simd_matches_scalar() {
        let width = 38; // two 16-px bulk iters (32) + 6-px (3-pair) tail
        let src = ramp_u8(width * 3);
        let mut simd = vec![0u8; width * 2];
        let mut scalar = vec![0u8; width * 2];
        yuyv_from_rgb_row(&src, &mut simd, width);
        yuyv_from_rgb_row_scalar(&src, &mut scalar, width);
        assert_eq!(simd, scalar);
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
        // cv2's Y'UV forward/inverse constants are independently rounded
        // (0.492/0.877 vs 2.032/1.140), so the round-trip is only ~1e-4
        // faithful — matching cv2's own round-trip error.
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!((a - b).abs() <= 5e-4, "{a} != {b}");
        }
    }

    // ----- Family B: packed 4:2:2 -----
    #[test]
    fn packed422_matches_scalar() {
        for fmt in [Packed422::Yuyv, Packed422::Uyvy, Packed422::Yvyu] {
            // 70 px = 35 groups/row -> 2×16-group NEON bodies (32 grp) + 3-group tail.
            let (w, h) = (70, 3);
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
    fn planar420_neon_bulk_matches_reference() {
        // w=64 → cw=32 exercises the 16-chroma NEON bulk loop twice; w=70 → cw=35
        // adds a 3-sample scalar tail. Both cover all four layouts vs an independent ref.
        for (w, h) in [(64usize, 6usize), (70usize, 4usize)] {
            let cw = w / 2;
            let ch = h / 2;
            let y: Vec<u8> = (0..w * h).map(|i| ((i * 7 + 16) % 240) as u8).collect();
            for fmt in [
                Planar420::Nv12,
                Planar420::Nv21,
                Planar420::I420,
                Planar420::Yv12,
            ] {
                let (c0, c1): (Vec<u8>, Vec<u8>) = match fmt {
                    Planar420::Nv12 | Planar420::Nv21 => (
                        (0..cw * ch * 2)
                            .map(|i| ((i * 5 + 90) % 250) as u8)
                            .collect(),
                        vec![],
                    ),
                    _ => (
                        (0..cw * ch).map(|i| ((i * 5 + 90) % 250) as u8).collect(),
                        (0..cw * ch).map(|i| ((i * 3 + 40) % 250) as u8).collect(),
                    ),
                };
                let mut dst = vec![0u8; w * h * 3];
                rgb_from_planar420(&y, &c0, &c1, &mut dst, w, h, fmt);
                let mut exp = vec![0u8; w * h * 3];
                for row in 0..h {
                    for col in 0..w {
                        let (u, v) = chroma_at(&c0, &c1, cw, row / 2, col / 2, fmt);
                        let yy = (y[row * w + col] as i32 - 16).max(0) * CY;
                        let (r, g, b) = decode_px(yy, u, v);
                        let d = (row * w + col) * 3;
                        exp[d] = r;
                        exp[d + 1] = g;
                        exp[d + 2] = b;
                    }
                }
                assert_eq!(dst, exp, "planar420 NEON mismatch (w={w}, h={h})");
            }
        }
    }

    // Ignored timing helper (not a correctness gate): `cargo test ... -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_throughput() {
        use std::time::Instant;
        let (w, h) = (1920usize, 1080usize);
        let npix = w * h;
        let rgb = ramp_u8(npix * 3);
        let yuyv = ramp_u8(npix * 2);
        let mut out3 = vec![0u8; npix * 3];
        let iters = 50;

        let t = Instant::now();
        for _ in 0..iters {
            ycc_from_rgb_u8(&rgb, &mut out3, npix, ChromaOrder::YCrCb);
        }
        let neon = t.elapsed().as_secs_f64() / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            ycc_from_rgb_u8_scalar(&rgb, &mut out3, npix, ChromaOrder::YCrCb);
        }
        let scalar = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "ycbcr_from_rgb u8 1080p: neon {:.3} ms, scalar {:.3} ms, speedup {:.2}x",
            neon * 1e3,
            scalar * 1e3,
            scalar / neon
        );

        let t = Instant::now();
        for _ in 0..iters {
            rgb_from_packed422(&yuyv, &mut out3, w, h, Packed422::Yuyv);
        }
        let neon = t.elapsed().as_secs_f64() / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            for row in 0..h {
                rgb_from_packed422_row_scalar(
                    &yuyv[row * w * 2..row * w * 2 + w * 2],
                    &mut out3[row * w * 3..row * w * 3 + w * 3],
                    w,
                    Packed422::Yuyv,
                );
            }
        }
        let scalar = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "yuyv->rgb 1080p: neon {:.3} ms, scalar {:.3} ms, speedup {:.2}x",
            neon * 1e3,
            scalar * 1e3,
            scalar / neon
        );
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
