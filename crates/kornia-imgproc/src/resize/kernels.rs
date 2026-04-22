//! Per-architecture row-level kernels for resize.
//!
//! This module centralizes the arch-specific fast paths used by the resize
//! algorithms (`pyramid::pyrdown_2x_rgb_u8`, `pyramid::pyrup_2x_rgb_u8`,
//! `separable::resize_separable_u8`, `bilinear::resize_bilinear_u8_nch`).
//! Each public kernel is a dispatcher that selects the best implementation at
//! compile time via `cfg`; algorithm code never carries `#[cfg(target_arch)]`
//! pairs of its own.
//!
//! # Kernels exposed
//!
//! - [`pyrdown_row_rgb_u8`]     — 2× horizontal-pair-and-vertical-pair average
//!   of two RGB u8 source rows (used by pyrdown). Scalar + aarch64 NEON.
//! - [`hinterp_row_rgb_u8`]     — 2× horizontal bilinear upscale of one RGB u8
//!   row (used by pyrup). Scalar + aarch64 NEON.
//! - [`blend_75_25_row`]        — byte-wise `0.75·a + 0.25·b` (rounded) via
//!   the `vrhaddq_u8(a, vrhaddq_u8(a, b))` identity. Scalar + aarch64 NEON.
//! - [`horizontal_row_rgb_u8`]  — Q14 separable horizontal pass for RGB.
//!   Generic over channels via the `const C: usize` parameter. Scalar +
//!   aarch64 NEON (C=3 only).
//! - [`horizontal_rows_rgb_u8_x4`] — 4-row NEON variant of the above that
//!   shares one coefficient LUT fetch across four src rows. aarch64 only.
//! - [`vertical_row`]           — Q14 separable vertical pass. Scalar +
//!   aarch64 NEON.
//!
//! # Adding a new backend
//!
//! To add (e.g.) an AVX2 / SSE4.1 / SVE / WASM-SIMD path:
//!
//! 1. Add a `#[cfg(target_arch = "…")]` block below the existing aarch64
//!    block in the relevant dispatch function.
//! 2. Implement the kernel as an `unsafe fn` with the right
//!    `#[target_feature]` attribute.
//! 3. Use the `_scalar` fallback as the reference for both semantics and
//!    numerical behavior. Add correctness tests that compare lane-equivalent
//!    output against the scalar kernel.

// ──────────────────────────────────────────────────────────────────────────
// pyrdown: 2× box-average of two RGB u8 rows → one RGB u8 row.
// ──────────────────────────────────────────────────────────────────────────

/// 2× box-averaging downsample row kernel for RGB u8.
///
/// Given two consecutive source rows `r0`/`r1` of width `2·dst_w` pixels, write
/// `dst_w` destination pixels where each output channel is the rounded mean of
/// the 2×2 source block of that channel.
#[inline(always)]
pub(super) fn pyrdown_row_rgb_u8(r0: &[u8], r1: &[u8], dst: &mut [u8], dst_w: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        pyrdown_row_rgb_u8_neon(r0, r1, dst, dst_w);
    }
    #[cfg(not(target_arch = "aarch64"))]
    pyrdown_row_rgb_u8_scalar(r0, r1, dst, dst_w);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn pyrdown_row_rgb_u8_scalar(r0: &[u8], r1: &[u8], dst: &mut [u8], dst_w: usize) {
    for x in 0..dst_w {
        for ch in 0..3 {
            let sum = r0[(2 * x) * 3 + ch] as u16
                + r0[(2 * x + 1) * 3 + ch] as u16
                + r1[(2 * x) * 3 + ch] as u16
                + r1[(2 * x + 1) * 3 + ch] as u16;
            dst[x * 3 + ch] = ((sum + 2) >> 2) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pyrdown_row_rgb_u8_neon(r0: &[u8], r1: &[u8], dst: &mut [u8], dst_w: usize) {
    use std::arch::aarch64::*;
    // Process 16 output pixels (= 48 bytes dst, 96 bytes per src row) per iter.
    let bulk = dst_w & !15;
    let mut x = 0usize;
    while x < bulk {
        let s0 = vld3q_u8(r0.as_ptr().add(x * 2 * 3));
        let s1 = vld3q_u8(r0.as_ptr().add(x * 2 * 3 + 48));
        let s2 = vld3q_u8(r1.as_ptr().add(x * 2 * 3));
        let s3 = vld3q_u8(r1.as_ptr().add(x * 2 * 3 + 48));
        // Each of s*.N is 16 u8 (two adjacent src pixels per output).
        // Sum adjacent pairs within a row via vpaddlq_u8 → 8 u16 per half,
        // then sum the two rows and rshift-narrow by 2 for /4 with rounding.
        let mut out = uint8x16x3_t(vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));
        let mut k = 0;
        while k < 3 {
            let a = match k {
                0 => s0.0,
                1 => s0.1,
                _ => s0.2,
            };
            let b = match k {
                0 => s1.0,
                1 => s1.1,
                _ => s1.2,
            };
            let c = match k {
                0 => s2.0,
                1 => s2.1,
                _ => s2.2,
            };
            let d = match k {
                0 => s3.0,
                1 => s3.1,
                _ => s3.2,
            };
            let ab_lo = vpaddlq_u8(a);
            let ab_hi = vpaddlq_u8(b);
            let cd_lo = vpaddlq_u8(c);
            let cd_hi = vpaddlq_u8(d);
            let sum_lo = vaddq_u16(ab_lo, cd_lo);
            let sum_hi = vaddq_u16(ab_hi, cd_hi);
            let o_lo = vrshrn_n_u16(sum_lo, 2);
            let o_hi = vrshrn_n_u16(sum_hi, 2);
            let o = vcombine_u8(o_lo, o_hi);
            match k {
                0 => out.0 = o,
                1 => out.1 = o,
                _ => out.2 = o,
            }
            k += 1;
        }
        vst3q_u8(dst.as_mut_ptr().add(x * 3), out);
        x += 16;
    }
    // Scalar tail.
    while x < dst_w {
        for ch in 0..3 {
            let sum = r0[(2 * x) * 3 + ch] as u16
                + r0[(2 * x + 1) * 3 + ch] as u16
                + r1[(2 * x) * 3 + ch] as u16
                + r1[(2 * x + 1) * 3 + ch] as u16;
            dst[x * 3 + ch] = ((sum + 2) >> 2) as u8;
        }
        x += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// pyrup: horizontal 2× bilinear upscale of one RGB u8 row.
// ──────────────────────────────────────────────────────────────────────────

/// Horizontally upscale one RGB u8 row by 2× with bilinear weights.
///
/// Emits `2·src_w` destination pixels: two edge pixels (clamped copies of
/// `src[0]` and `src[src_w-1]`) plus `(src_w-1)` interior `{0.75, 0.25}` pairs.
#[inline(always)]
pub(super) fn hinterp_row_rgb_u8(src: &[u8], dst: &mut [u8], src_w: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        hinterp_row_rgb_u8_neon(src, dst, src_w);
    }
    #[cfg(not(target_arch = "aarch64"))]
    hinterp_row_rgb_u8_scalar(src, dst, src_w);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn hinterp_row_rgb_u8_scalar(src: &[u8], dst: &mut [u8], src_w: usize) {
    dst[..3].copy_from_slice(&src[..3]);
    for j in 0..src_w - 1 {
        for ch in 0..3 {
            let a = src[j * 3 + ch] as u16;
            let b = src[(j + 1) * 3 + ch] as u16;
            let avg = (a + b + 1) >> 1;
            dst[(2 * j + 1) * 3 + ch] = ((a + avg + 1) >> 1) as u8;
            dst[(2 * j + 2) * 3 + ch] = ((b + avg + 1) >> 1) as u8;
        }
    }
    let tail = (2 * src_w - 1) * 3;
    dst[tail..tail + 3].copy_from_slice(&src[(src_w - 1) * 3..(src_w - 1) * 3 + 3]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hinterp_row_rgb_u8_neon(src: &[u8], dst: &mut [u8], src_w: usize) {
    use std::arch::aarch64::*;
    let s = src.as_ptr();
    let d = dst.as_mut_ptr();

    // Edge: dst[0] ← src[0] (clamped f=0 for dst_x=0).
    for ch in 0..3 {
        *d.add(ch) = *s.add(ch);
    }

    // Bulk: 16 src pixel pairs per iter → 32 dst pixels via vld3q / vst3q.
    // Each iter reads pixels [J..J+15] and [J+1..J+16], so needs J+16 ≤ src_w-1.
    let mut j = 0usize;
    while j + 17 <= src_w {
        let s_a = vld3q_u8(s.add(j * 3));
        let s_b = vld3q_u8(s.add((j + 1) * 3));

        // For each channel: compute both 0.75a+0.25b and 0.25a+0.75b via the
        // double-rhadd identity, then interleave so adjacent dst pixels
        // alternate (hi, lo) — matching the dst[2J+1], dst[2J+2] pattern.
        let blend = |ac: uint8x16_t, bc: uint8x16_t| -> (uint8x16_t, uint8x16_t) {
            let avg = vrhaddq_u8(ac, bc);
            let hi = vrhaddq_u8(ac, avg); // 0.75·ac + 0.25·bc
            let lo = vrhaddq_u8(bc, avg); // 0.25·ac + 0.75·bc
            (vzip1q_u8(hi, lo), vzip2q_u8(hi, lo))
        };

        let (r_lo, r_hi) = blend(s_a.0, s_b.0);
        let (g_lo, g_hi) = blend(s_a.1, s_b.1);
        let (b_lo, b_hi) = blend(s_a.2, s_b.2);

        let out_low = uint8x16x3_t(r_lo, g_lo, b_lo);
        let out_high = uint8x16x3_t(r_hi, g_hi, b_hi);

        vst3q_u8(d.add((2 * j + 1) * 3), out_low);
        vst3q_u8(d.add((2 * j + 17) * 3), out_high);

        j += 16;
    }

    // Scalar tail for the remaining (src_w - 1 - j) interior pairs.
    while j + 1 < src_w {
        let ja = j * 3;
        let jb = (j + 1) * 3;
        for ch in 0..3 {
            let a = *s.add(ja + ch) as u16;
            let b = *s.add(jb + ch) as u16;
            let avg = (a + b + 1) >> 1;
            *d.add((2 * j + 1) * 3 + ch) = ((a + avg + 1) >> 1) as u8;
            *d.add((2 * j + 2) * 3 + ch) = ((b + avg + 1) >> 1) as u8;
        }
        j += 1;
    }

    // Edge: dst[2·src_w-1] ← src[src_w-1] (clamped f=1 for last dst_x).
    for ch in 0..3 {
        *d.add((2 * src_w - 1) * 3 + ch) = *s.add((src_w - 1) * 3 + ch);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// pyrup: vertical 75/25 blend of two pre-hinterp'd RGB u8 rows.
// ──────────────────────────────────────────────────────────────────────────

/// Byte-wise `dst[i] = round(0.75·a[i] + 0.25·b[i])`.
///
/// Implemented via the `vrhaddq_u8(a, vrhaddq_u8(a, b))` identity on aarch64.
/// `a` and `b` must have equal length ≥ `dst.len()`.
#[inline(always)]
pub(super) fn blend_75_25_row(a: &[u8], b: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        blend_75_25_row_neon(a, b, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    blend_75_25_row_scalar(a, b, dst);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn blend_75_25_row_scalar(a: &[u8], b: &[u8], dst: &mut [u8]) {
    for i in 0..dst.len() {
        let av = a[i] as u16;
        let bv = b[i] as u16;
        let avg = (av + bv + 1) >> 1;
        dst[i] = ((av + avg + 1) >> 1) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blend_75_25_row_neon(a: &[u8], b: &[u8], dst: &mut [u8]) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let bulk = n & !31; // unroll 2× for Cortex-A78AE dual-issue load pair
    let mut i = 0;
    while i < bulk {
        let va0 = vld1q_u8(a.as_ptr().add(i));
        let vb0 = vld1q_u8(b.as_ptr().add(i));
        let va1 = vld1q_u8(a.as_ptr().add(i + 16));
        let vb1 = vld1q_u8(b.as_ptr().add(i + 16));
        let avg0 = vrhaddq_u8(va0, vb0);
        let avg1 = vrhaddq_u8(va1, vb1);
        let out0 = vrhaddq_u8(va0, avg0);
        let out1 = vrhaddq_u8(va1, avg1);
        vst1q_u8(dst.as_mut_ptr().add(i), out0);
        vst1q_u8(dst.as_mut_ptr().add(i + 16), out1);
        i += 32;
    }
    while i + 16 <= n {
        let va = vld1q_u8(a.as_ptr().add(i));
        let vb = vld1q_u8(b.as_ptr().add(i));
        let avg = vrhaddq_u8(va, vb);
        let out = vrhaddq_u8(va, avg);
        vst1q_u8(dst.as_mut_ptr().add(i), out);
        i += 16;
    }
    while i < n {
        let av = a[i] as u16;
        let bv = b[i] as u16;
        let avg = (av + bv + 1) >> 1;
        dst[i] = ((av + avg + 1) >> 1) as u8;
        i += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Q14 separable: horizontal row (one row, generic C).
// ──────────────────────────────────────────────────────────────────────────

/// Horizontal Q14 separable pass for one source row.
///
/// For `C == 3` on aarch64 this takes the NEON 4-wide-unroll path that hides
/// the ~4-cycle `vmlal_n_s16` latency with four independent accumulators.
/// All other configurations fall through to the scalar reference.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn horizontal_row_rgb_u8<const C: usize>(
    src_row: &[u8],
    out: &mut [i16],
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if C == 3 {
            unsafe {
                horizontal_row_c3_neon(src_row, out, dst_w, kx, xsrc, xw, last_sx_safe, round1);
            }
            return;
        }
    }
    horizontal_row_scalar::<C>(src_row, out, dst_w, kx, xsrc, xw, last_sx_safe, round1);
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn horizontal_row_scalar<const C: usize>(
    src_row: &[u8],
    out: &mut [i16],
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    _last_sx_safe: usize,
    round1: i32,
) {
    const Q: i32 = 14;
    for x in 0..dst_w {
        let ibase = x * kx;
        for ch in 0..C {
            let mut acc: i32 = 0;
            for t in 0..kx {
                let sx = xsrc[ibase + t] as usize;
                acc += src_row[sx * C + ch] as i32 * xw[ibase + t] as i32;
            }
            let v = (acc + round1) >> Q;
            out[x * C + ch] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_px(p: *const u8) -> std::arch::aarch64::int16x4_t {
    use std::arch::aarch64::*;
    vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(p))))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_px_edge(p: *const u8) -> std::arch::aarch64::int16x4_t {
    use std::arch::aarch64::*;
    let t: [i16; 4] = [*p as i16, *p.add(1) as i16, *p.add(2) as i16, 0];
    vld1_s16(t.as_ptr())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn horizontal_row_c3_neon(
    src_row: &[u8],
    out: &mut [i16],
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    use std::arch::aarch64::*;
    let round_v = vdupq_n_s32(round1);
    // 4-wide unroll: four independent accumulators hide the ~4-cycle
    // vmlal latency on A-class cores, nearly quadrupling MAC throughput
    // when kx is large (lanczos, bicubic at strong downscales).
    let mut x = 0;
    while x + 4 <= dst_w {
        let b0 = x * kx;
        let b1 = (x + 1) * kx;
        let b2 = (x + 2) * kx;
        let b3 = (x + 3) * kx;
        let mut a0 = round_v;
        let mut a1 = round_v;
        let mut a2 = round_v;
        let mut a3 = round_v;
        for t in 0..kx {
            let sx0 = *xsrc.get_unchecked(b0 + t) as usize;
            let sx1 = *xsrc.get_unchecked(b1 + t) as usize;
            let sx2 = *xsrc.get_unchecked(b2 + t) as usize;
            let sx3 = *xsrc.get_unchecked(b3 + t) as usize;
            let w0 = *xw.get_unchecked(b0 + t);
            let w1 = *xw.get_unchecked(b1 + t);
            let w2 = *xw.get_unchecked(b2 + t);
            let w3 = *xw.get_unchecked(b3 + t);
            let base = src_row.as_ptr();
            let p0 = if sx0 < last_sx_safe {
                load_px(base.add(sx0 * 3))
            } else {
                load_px_edge(base.add(sx0 * 3))
            };
            let p1 = if sx1 < last_sx_safe {
                load_px(base.add(sx1 * 3))
            } else {
                load_px_edge(base.add(sx1 * 3))
            };
            let p2 = if sx2 < last_sx_safe {
                load_px(base.add(sx2 * 3))
            } else {
                load_px_edge(base.add(sx2 * 3))
            };
            let p3 = if sx3 < last_sx_safe {
                load_px(base.add(sx3 * 3))
            } else {
                load_px_edge(base.add(sx3 * 3))
            };
            a0 = vmlal_n_s16(a0, p0, w0);
            a1 = vmlal_n_s16(a1, p1, w1);
            a2 = vmlal_n_s16(a2, p2, w2);
            a3 = vmlal_n_s16(a3, p3, w3);
        }
        let s0 = vqmovn_s32(vshrq_n_s32::<14>(a0));
        let s1 = vqmovn_s32(vshrq_n_s32::<14>(a1));
        let s2 = vqmovn_s32(vshrq_n_s32::<14>(a2));
        let s3 = vqmovn_s32(vshrq_n_s32::<14>(a3));
        let o = out.as_mut_ptr().add(x * 3);
        *o = vget_lane_s16::<0>(s0);
        *o.add(1) = vget_lane_s16::<1>(s0);
        *o.add(2) = vget_lane_s16::<2>(s0);
        *o.add(3) = vget_lane_s16::<0>(s1);
        *o.add(4) = vget_lane_s16::<1>(s1);
        *o.add(5) = vget_lane_s16::<2>(s1);
        *o.add(6) = vget_lane_s16::<0>(s2);
        *o.add(7) = vget_lane_s16::<1>(s2);
        *o.add(8) = vget_lane_s16::<2>(s2);
        *o.add(9) = vget_lane_s16::<0>(s3);
        *o.add(10) = vget_lane_s16::<1>(s3);
        *o.add(11) = vget_lane_s16::<2>(s3);
        x += 4;
    }
    while x < dst_w {
        let ibase = x * kx;
        let mut acc = round_v;
        for t in 0..kx {
            let sx = *xsrc.get_unchecked(ibase + t) as usize;
            let w = *xw.get_unchecked(ibase + t);
            let p = src_row.as_ptr().add(sx * 3);
            let px = if sx < last_sx_safe {
                load_px(p)
            } else {
                load_px_edge(p)
            };
            acc = vmlal_n_s16(acc, px, w);
        }
        let sat = vqmovn_s32(vshrq_n_s32::<14>(acc));
        let o = out.as_mut_ptr().add(x * 3);
        *o = vget_lane_s16::<0>(sat);
        *o.add(1) = vget_lane_s16::<1>(sat);
        *o.add(2) = vget_lane_s16::<2>(sat);
        x += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Q14 separable: horizontal over 4 rows (C=3) — NEON only.
// ──────────────────────────────────────────────────────────────────────────

/// Horizontal Q14 separable pass over 4 source rows simultaneously (C=3 only).
///
/// Shares one `xsrc`/`xw` load across all 4 rows so the coefficient LUT is
/// fetched once per tap rather than four times. Available on aarch64 only;
/// callers gate dispatch with `#[cfg(target_arch = "aarch64")]`.
///
/// Output storage uses `vst1_s16` (8-byte vector store) for interior pixels
/// and scalar lane extraction for the final pixel. Interior overflow (the 4th
/// lane's garbage byte) lands on the next pixel's `R` slot and is overwritten
/// by the next iteration's `lane 0`, so it's benign.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn horizontal_rows_rgb_u8_x4(
    src_rows: [&[u8]; 4],
    outs: [&mut [i16]; 4],
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    use std::arch::aarch64::*;
    let round_v = vdupq_n_s32(round1);
    let [s0p, s1p, s2p, s3p] = [
        src_rows[0].as_ptr(),
        src_rows[1].as_ptr(),
        src_rows[2].as_ptr(),
        src_rows[3].as_ptr(),
    ];
    let [mut o0, mut o1, mut o2, mut o3] = {
        let [a, b, c, d] = outs;
        [
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            c.as_mut_ptr(),
            d.as_mut_ptr(),
        ]
    };
    let last_x = dst_w.saturating_sub(1);
    for x in 0..dst_w {
        let ibase = x * kx;
        let mut a0 = round_v;
        let mut a1 = round_v;
        let mut a2 = round_v;
        let mut a3 = round_v;
        for t in 0..kx {
            let sx = *xsrc.get_unchecked(ibase + t) as usize;
            let w = *xw.get_unchecked(ibase + t);
            let safe = sx < last_sx_safe;
            let off = sx * 3;
            let p0 = if safe {
                load_px(s0p.add(off))
            } else {
                load_px_edge(s0p.add(off))
            };
            let p1 = if safe {
                load_px(s1p.add(off))
            } else {
                load_px_edge(s1p.add(off))
            };
            let p2 = if safe {
                load_px(s2p.add(off))
            } else {
                load_px_edge(s2p.add(off))
            };
            let p3 = if safe {
                load_px(s3p.add(off))
            } else {
                load_px_edge(s3p.add(off))
            };
            a0 = vmlal_n_s16(a0, p0, w);
            a1 = vmlal_n_s16(a1, p1, w);
            a2 = vmlal_n_s16(a2, p2, w);
            a3 = vmlal_n_s16(a3, p3, w);
        }
        let s0 = vqmovn_s32(vshrq_n_s32::<14>(a0));
        let s1 = vqmovn_s32(vshrq_n_s32::<14>(a1));
        let s2 = vqmovn_s32(vshrq_n_s32::<14>(a2));
        let s3 = vqmovn_s32(vshrq_n_s32::<14>(a3));
        if x < last_x {
            // Interior: store 8 bytes (4 × i16). Lane 3's garbage lands on
            // the next pixel's R slot and is overwritten by its lane 0 on
            // the next iteration, so it's benign.
            vst1_s16(o0, s0);
            vst1_s16(o1, s1);
            vst1_s16(o2, s2);
            vst1_s16(o3, s3);
        } else {
            // Last pixel: scalar stores (no room for overflow).
            *o0 = vget_lane_s16::<0>(s0);
            *o0.add(1) = vget_lane_s16::<1>(s0);
            *o0.add(2) = vget_lane_s16::<2>(s0);
            *o1 = vget_lane_s16::<0>(s1);
            *o1.add(1) = vget_lane_s16::<1>(s1);
            *o1.add(2) = vget_lane_s16::<2>(s1);
            *o2 = vget_lane_s16::<0>(s2);
            *o2.add(1) = vget_lane_s16::<1>(s2);
            *o2.add(2) = vget_lane_s16::<2>(s2);
            *o3 = vget_lane_s16::<0>(s3);
            *o3.add(1) = vget_lane_s16::<1>(s3);
            *o3.add(2) = vget_lane_s16::<2>(s3);
        }
        o0 = o0.add(3);
        o1 = o1.add(3);
        o2 = o2.add(3);
        o3 = o3.add(3);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Q14 separable: vertical row.
// ──────────────────────────────────────────────────────────────────────────

/// Q14 separable vertical pass for one destination row.
///
/// Consumes `rows.len()` i16 source rows (from the horizontal pass) weighted by
/// `w`, produces `n` u8 destination bytes.
#[inline(always)]
pub(super) fn vertical_row(rows: &[&[i16]], w: &[i16], dst_row: &mut [u8], n: usize, round2: i32) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is architectural on aarch64; kernel reads `rows`/`w` and writes `dst_row`
    // within the bounds enforced by `n` and the calling `resize_separable_u8`.
    unsafe {
        vertical_row_neon(rows, w, dst_row, n, round2);
    }
    #[cfg(not(target_arch = "aarch64"))]
    vertical_row_scalar(rows, w, dst_row, n, round2);
}

#[inline]
#[allow(dead_code)]
fn vertical_row_scalar(rows: &[&[i16]], w: &[i16], dst_row: &mut [u8], n: usize, round2: i32) {
    let ky = rows.len();
    for i in 0..n {
        let mut acc: i32 = 0;
        for k in 0..ky {
            acc += rows[k][i] as i32 * w[k] as i32;
        }
        dst_row[i] = (((acc + round2) >> 14).clamp(0, 255)) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vertical_row_neon(rows: &[&[i16]], w: &[i16], dst_row: &mut [u8], n: usize, round2: i32) {
    use std::arch::aarch64::*;
    let ky = rows.len();
    let round_v = vdupq_n_s32(round2);
    let zero = vdupq_n_s32(0);
    let mut i = 0usize;
    // 16-lane path: two parallel 8-lane halves (L=left, R=right), each with
    // its own 4-chain rolling accumulator set. Doubles MAC throughput per
    // outer-iter and hides loop overhead; register file is 32 v-regs so
    // 16 accumulators + scratch fits comfortably.
    while i + 16 <= n {
        let mut a0ll = round_v;
        let mut a1ll = zero;
        let mut a2ll = zero;
        let mut a3ll = zero;
        let mut a0lh = round_v;
        let mut a1lh = zero;
        let mut a2lh = zero;
        let mut a3lh = zero;
        let mut a0rl = round_v;
        let mut a1rl = zero;
        let mut a2rl = zero;
        let mut a3rl = zero;
        let mut a0rh = round_v;
        let mut a1rh = zero;
        let mut a2rh = zero;
        let mut a3rh = zero;

        let mut k = 0;
        while k + 4 <= ky {
            let p0 = rows.get_unchecked(k).as_ptr().add(i);
            let p1 = rows.get_unchecked(k + 1).as_ptr().add(i);
            let p2 = rows.get_unchecked(k + 2).as_ptr().add(i);
            let p3 = rows.get_unchecked(k + 3).as_ptr().add(i);
            let v0l = vld1q_s16(p0);
            let v1l = vld1q_s16(p1);
            let v2l = vld1q_s16(p2);
            let v3l = vld1q_s16(p3);
            let v0r = vld1q_s16(p0.add(8));
            let v1r = vld1q_s16(p1.add(8));
            let v2r = vld1q_s16(p2.add(8));
            let v3r = vld1q_s16(p3.add(8));
            let w0 = vdup_n_s16(*w.get_unchecked(k));
            let w1 = vdup_n_s16(*w.get_unchecked(k + 1));
            let w2 = vdup_n_s16(*w.get_unchecked(k + 2));
            let w3 = vdup_n_s16(*w.get_unchecked(k + 3));
            a0ll = vmlal_s16(a0ll, vget_low_s16(v0l), w0);
            a1ll = vmlal_s16(a1ll, vget_low_s16(v1l), w1);
            a2ll = vmlal_s16(a2ll, vget_low_s16(v2l), w2);
            a3ll = vmlal_s16(a3ll, vget_low_s16(v3l), w3);
            a0lh = vmlal_s16(a0lh, vget_high_s16(v0l), w0);
            a1lh = vmlal_s16(a1lh, vget_high_s16(v1l), w1);
            a2lh = vmlal_s16(a2lh, vget_high_s16(v2l), w2);
            a3lh = vmlal_s16(a3lh, vget_high_s16(v3l), w3);
            a0rl = vmlal_s16(a0rl, vget_low_s16(v0r), w0);
            a1rl = vmlal_s16(a1rl, vget_low_s16(v1r), w1);
            a2rl = vmlal_s16(a2rl, vget_low_s16(v2r), w2);
            a3rl = vmlal_s16(a3rl, vget_low_s16(v3r), w3);
            a0rh = vmlal_s16(a0rh, vget_high_s16(v0r), w0);
            a1rh = vmlal_s16(a1rh, vget_high_s16(v1r), w1);
            a2rh = vmlal_s16(a2rh, vget_high_s16(v2r), w2);
            a3rh = vmlal_s16(a3rh, vget_high_s16(v3r), w3);
            k += 4;
        }
        while k < ky {
            let p = rows.get_unchecked(k).as_ptr().add(i);
            let vl = vld1q_s16(p);
            let vr = vld1q_s16(p.add(8));
            let wk = vdup_n_s16(*w.get_unchecked(k));
            let vll = vget_low_s16(vl);
            let vlh = vget_high_s16(vl);
            let vrl = vget_low_s16(vr);
            let vrh = vget_high_s16(vr);
            match k & 3 {
                0 => {
                    a0ll = vmlal_s16(a0ll, vll, wk);
                    a0lh = vmlal_s16(a0lh, vlh, wk);
                    a0rl = vmlal_s16(a0rl, vrl, wk);
                    a0rh = vmlal_s16(a0rh, vrh, wk);
                }
                1 => {
                    a1ll = vmlal_s16(a1ll, vll, wk);
                    a1lh = vmlal_s16(a1lh, vlh, wk);
                    a1rl = vmlal_s16(a1rl, vrl, wk);
                    a1rh = vmlal_s16(a1rh, vrh, wk);
                }
                _ => {
                    a2ll = vmlal_s16(a2ll, vll, wk);
                    a2lh = vmlal_s16(a2lh, vlh, wk);
                    a2rl = vmlal_s16(a2rl, vrl, wk);
                    a2rh = vmlal_s16(a2rh, vrh, wk);
                }
            }
            k += 1;
        }

        let acc_ll = vaddq_s32(vaddq_s32(a0ll, a1ll), vaddq_s32(a2ll, a3ll));
        let acc_lh = vaddq_s32(vaddq_s32(a0lh, a1lh), vaddq_s32(a2lh, a3lh));
        let acc_rl = vaddq_s32(vaddq_s32(a0rl, a1rl), vaddq_s32(a2rl, a3rl));
        let acc_rh = vaddq_s32(vaddq_s32(a0rh, a1rh), vaddq_s32(a2rh, a3rh));
        let packed_l = vcombine_s16(
            vqmovn_s32(vshrq_n_s32::<14>(acc_ll)),
            vqmovn_s32(vshrq_n_s32::<14>(acc_lh)),
        );
        let packed_r = vcombine_s16(
            vqmovn_s32(vshrq_n_s32::<14>(acc_rl)),
            vqmovn_s32(vshrq_n_s32::<14>(acc_rh)),
        );
        // Saturate both halves to u8, pack into one q-reg, store 16 bytes.
        let out = vcombine_u8(vqmovun_s16(packed_l), vqmovun_s16(packed_r));
        vst1q_u8(dst_row.as_mut_ptr().add(i), out);
        i += 16;
    }
    // 8-lane fallback for a remaining half-vector.
    if i + 8 <= n {
        let mut a0l = round_v;
        let mut a1l = zero;
        let mut a2l = zero;
        let mut a3l = zero;
        let mut a0h = round_v;
        let mut a1h = zero;
        let mut a2h = zero;
        let mut a3h = zero;
        let mut k = 0;
        while k + 4 <= ky {
            let v0 = vld1q_s16(rows.get_unchecked(k).as_ptr().add(i));
            let v1 = vld1q_s16(rows.get_unchecked(k + 1).as_ptr().add(i));
            let v2 = vld1q_s16(rows.get_unchecked(k + 2).as_ptr().add(i));
            let v3 = vld1q_s16(rows.get_unchecked(k + 3).as_ptr().add(i));
            let w0 = vdup_n_s16(*w.get_unchecked(k));
            let w1 = vdup_n_s16(*w.get_unchecked(k + 1));
            let w2 = vdup_n_s16(*w.get_unchecked(k + 2));
            let w3 = vdup_n_s16(*w.get_unchecked(k + 3));
            a0l = vmlal_s16(a0l, vget_low_s16(v0), w0);
            a1l = vmlal_s16(a1l, vget_low_s16(v1), w1);
            a2l = vmlal_s16(a2l, vget_low_s16(v2), w2);
            a3l = vmlal_s16(a3l, vget_low_s16(v3), w3);
            a0h = vmlal_s16(a0h, vget_high_s16(v0), w0);
            a1h = vmlal_s16(a1h, vget_high_s16(v1), w1);
            a2h = vmlal_s16(a2h, vget_high_s16(v2), w2);
            a3h = vmlal_s16(a3h, vget_high_s16(v3), w3);
            k += 4;
        }
        while k < ky {
            let v = vld1q_s16(rows.get_unchecked(k).as_ptr().add(i));
            let wk = vdup_n_s16(*w.get_unchecked(k));
            let vl = vget_low_s16(v);
            let vh = vget_high_s16(v);
            match k & 3 {
                0 => {
                    a0l = vmlal_s16(a0l, vl, wk);
                    a0h = vmlal_s16(a0h, vh, wk);
                }
                1 => {
                    a1l = vmlal_s16(a1l, vl, wk);
                    a1h = vmlal_s16(a1h, vh, wk);
                }
                _ => {
                    a2l = vmlal_s16(a2l, vl, wk);
                    a2h = vmlal_s16(a2h, vh, wk);
                }
            }
            k += 1;
        }
        let acc_lo = vaddq_s32(vaddq_s32(a0l, a1l), vaddq_s32(a2l, a3l));
        let acc_hi = vaddq_s32(vaddq_s32(a0h, a1h), vaddq_s32(a2h, a3h));
        let packed = vcombine_s16(
            vqmovn_s32(vshrq_n_s32::<14>(acc_lo)),
            vqmovn_s32(vshrq_n_s32::<14>(acc_hi)),
        );
        vst1_u8(dst_row.as_mut_ptr().add(i), vqmovun_s16(packed));
        i += 8;
    }
    // Scalar tail.
    while i < n {
        let mut acc: i32 = 0;
        for k in 0..ky {
            acc += rows[k][i] as i32 * w[k] as i32;
        }
        dst_row[i] = (((acc + round2) >> 14).clamp(0, 255)) as u8;
        i += 1;
    }
}
