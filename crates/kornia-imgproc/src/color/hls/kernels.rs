//! RGB↔HLS f32 kernels (NEON / AVX2 / scalar).
//!
//! Convention (matches the historical kornia HLS): channel values are in `[0, 255]`.
//! H encodes `[0, 360)` degrees scaled to `[0, 255]`; L and S are `[0, 255]`. The image
//! channel order is `[H, L, S]`.
//!
//! Forward `hls_from_rgb_f32`: max/min/diff with bi-cone lightness and sextant hue.
//! Reverse `rgb_from_hls_f32`: standard hue2rgb reconstruction.

use super::super::kernel_common::par_strip_dispatch;

const INV_255: f32 = 1.0 / 255.0;
const DEG_TO_BYTE: f32 = 255.0 / 360.0; // h_degrees → [0,255]
const BYTE_TO_DEG: f32 = 360.0 / 255.0; // [0,255] → h_degrees

// ===== RGB f32 → HLS f32 ============================================================

/// Slice-level RGB f32 → HLS f32. Parallelized over row-strips for large images.
pub fn hls_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, hls_from_rgb_f32_kernel);
}

#[inline]
fn hls_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        hls_from_rgb_f32_neon(src, dst, npixels);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA confirmed by runtime probe.
            unsafe { hls_from_rgb_f32_avx2(src, dst, npixels) };
            return;
        }
    }
    #[allow(unreachable_code)]
    hls_from_rgb_f32_scalar(src, dst, npixels);
}

/// NEON RGB f32 → HLS f32: 4 px/iter via `vld3q_f32`, sextant hue via mask-select.
#[cfg(target_arch = "aarch64")]
fn hls_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let inv255 = vdupq_n_f32(INV_255);
        let v60 = vdupq_n_f32(60.0);
        let v2 = vdupq_n_f32(2.0);
        let v4 = vdupq_n_f32(4.0);
        let v360 = vdupq_n_f32(360.0);
        let v255 = vdupq_n_f32(255.0);
        let half = vdupq_n_f32(0.5);
        let deg2byte = vdupq_n_f32(DEG_TO_BYTE);
        let zero = vdupq_n_f32(0.0);
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        // Per-vector transform (4 px) as a closure so the driver runs two independent
        // instances per iteration — overlaps the three divide chains on the OoO core.
        let tx = |p: float32x4x3_t| -> float32x4x3_t {
            let r = vmulq_f32(p.0, inv255);
            let g = vmulq_f32(p.1, inv255);
            let b = vmulq_f32(p.2, inv255);

            let max = vmaxq_f32(vmaxq_f32(r, g), b);
            let min = vminq_f32(vminq_f32(r, g), b);
            let diff = vsubq_f32(max, min);
            let sum = vaddq_f32(max, min);
            let l = vmulq_f32(sum, half);

            // reciprocal of diff (guard against /0 via masks below)
            let rd = vdivq_f32(vdupq_n_f32(1.0), diff);

            // hue candidates (no %6 needed: (g-b)/diff ∈ [-1,1] on the r-max branch, etc.)
            let h_r = vmulq_f32(vsubq_f32(g, b), rd);
            let h_g = vaddq_f32(vmulq_f32(vsubq_f32(b, r), rd), v2);
            let h_b = vaddq_f32(vmulq_f32(vsubq_f32(r, g), rd), v4);

            let is_r = vceqq_f32(max, r);
            let is_g = vceqq_f32(max, g);
            // select: r-branch, else g-branch, else b-branch
            let mut h = vbslq_f32(is_r, h_r, vbslq_f32(is_g, h_g, h_b));
            h = vmulq_f32(h, v60); // degrees, possibly negative

            // h < 0 → h + 360
            let neg = vcltq_f32(h, zero);
            h = vbslq_f32(neg, vaddq_f32(h, v360), h);

            // diff == 0 → h = 0
            let diff0 = vceqq_f32(diff, zero);
            h = vbslq_f32(diff0, zero, h);
            // scale to [0,255]
            h = vmulq_f32(h, deg2byte);

            // s = if L<=0.5 { diff/sum } else { diff/(2-sum) }
            let s_lo = vmulq_f32(diff, vdivq_f32(vdupq_n_f32(1.0), sum));
            let s_hi = vmulq_f32(diff, vdivq_f32(vdupq_n_f32(1.0), vsubq_f32(v2, sum)));
            let l_le = vcleq_f32(l, half);
            let mut s = vbslq_f32(l_le, s_lo, s_hi);
            // diff == 0 → s = 0
            s = vbslq_f32(diff0, zero, s);
            s = vmulq_f32(s, v255);

            let lo = vmulq_f32(l, v255);
            float32x4x3_t(h, lo, s)
        };

        let mut i = 0usize;
        let bulk8 = npixels & !7;
        while i < bulk8 {
            let p0 = vld3q_f32(sp.add(i * 3));
            let p1 = vld3q_f32(sp.add((i + 4) * 3));
            let o0 = tx(p0);
            let o1 = tx(p1);
            vst3q_f32(dp.add(i * 3), o0);
            vst3q_f32(dp.add((i + 4) * 3), o1);
            i += 8;
        }
        if i + 4 <= npixels {
            let p = vld3q_f32(sp.add(i * 3));
            vst3q_f32(dp.add(i * 3), tx(p));
            i += 4;
        }
        // scalar tail
        while i < npixels {
            let si = i * 3;
            let (h, l, s) = hls_from_rgb_scalar_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
            *dp.add(si) = h;
            *dp.add(si + 1) = l;
            *dp.add(si + 2) = s;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hls_from_rgb_f32_avx2(src: &[f32], dst: &mut [f32], npixels: usize) {
    // x86 correctness via scalar; AVX2 specialization in the x86 perf pass.
    hls_from_rgb_f32_scalar(src, dst, npixels);
}

fn hls_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let si = i * 3;
        let (h, l, s) = hls_from_rgb_scalar_px(src[si], src[si + 1], src[si + 2]);
        dst[si] = h;
        dst[si + 1] = l;
        dst[si + 2] = s;
    }
}

/// Scalar per-pixel RGB→HLS oracle (values in [0,255], channel order [H, L, S]).
#[inline]
fn hls_from_rgb_scalar_px(r8: f32, g8: f32, b8: f32) -> (f32, f32, f32) {
    let r = r8 * INV_255;
    let g = g8 * INV_255;
    let b = b8 * INV_255;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let diff = max - min;
    let sum = max + min;
    let l = sum * 0.5;
    let (h, s) = if diff == 0.0 {
        (0.0, 0.0)
    } else {
        let s = if l <= 0.5 {
            diff / sum
        } else {
            diff / (2.0 - sum)
        };
        let h = if max == r {
            60.0 * (((g - b) / diff) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / diff) + 2.0)
        } else {
            60.0 * (((r - g) / diff) + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        (h, s)
    };
    (h * DEG_TO_BYTE, l * 255.0, s * 255.0)
}

// ===== HLS f32 → RGB f32 ============================================================

/// Slice-level HLS f32 → RGB f32. Parallelized over row-strips for large images.
pub fn rgb_from_hls_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, rgb_from_hls_f32_kernel);
}

#[inline]
fn rgb_from_hls_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_hls_f32_neon(src, dst, npixels);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA confirmed by runtime probe.
            unsafe { rgb_from_hls_f32_avx2(src, dst, npixels) };
            return;
        }
    }
    #[allow(unreachable_code)]
    rgb_from_hls_f32_scalar(src, dst, npixels);
}

/// NEON HLS f32 → RGB f32: 4 px/iter, hue2rgb branches via masks.
#[cfg(target_arch = "aarch64")]
fn rgb_from_hls_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let inv255 = vdupq_n_f32(INV_255);
        let v255 = vdupq_n_f32(255.0);
        let byte2deg_360 = vdupq_n_f32(BYTE_TO_DEG / 360.0); // H[0,255] → hue fraction [0,1)
        let v1 = vdupq_n_f32(1.0);
        let v2 = vdupq_n_f32(2.0);
        let v6 = vdupq_n_f32(6.0);
        let half = vdupq_n_f32(0.5);
        let third = vdupq_n_f32(1.0 / 3.0);
        let two_third = vdupq_n_f32(2.0 / 3.0);
        let one_sixth = vdupq_n_f32(1.0 / 6.0);
        let zero = vdupq_n_f32(0.0);
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        let mut i = 0usize;
        let bulk4 = npixels & !3;
        while i < bulk4 {
            let p = vld3q_f32(sp.add(i * 3));
            let l = vmulq_f32(p.1, inv255);
            let s = vmulq_f32(p.2, inv255);
            let hk = vmulq_f32(p.0, byte2deg_360); // hue fraction [0,1)

            // q = if L<0.5 { L*(1+S) } else { L+S-L*S }
            let q_lo = vmulq_f32(l, vaddq_f32(v1, s));
            let ls = vmulq_f32(l, s);
            let q_hi = vsubq_f32(vaddq_f32(l, s), ls);
            let l_lt = vcltq_f32(l, half);
            let q = vbslq_f32(l_lt, q_lo, q_hi);
            // p = 2L - q
            let pp = vsubq_f32(vmulq_f32(v2, l), q);

            let r = hue2rgb_neon(
                pp,
                q,
                vaddq_f32(hk, third),
                v1,
                v6,
                half,
                two_third,
                one_sixth,
            );
            let g = hue2rgb_neon(pp, q, hk, v1, v6, half, two_third, one_sixth);
            let b = hue2rgb_neon(
                pp,
                q,
                vsubq_f32(hk, third),
                v1,
                v6,
                half,
                two_third,
                one_sixth,
            );

            // S == 0 → r=g=b=L
            let s0 = vceqq_f32(s, zero);
            let r = vbslq_f32(s0, l, r);
            let g = vbslq_f32(s0, l, g);
            let b = vbslq_f32(s0, l, b);

            let r = vmulq_f32(r, v255);
            let g = vmulq_f32(g, v255);
            let b = vmulq_f32(b, v255);

            vst3q_f32(dp.add(i * 3), float32x4x3_t(r, g, b));
            i += 4;
        }
        while i < npixels {
            let si = i * 3;
            let (r, g, b) = rgb_from_hls_scalar_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
            *dp.add(si) = r;
            *dp.add(si + 1) = g;
            *dp.add(si + 2) = b;
            i += 1;
        }
    }
}

/// NEON hue2rgb: compute all 4 candidate branches, select with `t` masks.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn hue2rgb_neon(
    p: std::arch::aarch64::float32x4_t,
    q: std::arch::aarch64::float32x4_t,
    t: std::arch::aarch64::float32x4_t,
    v1: std::arch::aarch64::float32x4_t,
    v6: std::arch::aarch64::float32x4_t,
    half: std::arch::aarch64::float32x4_t,
    two_third: std::arch::aarch64::float32x4_t,
    one_sixth: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    unsafe {
        use std::arch::aarch64::*;
        let zero = vdupq_n_f32(0.0);
        // wrap t into [0,1): if t<0 t+=1; if t>1 t-=1
        let t = vbslq_f32(vcltq_f32(t, zero), vaddq_f32(t, v1), t);
        let t = vbslq_f32(vcgtq_f32(t, v1), vsubq_f32(t, v1), t);

        let qmp = vsubq_f32(q, p);
        // branch a: t<1/6 → p + (q-p)*6*t
        let cand_a = vaddq_f32(p, vmulq_f32(qmp, vmulq_f32(v6, t)));
        // branch b: t<1/2 → q
        // branch c: t<2/3 → p + (q-p)*(2/3 - t)*6
        let cand_c = vaddq_f32(p, vmulq_f32(qmp, vmulq_f32(vsubq_f32(two_third, t), v6)));
        // branch d (else): p

        // select from the bottom up: default p, then c if t<2/3, then q if t<1/2, then a if t<1/6
        let lt_two_third = vcltq_f32(t, two_third);
        let lt_half = vcltq_f32(t, half);
        let lt_sixth = vcltq_f32(t, one_sixth);
        let out = vbslq_f32(lt_two_third, cand_c, p);
        let out = vbslq_f32(lt_half, q, out);
        vbslq_f32(lt_sixth, cand_a, out)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb_from_hls_f32_avx2(src: &[f32], dst: &mut [f32], npixels: usize) {
    rgb_from_hls_f32_scalar(src, dst, npixels);
}

fn rgb_from_hls_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let si = i * 3;
        let (r, g, b) = rgb_from_hls_scalar_px(src[si], src[si + 1], src[si + 2]);
        dst[si] = r;
        dst[si + 1] = g;
        dst[si + 2] = b;
    }
}

/// Scalar per-pixel HLS→RGB oracle (values in [0,255], channel order [H, L, S]).
#[inline]
fn rgb_from_hls_scalar_px(h8: f32, l8: f32, s8: f32) -> (f32, f32, f32) {
    let l = l8 * INV_255;
    let s = s8 * INV_255;
    if s == 0.0 {
        let v = l * 255.0;
        return (v, v, v);
    }
    let h_deg = h8 * BYTE_TO_DEG;
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = h_deg / 360.0;
    let r = hue2rgb_scalar(p, q, hk + 1.0 / 3.0);
    let g = hue2rgb_scalar(p, q, hk);
    let b = hue2rgb_scalar(p, q, hk - 1.0 / 3.0);
    (r * 255.0, g * 255.0, b * 255.0)
}

#[inline]
fn hue2rgb_scalar(p: f32, q: f32, t: f32) -> f32 {
    let t = if t < 0.0 { t + 1.0 } else { t };
    let t = if t > 1.0 { t - 1.0 } else { t };
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 0.5 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}
