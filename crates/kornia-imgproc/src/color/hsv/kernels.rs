//! RGB↔HSV f32 kernels (NEON / AVX2 / scalar).
//!
//! Convention (matches the historical kornia HSV): channel values are in `[0, 255]`.
//! H encodes `[0, 360)` degrees scaled to `[0, 255]`; S and V are `[0, 255]`.
//!
//! Forward `hsv_from_rgb_f32`: BT.601-free max/min/delta with sextant hue.
//! Reverse `rgb_from_hsv_f32`: standard chroma/sextant reconstruction.

use super::super::kernel_common::par_strip_dispatch;

const INV_255: f32 = 1.0 / 255.0;
const DEG_TO_BYTE: f32 = 255.0 / 360.0; // h_degrees → [0,255]
const BYTE_TO_DEG: f32 = 360.0 / 255.0; // [0,255] → h_degrees

/// Fast NEON reciprocal `1/x` (vrecpe seed + 2 Newton steps, ~f32-exact). Higher
/// throughput than `vdivq_f32` on the A78AE for the per-pixel hue/sat divides.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn vrecip_f32x4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let r = vrecpeq_f32(x);
    let r = vmulq_f32(r, vrecpsq_f32(x, r));
    vmulq_f32(r, vrecpsq_f32(x, r))
}

// ===== RGB f32 → HSV f32 ============================================================

/// Slice-level RGB f32 → HSV f32. Parallelized over row-strips for large images.
pub fn hsv_from_rgb_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, hsv_from_rgb_f32_kernel);
}

#[inline]
fn hsv_from_rgb_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        hsv_from_rgb_f32_neon(src, dst, npixels);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA confirmed by runtime probe.
            unsafe { hsv_from_rgb_f32_avx2(src, dst, npixels) };
            return;
        }
    }
    #[allow(unreachable_code)]
    hsv_from_rgb_f32_scalar(src, dst, npixels);
}

/// NEON RGB f32 → HSV f32: 4 px/iter via `vld3q_f32`, sextant hue via mask-select.
#[cfg(target_arch = "aarch64")]
fn hsv_from_rgb_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let inv255 = vdupq_n_f32(INV_255);
        let v60 = vdupq_n_f32(60.0);
        let v2 = vdupq_n_f32(2.0);
        let v4 = vdupq_n_f32(4.0);
        let v360 = vdupq_n_f32(360.0);
        let v255 = vdupq_n_f32(255.0);
        let deg2byte = vdupq_n_f32(DEG_TO_BYTE);
        let zero = vdupq_n_f32(0.0);
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        // Per-vector transform (4 px). Pulled into a closure so the driver can run two
        // independent instances per iteration — the two reciprocal/divide chains then
        // overlap on the OoO core instead of serializing on FDIV latency.
        let tx = |p: float32x4x3_t| -> float32x4x3_t {
            let r = vmulq_f32(p.0, inv255);
            let g = vmulq_f32(p.1, inv255);
            let b = vmulq_f32(p.2, inv255);

            let max = vmaxq_f32(vmaxq_f32(r, g), b);
            let min = vminq_f32(vminq_f32(r, g), b);
            let delta = vsubq_f32(max, min);

            // reciprocals (guard against /0 via masks below)
            let rd = vrecip_f32x4(delta);
            let rmax = vrecip_f32x4(max);

            // hue candidates (no %6 needed: (g-b)/delta ∈ [-1,1] on the r-max branch, etc.)
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

            // delta == 0 → h = 0
            let delta0 = vceqq_f32(delta, zero);
            h = vbslq_f32(delta0, zero, h);
            // scale to [0,255]
            h = vmulq_f32(h, deg2byte);

            // s = (delta/max)*255, 0 where max==0
            let mut s = vmulq_f32(vmulq_f32(delta, rmax), v255);
            let max0 = vceqq_f32(max, zero);
            s = vbslq_f32(max0, zero, s);

            let v = vmulq_f32(max, v255);
            float32x4x3_t(h, s, v)
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
            let (h, s, v) = hsv_from_rgb_scalar_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
            *dp.add(si) = h;
            *dp.add(si + 1) = s;
            *dp.add(si + 2) = v;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsv_from_rgb_f32_avx2(src: &[f32], dst: &mut [f32], npixels: usize) {
    // x86 correctness via scalar; AVX2 specialization in the x86 perf pass.
    hsv_from_rgb_f32_scalar(src, dst, npixels);
}

fn hsv_from_rgb_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let si = i * 3;
        let (h, s, v) = hsv_from_rgb_scalar_px(src[si], src[si + 1], src[si + 2]);
        dst[si] = h;
        dst[si + 1] = s;
        dst[si + 2] = v;
    }
}

/// Scalar per-pixel RGB→HSV oracle (values in [0,255]).
#[inline]
fn hsv_from_rgb_scalar_px(r8: f32, g8: f32, b8: f32) -> (f32, f32, f32) {
    let r = r8 * INV_255;
    let g = g8 * INV_255;
    let b = b8 * INV_255;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let h = h * DEG_TO_BYTE;
    let s = if max == 0.0 {
        0.0
    } else {
        (delta / max) * 255.0
    };
    let v = max * 255.0;
    (h, s, v)
}

// ===== HSV f32 → RGB f32 ============================================================

/// Slice-level HSV f32 → RGB f32. Parallelized over row-strips for large images.
pub fn rgb_from_hsv_f32(src: &[f32], dst: &mut [f32], npixels: usize) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, rgb_from_hsv_f32_kernel);
}

#[inline]
fn rgb_from_hsv_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        rgb_from_hsv_f32_neon(src, dst, npixels);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA confirmed by runtime probe.
            unsafe { rgb_from_hsv_f32_avx2(src, dst, npixels) };
            return;
        }
    }
    #[allow(unreachable_code)]
    rgb_from_hsv_f32_scalar(src, dst, npixels);
}

/// NEON HSV f32 → RGB f32: 4 px/iter, 6-way sextant select via masks.
#[cfg(target_arch = "aarch64")]
fn rgb_from_hsv_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let inv255 = vdupq_n_f32(INV_255);
        let v255 = vdupq_n_f32(255.0);
        let byte2deg_60 = vdupq_n_f32(BYTE_TO_DEG / 60.0); // H[0,255] → sextant coordinate [0,6)
        let v1 = vdupq_n_f32(1.0);
        let v2 = vdupq_n_f32(2.0);
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        let mut i = 0usize;
        let bulk4 = npixels & !3;
        while i < bulk4 {
            let p = vld3q_f32(sp.add(i * 3));
            let s = vmulq_f32(p.1, inv255);
            let v = vmulq_f32(p.2, inv255);
            let hh = vmulq_f32(p.0, byte2deg_60); // [0,6)

            let c = vmulq_f32(v, s);
            // hp mod 2
            let hp_half = vmulq_f32(hh, vdupq_n_f32(0.5));
            let two_floor = vmulq_f32(vrndmq_f32(hp_half), v2);
            let hmod2 = vsubq_f32(hh, two_floor);
            // x = c * (1 - |hmod2 - 1|)
            let x = vmulq_f32(c, vsubq_f32(v1, vabsq_f32(vsubq_f32(hmod2, v1))));
            let m = vsubq_f32(v, c);

            let sext = vcvtq_u32_f32(vrndmq_f32(hh)); // floor(hh) in [0,6)

            // per-sextant (r1,g1,b1) before adding m:
            // 0:(c,x,0) 1:(x,c,0) 2:(0,c,x) 3:(0,x,c) 4:(x,0,c) 5:(c,0,x)
            let zero = vdupq_n_f32(0.0);
            let eq0 = vceqq_u32(sext, vdupq_n_u32(0));
            let eq1 = vceqq_u32(sext, vdupq_n_u32(1));
            let eq2 = vceqq_u32(sext, vdupq_n_u32(2));
            let eq3 = vceqq_u32(sext, vdupq_n_u32(3));
            let eq4 = vceqq_u32(sext, vdupq_n_u32(4));
            // r1
            let r1 = vbslq_f32(
                eq0,
                c,
                vbslq_f32(
                    eq1,
                    x,
                    vbslq_f32(eq2, zero, vbslq_f32(eq3, zero, vbslq_f32(eq4, x, c))),
                ),
            );
            // g1
            let g1 = vbslq_f32(
                eq0,
                x,
                vbslq_f32(
                    eq1,
                    c,
                    vbslq_f32(eq2, c, vbslq_f32(eq3, x, vbslq_f32(eq4, zero, zero))),
                ),
            );
            // b1
            let b1 = vbslq_f32(
                eq0,
                zero,
                vbslq_f32(
                    eq1,
                    zero,
                    vbslq_f32(eq2, x, vbslq_f32(eq3, c, vbslq_f32(eq4, c, x))),
                ),
            );

            let r = vmulq_f32(vaddq_f32(r1, m), v255);
            let g = vmulq_f32(vaddq_f32(g1, m), v255);
            let b = vmulq_f32(vaddq_f32(b1, m), v255);

            vst3q_f32(dp.add(i * 3), float32x4x3_t(r, g, b));
            i += 4;
        }
        while i < npixels {
            let si = i * 3;
            let (r, g, b) = rgb_from_hsv_scalar_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
            *dp.add(si) = r;
            *dp.add(si + 1) = g;
            *dp.add(si + 2) = b;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb_from_hsv_f32_avx2(src: &[f32], dst: &mut [f32], npixels: usize) {
    rgb_from_hsv_f32_scalar(src, dst, npixels);
}

fn rgb_from_hsv_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize) {
    for i in 0..npixels {
        let si = i * 3;
        let (r, g, b) = rgb_from_hsv_scalar_px(src[si], src[si + 1], src[si + 2]);
        dst[si] = r;
        dst[si + 1] = g;
        dst[si + 2] = b;
    }
}

/// Scalar per-pixel HSV→RGB oracle (values in [0,255]).
#[inline]
fn rgb_from_hsv_scalar_px(h8: f32, s8: f32, v8: f32) -> (f32, f32, f32) {
    let s = s8 * INV_255;
    let v = v8 * INV_255;
    let hh = h8 * (BYTE_TO_DEG / 60.0); // [0,6)
    let c = v * s;
    let hmod2 = hh - 2.0 * (hh * 0.5).floor();
    let x = c * (1.0 - (hmod2 - 1.0).abs());
    let m = v - c;
    let sext = hh.floor() as i32;
    let (r1, g1, b1) = match sext {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    ((r1 + m) * 255.0, (g1 + m) * 255.0, (b1 + m) * 255.0)
}
