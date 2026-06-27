//! CIE color-conversion f32 kernels (NEON / scalar) + f64 scalar oracles.
//!
//! Pipelines are fused in-register: an RGB→Lab pass does a single `vld3q_f32`,
//! runs gamma decode → linear→XYZ matrix → Lab combine entirely in vector
//! registers, then a single `vst3q_f32`. No intermediate round-trips through
//! memory between stages.
//!
//! Channel convention: RGB and linear-RGB in `[0, 1]`; XYZ in tristimulus units;
//! Lab `L ∈ [0,100]`, `a,b ∈ ~[-128,127]`; Luv `L ∈ [0,100]`, `u,v` similar.

use super::super::kernel_common::par_strip_dispatch;
// `nonlinear` is only used by the aarch64 NEON paths below; gate the import so it
// isn't flagged as unused on x86_64 (CI clippy runs on x86_64).
#[cfg(target_arch = "aarch64")]
use super::nonlinear;
use super::transfer;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ===== shared constants =============================================================

// linear-RGB → XYZ (row-major), and inverse. Canonical OpenCV/sRGB-D65 coefficients;
// keep the full published digits even where f32 rounds them (matches OpenCV bit-for-bit).
#[allow(clippy::excessive_precision)]
const M_RGB2XYZ: [f32; 9] = [
    0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227,
];
#[allow(clippy::excessive_precision)]
const M_XYZ2RGB: [f32; 9] = [
    3.240479, -1.537150, -0.498535, -0.969256, 1.875991, 0.041556, 0.055648, -0.204043, 1.057311,
];

// D65 white point.
const XN: f32 = 0.950456;
const YN: f32 = 1.0;
const ZN: f32 = 1.088754;
const INV_XN: f32 = 1.0 / XN;
const INV_ZN: f32 = 1.0 / ZN;

// Lab f(t) piecewise constants.
const LAB_DELTA: f32 = 0.008856; // (6/29)^3
const LAB_F_SLOPE: f32 = 1.0 / 0.128_418_55; // 1 / (3*(6/29)^2)  → t * this + offset
const LAB_F_OFFSET: f32 = 0.137_931_03; // 4/29
const LAB_FINV_THRESH: f32 = 0.206_896_55; // 6/29
const LAB_FINV_SLOPE: f32 = 0.128_418_55; // 3*(6/29)^2

// Luv constants.
const LUV_UN: f32 = 0.197_939_43;
const LUV_VN: f32 = 0.468_310_96;
const LUV_KAPPA: f32 = 903.3; // CIE κ

// ===== scalar helpers (oracle building blocks) ======================================

#[inline]
fn lab_f(t: f64) -> f64 {
    if t > LAB_DELTA as f64 {
        t.cbrt()
    } else {
        t * LAB_F_SLOPE as f64 + LAB_F_OFFSET as f64
    }
}

#[inline]
fn lab_finv(f: f64) -> f64 {
    if f > LAB_FINV_THRESH as f64 {
        f * f * f
    } else {
        LAB_FINV_SLOPE as f64 * (f - LAB_F_OFFSET as f64)
    }
}

#[inline]
fn matvec(m: &[f32; 9], a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    (
        m[0] as f64 * a + m[1] as f64 * b + m[2] as f64 * c,
        m[3] as f64 * a + m[4] as f64 * b + m[5] as f64 * c,
        m[6] as f64 * a + m[7] as f64 * b + m[8] as f64 * c,
    )
}

// ===== f64 scalar oracles (exact formulas) ==========================================

#[inline]
pub(crate) fn linear_from_srgb_scalar64(x: f64) -> f64 {
    let x = x.max(0.0);
    if x <= transfer::SRGB_THRESH as f64 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub(crate) fn srgb_from_linear_scalar64(l: f64) -> f64 {
    let l = l.max(0.0);
    if l <= transfer::SRGB_INV_THRESH as f64 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// Per-channel sRGB→linear over an RGB triple (f64 oracle, 3→3 form).
#[inline]
pub(crate) fn linear_from_srgb_scalar64_px(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        linear_from_srgb_scalar64(r),
        linear_from_srgb_scalar64(g),
        linear_from_srgb_scalar64(b),
    )
}

/// Per-channel linear→sRGB over a triple (f64 oracle, 3→3 form).
#[inline]
pub(crate) fn srgb_from_linear_scalar64_px(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        srgb_from_linear_scalar64(r),
        srgb_from_linear_scalar64(g),
        srgb_from_linear_scalar64(b),
    )
}

// Public RGB↔XYZ: a direct 3×3 matrix, NO gamma — matches OpenCV `COLOR_RGB2XYZ`
// and `kornia.color.rgb_to_xyz`. (Lab/Luv linearize internally via the `lin_*`
// helpers below before applying the matrix.)
#[inline]
pub(crate) fn xyz_from_rgb_scalar64(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    matvec(&M_RGB2XYZ, r, g, b)
}

#[inline]
pub(crate) fn rgb_from_xyz_scalar64(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    matvec(&M_XYZ2RGB, x, y, z)
}

// Gamma-aware XYZ (sRGB→linear→matrix), the colorimetric pipeline used by Lab/Luv.
#[inline]
fn lin_xyz_from_rgb_scalar64(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let lr = linear_from_srgb_scalar64(r);
    let lg = linear_from_srgb_scalar64(g);
    let lb = linear_from_srgb_scalar64(b);
    matvec(&M_RGB2XYZ, lr, lg, lb)
}

#[inline]
fn rgb_from_lin_xyz_scalar64(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (lr, lg, lb) = matvec(&M_XYZ2RGB, x, y, z);
    (
        srgb_from_linear_scalar64(lr),
        srgb_from_linear_scalar64(lg),
        srgb_from_linear_scalar64(lb),
    )
}

#[inline]
pub(crate) fn lab_from_rgb_scalar64(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = lin_xyz_from_rgb_scalar64(r, g, b);
    let fx = lab_f(x / XN as f64);
    let fy = lab_f(y / YN as f64);
    let fz = lab_f(z / ZN as f64);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

#[inline]
pub(crate) fn rgb_from_lab_scalar64(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = XN as f64 * lab_finv(fx);
    let y = YN as f64 * lab_finv(fy);
    let z = ZN as f64 * lab_finv(fz);
    rgb_from_lin_xyz_scalar64(x, y, z)
}

#[inline]
pub(crate) fn luv_from_rgb_scalar64(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = lin_xyz_from_rgb_scalar64(r, g, b);
    let yr = y / YN as f64;
    let l = if yr > LAB_DELTA as f64 {
        116.0 * yr.cbrt() - 16.0
    } else {
        LUV_KAPPA as f64 * yr
    };
    let d = x + 15.0 * y + 3.0 * z;
    let (up, vp) = if d == 0.0 {
        (0.0, 0.0)
    } else {
        (4.0 * x / d, 9.0 * y / d)
    };
    let u = 13.0 * l * (up - LUV_UN as f64);
    let v = 13.0 * l * (vp - LUV_VN as f64);
    (l, u, v)
}

#[inline]
pub(crate) fn rgb_from_luv_scalar64(l: f64, u: f64, v: f64) -> (f64, f64, f64) {
    if l <= 0.0 {
        return rgb_from_lin_xyz_scalar64(0.0, 0.0, 0.0);
    }
    let y = if l > 8.0 {
        YN as f64 * ((l + 16.0) / 116.0).powi(3)
    } else {
        YN as f64 * l / LUV_KAPPA as f64
    };
    let up = u / (13.0 * l) + LUV_UN as f64;
    let vp = v / (13.0 * l) + LUV_VN as f64;
    let x = y * 9.0 * up / (4.0 * vp);
    let z = y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp);
    rgb_from_lin_xyz_scalar64(x, y, z)
}

// f32 scalar per-pixel reference (mirrors the f32 SIMD math; used for tail + fallback).
#[inline]
fn lab_f_f32(t: f32) -> f32 {
    if t > LAB_DELTA {
        t.cbrt()
    } else {
        t * LAB_F_SLOPE + LAB_F_OFFSET
    }
}

#[inline]
fn lab_finv_f32(f: f32) -> f32 {
    if f > LAB_FINV_THRESH {
        f * f * f
    } else {
        LAB_FINV_SLOPE * (f - LAB_F_OFFSET)
    }
}

#[inline]
fn matvec32(m: &[f32; 9], a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    (
        m[0] * a + m[1] * b + m[2] * c,
        m[3] * a + m[4] * b + m[5] * c,
        m[6] * a + m[7] * b + m[8] * c,
    )
}

#[inline]
fn linear_from_srgb_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        transfer::srgb_to_linear_scalar(r),
        transfer::srgb_to_linear_scalar(g),
        transfer::srgb_to_linear_scalar(b),
    )
}

#[inline]
fn srgb_from_linear_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        transfer::linear_to_srgb_scalar(r),
        transfer::linear_to_srgb_scalar(g),
        transfer::linear_to_srgb_scalar(b),
    )
}

#[inline]
fn xyz_from_rgb_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    matvec32(&M_RGB2XYZ, r, g, b)
}

#[inline]
fn rgb_from_xyz_px(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    matvec32(&M_XYZ2RGB, x, y, z)
}

// Gamma-aware XYZ for the Lab/Luv f32 scalar tail.
#[inline]
fn lin_xyz_from_rgb_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (lr, lg, lb) = linear_from_srgb_px(r, g, b);
    matvec32(&M_RGB2XYZ, lr, lg, lb)
}

#[inline]
fn rgb_from_lin_xyz_px(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let (lr, lg, lb) = matvec32(&M_XYZ2RGB, x, y, z);
    srgb_from_linear_px(lr, lg, lb)
}

#[inline]
fn lab_from_rgb_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (x, y, z) = lin_xyz_from_rgb_px(r, g, b);
    let fx = lab_f_f32(x * INV_XN);
    let fy = lab_f_f32(y); // YN == 1
    let fz = lab_f_f32(z * INV_ZN);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

#[inline]
fn rgb_from_lab_px(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = XN * lab_finv_f32(fx);
    let y = YN * lab_finv_f32(fy);
    let z = ZN * lab_finv_f32(fz);
    rgb_from_lin_xyz_px(x, y, z)
}

#[inline]
fn luv_from_rgb_px(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (x, y, z) = lin_xyz_from_rgb_px(r, g, b);
    let yr = y; // YN == 1
    let l = if yr > LAB_DELTA {
        116.0 * yr.cbrt() - 16.0
    } else {
        LUV_KAPPA * yr
    };
    let d = x + 15.0 * y + 3.0 * z;
    let (up, vp) = if d == 0.0 {
        (0.0, 0.0)
    } else {
        (4.0 * x / d, 9.0 * y / d)
    };
    (l, 13.0 * l * (up - LUV_UN), 13.0 * l * (vp - LUV_VN))
}

#[inline]
fn rgb_from_luv_px(l: f32, u: f32, v: f32) -> (f32, f32, f32) {
    if l <= 0.0 {
        return rgb_from_lin_xyz_px(0.0, 0.0, 0.0);
    }
    let y = if l > 8.0 {
        let t = (l + 16.0) / 116.0;
        YN * t * t * t
    } else {
        YN * l / LUV_KAPPA
    };
    let inv13l = 1.0 / (13.0 * l);
    let up = u * inv13l + LUV_UN;
    let vp = v * inv13l + LUV_VN;
    let x = y * 9.0 * up / (4.0 * vp);
    let z = y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp);
    rgb_from_lin_xyz_px(x, y, z)
}

// ===== NEON building blocks =========================================================

/// NEON sRGB→linear on a vector (per channel). Branchless: compute both segments,
/// select with the breakpoint mask. Clamps negatives to 0 first.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn srgb_to_linear_v(x: float32x4_t) -> float32x4_t {
    let x = vmaxq_f32(x, vdupq_n_f32(0.0));
    let lin = vmulq_f32(x, vdupq_n_f32(transfer::SRGB_INV_1292));
    let t = vmulq_f32(
        vaddq_f32(x, vdupq_n_f32(transfer::SRGB_A)),
        vdupq_n_f32(transfer::SRGB_INV_1055),
    );
    let powed = transfer::pow_f32x4(t, vdupq_n_f32(transfer::SRGB_GAMMA));
    let mask = vcleq_f32(x, vdupq_n_f32(transfer::SRGB_THRESH));
    vbslq_f32(mask, lin, powed)
}

/// NEON linear→sRGB on a vector (per channel).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn linear_to_srgb_v(l: float32x4_t) -> float32x4_t {
    let l = vmaxq_f32(l, vdupq_n_f32(0.0));
    let lin = vmulq_f32(l, vdupq_n_f32(transfer::SRGB_1292));
    let powed = vsubq_f32(
        vmulq_f32(
            vdupq_n_f32(transfer::SRGB_1055),
            transfer::pow_f32x4(l, vdupq_n_f32(transfer::SRGB_INV_GAMMA)),
        ),
        vdupq_n_f32(transfer::SRGB_A),
    );
    let mask = vcleq_f32(l, vdupq_n_f32(transfer::SRGB_INV_THRESH));
    vbslq_f32(mask, lin, powed)
}

/// NEON 3×3 matrix-vector (row-major `m`) on deinterleaved channel vectors.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn matvec_v(
    m: &[f32; 9],
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let o0 = vfmaq_f32(
        vfmaq_f32(vmulq_f32(a, vdupq_n_f32(m[0])), b, vdupq_n_f32(m[1])),
        c,
        vdupq_n_f32(m[2]),
    );
    let o1 = vfmaq_f32(
        vfmaq_f32(vmulq_f32(a, vdupq_n_f32(m[3])), b, vdupq_n_f32(m[4])),
        c,
        vdupq_n_f32(m[5]),
    );
    let o2 = vfmaq_f32(
        vfmaq_f32(vmulq_f32(a, vdupq_n_f32(m[6])), b, vdupq_n_f32(m[7])),
        c,
        vdupq_n_f32(m[8]),
    );
    (o0, o1, o2)
}

/// NEON Lab `f(t)`: branchless cbrt vs linear segment.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn lab_f_v(t: float32x4_t) -> float32x4_t {
    let cube = nonlinear::cbrt_f32x4(t);
    let lin = vaddq_f32(
        vmulq_f32(t, vdupq_n_f32(LAB_F_SLOPE)),
        vdupq_n_f32(LAB_F_OFFSET),
    );
    let mask = vcgtq_f32(t, vdupq_n_f32(LAB_DELTA));
    vbslq_f32(mask, cube, lin)
}

/// NEON Lab `f^{-1}(f)`: branchless `f^3` vs linear segment.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn lab_finv_v(f: float32x4_t) -> float32x4_t {
    let cube = vmulq_f32(vmulq_f32(f, f), f);
    let lin = vmulq_f32(
        vdupq_n_f32(LAB_FINV_SLOPE),
        vsubq_f32(f, vdupq_n_f32(LAB_F_OFFSET)),
    );
    let mask = vcgtq_f32(f, vdupq_n_f32(LAB_FINV_THRESH));
    vbslq_f32(mask, cube, lin)
}

// ===== generic NEON driver ==========================================================

/// Run a per-vector transform `f(float32x4x3_t) -> float32x4x3_t` over the image,
/// 8 px/iter (2× unrolled so two independent `pow`/`cbrt` chains overlap on the
/// A78AE's dual SIMD pipes — hides transcendental latency), then a 4-px remainder,
/// then a scalar tail driven by `scalar_px`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_drive(
    src: &[f32],
    dst: &mut [f32],
    npixels: usize,
    f: impl Fn(float32x4x3_t) -> float32x4x3_t,
    scalar_px: impl Fn(f32, f32, f32) -> (f32, f32, f32),
) {
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    let bulk8 = npixels & !7;
    while i < bulk8 {
        // Two independent loads/transforms — no data dependency between them, so the
        // out-of-order core overlaps the two pow/cbrt chains.
        let p0 = vld3q_f32(sp.add(i * 3));
        let p1 = vld3q_f32(sp.add((i + 4) * 3));
        let o0 = f(p0);
        let o1 = f(p1);
        vst3q_f32(dp.add(i * 3), o0);
        vst3q_f32(dp.add((i + 4) * 3), o1);
        i += 8;
    }
    if i + 4 <= npixels {
        let p = vld3q_f32(sp.add(i * 3));
        vst3q_f32(dp.add(i * 3), f(p));
        i += 4;
    }
    while i < npixels {
        let si = i * 3;
        let (o0, o1, o2) = scalar_px(*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
        *dp.add(si) = o0;
        *dp.add(si + 1) = o1;
        *dp.add(si + 2) = o2;
        i += 1;
    }
}

// Scalar fallback driver (non-aarch64 / scalar leaf).
fn scalar_drive(
    src: &[f32],
    dst: &mut [f32],
    npixels: usize,
    scalar_px: impl Fn(f32, f32, f32) -> (f32, f32, f32),
) {
    for i in 0..npixels {
        let si = i * 3;
        let (o0, o1, o2) = scalar_px(src[si], src[si + 1], src[si + 2]);
        dst[si] = o0;
        dst[si + 1] = o1;
        dst[si + 2] = o2;
    }
}

// ===== public slice kernels =========================================================
//
// Each follows the 4-layer pattern: slice entry → strip dispatch → cfg leaf.

macro_rules! cie_kernel {
    ($name:ident, $neon_body:expr, $scalar_px:path) => {
        pub fn $name(src: &[f32], dst: &mut [f32], npixels: usize) {
            debug_assert!(src.len() >= npixels * 3);
            debug_assert!(dst.len() >= npixels * 3);
            par_strip_dispatch(src, dst, npixels, 3, 8, |s, d, n| {
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is baseline on aarch64; slices sized by debug_assert.
                    unsafe { neon_drive(s, d, n, $neon_body, $scalar_px) };
                    return;
                }
                #[allow(unreachable_code)]
                scalar_drive(s, d, n, $scalar_px);
            });
        }
    };
}

// --- sRGB ↔ linear-RGB ---
cie_kernel!(
    linear_rgb_from_rgb_f32,
    |p: float32x4x3_t| float32x4x3_t(
        srgb_to_linear_v(p.0),
        srgb_to_linear_v(p.1),
        srgb_to_linear_v(p.2)
    ),
    linear_from_srgb_px
);
cie_kernel!(
    rgb_from_linear_rgb_f32,
    |p: float32x4x3_t| float32x4x3_t(
        linear_to_srgb_v(p.0),
        linear_to_srgb_v(p.1),
        linear_to_srgb_v(p.2)
    ),
    srgb_from_linear_px
);

// --- RGB ↔ XYZ ---
cie_kernel!(
    xyz_from_rgb_f32,
    |p: float32x4x3_t| {
        // Direct matrix, no gamma — matches OpenCV/Kornia rgb_to_xyz convention.
        let (x, y, z) = matvec_v(&M_RGB2XYZ, p.0, p.1, p.2);
        float32x4x3_t(x, y, z)
    },
    xyz_from_rgb_px
);
cie_kernel!(
    rgb_from_xyz_f32,
    |p: float32x4x3_t| {
        let (r, g, b) = matvec_v(&M_XYZ2RGB, p.0, p.1, p.2);
        float32x4x3_t(r, g, b)
    },
    rgb_from_xyz_px
);

// --- RGB ↔ Lab (fully fused) ---
cie_kernel!(
    lab_from_rgb_f32,
    |p: float32x4x3_t| {
        let lr = srgb_to_linear_v(p.0);
        let lg = srgb_to_linear_v(p.1);
        let lb = srgb_to_linear_v(p.2);
        let (x, y, z) = matvec_v(&M_RGB2XYZ, lr, lg, lb);
        let fx = lab_f_v(vmulq_f32(x, vdupq_n_f32(INV_XN)));
        let fy = lab_f_v(y); // YN == 1
        let fz = lab_f_v(vmulq_f32(z, vdupq_n_f32(INV_ZN)));
        let l = vsubq_f32(vmulq_f32(fy, vdupq_n_f32(116.0)), vdupq_n_f32(16.0));
        let a = vmulq_f32(vsubq_f32(fx, fy), vdupq_n_f32(500.0));
        let b = vmulq_f32(vsubq_f32(fy, fz), vdupq_n_f32(200.0));
        float32x4x3_t(l, a, b)
    },
    lab_from_rgb_px
);
cie_kernel!(
    rgb_from_lab_f32,
    |p: float32x4x3_t| {
        let fy = vmulq_f32(vaddq_f32(p.0, vdupq_n_f32(16.0)), vdupq_n_f32(1.0 / 116.0));
        let fx = vfmaq_f32(fy, p.1, vdupq_n_f32(1.0 / 500.0));
        let fz = vfmsq_f32(fy, p.2, vdupq_n_f32(1.0 / 200.0));
        let x = vmulq_f32(lab_finv_v(fx), vdupq_n_f32(XN));
        let y = vmulq_f32(lab_finv_v(fy), vdupq_n_f32(YN));
        let z = vmulq_f32(lab_finv_v(fz), vdupq_n_f32(ZN));
        let (lr, lg, lb) = matvec_v(&M_XYZ2RGB, x, y, z);
        float32x4x3_t(
            linear_to_srgb_v(lr),
            linear_to_srgb_v(lg),
            linear_to_srgb_v(lb),
        )
    },
    rgb_from_lab_px
);

// --- RGB ↔ Luv (fully fused) ---
cie_kernel!(
    luv_from_rgb_f32,
    |p: float32x4x3_t| {
        let lr = srgb_to_linear_v(p.0);
        let lg = srgb_to_linear_v(p.1);
        let lb = srgb_to_linear_v(p.2);
        let (x, y, z) = matvec_v(&M_RGB2XYZ, lr, lg, lb);
        // L: 116*cbrt(yr)-16 if yr>delta else kappa*yr; YN==1 so yr==y.
        let l_big = vsubq_f32(
            vmulq_f32(nonlinear::cbrt_f32x4(y), vdupq_n_f32(116.0)),
            vdupq_n_f32(16.0),
        );
        let l_small = vmulq_f32(y, vdupq_n_f32(LUV_KAPPA));
        let l = vbslq_f32(vcgtq_f32(y, vdupq_n_f32(LAB_DELTA)), l_big, l_small);
        // d = x + 15y + 3z; u'=4x/d, v'=9y/d (0 where d==0).
        let d = vfmaq_f32(vfmaq_f32(x, y, vdupq_n_f32(15.0)), z, vdupq_n_f32(3.0));
        let rd = nonlinear::vrecip_fast_f32x4(d);
        let up = vmulq_f32(vmulq_f32(x, vdupq_n_f32(4.0)), rd);
        let vp = vmulq_f32(vmulq_f32(y, vdupq_n_f32(9.0)), rd);
        let dz = vceqq_f32(d, vdupq_n_f32(0.0));
        let up = vbslq_f32(dz, vdupq_n_f32(0.0), up);
        let vp = vbslq_f32(dz, vdupq_n_f32(0.0), vp);
        let l13 = vmulq_f32(l, vdupq_n_f32(13.0));
        let u = vmulq_f32(l13, vsubq_f32(up, vdupq_n_f32(LUV_UN)));
        let v = vmulq_f32(l13, vsubq_f32(vp, vdupq_n_f32(LUV_VN)));
        float32x4x3_t(l, u, v)
    },
    luv_from_rgb_px
);
cie_kernel!(
    rgb_from_luv_f32,
    |p: float32x4x3_t| {
        let l = p.0;
        // Y: kappa branch for L<=8, cube branch otherwise.
        let t = vmulq_f32(vaddq_f32(l, vdupq_n_f32(16.0)), vdupq_n_f32(1.0 / 116.0));
        let y_big = vmulq_f32(vmulq_f32(t, t), t);
        let y_small = vmulq_f32(l, vdupq_n_f32(1.0 / LUV_KAPPA));
        let y = vbslq_f32(vcgtq_f32(l, vdupq_n_f32(8.0)), y_big, y_small);
        let inv13l = nonlinear::vrecip_fast_f32x4(vmulq_f32(l, vdupq_n_f32(13.0)));
        let up = vfmaq_f32(vdupq_n_f32(LUV_UN), p.1, inv13l);
        let vp = vfmaq_f32(vdupq_n_f32(LUV_VN), p.2, inv13l);
        let inv4vp = nonlinear::vrecip_fast_f32x4(vmulq_f32(vp, vdupq_n_f32(4.0)));
        let x = vmulq_f32(vmulq_f32(y, vmulq_f32(vdupq_n_f32(9.0), up)), inv4vp);
        // z = y*(12 - 3u' - 20v')/(4v')
        let zn = vsubq_f32(
            vsubq_f32(vdupq_n_f32(12.0), vmulq_f32(up, vdupq_n_f32(3.0))),
            vmulq_f32(vp, vdupq_n_f32(20.0)),
        );
        let z = vmulq_f32(vmulq_f32(y, zn), inv4vp);
        // L<=0 → black.
        let lpos = vcgtq_f32(l, vdupq_n_f32(0.0));
        let zero = vdupq_n_f32(0.0);
        let x = vbslq_f32(lpos, x, zero);
        let y = vbslq_f32(lpos, y, zero);
        let z = vbslq_f32(lpos, z, zero);
        let (lr, lg, lb) = matvec_v(&M_XYZ2RGB, x, y, z);
        float32x4x3_t(
            linear_to_srgb_v(lr),
            linear_to_srgb_v(lg),
            linear_to_srgb_v(lb),
        )
    },
    rgb_from_luv_px
);
