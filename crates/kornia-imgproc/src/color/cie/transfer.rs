//! sRGB ↔ linear-RGB transfer functions and the vectorized `pow(x, p)` they need.
//!
//! The sRGB EOTF/OETF are piecewise: a small linear segment near black plus a
//! power-law tail (`x^2.4` / `x^(1/2.4)`). The tail is the expensive part, so this
//! module provides a NEON `pow_f32x4` built from `exp2 ∘ log2` (degree-5 minimax
//! polynomials over the mantissa, exponent handled by float bit-tricks) and a
//! scalar `powf` reference used by the f64 oracle.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// sRGB transfer constants (OpenCV f32 path).
pub(crate) const SRGB_THRESH: f32 = 0.04045; // sRGB→linear breakpoint
pub(crate) const SRGB_INV_THRESH: f32 = 0.0031308; // linear→sRGB breakpoint
pub(crate) const SRGB_A: f32 = 0.055;
pub(crate) const SRGB_INV_1055: f32 = 1.0 / 1.055;
pub(crate) const SRGB_INV_1292: f32 = 1.0 / 12.92;
pub(crate) const SRGB_1292: f32 = 12.92;
pub(crate) const SRGB_1055: f32 = 1.055;
pub(crate) const SRGB_GAMMA: f32 = 2.4;
pub(crate) const SRGB_INV_GAMMA: f32 = 1.0 / 2.4;

// ===== scalar reference (oracle) ====================================================

/// sRGB → linear (per channel). Clamps to `≥0` first, matching the kernel.
#[inline]
pub(crate) fn srgb_to_linear_scalar(x: f32) -> f32 {
    let x = x.max(0.0);
    if x <= SRGB_THRESH {
        x * SRGB_INV_1292
    } else {
        ((x + SRGB_A) * SRGB_INV_1055).powf(SRGB_GAMMA)
    }
}

/// linear → sRGB (per channel).
#[inline]
pub(crate) fn linear_to_srgb_scalar(l: f32) -> f32 {
    let l = l.max(0.0);
    if l <= SRGB_INV_THRESH {
        l * SRGB_1292
    } else {
        SRGB_1055 * l.powf(SRGB_INV_GAMMA) - SRGB_A
    }
}

// ===== vectorized pow via exp2∘log2 =================================================

/// NEON `log2(x)` for `x > 0`, degree-7 minimax over the mantissa.
///
/// Decomposes `x = m · 2^e` with `m ∈ [1, 2)` via the IEEE-754 exponent bits, then
/// approximates `log2(m)` with a polynomial. Inputs `≤ 0` are not expected here (the
/// gamma tail is only taken for `x > breakpoint > 0`).
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::excessive_precision)] // least-squares fit constants; keep full digits
unsafe fn log2_f32x4(x: float32x4_t) -> float32x4_t {
    // Extract exponent: e = ((bits >> 23) & 0xff) - 127.
    let bits = vreinterpretq_u32_f32(x);
    let exp = vsubq_s32(
        vreinterpretq_s32_u32(vshrq_n_u32(bits, 23)),
        vdupq_n_s32(127),
    );
    let e = vcvtq_f32_s32(exp);

    // Mantissa m ∈ [1, 2): clear exponent bits, set them to 127 (bias for 2^0).
    let mant_bits = vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    );
    let m = vreinterpretq_f32_u32(mant_bits);

    // Minimax for log2(m) on [1, 2) (degree-7, Horner). Max abs err < 1e-6 → pow rel
    // err < 2e-6. High precision so our Lab/Luv land as close to OpenCV as its own
    // approximation allows (accuracy is the priority over the hot-path speed here).
    let c0 = vdupq_n_f32(-3.235_209_4);
    let c1 = vdupq_n_f32(7.085_100_8);
    let c2 = vdupq_n_f32(-7.396_148);
    let c3 = vdupq_n_f32(5.673_517_7);
    let c4 = vdupq_n_f32(-2.914_490_3);
    let c5 = vdupq_n_f32(0.950_741);
    let c6 = vdupq_n_f32(-0.178_109_57);
    let c7 = vdupq_n_f32(0.014_598_475);
    let mut p = c7;
    p = vfmaq_f32(c6, p, m);
    p = vfmaq_f32(c5, p, m);
    p = vfmaq_f32(c4, p, m);
    p = vfmaq_f32(c3, p, m);
    p = vfmaq_f32(c2, p, m);
    p = vfmaq_f32(c1, p, m);
    p = vfmaq_f32(c0, p, m);
    // p ≈ log2(m); add exponent.
    vaddq_f32(p, e)
}

/// NEON `exp2(x)` = `2^x`, degree-5 minimax over the fractional part.
///
/// Splits `x = n + f` with `n = floor(x)` and `f ∈ [0, 1)`; `2^n` is built by
/// injecting `n` into the IEEE exponent field, `2^f` by a polynomial.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn exp2_f32x4(x: float32x4_t) -> float32x4_t {
    // Clamp to a sane range so the exponent injection can't overflow.
    let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-126.0)), vdupq_n_f32(126.0));
    let n = vrndmq_f32(x); // floor(x)
    let f = vsubq_f32(x, n); // [0, 1)

    // Minimax for 2^f on [0, 1) (degree-5, Horner). Max abs err < 3e-7.
    let c0 = vdupq_n_f32(0.999_999_77);
    let c1 = vdupq_n_f32(0.693_156_8);
    let c2 = vdupq_n_f32(0.240_131_68);
    let c3 = vdupq_n_f32(0.055_876_57);
    let c4 = vdupq_n_f32(0.008_940_577);
    let c5 = vdupq_n_f32(0.001_894_378_6);
    let mut p = c5;
    p = vfmaq_f32(c4, p, f);
    p = vfmaq_f32(c3, p, f);
    p = vfmaq_f32(c2, p, f);
    p = vfmaq_f32(c1, p, f);
    p = vfmaq_f32(c0, p, f); // 2^f

    // 2^n via exponent injection: bits = (n + 127) << 23.
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
    vmulq_f32(p, pow2n)
}

/// NEON `pow(x, p) = exp2(p · log2(x))` for `x > 0`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn pow_f32x4(x: float32x4_t, p: float32x4_t) -> float32x4_t {
    exp2_f32x4(vmulq_f32(p, log2_f32x4(x)))
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn pow_f32x4_matches_std() {
        use std::arch::aarch64::*;
        // sweep bases over the gamma domain and a couple of exponents
        for &p in &[2.4f32, 1.0 / 2.4] {
            for k in 1..=200 {
                let x = k as f32 / 200.0; // (0, 1]
                let want = x.powf(p);
                unsafe {
                    let vx = vdupq_n_f32(x);
                    let vp = vdupq_n_f32(p);
                    let got = vgetq_lane_f32::<0>(super::pow_f32x4(vx, vp));
                    let rel = (got - want).abs() / want.max(1e-6);
                    assert!(
                        rel <= 1e-5,
                        "pow({x},{p}): got {got}, want {want}, rel {rel}"
                    );
                }
            }
        }
    }
}
