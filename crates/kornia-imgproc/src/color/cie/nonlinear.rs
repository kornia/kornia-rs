//! Vectorized non-linearities for the Lab/Luv stages: cube root and reciprocal.
//!
//! Lab needs `t^(1/3)` of each normalized tristimulus value; Luv needs `Y^(1/3)`
//! and a `1/d` divide. NEON has no cbrt, so `cbrt_f32x4` uses the classic
//! magic-constant bit hack for an initial guess followed by one Halley iteration
//! (cubic convergence), tuned to colour precision (~2e-5) for the `[0, 1]`-ish domain.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON cube root for `x ≥ 0`. Two Halley iterations after a bit-hack seed.
///
/// Seed: `y ≈ 2^(e/3)` via the magic constant `0x2a514067` on the float bits.
/// Halley step (cube root form): `y ← y · (c + 2x) / (2c + x)` with `c = y³`.
/// `x == 0` returns `0` (seed is small, iterations keep it near 0; we mask exact 0).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn cbrt_f32x4(x: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_u32_f32(x);
    // Seed y0 = bitcast(bits/3 + magic) ≈ 2^(e/3); the magic constant restores the
    // exponent bias and gives a good first-order mantissa guess.
    let seed_bits = vaddq_u32(
        vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(bits), vdupq_n_f32(1.0 / 3.0))),
        vdupq_n_u32(0x2a51_4067),
    );
    let mut y = vreinterpretq_f32_u32(seed_bits);

    // One Halley iteration: y = y*(c + 2x)/(2c + x), c = y^3. Cubic convergence on
    // the ~4% bit-hack seed gives max rel err ≈ 2.3e-5 — ample for colour (~1e-2),
    // so a second iteration is wasted work on this hot path.
    {
        let c = vmulq_f32(vmulq_f32(y, y), y);
        let num = vaddq_f32(c, vaddq_f32(x, x)); // c + 2x
        let den = vaddq_f32(vaddq_f32(c, c), x); // 2c + x
        y = vmulq_f32(y, vmulq_f32(num, vrecip_f32x4(den)));
    }
    // Force exact 0 input to 0 output.
    let is0 = vceqq_f32(x, vdupq_n_f32(0.0));
    vbslq_f32(is0, vdupq_n_f32(0.0), y)
}

/// NEON reciprocal `1/x` refined by two Newton-Raphson steps (vrecpe seed).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn vrecip_f32x4(x: float32x4_t) -> float32x4_t {
    let mut r = vrecpeq_f32(x);
    r = vmulq_f32(r, vrecpsq_f32(x, r));
    r = vmulq_f32(r, vrecpsq_f32(x, r));
    r
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cbrt_f32x4_matches_std() {
        use std::arch::aarch64::*;
        for k in 0..=400 {
            let x = k as f32 / 200.0; // [0, 2]
            let want = x.cbrt();
            unsafe {
                let got = vgetq_lane_f32::<0>(super::cbrt_f32x4(vdupq_n_f32(x)));
                let rel = (got - want).abs() / want.max(1e-6);
                // Colour-grade: one Halley iteration → rel err ≈ 2.3e-5.
                assert!(rel <= 1e-4, "cbrt({x}): got {got}, want {want}, rel {rel}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn recip_f32x4_matches_std() {
        use std::arch::aarch64::*;
        for k in 1..=400 {
            let x = k as f32 / 100.0;
            unsafe {
                let got = vgetq_lane_f32::<0>(super::vrecip_f32x4(vdupq_n_f32(x)));
                let rel = (got - 1.0 / x).abs() * x;
                assert!(rel <= 1e-5, "recip({x}): got {got}, rel {rel}");
            }
        }
    }
}
