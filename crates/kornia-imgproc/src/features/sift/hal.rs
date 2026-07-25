//! The three scalar-math primitives the orientation and descriptor stages
//! depend on, reproduced bit for bit.
//!
//! On this platform the reference's `exp32f`, `fastAtan2` and `magnitude32f` are
//! overridden by an accelerated backend built on ARM's *estimate* instructions —
//! `FRECPE`/`FRECPS` and `FRSQRTE`/`FRSQRTS` — not on exact division or square
//! root. `atan2f`, `expf` and `sqrtf` do not reproduce them.
//!
//! The CUDA port had to emulate those instructions with baked lookup tables and
//! integer exponent assembly. Here they are simply *called*: this is aarch64, so
//! `vrecpeq_f32` and friends are the hardware, which also means the zero and
//! infinity special cases come free rather than needing to be hand-written (a
//! missing one of those was a real bug on the GPU side).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Reciprocal: estimate plus exactly two Newton-Raphson steps.
///
/// Two, not three: measured against the reference, one step leaves 2540
/// mismatches per 4096, two leaves 0, and three leaves 262. More refinement is
/// *worse*, because the target is the backend's approximation, not `1/x`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn recip(v: f32) -> f32 {
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        let x = vdupq_n_f32(v);
        let mut r = vrecpeq_f32(x);
        r = vmulq_f32(vrecpsq_f32(x, r), r);
        r = vmulq_f32(vrecpsq_f32(x, r), r);
        vgetq_lane_f32(r, 0)
    }
}

/// Reciprocal square root: estimate plus exactly two refinement steps.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn rsqrt(v: f32) -> f32 {
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        let x = vdupq_n_f32(v);
        let mut e = vrsqrteq_f32(x);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), x), e);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), x), e);
        vgetq_lane_f32(e, 0)
    }
}

/// Magnitude, as the backend computes it: `recip(rsqrt(x*x + y*y))`.
///
/// It never issues a square root. The sum must be a single fused
/// multiply-add — computing it with two roundings disagrees on ~7% of inputs.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn magnitude(x: f32, y: f32) -> f32 {
    recip(rsqrt(x.mul_add(x, y * y)))
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn magnitude(x: f32, y: f32) -> f32 {
    x.mul_add(x, y * y).sqrt()
}

/// Angle in degrees, clockwise from +x, in `[0, 360)`.
///
/// The polynomial must be a **fused Horner chain**. The backend's own source
/// spells it as separate multiplies and adds, but only the fused form reproduces
/// the reference's bits — they differ by one ULP, which was a long-standing
/// residual on the CUDA side. Constants are products evaluated in `f64` then
/// narrowed; computing them from `f32` literals changes `p1`'s last bit.
#[inline]
pub fn atan2_deg(y: f32, x: f32) -> f32 {
    let AtanConsts {
        p1,
        p3,
        p5,
        p7,
        eps,
    } = *ATAN_C;

    let (ax, ay) = (x.abs(), y.abs());
    let (tmin, tmax) = (ax.min(ay), ax.max(ay));
    #[cfg(target_arch = "aarch64")]
    let c = tmin * recip(tmax + eps);
    #[cfg(not(target_arch = "aarch64"))]
    let c = tmin / (tmax + eps);
    let c2 = c * c;

    let mut a = c2.mul_add(p7, p5);
    a = a.mul_add(c2, p3);
    a = a.mul_add(c2, p1);
    a *= c;

    // Deliberately the negated form, not `ax < ay`: the reference spells it this
    // way and the two differ when either operand is NaN, which decides the
    // quadrant fixup. Not a readability wart.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(ax >= ay) {
        a = 90.0 - a;
    }
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}

/// `2^(i/64) * A0`, where `A0` is the constant the polynomial's coefficients are
/// divided by. The factor cancels algebraically, but the intermediate rounding
/// does not, so it has to be kept.
///
/// Built **once**. Rebuilding it per call means 64 `powf` evaluations for every
/// exponential, and this is called once per patch sample — tens of millions of
/// times an image. That is the same mistake, in a different guise, as the baked
/// tables that ended up in per-thread local memory on the CUDA side.
pub(crate) static EXP_TAB: std::sync::LazyLock<[f32; 64]> = std::sync::LazyLock::new(|| {
    const A0: f64 = 0.009_670_371_139_572_338;
    let mut t = [0.0f32; 64];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = (2f64.powf(i as f64 / 64.0) * A0) as f32;
    }
    t
});

/// The exponential's polynomial coefficients and range, also built once.
pub(crate) struct ExpConsts {
    pub(crate) a1: f32,
    pub(crate) a2: f32,
    pub(crate) a3: f32,
    pub(crate) a4: f32,
    pub(crate) prescale: f32,
    pub(crate) postscale: f32,
    pub(crate) minval: f32,
    pub(crate) maxval: f32,
}

pub(crate) static EXP_C: std::sync::LazyLock<ExpConsts> = std::sync::LazyLock::new(|| {
    const A0: f64 = 0.009_670_371_139_572_338;
    let pre = std::f64::consts::LOG2_E * 64.0;
    let exp_max = 3000.0f64 * 64.0;
    ExpConsts {
        a1: (0.055_503_393_667_531_25 / A0) as f32,
        a2: (0.240_226_510_951_330_15 / A0) as f32,
        a3: (0.693_147_180_552_144_8 / A0) as f32,
        a4: (1.0f64 / A0) as f32,
        prescale: pre as f32,
        postscale: (1.0f64 / 64.0) as f32,
        minval: (-exp_max / pre) as f32,
        maxval: (exp_max / pre) as f32,
    }
});

/// The angle polynomial's constants: products evaluated in `f64` then narrowed.
pub(crate) struct AtanConsts {
    pub(crate) p1: f32,
    pub(crate) p3: f32,
    pub(crate) p5: f32,
    pub(crate) p7: f32,
    pub(crate) eps: f32,
}

pub(crate) static ATAN_C: std::sync::LazyLock<AtanConsts> = std::sync::LazyLock::new(|| {
    const D: f64 = 180.0 / std::f64::consts::PI;
    AtanConsts {
        p1: (0.999_787_841_279_480_7 * D) as f32,
        p3: (-0.325_808_397_464_097_5 * D) as f32,
        p5: (0.155_578_651_846_328_1 * D) as f32,
        p7: (-0.044_326_555_547_921_28 * D) as f32,
        eps: f64::EPSILON as f32,
    }
});

/// The reference's exponential: 64-entry table, exponent injection, and a
/// four-term FMA polynomial. Not `expf`.
#[inline]
pub fn exp(x: f32) -> f32 {
    let c = &*EXP_C;
    let (a1, a2, a3, a4) = (c.a1, c.a2, c.a3, c.a4);
    let tab = &*EXP_TAB;
    let mut x = x.clamp(c.minval, c.maxval);
    x *= c.prescale;
    let xi = x.round_ties_even() as i32;
    let xf = (x - xi as f32) * c.postscale;
    let mut yf = tab[(xi & 63) as usize];
    let t = ((xi >> 6) + 127).clamp(0, 255) as u32;
    yf *= f32::from_bits(t << 23);

    let mut z = xf + a1;
    z = z.mul_add(xf, a2);
    z = z.mul_add(xf, a3);
    z = z.mul_add(xf, a4);
    z * yf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compared against vectors dumped from the reference's own entry points —
    /// never against a language binding, which may resolve to a different
    /// accelerated backend and give different bits.
    #[test]
    fn hal_matches_reference_bitwise() {
        let Ok(path) = std::env::var("KORNIA_SIFT_HALREF") else {
            eprintln!("KORNIA_SIFT_HALREF unset; skipping");
            return;
        };
        let b = std::fs::read(&path).expect("halref");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let f = |off: usize, i: usize| {
            let o = off + i * 4;
            f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
        };
        let (xo, yo, eo, ao) = (4, 4 + 4 * n, 4 + 8 * n, 4 + 12 * n);

        let mut bad_exp = 0;
        let mut bad_atan = 0;
        for i in 0..n {
            let (x, y) = (f(xo, i), f(yo, i));
            if exp(x).to_bits() != f(eo, i).to_bits() {
                bad_exp += 1;
            }
            if atan2_deg(y, x).to_bits() != f(ao, i).to_bits() {
                bad_atan += 1;
            }
        }
        eprintln!("  cpu hal: exp {bad_exp}/{n} atan2 {bad_atan}/{n} mismatched");
        assert_eq!(bad_exp, 0, "exp differs from the reference");
        assert_eq!(bad_atan, 0, "atan2 differs from the reference");
    }
}

/// Four-lane forms of the three primitives.
///
/// Lane-wise identical to the scalar versions — the same instructions, four at a
/// time — so a caller can vectorise the *evaluation* of a patch while still
/// scattering the results sequentially, and stay bit-exact. That split is what
/// makes the descriptor and orientation loops vectorisable at all: their
/// accumulation order is fixed, but the work feeding it is not.
#[cfg(target_arch = "aarch64")]
pub mod x4 {
    use super::{ATAN_C, EXP_C, EXP_TAB};
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn recip(x: float32x4_t) -> float32x4_t {
        let mut r = vrecpeq_f32(x);
        r = vmulq_f32(vrecpsq_f32(x, r), r);
        r = vmulq_f32(vrecpsq_f32(x, r), r);
        r
    }

    #[inline]
    unsafe fn rsqrt(x: float32x4_t) -> float32x4_t {
        let mut e = vrsqrteq_f32(x);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), x), e);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), x), e);
        e
    }

    /// `recip(rsqrt(x*x + y*y))`, with the sum as one fused multiply-add.
    ///
    /// # Safety
    /// Requires NEON, which is baseline on aarch64.
    #[inline]
    pub unsafe fn magnitude(x: float32x4_t, y: float32x4_t) -> float32x4_t {
        recip(rsqrt(vfmaq_f32(vmulq_f32(y, y), x, x)))
    }

    /// Angle in degrees, clockwise from +x, in `[0, 360)`.
    ///
    /// # Safety
    /// Requires NEON, which is baseline on aarch64.
    #[inline]
    pub unsafe fn atan2_deg(y: float32x4_t, x: float32x4_t) -> float32x4_t {
        let c = &*ATAN_C;
        let ax = vabsq_f32(x);
        let ay = vabsq_f32(y);
        let tmin = vminq_f32(ax, ay);
        let tmax = vmaxq_f32(ax, ay);
        let cc = vmulq_f32(tmin, recip(vaddq_f32(tmax, vdupq_n_f32(c.eps))));
        let c2 = vmulq_f32(cc, cc);

        let mut a = vfmaq_n_f32(vdupq_n_f32(c.p5), c2, c.p7);
        a = vfmaq_f32(vdupq_n_f32(c.p3), a, c2);
        a = vfmaq_f32(vdupq_n_f32(c.p1), a, c2);
        a = vmulq_f32(a, cc);

        // `!(ax >= ay)` — the negated form, matching the reference, which
        // differs from `ax < ay` when an operand is NaN.
        let swap = vmvnq_u32(vcgeq_f32(ax, ay));
        a = vbslq_f32(swap, vsubq_f32(vdupq_n_f32(90.0), a), a);
        let negx = vcltq_f32(x, vdupq_n_f32(0.0));
        a = vbslq_f32(negx, vsubq_f32(vdupq_n_f32(180.0), a), a);
        let negy = vcltq_f32(y, vdupq_n_f32(0.0));
        vbslq_f32(negy, vsubq_f32(vdupq_n_f32(360.0), a), a)
    }

    /// Table-and-polynomial exponential.
    ///
    /// The table index is data-dependent, so those four lookups stay scalar;
    /// everything around them is vector.
    ///
    /// # Safety
    /// Requires NEON, which is baseline on aarch64.
    #[inline]
    pub unsafe fn exp(x: float32x4_t) -> float32x4_t {
        let c = &*EXP_C;
        let tab = &*EXP_TAB;
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(c.minval)), vdupq_n_f32(c.maxval));
        let x = vmulq_f32(x, vdupq_n_f32(c.prescale));
        let xi = vcvtnq_s32_f32(x); // round to nearest, ties to even
        let xf = vmulq_n_f32(vsubq_f32(x, vcvtq_f32_s32(xi)), c.postscale);

        let mut idx = [0i32; 4];
        vst1q_s32(idx.as_mut_ptr(), xi);
        let mut yf = [0.0f32; 4];
        for (l, &i) in idx.iter().enumerate() {
            let t = ((i >> 6) + 127).clamp(0, 255) as u32;
            yf[l] = tab[(i & 63) as usize] * f32::from_bits(t << 23);
        }
        let yf = vld1q_f32(yf.as_ptr());

        let mut z = vaddq_f32(xf, vdupq_n_f32(c.a1));
        z = vfmaq_f32(vdupq_n_f32(c.a2), z, xf);
        z = vfmaq_f32(vdupq_n_f32(c.a3), z, xf);
        z = vfmaq_f32(vdupq_n_f32(c.a4), z, xf);
        vmulq_f32(z, yf)
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod x4_tests {
    use super::*;
    use std::arch::aarch64::{vld1q_f32, vmulq_n_f32, vst1q_f32};

    /// The four-lane forms must be lane-wise identical to the scalar ones —
    /// that identity is what lets the descriptor vectorise its evaluation while
    /// keeping the reference's scatter order.
    #[test]
    fn x4_matches_scalar_bitwise() {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 33) as f32 / (1u32 << 31) as f32 - 0.5) * 400.0
        };
        for _ in 0..4096 {
            let xs = [next(), next(), next(), next()];
            let ys = [next(), next(), next(), next()];
            // SAFETY: NEON is baseline on aarch64.
            unsafe {
                let vx = vld1q_f32(xs.as_ptr());
                let vy = vld1q_f32(ys.as_ptr());
                let mut m = [0.0f32; 4];
                let mut a = [0.0f32; 4];
                let mut e = [0.0f32; 4];
                vst1q_f32(m.as_mut_ptr(), x4::magnitude(vx, vy));
                vst1q_f32(a.as_mut_ptr(), x4::atan2_deg(vy, vx));
                vst1q_f32(e.as_mut_ptr(), x4::exp(vmulq_n_f32(vx, 0.01)));
                for l in 0..4 {
                    assert_eq!(
                        m[l].to_bits(),
                        magnitude(xs[l], ys[l]).to_bits(),
                        "magnitude"
                    );
                    assert_eq!(a[l].to_bits(), atan2_deg(ys[l], xs[l]).to_bits(), "atan2");
                    assert_eq!(e[l].to_bits(), exp(xs[l] * 0.01).to_bits(), "exp");
                }
            }
        }
    }
}
