//! M-estimator robust kernels.
//!
//! These are pluggable weight functions that downweight outlier residuals
//! during iterated least-squares refinement (LM normal equations, BA, the
//! eventual MAGSAC++ scorer). Each kernel takes a *squared residual* and a
//! *squared scale* (the kernel's transition point, sometimes called `c²`)
//! and returns a non-negative weight in `[0, 1]`, plus the robust objective
//! `ρ` that weight is the derivative of.
//!
//! The squared-units convention matches the rest of `kornia-3d`
//! (`sampson_distance`, transfer error, reprojection RMSE all return
//! squared quantities), so call sites don't have to take a square root in
//! the hot loop just to apply the kernel.
//!
//! # Choosing a kernel
//!
//! - [`IdentityKernel`] — weight ≡ 1. No robustification. Use as the
//!   default in places that already round through a hard inlier threshold
//!   (RANSAC) where soft weighting would only cost FLOPs.
//! - [`HuberKernel`] — quadratic inside `c`, linear outside. Bounded
//!   *influence*. Good first choice when you don't know the outlier mix.
//! - [`CauchyKernel`] — strictly redescending; outliers contribute less
//!   the further they are from `c`, but never zero. Tolerates moderate
//!   outliers without collapsing.
//! - [`TukeyKernel`] — hard redescender, weight reaches 0 at `c`. Best for
//!   gross outliers but needs a decent initialisation or it can lock into
//!   a wrong basin.

/// A pluggable robust weight function and its objective.
///
/// `weight(r², c²)` returns a non-negative scalar in `[0, 1]` that scales
/// the contribution of a residual to a quadratic loss. Multiplying both
/// sides of the normal equations by this weight implements iteratively
/// reweighted least squares (IRLS).
///
/// `rho(r², c²)` is the objective those weights minimise: the cost of a
/// residual is `½ρ(r²)`, and its gradient is the weighted `w·r` of the IRLS
/// normal equations. Use it wherever the cost itself is compared, e.g. to
/// accept or reject a Levenberg–Marquardt step.
pub trait RobustKernel {
    /// Compute the weight for a residual whose squared magnitude is `r_sq`,
    /// given the kernel's squared scale parameter `c_sq`.
    fn weight(&self, r_sq: f64, c_sq: f64) -> f64;

    /// Robust objective `ρ(s)` at `s = r_sq`, in squared units, given the
    /// kernel's squared scale parameter `c_sq`.
    ///
    /// Defined by `ρ(0) = 0` and `dρ/ds = weight(s, c_sq)`, so `ρ(s) ≤ s`
    /// and `ρ(s) = s` while the weight stays 1 (Huber inside the knee, and
    /// every kernel at a degenerate `c² ≤ 0` or infinite scale).
    ///
    /// The reweighted cost `½·w·s` is NOT this objective: for Huber past
    /// the knee it is `½√(c² s)`, with half the slope of `½ρ`.
    ///
    /// # Arguments
    ///
    /// * `r_sq` - Squared residual magnitude `s`.
    /// * `c_sq` - Squared scale of the kernel, in the units of `r_sq`.
    ///
    /// # Returns
    ///
    /// The objective `ρ(s)` in the units of `r_sq`; the cost is `½ρ(s)`.
    ///
    /// # Errors
    ///
    /// None: `rho` is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::ransac::{HuberKernel, RobustKernel};
    ///
    /// assert_eq!(HuberKernel.rho(16.0, 25.0), 16.0);
    /// assert_eq!(HuberKernel.rho(100.0, 25.0), 75.0);
    /// ```
    fn rho(&self, r_sq: f64, c_sq: f64) -> f64;
}

/// Pass-through kernel: every residual gets weight 1.
///
/// Use to disable robustification at the call site without branching on
/// `Option<&dyn RobustKernel>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityKernel;

impl RobustKernel for IdentityKernel {
    #[inline]
    fn weight(&self, _r_sq: f64, _c_sq: f64) -> f64 {
        1.0
    }

    #[inline]
    fn rho(&self, r_sq: f64, _c_sq: f64) -> f64 {
        r_sq
    }
}

/// Huber's M-estimator: quadratic loss inside `|r| ≤ c`, linear outside.
///
/// Weight: `1` if `r² ≤ c²` else `c / |r|`. Implemented via squared
/// quantities to avoid a `sqrt` in the inlier branch.
///
/// Objective: `ρ = r²` if `r² ≤ c²` else `2·c·|r| − c²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuberKernel;

impl RobustKernel for HuberKernel {
    #[inline]
    fn weight(&self, r_sq: f64, c_sq: f64) -> f64 {
        if r_sq <= c_sq {
            1.0
        } else if c_sq <= 0.0 {
            // Degenerate scale → treat all residuals as inliers; matches
            // IdentityKernel behaviour rather than NaN'ing.
            1.0
        } else {
            // c / |r| = sqrt(c² / r²). One sqrt instead of two.
            (c_sq / r_sq).sqrt()
        }
    }

    #[inline]
    fn rho(&self, r_sq: f64, c_sq: f64) -> f64 {
        if r_sq <= c_sq || c_sq <= 0.0 {
            r_sq
        } else {
            // `c·|r|` from separate roots, and no doubling: `c² · r²` and
            // `2·c·|r|` can leave the f64 range while `ρ` is still inside it.
            let c_r = c_sq.sqrt() * r_sq.sqrt();
            c_r + (c_r - c_sq)
        }
    }
}

/// Cauchy / Lorentzian kernel — strictly redescending.
///
/// Weight: `1 / (1 + r²/c²)`. Smooth; never reaches zero. Good middle
/// ground between Huber (bounded influence) and Tukey (hard rejection).
///
/// Objective: `ρ = c² · ln(1 + r²/c²)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CauchyKernel;

impl RobustKernel for CauchyKernel {
    #[inline]
    fn weight(&self, r_sq: f64, c_sq: f64) -> f64 {
        if c_sq <= 0.0 {
            return 1.0;
        }
        1.0 / (1.0 + r_sq / c_sq)
    }

    #[inline]
    fn rho(&self, r_sq: f64, c_sq: f64) -> f64 {
        // `c² = ∞` has weight 1, but `∞ · ln_1p(0)` is NaN.
        if c_sq <= 0.0 || c_sq == f64::INFINITY {
            return r_sq;
        }
        let t = r_sq / c_sq;
        if t < f64::EPSILON {
            // `ln_1p(t) = t` to rounding, and `t` may have underflowed.
            r_sq
        } else if t == f64::INFINITY {
            c_sq * (r_sq.ln() - c_sq.ln())
        } else {
            c_sq * t.ln_1p()
        }
    }
}

/// Tukey biweight — hard redescender. Weight reaches 0 at `|r| = c`.
///
/// Weight: `(1 - r²/c²)²` for `r² ≤ c²`, else `0`. Outliers beyond `c`
/// contribute zero gradient — makes Tukey effective at gross outliers but
/// sensitive to the initial estimate (a bad initialisation can leave good
/// observations marked as zero-weight).
///
/// Objective: `ρ = (c²/3)·(1 − (1 − r²/c²)³)` for `r² ≤ c²`, else `c²/3`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TukeyKernel;

impl RobustKernel for TukeyKernel {
    #[inline]
    fn weight(&self, r_sq: f64, c_sq: f64) -> f64 {
        if c_sq <= 0.0 || r_sq >= c_sq {
            if c_sq <= 0.0 {
                return 1.0;
            }
            return 0.0;
        }
        let u = 1.0 - r_sq / c_sq;
        u * u
    }

    #[inline]
    fn rho(&self, r_sq: f64, c_sq: f64) -> f64 {
        if c_sq <= 0.0 {
            return r_sq;
        }
        if r_sq >= c_sq {
            return c_sq / 3.0;
        }
        // Expanded in t = r²/c² to avoid cancellation in `1 − u³` near 0;
        // this form also gives `r²` (not NaN) for `c² = ∞`.
        let t = r_sq / c_sq;
        r_sq * (1.0 - t + t * t / 3.0)
    }
}

/// Tagged enum dispatch for use in config structs that don't want the
/// `dyn`-trait overhead. Maps to one of the concrete kernels above.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RobustKernelKind {
    /// No robustification (weight ≡ 1).
    #[default]
    Identity,
    /// Huber's bounded-influence kernel.
    Huber,
    /// Cauchy / Lorentzian smooth redescender.
    Cauchy,
    /// Tukey biweight (hard redescender).
    Tukey,
}

impl RobustKernel for RobustKernelKind {
    #[inline]
    fn weight(&self, r_sq: f64, c_sq: f64) -> f64 {
        match self {
            RobustKernelKind::Identity => IdentityKernel.weight(r_sq, c_sq),
            RobustKernelKind::Huber => HuberKernel.weight(r_sq, c_sq),
            RobustKernelKind::Cauchy => CauchyKernel.weight(r_sq, c_sq),
            RobustKernelKind::Tukey => TukeyKernel.weight(r_sq, c_sq),
        }
    }

    #[inline]
    fn rho(&self, r_sq: f64, c_sq: f64) -> f64 {
        match self {
            RobustKernelKind::Identity => IdentityKernel.rho(r_sq, c_sq),
            RobustKernelKind::Huber => HuberKernel.rho(r_sq, c_sq),
            RobustKernelKind::Cauchy => CauchyKernel.rho(r_sq, c_sq),
            RobustKernelKind::Tukey => TukeyKernel.rho(r_sq, c_sq),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At zero residual, every kernel weights 1 (or near 1 for asymptotic
    /// kernels) — outliers don't exist at the centre.
    #[test]
    fn weight_at_zero_is_one_for_all() {
        let c_sq = 1.0;
        assert!((IdentityKernel.weight(0.0, c_sq) - 1.0).abs() < 1e-15);
        assert!((HuberKernel.weight(0.0, c_sq) - 1.0).abs() < 1e-15);
        assert!((CauchyKernel.weight(0.0, c_sq) - 1.0).abs() < 1e-15);
        assert!((TukeyKernel.weight(0.0, c_sq) - 1.0).abs() < 1e-15);
    }

    /// Tukey is the only kernel that hits exactly 0 outside its scale.
    #[test]
    fn tukey_is_hard_redescender() {
        assert_eq!(TukeyKernel.weight(1.0, 1.0), 0.0);
        assert_eq!(TukeyKernel.weight(4.0, 1.0), 0.0);
    }

    /// Cauchy never hits zero, only asymptotes — even at 100x scale.
    #[test]
    fn cauchy_never_hits_zero() {
        let w = CauchyKernel.weight(10000.0, 1.0);
        assert!(w > 0.0, "Cauchy weight should be strictly positive: {w}");
        assert!(
            w < 1e-3,
            "Cauchy weight should be small at far residuals: {w}"
        );
    }

    /// Huber weight at the transition equals 1 (boundary inclusive).
    #[test]
    fn huber_weight_continuous_at_transition() {
        let c_sq = 4.0;
        let w_in = HuberKernel.weight(c_sq - 1e-12, c_sq);
        let w_out = HuberKernel.weight(c_sq + 1e-12, c_sq);
        assert!((w_in - 1.0).abs() < 1e-9);
        assert!((w_out - 1.0).abs() < 1e-6);
    }

    /// Weights are monotonically non-increasing with residual magnitude
    /// for every kernel except Identity. Spot-check at several points.
    #[test]
    fn weights_monotonic_in_residual() {
        let c_sq = 1.0;
        for kernel in [
            RobustKernelKind::Huber,
            RobustKernelKind::Cauchy,
            RobustKernelKind::Tukey,
        ] {
            let pts = [0.1, 0.5, 0.9, 1.5, 4.0, 16.0];
            let mut prev = f64::INFINITY;
            for &r_sq in &pts {
                let w = kernel.weight(r_sq, c_sq);
                assert!(
                    w <= prev + 1e-12,
                    "{kernel:?} not monotonic at r²={r_sq}: w={w}, prev={prev}"
                );
                prev = w;
            }
        }
    }

    /// All kernels return weights in [0, 1].
    #[test]
    fn weights_bounded() {
        let c_sq = 1.0;
        for kernel in [
            RobustKernelKind::Identity,
            RobustKernelKind::Huber,
            RobustKernelKind::Cauchy,
            RobustKernelKind::Tukey,
        ] {
            for &r_sq in &[0.0, 0.01, 0.5, 1.0, 1.5, 100.0] {
                let w = kernel.weight(r_sq, c_sq);
                assert!(
                    (0.0..=1.0 + 1e-12).contains(&w),
                    "{kernel:?} produced out-of-range weight {w} at r²={r_sq}"
                );
            }
        }
    }

    /// Degenerate scale (c² ≤ 0) doesn't NaN any kernel.
    #[test]
    fn degenerate_scale_is_safe() {
        for kernel in [
            RobustKernelKind::Identity,
            RobustKernelKind::Huber,
            RobustKernelKind::Cauchy,
            RobustKernelKind::Tukey,
        ] {
            let w = kernel.weight(1.0, 0.0);
            assert!(w.is_finite(), "{kernel:?} produced non-finite weight: {w}");
        }
    }

    const ALL_KERNELS: [RobustKernelKind; 4] = [
        RobustKernelKind::Identity,
        RobustKernelKind::Huber,
        RobustKernelKind::Cauchy,
        RobustKernelKind::Tukey,
    ];

    /// Squared residuals on both sides of the knee at `c² = 4`.
    const RHO_SAMPLES: [f64; 10] = [0.01, 0.5, 1.0, 2.0, 3.5, 3.99, 4.01, 6.0, 25.0, 100.0];

    #[test]
    fn rho_at_zero_is_zero() {
        for kernel in ALL_KERNELS {
            assert_eq!(kernel.rho(0.0, 4.0), 0.0, "{kernel:?}");
        }
    }

    /// `dρ/ds = weight(s)`, checked by central differences.
    #[test]
    fn rho_derivative_is_weight() {
        let c_sq = 4.0;
        let h = 1e-6;
        for kernel in ALL_KERNELS {
            for &s in &RHO_SAMPLES {
                let d_rho = (kernel.rho(s + h, c_sq) - kernel.rho(s - h, c_sq)) / (2.0 * h);
                let w = kernel.weight(s, c_sq);
                assert!(
                    (d_rho - w).abs() < 1e-6,
                    "{kernel:?} at s={s}: dρ/ds={d_rho}, weight={w}"
                );
            }
        }
    }

    #[test]
    fn rho_continuous_at_knee() {
        let c_sq = 4.0;
        let eps = 1e-9;
        for kernel in [RobustKernelKind::Huber, RobustKernelKind::Tukey] {
            let inside = kernel.rho(c_sq - eps, c_sq);
            let at = kernel.rho(c_sq, c_sq);
            let outside = kernel.rho(c_sq + eps, c_sq);
            assert!((inside - at).abs() < 1e-8, "{kernel:?}: {inside} vs {at}");
            assert!((outside - at).abs() < 1e-8, "{kernel:?}: {outside} vs {at}");
        }
    }

    /// Closed-form values of Ceres' `HuberLoss` and `CauchyLoss` (`a² = c²`),
    /// and twice its `TukeyLoss`, which is normalised to `ρ'(0) = ½`.
    #[test]
    fn rho_reference_values() {
        assert!((HuberKernel.rho(100.0, 25.0) - 75.0).abs() < 1e-12);
        assert!((HuberKernel.rho(16.0, 25.0) - 16.0).abs() < 1e-12);
        assert!((CauchyKernel.rho(4.0, 4.0) - 4.0 * std::f64::consts::LN_2).abs() < 1e-12);
        assert!((TukeyKernel.rho(2.0, 4.0) - 7.0 / 6.0).abs() < 1e-12);
    }

    /// Scales where `c² · r²` or `r² / c²` leave the f64 range while `ρ`
    /// does not.
    #[test]
    fn rho_is_finite_at_extreme_scales() {
        let close = |rho: f64, expected: f64| (rho - expected).abs() <= 1e-12 * expected;
        assert!(close(HuberKernel.rho(4e200, 1e200), 3e200));
        assert!(close(HuberKernel.rho(4e-200, 1e-200), 3e-200));
        let ln_ratio = 400.0 * std::f64::consts::LN_10;
        assert!(close(CauchyKernel.rho(1e200, 1e-200), 1e-200 * ln_ratio));
        assert!(close(CauchyKernel.rho(1e-200, 1e200), 1e-200));
    }

    /// Tukey saturates at `c²/3` beyond the scale.
    #[test]
    fn tukey_rho_saturates() {
        assert!((TukeyKernel.rho(4.0, 4.0) - 4.0 / 3.0).abs() < 1e-12);
        assert!((TukeyKernel.rho(100.0, 4.0) - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rho_bounded_by_squared_residual() {
        let c_sq = 4.0;
        for kernel in ALL_KERNELS {
            for &s in &RHO_SAMPLES {
                let rho = kernel.rho(s, c_sq);
                assert!(rho <= s + 1e-12, "{kernel:?} at s={s}: ρ={rho} > s");
                assert!(rho >= 0.0, "{kernel:?} at s={s}: ρ={rho} < 0");
            }
        }
    }

    /// A degenerate (`c² ≤ 0`) or infinite scale weights every residual 1,
    /// so `ρ(s) = s`, never NaN.
    #[test]
    fn rho_degenerate_and_infinite_scale_is_squared_residual() {
        for kernel in ALL_KERNELS {
            for c_sq in [0.0, -1.0, f64::INFINITY] {
                for s in [0.0, 0.5, 4.0, 1e6] {
                    let rho = kernel.rho(s, c_sq);
                    assert_eq!(rho, s, "{kernel:?} at s={s}, c²={c_sq}: ρ={rho}");
                }
            }
        }
    }
}
