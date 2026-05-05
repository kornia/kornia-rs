//! M-estimator robust kernels.
//!
//! These are pluggable weight functions that downweight outlier residuals
//! during iterated least-squares refinement (LM normal equations, BA, the
//! eventual MAGSAC++ scorer). Each kernel takes a *squared residual* and a
//! *squared scale* (the kernel's transition point, sometimes called `c²`)
//! and returns a non-negative weight in `[0, 1]`.
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

/// A pluggable robust weight function.
///
/// `weight(r², c²)` returns a non-negative scalar in `[0, 1]` that scales
/// the contribution of a residual to a quadratic loss. Multiplying both
/// sides of the normal equations by this weight implements iteratively
/// reweighted least squares (IRLS).
pub trait RobustKernel {
    /// Compute the weight for a residual whose squared magnitude is `r_sq`,
    /// given the kernel's squared scale parameter `c_sq`.
    fn weight(&self, r_sq: f64, c_sq: f64) -> f64;
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
}

/// Huber's M-estimator: quadratic loss inside `|r| ≤ c`, linear outside.
///
/// Weight: `1` if `r² ≤ c²` else `c / |r|`. Implemented via squared
/// quantities to avoid a `sqrt` in the inlier branch.
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
}

/// Cauchy / Lorentzian kernel — strictly redescending.
///
/// Weight: `1 / (1 + r²/c²)`. Smooth; never reaches zero. Good middle
/// ground between Huber (bounded influence) and Tukey (hard rejection).
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
}

/// Tukey biweight — hard redescender. Weight reaches 0 at `|r| = c`.
///
/// Weight: `(1 - r²/c²)²` for `r² ≤ c²`, else `0`. Outliers beyond `c`
/// contribute zero gradient — makes Tukey effective at gross outliers but
/// sensitive to the initial estimate (a bad initialisation can leave good
/// observations marked as zero-weight).
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
        assert!(w < 1e-3, "Cauchy weight should be small at far residuals: {w}");
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
}
