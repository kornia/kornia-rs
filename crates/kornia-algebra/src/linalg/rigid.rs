//! Rigid alignment utilities (Kabsch / Umeyama), with optional per-point weights.

use crate::linalg::svd::svd3_f64;
use crate::{Mat3AF32, Mat3F32, Mat3F64, Vec3AF32, Vec3F32, Vec3F64};
use nalgebra::{Matrix3, SVD};
use thiserror::Error;

/// Rotation (R), translation (t), and scale (s) output of Umeyama without scaling (s = 1).
pub type UmeyamaOutput = (Mat3AF32, Vec3AF32, f32);

/// Error type for Umeyama rigid alignment operations.
#[derive(Debug, Error)]
pub enum UmeyamaError {
    /// Source and destination arrays must have the same length
    #[error("Source and destination arrays must have the same length")]
    MismatchedInputLengths,
    /// Source and destination arrays must not be empty
    #[error("Source and destination arrays must not be empty")]
    EmptyInput,
    /// Failed to compute U in SVD
    #[error("Failed to compute U in SVD")]
    SvdU,
    /// Failed to compute V^T in SVD
    #[error("Failed to compute V^T in SVD")]
    SvdVT,
    /// Source, destination and weight arrays must have the same length
    #[error("Source, destination and weight arrays must have the same length")]
    MismatchedWeightsLength,
    /// Weights must be finite, non-negative, and have a positive sum
    #[error("Weights must be finite, non-negative, and have a positive sum")]
    InvalidWeights,
    /// Robust-fit Huber threshold must be finite and positive.
    #[error("Robust Umeyama delta must be finite and positive, got {delta}")]
    InvalidDelta {
        /// The invalid Huber threshold.
        delta: f64,
    },
    /// Robust-fit iteration cap must be non-zero.
    #[error("Robust Umeyama max_iters must be greater than zero, got {max_iters}")]
    InvalidMaxIters {
        /// The invalid iteration cap.
        max_iters: usize,
    },
    /// Robust-fit convergence tolerance must be finite and positive.
    #[error("Robust Umeyama tol must be finite and positive, got {tol}")]
    InvalidTol {
        /// The invalid convergence tolerance.
        tol: f64,
    },
}

/// Result type alias for Umeyama.
pub type UmeyamaResult = Result<UmeyamaOutput, UmeyamaError>;

/// Umeyama/Kabsch algorithm without scale.
/// Returns (R, t, s) where s == 1.0.
pub fn umeyama(src: &[Vec3AF32], dst: &[Vec3AF32]) -> UmeyamaResult {
    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }
    let n = src.len() as f32;

    // Centroids
    let mut mu_s = [0.0; 3];
    let mut mu_d = [0.0; 3];
    for i in 0..src.len() {
        mu_s[0] += src[i].x;
        mu_s[1] += src[i].y;
        mu_s[2] += src[i].z;

        mu_d[0] += dst[i].x;
        mu_d[1] += dst[i].y;
        mu_d[2] += dst[i].z;
    }
    for x in &mut mu_s {
        *x /= n;
    }
    for x in &mut mu_d {
        *x /= n;
    }

    // Covariance matrix H = (dst_c)^T * src_c / n
    let mut h = [[0.0f32; 3]; 3];
    for i in 0..src.len() {
        let sc = [src[i].x - mu_s[0], src[i].y - mu_s[1], src[i].z - mu_s[2]];
        let dc = [dst[i].x - mu_d[0], dst[i].y - mu_d[1], dst[i].z - mu_d[2]];
        for (r, &dc_r) in dc.iter().enumerate() {
            for (c, &sc_c) in sc.iter().enumerate() {
                h[r][c] += dc_r * sc_c;
            }
        }
    }
    for row in &mut h {
        for val in row {
            *val /= n;
        }
    }

    let h_na = Matrix3::<f32>::from_row_slice(&[
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    ]);

    let svd = SVD::new(h_na, true, true);
    let Some(u) = svd.u else {
        return Err(UmeyamaError::SvdU);
    };
    let Some(v_t) = svd.v_t else {
        return Err(UmeyamaError::SvdVT);
    };

    let mut r_na = u * v_t;
    if r_na.determinant() < 0.0 {
        r_na.column_mut(2).scale_mut(-1.0);
    }

    let r_arr = [
        [r_na[(0, 0)], r_na[(0, 1)], r_na[(0, 2)]],
        [r_na[(1, 0)], r_na[(1, 1)], r_na[(1, 2)]],
        [r_na[(2, 0)], r_na[(2, 1)], r_na[(2, 2)]],
    ];

    let mut t = [0.0; 3];
    for r in 0..3 {
        t[r] = mu_d[r] - (r_arr[r][0] * mu_s[0] + r_arr[r][1] * mu_s[1] + r_arr[r][2] * mu_s[2]);
    }

    // Convert row-major array to column-major for Mat3F32::from_cols_array
    let r_flat = [
        r_arr[0][0],
        r_arr[1][0],
        r_arr[2][0], // first column
        r_arr[0][1],
        r_arr[1][1],
        r_arr[2][1], // second column
        r_arr[0][2],
        r_arr[1][2],
        r_arr[2][2], // third column
    ];
    let r = Mat3F32::from_cols_array(&r_flat);
    let t = Vec3F32::from_array(t);

    let r_cols: [f32; 9] = r.into();
    Ok((
        Mat3AF32::from_cols_array(&r_cols),
        Vec3AF32::new(t.x, t.y, t.z),
        1.0,
    ))
}

/// Rotation (R), translation (t), and scale (s) output of Umeyama (f64).
pub type UmeyamaOutputF64 = (Mat3F64, Vec3F64, f64);

/// Result type alias for Umeyama (f64).
pub type UmeyamaResultF64 = Result<UmeyamaOutputF64, UmeyamaError>;

/// Umeyama alignment with optional per-point weights and scale estimation (f64).
///
/// Solves for the rotation `R`, translation `t`, and optional scale `s` that
/// minimize the sum of weighted residuals `Σ wᵢ ‖dstᵢ − (s·R·srcᵢ + t)‖²`. With
/// `weights == None` every point is weighted equally, i.e. the unweighted
/// Umeyama/Kabsch solution.
///
/// Point coordinates are assumed finite; non-finite coordinates of points with
/// a non-zero weight are not validated and propagate to the result. A point with
/// weight `0` is ignored entirely.
///
/// # Arguments
///
/// * `src` - Source points. Must have the same length as `dst` (and `weights`,
///   when provided).
/// * `dst` - Destination points. Must have the same length as `src`.
/// * `weights` - Optional per-point weights, one entry per point. `None` is the
///   unweighted case. When `Some`, weights must be finite, non-negative, and have
///   a positive sum; they are normalized internally (only their relative values
///   matter), and a point with weight `0` is ignored.
/// * `with_scale` - If `true`, estimates the scale `s`; otherwise `s` is fixed
///   to `1.0` (rigid alignment only). If the source points have (near-)zero
///   variance, `s` falls back to `1.0`.
///
/// # Returns
///
/// `Ok((R, t, s))` on success, where `R` is a proper rotation matrix
/// (`det(R) = 1`), `t` is the translation vector, and `s` is the estimated scale
/// (`1.0` when `with_scale` is `false`).
///
/// # Errors
///
/// Returns [`UmeyamaError::MismatchedInputLengths`] if `src` and `dst` differ in
/// length.
/// Returns [`UmeyamaError::EmptyInput`] if `src` and `dst` are both empty.
/// Returns [`UmeyamaError::InvalidWeights`] if `weights` is empty, contains a
/// non-finite or negative value, or is all zero. This is checked before the
/// weights-length check, so it takes precedence over
/// [`UmeyamaError::MismatchedWeightsLength`] (but not over
/// [`UmeyamaError::MismatchedInputLengths`] or [`UmeyamaError::EmptyInput`]).
/// Returns [`UmeyamaError::MismatchedWeightsLength`] if otherwise-valid
/// `weights` do not have exactly one entry per point.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::linalg::rigid::umeyama_f64;
/// use kornia_algebra::Vec3F64;
///
/// let src = [
///     Vec3F64::new(0.0, 0.0, 0.0),
///     Vec3F64::new(1.0, 0.0, 0.0),
///     Vec3F64::new(0.0, 1.0, 0.0),
///     Vec3F64::new(0.0, 0.0, 1.0),
/// ];
/// // Pure translation by (1, 2, 3).
/// let dst: Vec<Vec3F64> = src.iter().map(|p| *p + Vec3F64::new(1.0, 2.0, 3.0)).collect();
///
/// // Unweighted.
/// let (_r, t, _s) = umeyama_f64(&src, &dst, None, false)?;
/// assert!((t.x - 1.0).abs() < 1e-12);
///
/// // Weighted (weights only rescale the residuals).
/// let weights = [1.0, 2.0, 3.0, 4.0];
/// let (_r, t, _s) = umeyama_f64(&src, &dst, Some(&weights), false)?;
/// assert!((t.x - 1.0).abs() < 1e-12);
///
/// // A zero weight ignores a point entirely (here a corrupted destination),
/// // and the scale can still be estimated from the remaining points.
/// let mut noisy = dst.clone();
/// noisy[3] = Vec3F64::new(1e6, -1e6, 1e6);
/// let weights = [1.0, 1.0, 1.0, 0.0];
/// let (_r, t, s) = umeyama_f64(&src, &noisy, Some(&weights), true)?;
/// assert!((t.x - 1.0).abs() < 1e-9 && (s - 1.0).abs() < 1e-9);
/// # Ok::<(), kornia_algebra::linalg::rigid::UmeyamaError>(())
/// ```
pub fn umeyama_f64(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    weights: Option<&[f64]>,
    with_scale: bool,
) -> UmeyamaResultF64 {
    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }
    if src.is_empty() {
        return Err(UmeyamaError::EmptyInput);
    }

    let Moments {
        mu_s,
        mu_d,
        h,
        var_src,
    } = moments(src, dst, weights)?;

    // SVD of H via the crate's `svd3_f64` (handles repeated singular values).
    let svd = svd3_f64(&h);
    let (u, sigma, v) = (svd.u(), svd.s(), svd.v());

    // Reflection correction: S = diag(1, 1, det(U)·det(V)) (det(V^T) == det(V))
    let s_sign = if u.determinant() * v.determinant() < 0.0 {
        -1.0
    } else {
        1.0
    };

    // R = U · S · V^T
    let r = *u * Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, s_sign)) * v.transpose();

    // Scale: s = tr(S · Σ) / σ²_src
    let scale = if with_scale && var_src > 1e-15 {
        (sigma.x_axis.x + sigma.y_axis.y + s_sign * sigma.z_axis.z) / var_src
    } else {
        1.0
    };

    // Translation: t = μ_d − s·R·μ_s
    let t = mu_d - scale * r * mu_s;

    Ok((r, t, scale))
}

/// σ̂ = MAD / Φ⁻¹(3/4): the MAD → pseudo-σ consistency factor for Gaussian
/// residuals (`Φ⁻¹(3/4) = 0.6744897501960817…`, so the factor is
/// `1.4826022185056018…`). Distribution-dependent for non-Gaussian data.
const MAD_TO_SIGMA: f64 = 1.0 / 0.674_489_750_196_081_7;

/// Hard-rejection threshold in units of the MAD pseudo-σ: residuals farther
/// than this from the median residual are dropped before the final refit.
const OUTLIER_SIGMA_MULT: f64 = 3.5;

/// Median of a slice: sorts a copy with [`f64::total_cmp`] (NaN gets a
/// deterministic position); empty input yields `0.0`.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let m = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[m]
    } else {
        0.5 * (sorted[m - 1] + sorted[m])
    }
}

/// Configuration for the robust weighted Umeyama fit.
#[derive(Debug, Clone, Copy)]
pub struct RobustUmeyamaConfig {
    /// Huber threshold in the residual's units: residuals below `delta` keep
    /// full weight, above it are down-weighted as `delta / residual`.
    pub delta: f64,
    /// Maximum number of IRLS iterations.
    pub max_iters: usize,
    /// Convergence tolerance on the maximum relative change of scale,
    /// rotation and translation between iterations.
    pub tol: f64,
}

impl Default for RobustUmeyamaConfig {
    fn default() -> Self {
        Self {
            delta: 1.0,
            max_iters: 20,
            tol: 1e-6,
        }
    }
}

impl RobustUmeyamaConfig {
    /// Validates the config fields, one specific error variant per field.
    fn validate(&self) -> Result<(), UmeyamaError> {
        if !self.delta.is_finite() || self.delta <= 0.0 {
            return Err(UmeyamaError::InvalidDelta { delta: self.delta });
        }
        if self.max_iters == 0 {
            return Err(UmeyamaError::InvalidMaxIters {
                max_iters: self.max_iters,
            });
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(UmeyamaError::InvalidTol { tol: self.tol });
        }
        Ok(())
    }
}

/// Huber-weighted IRLS robust Umeyama fit with a MAD hard-rejection pass (f64).
///
/// Solves `(R, t, s)` minimizing `Σ ρ(‖dstᵢ − (s·R·srcᵢ + t)‖)` where `ρ` is
/// the Huber loss, by iteratively reweighted least squares over
/// [`umeyama_f64`] solves. After convergence, the residual scale is estimated
/// robustly as `σ̂ = MAD / Φ⁻¹(3/4) ≈ 1.4826 · MAD` of the residuals around
/// their median, and correspondences whose residual deviates from the median
/// by more than `3.5 · σ̂` are dropped before a final unweighted fit on the
/// survivors. Returns `(R, t, s)`.
///
/// The rejection window is centered on the median residual: it tracks a
/// biased intermediate fit and always keeps at least half of the
/// correspondences for finite inputs, so a heavy outlier cluster cannot
/// empty the inlier set. The fit is deterministic (no randomness, no
/// parallelism).
///
/// With `with_scale = false` the estimator assumes the true scale is `1.0`:
/// a genuine scale mismatch surfaces as inflated inlier residuals, which
/// loosens the MAD threshold and weakens rejection.
///
/// Point coordinates are assumed finite; non-finite coordinates propagate to
/// the result, as in [`umeyama_f64`].
///
/// # Arguments
///
/// * `src` - Source points. Must have the same length as `dst`.
/// * `dst` - Destination points. Must have the same length as `src`.
/// * `config` - Robust-fit configuration (Huber threshold, iteration cap,
///   convergence tolerance). Invalid field values are rejected before the
///   input is inspected.
/// * `with_scale` - If `true`, estimates the scale `s`; otherwise `s` is
///   fixed to `1.0` (rigid alignment only).
///
/// # Returns
///
/// `Ok((R, t, s))` on success, where `R` is a proper rotation matrix
/// (`det(R) = 1`), `t` is the translation, and `s` is the estimated scale
/// (exactly `1.0` when `with_scale` is `false`).
///
/// # Errors
///
/// Returns [`UmeyamaError::InvalidDelta`], [`UmeyamaError::InvalidMaxIters`],
/// or [`UmeyamaError::InvalidTol`] if the corresponding config field is
/// invalid. Config validation runs before the input checks below, so it
/// takes precedence over them.
/// Returns [`UmeyamaError::MismatchedInputLengths`] if `src` and `dst`
/// differ in length, and [`UmeyamaError::EmptyInput`] if both are empty.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::linalg::rigid::{umeyama_f64_robust, RobustUmeyamaConfig};
/// use kornia_algebra::Vec3F64;
///
/// let src = [
///     Vec3F64::new(0.0, 0.0, 0.0),
///     Vec3F64::new(1.0, 0.0, 0.0),
///     Vec3F64::new(0.0, 1.0, 0.0),
///     Vec3F64::new(1.0, 1.0, 0.0),
///     Vec3F64::new(0.5, 0.5, 1.0),
///     Vec3F64::new(1.0, 0.5, 0.5),
/// ];
/// // True transform (scale 2, translation (1, 2, 3)), one corrupted corner.
/// let mut dst: Vec<Vec3F64> = src
///     .iter()
///     .map(|p| *p * 2.0 + Vec3F64::new(1.0, 2.0, 3.0))
///     .collect();
/// dst[0] = Vec3F64::new(-50.0, 50.0, 50.0);
///
/// let (r, t, s) = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
/// // The corrupted corner keeps a large residual while the good corners map tightly.
/// let bad_res = ((r * src[0]) * s + t - dst[0]).length();
/// let good_res = ((r * src[1]) * s + t - dst[1]).length();
/// assert!(good_res < 0.5 && bad_res > 20.0 && (s - 2.0).abs() < 0.5);
/// # Ok::<(), kornia_algebra::linalg::rigid::UmeyamaError>(())
/// ```
pub fn umeyama_f64_robust(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    config: &RobustUmeyamaConfig,
    with_scale: bool,
) -> UmeyamaResultF64 {
    config.validate()?;

    let n = src.len();
    let mut weights = vec![1.0_f64; n];

    // Initial unweighted fit; also enforces the length/empty checks.
    let mut fit = umeyama_f64(src, dst, Some(&weights), with_scale)?;

    // r_i = |(s·R·src_i + t) - dst_i| for the current fit.
    let write_residuals = |fit: &UmeyamaOutputF64, out: &mut [f64]| {
        for (i, (p, d)) in src.iter().zip(dst).enumerate() {
            let mapped = (fit.0 * *p) * fit.2 + fit.1;
            out[i] = (mapped - *d).length();
        }
    };

    let mut residuals = vec![0.0_f64; n];
    for _ in 0..config.max_iters {
        write_residuals(&fit, &mut residuals);
        // Huber weights: full weight below `delta`, `delta / r` above.
        for (w, &r) in weights.iter_mut().zip(&residuals) {
            *w = if r > 0.0 {
                (config.delta / r).min(1.0)
            } else {
                1.0
            };
        }
        let new_fit = umeyama_f64(src, dst, Some(&weights), with_scale)?;

        // Convergence: maximum relative change of scale, rotation columns
        // and translation between the two fits. `scale_rel` is 0 in rigid
        // mode, so those fits converge on R/t alone.
        let eps = f64::EPSILON;
        let scale_rel = ((new_fit.2 - fit.2) / fit.2.max(eps)).abs();
        let rot_rel = (0..3)
            .map(|c| (new_fit.0.col(c) - fit.0.col(c)).length() / fit.0.col(c).length().max(eps))
            .fold(0.0_f64, f64::max);
        let t_rel = (new_fit.1 - fit.1).length() / fit.1.length().max(eps);

        fit = new_fit;
        if scale_rel.max(rot_rel).max(t_rel) < config.tol {
            break;
        }
    }

    // Residuals under the converged fit.
    write_residuals(&fit, &mut residuals);

    // MAD hard rejection around the median residual:
    // σ̂ = MAD / Φ⁻¹(3/4) ≈ 1.4826 · MAD.
    // Outliers: |r_i - med| > 3.5 · σ̂. The centered window (unlike a raw
    // residual threshold anchored at zero) tracks a biased intermediate
    // fit, so a heavy outlier cluster cannot empty the inlier set.
    let med = median(&residuals);
    let abs_dev: Vec<f64> = residuals.iter().map(|&r| (r - med).abs()).collect();
    let mad = median(&abs_dev);
    let sigma = MAD_TO_SIGMA * mad;

    if sigma > f64::EPSILON {
        let thr = OUTLIER_SIGMA_MULT * sigma;
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| (residuals[i] - med).abs() <= thr)
            .collect();
        // The centered window keeps at least half the residuals by
        // construction (half of all |r - med| are <= MAD < thr), so the
        // inlier set is never empty for finite inputs.
        if inliers.len() < n {
            let src_in: Vec<Vec3F64> = inliers.iter().map(|&i| src[i]).collect();
            let dst_in: Vec<Vec3F64> = inliers.iter().map(|&i| dst[i]).collect();
            fit = umeyama_f64(&src_in, &dst_in, None, with_scale)?;
        }
    }

    Ok(fit)
}

/// Accumulated first- and second-order statistics of the two point sets.
#[derive(Debug)]
struct Moments {
    /// Source centroid.
    mu_s: Vec3F64,
    /// Destination centroid.
    mu_d: Vec3F64,
    /// Cross-covariance columns, normalized by the total weight.
    h: Mat3F64,
    /// Source variance `Σ wᵢ·‖src_c‖²`, normalized by the total weight.
    var_src: f64,
}

/// Accumulates the centroids, cross-covariance, and source variance, weighted by
/// `weights` or uniformly when it is `None`.
fn moments(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    weights: Option<&[f64]>,
) -> Result<Moments, UmeyamaError> {
    // Unit weights when unweighted; otherwise validate and normalize by the
    // maximum. Invalid weights take precedence over the length check (see
    // `umeyama_f64`'s `# Errors`).
    let max_w = match weights {
        None => 1.0,
        Some(w) => {
            let mut max_w = 0.0f64;
            for &wi in w {
                if !wi.is_finite() || wi < 0.0 {
                    return Err(UmeyamaError::InvalidWeights);
                }
                max_w = max_w.max(wi);
            }
            if max_w <= 0.0 {
                return Err(UmeyamaError::InvalidWeights);
            }
            if w.len() != src.len() {
                return Err(UmeyamaError::MismatchedWeightsLength);
            }
            max_w
        }
    };

    // Normalize by max_w (cancels in every ratio): keeps Σw in [1, n] and
    // w_i * p_i bounded, so extreme finite weights cannot overflow. Divide
    // rather than multiply by 1/max_w so a subnormal max_w stays exact.
    let weight_at = |i: usize| weights.map_or(1.0, |w| w[i] / max_w);

    // glam's f64 types are scalar (`DVec3`/`DMat3` set `USES_SCALAR_MATH`), so
    // this accumulation is deliberately scalar — there is no f64 SIMD to use.
    //
    // Weighted centroids μ = Σ wᵢ pᵢ / Σ wᵢ.
    let mut mu_s = Vec3F64::ZERO;
    let mut mu_d = Vec3F64::ZERO;
    let mut sum_w = 0.0f64;
    for (i, (s, d)) in src.iter().zip(dst).enumerate() {
        let wi = weight_at(i);
        // A zero-weight point is ignored entirely: `inf * 0.0 = NaN` would
        // otherwise poison the centroid.
        if wi == 0.0 {
            continue;
        }
        sum_w += wi;
        mu_s += *s * wi;
        mu_d += *d * wi;
    }
    mu_s /= sum_w;
    mu_d /= sum_w;

    // Weighted cross-covariance columns and source variance.
    let mut h_x = Vec3F64::ZERO;
    let mut h_y = Vec3F64::ZERO;
    let mut h_z = Vec3F64::ZERO;
    let mut var_src = 0.0f64;
    for (i, (s, d)) in src.iter().zip(dst).enumerate() {
        let wi = weight_at(i);
        // Zero-weight points are ignored; a centered extreme coordinate would
        // otherwise give `0.0 * inf = NaN`.
        if wi == 0.0 {
            continue;
        }
        let sc = *s - mu_s;
        let dc = *d - mu_d;
        h_x += dc * sc.x * wi;
        h_y += dc * sc.y * wi;
        h_z += dc * sc.z * wi;
        var_src += sc.dot(sc) * wi;
    }
    h_x /= sum_w;
    h_y /= sum_w;
    h_z /= sum_w;
    var_src /= sum_w;

    Ok(Moments {
        mu_s,
        mu_d,
        h: Mat3F64::from_cols(h_x, h_y, h_z),
        var_src,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    /// 90° rotation about Z, column-major (`[[0, -1, 0], [1, 0, 0], [0, 0, 1]]`).
    fn rot_z90() -> Mat3F64 {
        Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    }

    #[test]
    #[cfg(feature = "approx")]
    fn test_umeyama_synthetic_z90() -> Result<(), UmeyamaError> {
        use approx::assert_relative_eq;
        // Source points (square in XY plane, z=0)
        let src: [Vec3AF32; 4] = [
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
        ];
        // True transform: 90° about Z, plus translation
        // Rotation matrix (row-major): [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        // Converted to column-major for from_cols
        let r = Mat3AF32::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t = Vec3AF32::new(0.5, -0.3, 2.0);
        // Create dst
        let mut dst = [Vec3AF32::new(0.0, 0.0, 0.0); 4];
        for i in 0..4 {
            dst[i] = r * src[i] + t
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        assert_relative_eq!(r_est, r, epsilon = 1e-6);
        assert_relative_eq!(t_est, t, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_rigid() -> Result<(), UmeyamaError> {
        // 90° rotation about Z + translation, no scale
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(1.0, 1.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();

        let (r_est, t_est, s_est) = umeyama_f64(&src, &dst, None, false)?;
        assert!((s_est - 1.0).abs() < 1e-12);
        for i in 0..9 {
            let r_true_arr: [f64; 9] = r_true.into();
            let r_est_arr: [f64; 9] = r_est.into();
            assert!(
                (r_est_arr[i] - r_true_arr[i]).abs() < 1e-10,
                "R mismatch at {i}"
            );
        }
        assert!((t_est.x - t_true.x).abs() < 1e-10);
        assert!((t_est.y - t_true.y).abs() < 1e-10);
        assert!((t_est.z - t_true.z).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_with_scale() -> Result<(), UmeyamaError> {
        // Identity rotation, no translation, scale = 2.5
        let src = [
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(1.0, 1.0, 1.0),
        ];
        let scale_true = 2.5;
        let dst: Vec<Vec3F64> = src.iter().map(|p| *p * scale_true).collect();

        let (_r_est, t_est, s_est) = umeyama_f64(&src, &dst, None, true)?;
        assert!(
            (s_est - scale_true).abs() < 1e-10,
            "scale: got {s_est}, expected {scale_true}"
        );
        assert!(t_est.length() < 1e-10, "translation should be near zero");
        Ok(())
    }

    /// Assert two Umeyama outputs agree component-wise within `eps`.
    fn assert_outputs_close(a: &UmeyamaOutputF64, b: &UmeyamaOutputF64, eps: f64) {
        let ar: [f64; 9] = a.0.into();
        let br: [f64; 9] = b.0.into();
        for i in 0..9 {
            assert!(
                (ar[i] - br[i]).abs() < eps,
                "R mismatch at {i}: {} vs {}",
                ar[i],
                br[i]
            );
        }
        assert!((a.1.x - b.1.x).abs() < eps, "t.x: {} vs {}", a.1.x, b.1.x);
        assert!((a.1.y - b.1.y).abs() < eps, "t.y: {} vs {}", a.1.y, b.1.y);
        assert!((a.1.z - b.1.z).abs() < eps, "t.z: {} vs {}", a.1.z, b.1.z);
        assert!((a.2 - b.2).abs() < eps, "s: {} vs {}", a.2, b.2);
    }

    /// Assert a recovered `(R, t, s)` matches the true transform within `eps`.
    fn assert_matches_true(
        got: &UmeyamaOutputF64,
        r_true: &Mat3F64,
        t_true: &Vec3F64,
        s_true: f64,
        eps: f64,
    ) {
        assert_outputs_close(got, &(*r_true, *t_true, s_true), eps);
    }

    /// Assert two [`Moments`] agree component-wise within `eps`.
    fn assert_moments_close(a: &Moments, b: &Moments, eps: f64) {
        let (a_mu_s, b_mu_s) = (a.mu_s.to_array(), b.mu_s.to_array());
        let (a_mu_d, b_mu_d) = (a.mu_d.to_array(), b.mu_d.to_array());
        for k in 0..3 {
            assert!((a_mu_s[k] - b_mu_s[k]).abs() < eps, "mu_s[{k}]");
            assert!((a_mu_d[k] - b_mu_d[k]).abs() < eps, "mu_d[{k}]");
        }
        let a_cols = [a.h.x_axis(), a.h.y_axis(), a.h.z_axis()];
        let b_cols = [b.h.x_axis(), b.h.y_axis(), b.h.z_axis()];
        for c in 0..3 {
            let (a_col, b_col) = (a_cols[c].to_array(), b_cols[c].to_array());
            for r in 0..3 {
                assert!(
                    (a_col[r] - b_col[r]).abs() < eps,
                    "h[{r}][{c}]: {} vs {}",
                    a_col[r],
                    b_col[r]
                );
            }
        }
        assert!(
            (a.var_src - b.var_src).abs() < eps,
            "var_src: {} vs {}",
            a.var_src,
            b.var_src
        );
    }

    /// Assert `H` is symmetric and `trace(H) == var_src` (both equal
    /// `Σ wᵢ‖src_c‖² / Σ wᵢ`).
    fn assert_covariance_symmetric_with_trace(m: &Moments) {
        let cols: [[f64; 3]; 3] = [
            m.h.x_axis().to_array(),
            m.h.y_axis().to_array(),
            m.h.z_axis().to_array(),
        ];
        let trace = cols[0][0] + cols[1][1] + cols[2][2];
        for (r, row_r) in cols.iter().enumerate() {
            for (c, row_c) in cols.iter().enumerate() {
                assert!((row_c[r] - row_r[c]).abs() < 1e-12, "H not symmetric");
            }
        }
        assert!((trace - m.var_src).abs() < 1e-12, "trace(H) != var_src");
    }

    #[test]
    fn test_unweighted_moments_centroids() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        ];
        let m = moments(&src, &src, None)?;
        let (mu_s, mu_d) = (m.mu_s.to_array(), m.mu_d.to_array());

        for k in 0..3 {
            assert!((mu_s[k] - 0.25).abs() < 1e-12, "mu_s[{k}]");
            assert!((mu_d[k] - 0.25).abs() < 1e-12, "mu_d[{k}]");
        }
        Ok(())
    }

    #[test]
    fn test_unweighted_moments_covariance_is_symmetric_with_matching_trace(
    ) -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        ];
        let m = moments(&src, &src, None)?;

        // Σ‖sc‖² / 4 = 2.25 / 4.
        assert!((m.var_src - 0.5625).abs() < 1e-12, "var_src: {}", m.var_src);

        // H is symmetric and trace(H) == var_src.
        assert_covariance_symmetric_with_trace(&m);
        Ok(())
    }

    #[test]
    fn test_weighted_moments_weighted_centroids() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let w = [1.0, 2.0, 3.0];
        let m = moments(&src, &src, Some(&w))?;
        let mu_s = m.mu_s.to_array();

        // μ = Σ w_i p_i / Σ w_i, with Σ w_i = 6.
        assert!((mu_s[0] - 2.0 / 6.0).abs() < 1e-12, "mu_s[0]");
        assert!((mu_s[1] - 3.0 / 6.0).abs() < 1e-12, "mu_s[1]");
        assert!(mu_s[2].abs() < 1e-12, "mu_s[2]");
        Ok(())
    }

    #[test]
    fn test_weighted_moments_covariance_is_symmetric_with_matching_trace(
    ) -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let w = [1.0, 2.0, 3.0];
        let m = moments(&src, &src, Some(&w))?;

        assert_covariance_symmetric_with_trace(&m);
        Ok(())
    }

    #[test]
    fn test_weighted_moments_normalization_invariant() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
        ];
        let dst = [
            Vec3F64::new(0.5, 0.1, 1.0),
            Vec3F64::new(1.1, 0.9, -0.2),
            Vec3F64::new(-0.3, 2.4, 0.6),
        ];
        let w = [0.5, 3.0, 1.0];
        let scaled: Vec<f64> = w.iter().map(|x| x * 7.5).collect();

        let a = moments(&src, &dst, Some(&w))?;
        let b = moments(&src, &dst, Some(&scaled))?;
        assert_moments_close(&a, &b, 1e-12);
        Ok(())
    }

    #[test]
    fn test_weighted_moments_uniform_equals_unweighted() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
        ];
        let dst = [
            Vec3F64::new(0.5, 0.1, 1.0),
            Vec3F64::new(1.1, 0.9, -0.2),
            Vec3F64::new(-0.3, 2.4, 0.6),
            Vec3F64::new(1.2, -1.0, 2.0),
        ];
        let unweighted = moments(&src, &dst, None)?;
        let weights = vec![1.0; src.len()];
        let weighted = moments(&src, &dst, Some(&weights))?;
        assert_moments_close(&unweighted, &weighted, 1e-12);
        Ok(())
    }

    #[test]
    fn test_weighted_moments_errors() {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];

        // empty weights
        let err = moments(&src, &src, Some(&[])).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // non-finite weights
        let err = moments(&src, &src, Some(&[1.0, f64::NAN, 1.0])).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // all-zero weights
        let err = moments(&src, &src, Some(&[0.0, 0.0, 0.0])).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // negative sum
        let err = moments(&src, &src, Some(&[1.0, -5.0, 1.0])).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // mixed-sign weights whose sum is positive
        let err = moments(&src, &src, Some(&[2.0, -1.0, 1.0])).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // a single negative weight
        let err = moments(&src, &src, Some(&[-1.0, 1.0, 1.0])).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // wrong length (otherwise valid weights)
        let err = moments(&src, &src, Some(&[1.0])).unwrap_err();
        assert!(
            matches!(err, UmeyamaError::MismatchedWeightsLength),
            "got {err:?}"
        );
    }

    #[test]
    fn test_weighted_moments_extreme_weights_are_scale_invariant() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
        ];
        let dst = [
            Vec3F64::new(0.5, 0.1, 1.0),
            Vec3F64::new(1.1, 0.9, -0.2),
            Vec3F64::new(-0.3, 2.4, 0.6),
        ];
        // These raw weights sum to +inf, but only their ratios matter.
        let huge = [f64::MAX, f64::MAX, f64::MAX / 2.0];
        let unit = [1.0, 1.0, 0.5];

        let a = moments(&src, &dst, Some(&huge))?;
        let b = moments(&src, &dst, Some(&unit))?;
        assert_moments_close(&a, &b, 1e-12);
        Ok(())
    }

    #[test]
    fn test_weighted_moments_zero_weight_extreme_point_ignored() -> Result<(), UmeyamaError> {
        // Two points at +f64::MAX/2 put the centroid at +f64::MAX/2 (their sum is
        // still finite). A third, zero-weight point at -f64::MAX then overflows to
        // -inf when centered, so it must be skipped before its `0.0 * inf = NaN`
        // can poison the sums.
        let big = f64::MAX / 2.0;
        let src = [
            Vec3F64::new(big, 0.0, 0.0),
            Vec3F64::new(big, 1.0, 0.0),
            Vec3F64::new(-f64::MAX, 0.0, 0.0),
        ];
        let dst = src;
        let weights = [1.0, 1.0, 0.0];

        let base = moments(&src[..2], &dst[..2], Some(&[1.0, 1.0]))?;
        let with_extreme = moments(&src, &dst, Some(&weights))?;
        assert_moments_close(&base, &with_extreme, 1e-12);
        Ok(())
    }

    #[test]
    fn test_weighted_moments_zero_weight_non_finite_point_ignored() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(f64::INFINITY, f64::NAN, f64::NEG_INFINITY),
        ];
        let dst = src;
        let base = moments(&src[..2], &dst[..2], Some(&[1.0, 1.0]))?;
        let with_non_finite = moments(&src, &dst, Some(&[1.0, 1.0, 0.0]))?;
        assert_moments_close(&base, &with_non_finite, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_recovers_exact_transform() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(1.0, 1.0, 1.0),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 2.0;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        let weights = [0.5, 3.0, 1.0, 2.0, 4.0];

        let got = umeyama_f64(&src, &dst, Some(&weights), true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_uniform_equals_unweighted() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
            Vec3F64::new(1.7, 1.5, -1.9),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();

        let unweighted = umeyama_f64(&src, &dst, None, true)?;
        let weights = vec![1.0; src.len()];
        let weighted = umeyama_f64(&src, &dst, Some(&weights), true)?;

        assert_outputs_close(&unweighted, &weighted, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_normalization_invariant() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
            Vec3F64::new(1.7, 1.5, -1.9),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();

        let weights = [0.5, 3.0, 1.0, 2.0, 4.0];
        let scaled_weights: Vec<f64> = weights.iter().map(|w| w * 7.5).collect();

        let a = umeyama_f64(&src, &dst, Some(&weights), true)?;
        let b = umeyama_f64(&src, &dst, Some(&scaled_weights), true)?;
        assert_outputs_close(&a, &b, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_zero_weight_point_ignored() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(0.5, 0.5, 0.5),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let mut dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();
        // Corrupt the last destination wildly; its weight is zero so it must be
        // ignored entirely.
        dst[4] = Vec3F64::new(1e6, -1e6, 1e6);
        let weights = [1.0, 2.0, 0.5, 1.5, 0.0];

        let got = umeyama_f64(&src, &dst, Some(&weights), false)?;
        assert_matches_true(&got, &r_true, &t_true, 1.0, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_coplanar_input() -> Result<(), UmeyamaError> {
        // Eight tag-like corners, all on the z = 0 plane (rank-deficient H).
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(1.0, 1.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.5, 0.0, 0.0),
            Vec3F64::new(1.0, 0.5, 0.0),
            Vec3F64::new(0.5, 1.0, 0.0),
            Vec3F64::new(0.0, 0.5, 0.0),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.3;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        let weights = [1.0, 2.0, 1.5, 0.7, 1.1, 3.0, 0.9, 1.3];

        let got = umeyama_f64(&src, &dst, Some(&weights), true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_collinear_input() -> Result<(), UmeyamaError> {
        // Points along the x-axis: the covariance has rank 1.
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(2.0, 0.0, 0.0),
            Vec3F64::new(3.0, 0.0, 0.0),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 2.0;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        let weights = [1.0, 2.0, 0.5, 1.5];

        let (r, t, s) = umeyama_f64(&src, &dst, Some(&weights), true)?;

        // Rank-deficient: (R, t, s) is not unique, but the recovered transform
        // must still reproduce every destination point exactly.
        assert!((s - s_true).abs() < 1e-9, "scale: got {s}");
        for (i, p) in src.iter().enumerate() {
            let mapped = (r * *p) * s + t;
            assert!(
                (mapped.x - dst[i].x).abs() < 1e-9
                    && (mapped.y - dst[i].y).abs() < 1e-9
                    && (mapped.z - dst[i].z).abs() < 1e-9,
                "point {i} not reproduced"
            );
        }
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_identical_points() -> Result<(), UmeyamaError> {
        // All source points coincide: zero variance, rank-0 covariance.
        let p = Vec3F64::new(1.0, 2.0, 3.0);
        let src = [p, p, p];
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|q| *q + t_true).collect();
        let weights = [1.0, 2.0, 3.0];

        let (r, t, s) = umeyama_f64(&src, &dst, Some(&weights), true)?;

        assert!(
            (s - 1.0).abs() < 1e-12,
            "scale must fall back to 1.0, got {s}"
        );
        // Zero variance leaves the rotation underdetermined; the translation
        // must still map the common point onto its destination.
        let mapped = (r * p) * s + t;
        assert!((mapped.x - dst[0].x).abs() < 1e-9, "x");
        assert!((mapped.y - dst[0].y).abs() < 1e-9, "y");
        assert!((mapped.z - dst[0].z).abs() < 1e-9, "z");
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_repeated_singular_values() -> Result<(), UmeyamaError> {
        // Regular hexagon, pre-rotated so the cross-covariance is dense with two
        // equal non-zero singular values — the degenerate case `svd3_f64`'s Jacobi
        // solver must actually diagonalize (axis-aligned would already be diagonal).
        use crate::SO3F64;
        let pre = SO3F64::exp(Vec3F64::new(0.37, -0.52, 0.61)).matrix();
        let src: Vec<Vec3F64> = (0..6)
            .map(|k| {
                let a = k as f64 * std::f64::consts::FRAC_PI_3;
                pre * Vec3F64::new(a.cos(), a.sin(), 0.0)
            })
            .collect();

        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();

        let (r, t, s) = umeyama_f64(&src, &dst, None, true)?;

        // In-plane singular vectors are ambiguous; assert the transform reproduces
        // the data rather than matching the planted rotation.
        assert!(
            (s - s_true).abs() < 1e-12,
            "scale: got {s}, expected {s_true}"
        );
        assert!(
            (r.determinant() - 1.0).abs() < 1e-12,
            "R must be a proper rotation"
        );
        for (i, p) in src.iter().enumerate() {
            let mapped = (r * *p) * s + t;
            assert!(
                (mapped - dst[i]).length() < 1e-12,
                "point {i} not reproduced: {mapped:?} vs {:?}",
                dst[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_errors() {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let dst: Vec<Vec3F64> = src.to_vec();

        // src/dst length mismatch takes precedence over weight validation.
        let err = umeyama_f64(&src, &dst[..2], Some(&[1.0, 1.0]), true).unwrap_err();
        assert!(
            matches!(err, UmeyamaError::MismatchedInputLengths),
            "got {err:?}"
        );

        // weights length mismatch
        let err = umeyama_f64(&src, &dst, Some(&[1.0]), true).unwrap_err();
        assert!(
            matches!(err, UmeyamaError::MismatchedWeightsLength),
            "got {err:?}"
        );

        // empty weights
        let err = umeyama_f64(&src, &dst, Some(&[]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // non-finite weights
        let err = umeyama_f64(&src, &dst, Some(&[1.0, f64::NAN, 1.0]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");
        let err = umeyama_f64(&src, &dst, Some(&[1.0, f64::INFINITY, 1.0]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // non-positive weight sums
        let err = umeyama_f64(&src, &dst, Some(&[0.0, 0.0, 0.0]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");
        let err = umeyama_f64(&src, &dst, Some(&[1.0, -5.0, 1.0]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // a negative weight even when the sum stays positive
        let err = umeyama_f64(&src, &dst, Some(&[2.0, -1.0, 1.0]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");
    }

    #[test]
    fn test_umeyama_f64_empty_input_is_rejected() {
        let err = umeyama_f64(&[], &[], None, true).unwrap_err();
        assert!(matches!(err, UmeyamaError::EmptyInput), "got {err:?}");

        // Empty input is rejected before weight validation.
        let err = umeyama_f64(&[], &[], Some(&[]), true).unwrap_err();
        assert!(matches!(err, UmeyamaError::EmptyInput), "got {err:?}");
    }

    #[test]
    fn test_umeyama_f64_no_scale_returns_unit() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
        ];
        // dst is scaled by 3.0, but with_scale = false must ignore that.
        let dst: Vec<Vec3F64> = src.iter().map(|p| *p * 3.0).collect();
        let weights = [1.0, 2.0, 0.5, 1.5];

        let (_r, _t, s) = umeyama_f64(&src, &dst, Some(&weights), false)?;
        assert!((s - 1.0).abs() < 1e-15, "scale should be 1.0, got {s}");
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_is_stationary() -> Result<(), UmeyamaError> {
        // Noisy (non-exact) data with non-uniform weights. The exact-recovery
        // tests can pass even if the objective is subtly wrong, so check the
        // first-order optimality conditions instead:
        //   ∂/∂t: Σ wᵢ·rᵢ = 0,   ∂/∂s: Σ wᵢ·rᵢ·(R·srcᵢ) = 0,
        //   ∂/∂R: Σ wᵢ·(R·srcᵢ) × rᵢ = 0,   rᵢ = dstᵢ − (s·R·srcᵢ + t).
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
            Vec3F64::new(1.7, 1.5, -1.9),
            Vec3F64::new(-1.4, 0.6, 0.8),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        let noise = [
            Vec3F64::new(0.01, -0.02, 0.03),
            Vec3F64::new(-0.04, 0.02, -0.01),
            Vec3F64::new(0.02, 0.03, -0.02),
            Vec3F64::new(-0.03, -0.01, 0.04),
            Vec3F64::new(0.01, 0.02, 0.01),
            Vec3F64::new(-0.02, 0.04, -0.03),
        ];
        let dst: Vec<Vec3F64> = src
            .iter()
            .zip(noise)
            .map(|(p, n)| (r_true * *p) * s_true + t_true + n)
            .collect();
        let weights = [0.5, 3.0, 1.0, 2.0, 4.0, 1.5];

        let (r, t, s) = umeyama_f64(&src, &dst, Some(&weights), true)?;

        let mut mean_res = Vec3F64::ZERO;
        let mut rot = Vec3F64::ZERO;
        let mut scale_cond = 0.0f64;
        let mut bound = 0.0f64;
        for (i, (p, d)) in src.iter().zip(&dst).enumerate() {
            let wi = weights[i];
            let rs = r * *p;
            let res = *d - (rs * s + t);
            mean_res += res * wi;
            rot += rs.cross(res) * wi;
            scale_cond += res.dot(rs) * wi;
            bound += wi * res.length() * rs.length();
        }
        let tol = 1e-9 * (bound + 1.0);
        assert!(mean_res.length() <= tol, "Σw·r = 0 violated: {mean_res:?}");
        assert!(
            rot.length() <= tol,
            "rotation stationarity violated: {rot:?}"
        );
        assert!(
            scale_cond.abs() <= tol,
            "scale stationarity violated: {scale_cond}"
        );
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_small_scale_cloud_falls_back_to_unit_scale() -> Result<(), UmeyamaError> {
        // Coordinates ~1e-8 put `var_src` below the 1e-15 guard, so the scale
        // falls back to 1.0 even though the destination is scaled by 2.
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1e-8, 0.0, 0.0),
            Vec3F64::new(0.0, 1e-8, 0.0),
            Vec3F64::new(1e-8, 1e-8, 0.0),
        ];
        let dst: Vec<Vec3F64> = src.iter().map(|p| *p * 2.0).collect();

        let (_r, _t, s) = umeyama_f64(&src, &dst, None, true)?;
        assert!((s - 1.0).abs() < 1e-12, "documented fallback, got {s}");
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_single_nonzero_weight_is_reproduced() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
        ];
        let dst = [
            Vec3F64::new(0.5, 0.1, 1.0),
            Vec3F64::new(1.1, 0.9, -0.2),
            Vec3F64::new(-0.3, 2.4, 0.6),
        ];

        // Only the third correspondence counts: `var_src` is zero, so the scale
        // falls back to 1.0 and the single point must be reproduced.
        let weights = [0.0, 0.0, 1.0];
        let (r, t, s) = umeyama_f64(&src, &dst, Some(&weights), true)?;

        assert!((s - 1.0).abs() < 1e-12, "s: {s}");
        let mapped = (r * src[2]) * s + t;
        assert!(
            (mapped - dst[2]).length() < 1e-9,
            "single weighted point not reproduced: {mapped:?} vs {:?}",
            dst[2]
        );
        Ok(())
    }

    /// `n` seeded-random points in `[-half, half]^3` (deterministic per seed).
    fn random_points(seed: u64, n: usize, half: f64) -> Vec<Vec3F64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                Vec3F64::new(
                    (rng.random::<f64>() * 2.0 - 1.0) * half,
                    (rng.random::<f64>() * 2.0 - 1.0) * half,
                    (rng.random::<f64>() * 2.0 - 1.0) * half,
                )
            })
            .collect()
    }

    /// Adds seeded uniform noise in `[-amp, amp]^3` to each point.
    fn add_noise(seed: u64, points: &mut [Vec3F64], amp: f64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for p in points {
            *p += Vec3F64::new(
                (rng.random::<f64>() * 2.0 - 1.0) * amp,
                (rng.random::<f64>() * 2.0 - 1.0) * amp,
                (rng.random::<f64>() * 2.0 - 1.0) * amp,
            );
        }
    }

    #[test]
    fn test_umeyama_f64_robust_invalid_config() {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let dst: Vec<Vec3F64> = src.to_vec();
        let default = RobustUmeyamaConfig::default();

        let err = umeyama_f64_robust(
            &src,
            &dst,
            &RobustUmeyamaConfig {
                delta: 0.0,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidDelta { delta: 0.0 }),
            "got {err:?}"
        );

        let err = umeyama_f64_robust(
            &src,
            &dst,
            &RobustUmeyamaConfig {
                delta: -1.0,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidDelta { .. }),
            "got {err:?}"
        );

        let err = umeyama_f64_robust(
            &src,
            &dst,
            &RobustUmeyamaConfig {
                delta: f64::NAN,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidDelta { delta } if delta.is_nan()),
            "got {err:?}"
        );

        let err = umeyama_f64_robust(
            &src,
            &dst,
            &RobustUmeyamaConfig {
                max_iters: 0,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidMaxIters { max_iters: 0 }),
            "got {err:?}"
        );

        let err = umeyama_f64_robust(
            &src,
            &dst,
            &RobustUmeyamaConfig {
                tol: 0.0,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidTol { tol: 0.0 }),
            "got {err:?}"
        );

        let err = umeyama_f64_robust(
            &src,
            &dst,
            &RobustUmeyamaConfig {
                tol: f64::INFINITY,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidTol { .. }),
            "got {err:?}"
        );

        // Config validation takes precedence over the input-shape checks.
        let err = umeyama_f64_robust(
            &src,
            &dst[..2],
            &RobustUmeyamaConfig {
                delta: 0.0,
                ..default
            },
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, UmeyamaError::InvalidDelta { .. }),
            "got {err:?}"
        );

        // With a valid config, the core's errors surface unchanged.
        let err = umeyama_f64_robust(&src, &dst[..2], &default, true).unwrap_err();
        assert!(
            matches!(err, UmeyamaError::MismatchedInputLengths),
            "got {err:?}"
        );
        let err = umeyama_f64_robust(&[], &[], &default, true).unwrap_err();
        assert!(matches!(err, UmeyamaError::EmptyInput), "got {err:?}");
    }

    #[test]
    fn test_umeyama_f64_robust_exact_recovery_equals_unweighted() -> Result<(), UmeyamaError> {
        let src = random_points(0x9e37_79b9_7f4a_7c15, 12, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();

        let got = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 1e-9);

        // The IRLS loop converges in one iteration (residuals sit at the SVD
        // error floor, so every Huber weight stays exactly 1). The MAD pass
        // may still refit on those SVD-noise-level residuals, so the robust
        // fit agrees with the plain unweighted fit only to the module's
        // ~1e-9 recovery precision, not bit-for-bit.
        let plain = umeyama_f64(&src, &dst, None, true)?;
        assert_outputs_close(&got, &plain, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_clean_data_is_stable_across_iteration_budgets(
    ) -> Result<(), UmeyamaError> {
        let src = random_points(0x5eed_5eed_5eed_5eed, 10, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();

        // The fit on clean data must not depend on the iteration budget.
        // (The early exit is not observable through the public API; the IRLS
        // loop itself is guarded by the heavy-outlier test below.)
        let full = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
        let cfg_one = RobustUmeyamaConfig {
            max_iters: 1,
            ..RobustUmeyamaConfig::default()
        };
        let one = umeyama_f64_robust(&src, &dst, &cfg_one, true)?;
        assert_outputs_close(&full, &one, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_coplanar_corners() -> Result<(), UmeyamaError> {
        // Eight tag-like corners on the z = 0 plane (rank-deficient
        // covariance; the crate's `svd3_f64` must carry the fit, no
        // conditioning hacks).
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(1.0, 1.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.5, 0.0, 0.0),
            Vec3F64::new(1.0, 0.5, 0.0),
            Vec3F64::new(0.5, 1.0, 0.0),
            Vec3F64::new(0.0, 0.5, 0.0),
        ];
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.3;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();

        let got = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_rejects_gross_outliers() -> Result<(), UmeyamaError> {
        let mut src = random_points(0x1001, 20, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;

        // Realistic inlier noise so the MAD pass has a meaningful positive
        // scale to work with (on noiseless synthetic inliers the scale sits
        // at the SVD error floor, where rejection adds little).
        add_noise(0x1002, &mut src, 5e-4);
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        add_noise(0x1003, &mut dst, 5e-4);

        // 10% gross outliers: displace 2 of 20 by 40-60 units.
        dst[3] += Vec3F64::new(50.0, 0.0, 0.0);
        dst[17] += Vec3F64::new(0.0, -40.0, 30.0);

        let got = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 5e-3);

        // Good points keep tiny residuals; the corrupted ones keep large ones
        // (proof of down-weighting and rejection).
        let mut max_good = 0.0_f64;
        for (i, (p, d)) in src.iter().zip(&dst).enumerate() {
            let r = ((got.0 * *p) * got.2 + got.1 - *d).length();
            if i == 3 || i == 17 {
                assert!(r > 20.0, "outlier {i} residual too small: {r}");
            } else {
                max_good = max_good.max(r);
            }
        }
        assert!(max_good < 5e-3, "good-point residual too large: {max_good}");
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_is_order_invariant() -> Result<(), UmeyamaError> {
        let mut src = random_points(0x2001, 24, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        // Noisy inliers and outliers so the MAD refit path is order-checked too.
        add_noise(0x2002, &mut src, 5e-4);
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        add_noise(0x2003, &mut dst, 5e-4);
        for i in [2usize, 11, 20] {
            dst[i] += Vec3F64::new(45.0, -45.0, 45.0);
        }

        let a = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;

        // Identical permutation of both sets: the fit is a symmetric function
        // of the correspondences, so only floating-point reassociation differs.
        let mut perm: Vec<usize> = (0..src.len()).collect();
        perm.reverse();
        perm.rotate_left(5);
        let src_b: Vec<Vec3F64> = perm.iter().map(|&i| src[i]).collect();
        let dst_b: Vec<Vec3F64> = perm.iter().map(|&i| dst[i]).collect();
        let b = umeyama_f64_robust(&src_b, &dst_b, &RobustUmeyamaConfig::default(), true)?;

        assert_outputs_close(&a, &b, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_is_deterministic() -> Result<(), UmeyamaError> {
        let mut src = random_points(0x3001, 16, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        add_noise(0x3002, &mut src, 5e-4);
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        add_noise(0x3003, &mut dst, 5e-4);
        dst[7] += Vec3F64::new(0.0, 60.0, -30.0);

        let a = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
        let b = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;

        let ar: [f64; 9] = a.0.into();
        let br: [f64; 9] = b.0.into();
        assert_eq!(ar, br, "R must be bit-for-bit identical");
        assert_eq!(a.1, b.1, "t must be bit-for-bit identical");
        assert_eq!(a.2, b.2, "s must be bit-for-bit identical");
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_mad_refit_rejects_heavy_outliers() -> Result<(), UmeyamaError> {
        // 40% gross outliers: the IRLS-only Huber fixpoint stays visibly
        // biased (bias ~ delta * outlier fraction), so passing the tight
        // tolerance below proves the centered MAD pass fired and the
        // unweighted refit on survivors landed on the clean inliers. The
        // centered window is what makes this work: at the biased fixpoint
        // the inlier residuals sit ~bias away from zero, where a raw
        // residual threshold would reject them too. This case also requires
        // the IRLS loop: with the loop disabled, the unweighted seed fit is
        // too biased for the MAD pass to recover (verified during review).
        let mut src = random_points(0x4001, 20, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.7;
        add_noise(0x4002, &mut src, 5e-4);
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        add_noise(0x4003, &mut dst, 5e-4);
        for i in [1usize, 4, 7, 9, 12, 14, 16, 19] {
            dst[i] += Vec3F64::new(30.0, 40.0, -50.0);
        }

        let got = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 5e-3);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_robust_rigid_rejects_gross_outliers() -> Result<(), UmeyamaError> {
        // Rigid mode (`with_scale = false`): the true scale is 1, so the fit
        // must return exactly 1.0 while still rejecting gross outliers.
        let mut src = random_points(0x5001, 20, 2.0);
        let r_true = rot_z90();
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        add_noise(0x5002, &mut src, 5e-4);
        let mut dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();
        add_noise(0x5003, &mut dst, 5e-4);
        // 3 of 20 corrupted by 40-60 units.
        dst[2] += Vec3F64::new(50.0, 0.0, 0.0);
        dst[9] += Vec3F64::new(0.0, -40.0, 30.0);
        dst[15] += Vec3F64::new(45.0, 45.0, 0.0);

        let (r, t, s) = umeyama_f64_robust(&src, &dst, &RobustUmeyamaConfig::default(), false)?;

        assert_eq!(s, 1.0, "rigid mode must return s = 1.0 exactly");
        assert_matches_true(&(r, t, s), &r_true, &t_true, 1.0, 5e-3);

        // Good points map tightly; the corrupted ones keep large residuals.
        let mut max_good = 0.0_f64;
        for (i, (p, d)) in src.iter().zip(&dst).enumerate() {
            let res = (r * *p + t - *d).length();
            if i == 2 || i == 9 || i == 15 {
                assert!(res > 20.0, "outlier {i} residual too small: {res}");
            } else {
                max_good = max_good.max(res);
            }
        }
        assert!(max_good < 5e-3, "good-point residual too large: {max_good}");
        Ok(())
    }
}
