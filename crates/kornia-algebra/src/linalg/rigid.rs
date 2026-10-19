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
}
