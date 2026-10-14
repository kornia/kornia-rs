//! Rigid alignment utilities (Kabsch / Umeyama)

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

/// Umeyama alignment with optional scale estimation (f64 precision).
///
/// Solves for the rotation `R`, translation `t`, and optional scale `s` that
/// minimize the sum of squared residuals `Σ ‖dst_i − (s·R·src_i + t)‖²`.
///
/// This is the unweighted case of [`umeyama_f64_weighted`] (every weight equal).
/// It produces byte-identical output to previous releases; the weighted variant
/// agrees with it to within floating-point rounding for uniform weights.
///
/// Point coordinates are assumed finite; non-finite coordinates are not
/// validated and propagate to the result.
///
/// # Arguments
///
/// * `src` - Source points. Must have the same length as `dst`.
/// * `dst` - Destination points. Must have the same length as `src`.
/// * `with_scale` - If `true`, estimates the scale `s`; otherwise `s` is fixed
///   to `1.0` (rigid alignment only).
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
/// Returns [`UmeyamaError::SvdU`] or [`UmeyamaError::SvdVT`] if the SVD fails.
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
/// let (_r_est, t_est, _s) = umeyama_f64(&src, &dst, false)?;
/// assert!((t_est.x - 1.0).abs() < 1e-12);
/// assert!((t_est.y - 2.0).abs() < 1e-12);
/// assert!((t_est.z - 3.0).abs() < 1e-12);
/// # Ok::<(), kornia_algebra::linalg::rigid::UmeyamaError>(())
/// ```
pub fn umeyama_f64(src: &[Vec3F64], dst: &[Vec3F64], with_scale: bool) -> UmeyamaResultF64 {
    umeyama_f64_core(src, dst, None, with_scale)
}

/// Weighted Umeyama alignment with optional scale estimation (f64 precision).
///
/// Solves for the rotation `R`, translation `t`, and optional scale `s` that
/// minimize the weighted sum of squared residuals
/// `Σ w_i ‖dst_i − (s·R·src_i + t)‖²`. Weights need not be normalized; they are
/// normalized internally by their sum `Σ w_i`, so only their relative values
/// matter. Uniform weights reproduce the result of [`umeyama_f64`] with the same
/// `with_scale` flag, and a point with weight `0` is ignored by the fit.
///
/// Point coordinates are assumed finite; non-finite coordinates are not
/// validated and propagate to the result.
///
/// # Arguments
///
/// * `src` - Source points. Must have the same length as `dst` and `weights`.
/// * `dst` - Destination points. Must have the same length as `src` and `weights`.
/// * `weights` - Per-point weights, one entry per point. Must be finite,
///   non-negative, and have a positive sum.
/// * `with_scale` - If `true`, estimates the scale `s`; otherwise `s` is fixed
///   to `1.0` (rigid alignment only).
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
/// Returns [`UmeyamaError::MismatchedWeightsLength`] if `weights` does not have
/// exactly one entry per point.
/// Returns [`UmeyamaError::InvalidWeights`] if `weights` contains a non-finite
/// or negative value, or has a non-positive sum.
/// Returns [`UmeyamaError::SvdU`] or [`UmeyamaError::SvdVT`] if the SVD fails.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::linalg::rigid::umeyama_f64_weighted;
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
/// let weights = [1.0, 1.0, 1.0, 1.0];
///
/// let (_r_est, t_est, _s) = umeyama_f64_weighted(&src, &dst, &weights, false)?;
/// assert!((t_est.x - 1.0).abs() < 1e-12);
/// assert!((t_est.y - 2.0).abs() < 1e-12);
/// assert!((t_est.z - 3.0).abs() < 1e-12);
/// # Ok::<(), kornia_algebra::linalg::rigid::UmeyamaError>(())
/// ```
pub fn umeyama_f64_weighted(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    weights: &[f64],
    with_scale: bool,
) -> UmeyamaResultF64 {
    umeyama_f64_core(src, dst, Some(weights), with_scale)
}

/// Accumulated first- and second-order statistics shared by the weighted and
/// unweighted Umeyama paths.
#[derive(Debug)]
struct Moments {
    /// Source centroid.
    mu_s: [f64; 3],
    /// Destination centroid.
    mu_d: [f64; 3],
    /// Cross-covariance of the centered points, normalized by the total weight
    /// (the point count when unweighted).
    h: [[f64; 3]; 3],
    /// Variance of the centered source points, normalized by the total weight
    /// (the point count when unweighted).
    var_src: f64,
}

/// Unweighted moments (unit weights), matching the original `umeyama_f64`
/// accumulation exactly. Kept separate from [`weighted_moments`] so that
/// `umeyama_f64` keeps producing byte-identical output.
fn unweighted_moments(src: &[Vec3F64], dst: &[Vec3F64]) -> Moments {
    let n = src.len() as f64;

    // Centroids
    let mut mu_s = [0.0f64; 3];
    let mut mu_d = [0.0f64; 3];
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

    // Cross-covariance H = Σ (dst_c)^T · src_c / n
    // and source variance σ²_src = Σ ‖src_c‖² / n
    let mut h = [[0.0f64; 3]; 3];
    let mut var_src = 0.0f64;
    for i in 0..src.len() {
        let sc = [src[i].x - mu_s[0], src[i].y - mu_s[1], src[i].z - mu_s[2]];
        let dc = [dst[i].x - mu_d[0], dst[i].y - mu_d[1], dst[i].z - mu_d[2]];
        for (r, &dc_r) in dc.iter().enumerate() {
            for (c, &sc_c) in sc.iter().enumerate() {
                h[r][c] += dc_r * sc_c;
            }
        }
        var_src += sc[0] * sc[0] + sc[1] * sc[1] + sc[2] * sc[2];
    }
    for row in &mut h {
        for val in row {
            *val /= n;
        }
    }
    var_src /= n;

    Moments {
        mu_s,
        mu_d,
        h,
        var_src,
    }
}

/// Weighted moments with per-point weights. Validates the weights, then scales
/// them by their maximum before accumulating; the scale cancels in every ratio,
/// so the result is independent of the weight magnitude.
fn weighted_moments(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    weights: &[f64],
) -> Result<Moments, UmeyamaError> {
    // Validate the weights and find the maximum in a single pass. This runs
    // before the length check so that an empty or otherwise unusable weight set
    // is reported as `InvalidWeights` rather than a length mismatch. The
    // closed-form Umeyama minimizer is only valid for `w_i >= 0`.
    let mut max_w = 0.0f64;
    for &wi in weights {
        if !wi.is_finite() || wi < 0.0 {
            return Err(UmeyamaError::InvalidWeights);
        }
        max_w = max_w.max(wi);
    }
    // Empty and all-zero weights both leave `max_w == 0`.
    if max_w <= 0.0 {
        return Err(UmeyamaError::InvalidWeights);
    }
    if weights.len() != src.len() {
        return Err(UmeyamaError::MismatchedWeightsLength);
    }

    // Normalize by the maximum so every `w_i' <= 1`: the weight sum stays in
    // `[1, n]` and every `w_i' * p_i` product is bounded by `|p_i|`, so neither
    // can overflow for extreme-but-finite weights. `max_w` cancels in each ratio
    // below, so the normalized result equals the unnormalized one. Divide
    // directly (rather than by a precomputed reciprocal) so the maximum weight
    // maps to exactly `1.0` and no subnormal reciprocal rounds the result.

    // Weighted centroids μ = Σ w_i p_i / Σ w_i
    let mut mu_s = [0.0f64; 3];
    let mut mu_d = [0.0f64; 3];
    let mut sum_w = 0.0f64;
    for i in 0..src.len() {
        let wi = weights[i] / max_w;
        sum_w += wi;
        mu_s[0] += wi * src[i].x;
        mu_s[1] += wi * src[i].y;
        mu_s[2] += wi * src[i].z;
        mu_d[0] += wi * dst[i].x;
        mu_d[1] += wi * dst[i].y;
        mu_d[2] += wi * dst[i].z;
    }
    for x in &mut mu_s {
        *x /= sum_w;
    }
    for x in &mut mu_d {
        *x /= sum_w;
    }

    // Weighted cross-covariance H = Σ w_i (dst_c)^T · src_c / Σ w_i
    // and weighted source variance σ²_src = Σ w_i ‖src_c‖² / Σ w_i
    let mut h = [[0.0f64; 3]; 3];
    let mut var_src = 0.0f64;
    for i in 0..src.len() {
        let wi = weights[i] / max_w;
        let sc = [src[i].x - mu_s[0], src[i].y - mu_s[1], src[i].z - mu_s[2]];
        let dc = [dst[i].x - mu_d[0], dst[i].y - mu_d[1], dst[i].z - mu_d[2]];
        for (r, &dc_r) in dc.iter().enumerate() {
            for (c, &sc_c) in sc.iter().enumerate() {
                h[r][c] += wi * dc_r * sc_c;
            }
        }
        var_src += wi * (sc[0] * sc[0] + sc[1] * sc[1] + sc[2] * sc[2]);
    }
    for row in &mut h {
        for val in row {
            *val /= sum_w;
        }
    }
    var_src /= sum_w;

    Ok(Moments {
        mu_s,
        mu_d,
        h,
        var_src,
    })
}

/// Shared core for [`umeyama_f64`] and [`umeyama_f64_weighted`].
///
/// Both paths are allocation-free (stack arrays + borrowed slices) and share
/// the nalgebra SVD tail below.
fn umeyama_f64_core(
    src: &[Vec3F64],
    dst: &[Vec3F64],
    weights: Option<&[f64]>,
    with_scale: bool,
) -> UmeyamaResultF64 {
    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }

    // Do NOT unify these two paths: the `None` arm is byte-identical to the
    // historical `umeyama_f64` accumulation, and folding in a `w_i = 1.0`
    // multiply or a `sum_w` divisor would shift its output in the last ULP.
    let Moments {
        mu_s,
        mu_d,
        h,
        var_src,
    } = match weights {
        None => unweighted_moments(src, dst),
        Some(w) => weighted_moments(src, dst, w)?,
    };

    // SVD of H via nalgebra (avoids the known svd3_f64 Jacobi bug)
    let h_na = Matrix3::<f64>::from_row_slice(&[
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    ]);
    let svd = SVD::new(h_na, true, true);
    let Some(u) = svd.u else {
        return Err(UmeyamaError::SvdU);
    };
    let Some(v_t) = svd.v_t else {
        return Err(UmeyamaError::SvdVT);
    };

    // Reflection correction: S = diag(1, 1, det(U)·det(V^T))
    let d = u.determinant() * v_t.determinant();
    let s_sign = if d < 0.0 { -1.0 } else { 1.0 };
    let s_diag = Matrix3::<f64>::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, s_sign));

    // R = U · S · V^T
    let r_na = u * s_diag * v_t;

    // Scale
    let scale = if with_scale && var_src > 1e-15 {
        // s = tr(S · Σ) / σ²_src
        let sigma = &svd.singular_values;
        (sigma[0] + sigma[1] + s_sign * sigma[2]) / var_src
    } else {
        1.0
    };

    // Translation: t = μ_d − s·R·μ_s
    let mu_s_na = nalgebra::Vector3::new(mu_s[0], mu_s[1], mu_s[2]);
    let mu_d_na = nalgebra::Vector3::new(mu_d[0], mu_d[1], mu_d[2]);
    let t_na = mu_d_na - scale * r_na * mu_s_na;

    // Convert to glam types (row-major → column-major for from_cols_array)
    let r_flat = [
        r_na[(0, 0)],
        r_na[(1, 0)],
        r_na[(2, 0)],
        r_na[(0, 1)],
        r_na[(1, 1)],
        r_na[(2, 1)],
        r_na[(0, 2)],
        r_na[(1, 2)],
        r_na[(2, 2)],
    ];

    Ok((
        Mat3F64::from_cols_array(&r_flat),
        Vec3F64::new(t_na[0], t_na[1], t_na[2]),
        scale,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();

        let (r_est, t_est, s_est) = umeyama_f64(&src, &dst, false)?;
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

        let (_r_est, t_est, s_est) = umeyama_f64(&src, &dst, true)?;
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
        let r_true_arr: [f64; 9] = (*r_true).into();
        let r_got_arr: [f64; 9] = got.0.into();
        for i in 0..9 {
            assert!(
                (r_got_arr[i] - r_true_arr[i]).abs() < eps,
                "R mismatch at {i}: {} vs {}",
                r_got_arr[i],
                r_true_arr[i]
            );
        }
        assert!((got.1.x - t_true.x).abs() < eps, "t.x: {}", got.1.x);
        assert!((got.1.y - t_true.y).abs() < eps, "t.y: {}", got.1.y);
        assert!((got.1.z - t_true.z).abs() < eps, "t.z: {}", got.1.z);
        assert!((got.2 - s_true).abs() < eps, "s: {} vs {}", got.2, s_true);
    }

    /// Assert two [`Moments`] agree component-wise within `eps`.
    fn assert_moments_close(a: &Moments, b: &Moments, eps: f64) {
        for k in 0..3 {
            assert!((a.mu_s[k] - b.mu_s[k]).abs() < eps, "mu_s[{k}]");
            assert!((a.mu_d[k] - b.mu_d[k]).abs() < eps, "mu_d[{k}]");
        }
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (a.h[r][c] - b.h[r][c]).abs() < eps,
                    "h[{r}][{c}]: {} vs {}",
                    a.h[r][c],
                    b.h[r][c]
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

    #[test]
    fn test_unweighted_moments_centroids() {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        ];
        let m = unweighted_moments(&src, &src);

        for k in 0..3 {
            assert!((m.mu_s[k] - 0.25).abs() < 1e-12, "mu_s[{k}]");
            assert!((m.mu_d[k] - 0.25).abs() < 1e-12, "mu_d[{k}]");
        }
    }

    #[test]
    fn test_unweighted_moments_covariance_is_symmetric_with_matching_trace() {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        ];
        let m = unweighted_moments(&src, &src);

        // Σ‖sc‖² / 4 = 2.25 / 4.
        assert!((m.var_src - 0.5625).abs() < 1e-12, "var_src: {}", m.var_src);

        // H is symmetric and trace(H) == var_src (both equal Σ‖sc‖² / n).
        let mut trace = 0.0;
        for r in 0..3 {
            trace += m.h[r][r];
            for c in 0..3 {
                assert!((m.h[r][c] - m.h[c][r]).abs() < 1e-12, "H not symmetric");
            }
        }
        assert!((trace - m.var_src).abs() < 1e-12, "trace(H) != var_src");
    }

    #[test]
    fn test_weighted_moments_weighted_centroids() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let w = [1.0, 2.0, 3.0];
        let m = weighted_moments(&src, &src, &w)?;

        // μ = Σ w_i p_i / Σ w_i, with Σ w_i = 6.
        assert!((m.mu_s[0] - 2.0 / 6.0).abs() < 1e-12, "mu_s[0]");
        assert!((m.mu_s[1] - 3.0 / 6.0).abs() < 1e-12, "mu_s[1]");
        assert!(m.mu_s[2].abs() < 1e-12, "mu_s[2]");
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
        let m = weighted_moments(&src, &src, &w)?;

        let mut trace = 0.0;
        for r in 0..3 {
            trace += m.h[r][r];
            for c in 0..3 {
                assert!((m.h[r][c] - m.h[c][r]).abs() < 1e-12, "H not symmetric");
            }
        }
        assert!((trace - m.var_src).abs() < 1e-12, "trace(H) != var_src");
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

        let a = weighted_moments(&src, &dst, &w)?;
        let b = weighted_moments(&src, &dst, &scaled)?;
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
        let unweighted = unweighted_moments(&src, &dst);
        let weights = vec![1.0; src.len()];
        let weighted = weighted_moments(&src, &dst, &weights)?;
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
        let err = weighted_moments(&src, &src, &[]).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // non-finite weights
        let err = weighted_moments(&src, &src, &[1.0, f64::NAN, 1.0]).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // all-zero weights
        let err = weighted_moments(&src, &src, &[0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // negative sum
        let err = weighted_moments(&src, &src, &[1.0, -5.0, 1.0]).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // mixed-sign weights whose sum is positive
        let err = weighted_moments(&src, &src, &[2.0, -1.0, 1.0]).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // a single negative weight
        let err = weighted_moments(&src, &src, &[-1.0, 1.0, 1.0]).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // wrong length (otherwise valid weights)
        let err = weighted_moments(&src, &src, &[1.0]).unwrap_err();
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

        let a = weighted_moments(&src, &dst, &huge)?;
        let b = weighted_moments(&src, &dst, &unit)?;
        assert_moments_close(&a, &b, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_recovers_exact_transform() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(1.0, 1.0, 1.0),
        ];
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 2.0;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        let weights = [0.5, 3.0, 1.0, 2.0, 4.0];

        let got = umeyama_f64_weighted(&src, &dst, &weights, true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_uniform_equals_unweighted() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
            Vec3F64::new(1.7, 1.5, -1.9),
        ];
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();

        let unweighted = umeyama_f64(&src, &dst, true)?;
        let weights = vec![1.0; src.len()];
        let weighted = umeyama_f64_weighted(&src, &dst, &weights, true)?;

        assert_outputs_close(&unweighted, &weighted, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_normalization_invariant() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
            Vec3F64::new(1.7, 1.5, -1.9),
        ];
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();

        let weights = [0.5, 3.0, 1.0, 2.0, 4.0];
        let scaled_weights: Vec<f64> = weights.iter().map(|w| w * 7.5).collect();

        let a = umeyama_f64_weighted(&src, &dst, &weights, true)?;
        let b = umeyama_f64_weighted(&src, &dst, &scaled_weights, true)?;
        assert_outputs_close(&a, &b, 1e-12);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_zero_weight_point_ignored() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(0.5, 0.5, 0.5),
        ];
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let mut dst: Vec<Vec3F64> = src.iter().map(|p| r_true * *p + t_true).collect();
        // Corrupt the last destination wildly; its weight is zero so it must be
        // ignored entirely.
        dst[4] = Vec3F64::new(1e6, -1e6, 1e6);
        let weights = [1.0, 2.0, 0.5, 1.5, 0.0];

        let got = umeyama_f64_weighted(&src, &dst, &weights, false)?;
        assert_matches_true(&got, &r_true, &t_true, 1.0, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_coplanar_input() -> Result<(), UmeyamaError> {
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
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 1.3;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        let weights = [1.0, 2.0, 1.5, 0.7, 1.1, 3.0, 0.9, 1.3];

        let got = umeyama_f64_weighted(&src, &dst, &weights, true)?;
        assert_matches_true(&got, &r_true, &t_true, s_true, 1e-9);
        Ok(())
    }

    #[test]
    fn test_umeyama_f64_weighted_collinear_input() -> Result<(), UmeyamaError> {
        // Points along the x-axis: the covariance has rank 1.
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(2.0, 0.0, 0.0),
            Vec3F64::new(3.0, 0.0, 0.0),
        ];
        let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let s_true = 2.0;
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        let weights = [1.0, 2.0, 0.5, 1.5];

        let (r, t, s) = umeyama_f64_weighted(&src, &dst, &weights, true)?;

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
    fn test_umeyama_f64_weighted_identical_points() -> Result<(), UmeyamaError> {
        // All source points coincide: zero variance, rank-0 covariance.
        let p = Vec3F64::new(1.0, 2.0, 3.0);
        let src = [p, p, p];
        let t_true = Vec3F64::new(0.5, -0.3, 2.0);
        let dst: Vec<Vec3F64> = src.iter().map(|q| *q + t_true).collect();
        let weights = [1.0, 2.0, 3.0];

        let (r, t, s) = umeyama_f64_weighted(&src, &dst, &weights, true)?;

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
    fn test_umeyama_f64_weighted_errors() {
        let src = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
        ];
        let dst: Vec<Vec3F64> = src.to_vec();

        // src/dst length mismatch takes precedence over weight validation.
        let err = umeyama_f64_weighted(&src, &dst[..2], &[1.0, 1.0], true).unwrap_err();
        assert!(
            matches!(err, UmeyamaError::MismatchedInputLengths),
            "got {err:?}"
        );

        // weights length mismatch
        let err = umeyama_f64_weighted(&src, &dst, &[1.0], true).unwrap_err();
        assert!(
            matches!(err, UmeyamaError::MismatchedWeightsLength),
            "got {err:?}"
        );

        // empty weights
        let err = umeyama_f64_weighted(&src, &dst, &[], true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // non-finite weights
        let err = umeyama_f64_weighted(&src, &dst, &[1.0, f64::NAN, 1.0], true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");
        let err = umeyama_f64_weighted(&src, &dst, &[1.0, f64::INFINITY, 1.0], true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // non-positive weight sums
        let err = umeyama_f64_weighted(&src, &dst, &[0.0, 0.0, 0.0], true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");
        let err = umeyama_f64_weighted(&src, &dst, &[1.0, -5.0, 1.0], true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");

        // a negative weight even when the sum stays positive
        let err = umeyama_f64_weighted(&src, &dst, &[2.0, -1.0, 1.0], true).unwrap_err();
        assert!(matches!(err, UmeyamaError::InvalidWeights), "got {err:?}");
    }

    #[test]
    fn test_umeyama_f64_weighted_no_scale_returns_unit() -> Result<(), UmeyamaError> {
        let src = [
            Vec3F64::new(0.2, -0.4, 1.1),
            Vec3F64::new(1.3, 0.7, -0.5),
            Vec3F64::new(-0.8, 2.1, 0.3),
            Vec3F64::new(0.9, -1.2, 2.4),
        ];
        // dst is scaled by 3.0, but with_scale = false must ignore that.
        let dst: Vec<Vec3F64> = src.iter().map(|p| *p * 3.0).collect();
        let weights = [1.0, 2.0, 0.5, 1.5];

        let (_r, _t, s) = umeyama_f64_weighted(&src, &dst, &weights, false)?;
        assert!((s - 1.0).abs() < 1e-15, "scale should be 1.0, got {s}");
        Ok(())
    }
}
