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
/// Solves for (R, t, s) minimizing `Σ ‖dst_i − (s·R·src_i + t)‖²`.
/// When `with_scale` is false, s is fixed to 1.0 (rigid alignment only).
pub fn umeyama_f64(src: &[Vec3F64], dst: &[Vec3F64], with_scale: bool) -> UmeyamaResultF64 {
    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }
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
}
