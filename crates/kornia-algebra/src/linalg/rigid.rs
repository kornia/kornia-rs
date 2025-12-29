//! Rigid alignment utilities (Kabsch / Umeyama)

use crate::{Mat3AF32, Mat3F32, Vec3AF32, Vec3F32};
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

#[cfg(test)]
mod tests {

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
}
