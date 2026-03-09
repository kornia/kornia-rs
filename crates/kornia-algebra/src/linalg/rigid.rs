//! Rigid alignment utilities (Kabsch / Umeyama)

use crate::{linalg::svd::svd3_f64, Mat3AF32, Mat3F64, Vec3AF32, Vec3F64};
use thiserror::Error;

/// Rotation (R), translation (t), and scale (s) output of Umeyama without scaling (s = 1).
pub type UmeyamaOutput = (Mat3AF32, Vec3AF32, f32);

/// Error type for Umeyama rigid alignment operations.
#[derive(Debug, Error)]
pub enum UmeyamaError {
    /// Source and destination arrays must have the same length
    #[error("Source and destination arrays must have the same length")]
    MismatchedInputLengths,
    #[error("Input arrays must not be empty")]
    EmptyInput,
}

/// Result type alias for Umeyama.
pub type UmeyamaResult = Result<UmeyamaOutput, UmeyamaError>;

/// Umeyama/Kabsch algorithm without scale.
/// Returns (R, t, s) where s == 1.0.
pub fn umeyama(src: &[Vec3AF32], dst: &[Vec3AF32]) -> UmeyamaResult {
    if src.is_empty() {
        return Err(UmeyamaError::EmptyInput);
    }

    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }
    let n = src.len() as f64;

    // 1. Calculate Centroids using raw f64 to bypass missing wrapper traits
    let mut mu_s_x = 0.0;
    let mut mu_s_y = 0.0;
    let mut mu_s_z = 0.0;
    let mut mu_d_x = 0.0;
    let mut mu_d_y = 0.0;
    let mut mu_d_z = 0.0;

    for i in 0..src.len() {
        mu_s_x += src[i].x as f64;
        mu_s_y += src[i].y as f64;
        mu_s_z += src[i].z as f64;
        mu_d_x += dst[i].x as f64;
        mu_d_y += dst[i].y as f64;
        mu_d_z += dst[i].z as f64;
    }

    mu_s_x /= n;
    mu_s_y /= n;
    mu_s_z /= n;
    mu_d_x /= n;
    mu_d_y /= n;
    mu_d_z /= n;

    // 2. Compute Covariance matrix H in raw f64
    let mut h_00 = 0.0;
    let mut h_01 = 0.0;
    let mut h_02 = 0.0;
    let mut h_10 = 0.0;
    let mut h_11 = 0.0;
    let mut h_12 = 0.0;
    let mut h_20 = 0.0;
    let mut h_21 = 0.0;
    let mut h_22 = 0.0;

    for i in 0..src.len() {
        let sc_x = src[i].x as f64 - mu_s_x;
        let sc_y = src[i].y as f64 - mu_s_y;
        let sc_z = src[i].z as f64 - mu_s_z;

        let dc_x = dst[i].x as f64 - mu_d_x;
        let dc_y = dst[i].y as f64 - mu_d_y;
        let dc_z = dst[i].z as f64 - mu_d_z;

        // Outer product H = dc * sc^T
        h_00 += dc_x * sc_x;
        h_01 += dc_x * sc_y;
        h_02 += dc_x * sc_z;
        h_10 += dc_y * sc_x;
        h_11 += dc_y * sc_y;
        h_12 += dc_y * sc_z;
        h_20 += dc_z * sc_x;
        h_21 += dc_z * sc_y;
        h_22 += dc_z * sc_z;
    }

    let h = Mat3F64::from_cols(
        Vec3F64::new(h_00 / n, h_10 / n, h_20 / n), // Col 0
        Vec3F64::new(h_01 / n, h_11 / n, h_21 / n), // Col 1
        Vec3F64::new(h_02 / n, h_12 / n, h_22 / n), // Col 2
    );

    // 3. Internal f64 SVD path.
    let svd = svd3_f64(&h);

    // Use getter methods instead of private fields.
    // We use * to dereference because the methods return &Mat3F64.
    let u = *svd.u();
    let v = *svd.v();

    // Keep behavior consistent with existing EPnP regression expectations:
    // if det(R) < 0, flip the third column of R.
    let mut r = u * v.transpose();
    if r.determinant() < 0.0 {
        r.z_axis = -r.z_axis;
    }
    let tx = mu_d_x - (r.x_axis.x * mu_s_x + r.y_axis.x * mu_s_y + r.z_axis.x * mu_s_z);
    let ty = mu_d_y - (r.x_axis.y * mu_s_x + r.y_axis.y * mu_s_y + r.z_axis.y * mu_s_z);
    let tz = mu_d_z - (r.x_axis.z * mu_s_x + r.y_axis.z * mu_s_y + r.z_axis.z * mu_s_z);

    // 6. Cast back to f32 for the output
    let r_cols = [
        r.x_axis.x as f32,
        r.x_axis.y as f32,
        r.x_axis.z as f32, // Col 1
        r.y_axis.x as f32,
        r.y_axis.y as f32,
        r.y_axis.z as f32, // Col 2
        r.z_axis.x as f32,
        r.z_axis.y as f32,
        r.z_axis.z as f32, // Col 3
    ];

    Ok((
        Mat3AF32::from_cols_array(&r_cols),
        Vec3AF32::new(tx as f32, ty as f32, tz as f32),
        1.0,
    ))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "approx")]
    fn test_umeyama_translation_only() -> Result<(), UmeyamaError> {
        use approx::assert_relative_eq;
        let src: [Vec3AF32; 4] = [
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
        ];
        // True transform: Identity rotation, only translation
        let r = Mat3AF32::IDENTITY;
        let t = Vec3AF32::new(10.5, -5.2, 3.14);

        let mut dst = [Vec3AF32::ZERO; 4];
        for i in 0..4 {
            dst[i] = r * src[i] + t;
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        assert_relative_eq!(r_est, r, epsilon = 1e-5);
        assert_relative_eq!(t_est, t, epsilon = 1e-5);
        Ok(())
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
        let r = Mat3AF32::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let t = Vec3AF32::new(0.5, -0.3, 2.0);

        let mut dst = [Vec3AF32::ZERO; 4];
        for i in 0..4 {
            dst[i] = r * src[i] + t;
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        assert_relative_eq!(r_est, r, epsilon = 1e-5);
        assert_relative_eq!(t_est, t, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    #[cfg(feature = "approx")]
    fn test_umeyama_complex_transform() -> Result<(), UmeyamaError> {
        use approx::assert_relative_eq;
        // Arbitrary 3D point cloud
        let src: [Vec3AF32; 5] = [
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 2.0, 3.0),
            Vec3AF32::new(-1.0, 5.0, -2.0),
            Vec3AF32::new(4.0, -1.0, 7.0),
            Vec3AF32::new(2.5, 3.5, 1.5),
        ];

        // True transform: A valid orthonormal rotation matrix across all 3 axes
        // Col 1: [0.36, 0.48, 0.8]
        // Col 2: [-0.80, 0.60, 0.0]
        // Col 3: [-0.48, -0.64, 0.60]
        let r =
            Mat3AF32::from_cols_array(&[0.36, 0.48, 0.80, -0.80, 0.60, 0.00, -0.48, -0.64, 0.60]);
        let t = Vec3AF32::new(-10.5, 20.2, -5.5);

        let mut dst = [Vec3AF32::ZERO; 5];
        for i in 0..5 {
            dst[i] = r * src[i] + t;
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        assert_relative_eq!(r_est, r, epsilon = 1e-5);
        assert_relative_eq!(t_est, t, epsilon = 1e-5);
        Ok(())
    }
}
