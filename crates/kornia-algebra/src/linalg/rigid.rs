//! Rigid alignment utilities (Kabsch / Umeyama)

use crate::{linalg::svd::svd3_f64, Mat3AF32, Mat3F64, Vec3AF32, Vec3F64};
use thiserror::Error;

/// Rotation (R), translation (t), and scale (s) output of Umeyama without scaling (s = 1).
pub type UmeyamaOutput = (Mat3AF32, Vec3AF32, f32);

/// Error type for Umeyama rigid alignment operations.
#[derive(Debug, Error)]
pub enum UmeyamaError {
    #[error("Source and destination arrays must have the same length")]
    MismatchedInputLengths,

    #[error("Input arrays must not be empty")]
    EmptyInput,

    #[deprecated(note = "Internal SVD implementation no longer fails. This variant is obsolete.")]
    #[error("Failed to compute U in SVD")]
    SvdU,

    #[deprecated(note = "Internal SVD implementation no longer fails. This variant is obsolete.")]
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

    if src.is_empty() {
        return Err(UmeyamaError::EmptyInput);
    }

    let n = src.len() as f64;

    // 1. Calculate Centroids using vector accumulation
    let mut mu_s = Vec3F64::ZERO;
    let mut mu_d = Vec3F64::ZERO;

    for i in 0..src.len() {
        mu_s += Vec3F64::new(src[i].x as f64, src[i].y as f64, src[i].z as f64);
        mu_d += Vec3F64::new(dst[i].x as f64, dst[i].y as f64, dst[i].z as f64);
    }

    mu_s /= n;
    mu_d /= n;

    // 2. Compute Covariance matrix H
    let mut h = Mat3F64::ZERO;

    for i in 0..src.len() {
        let sc = Vec3F64::new(src[i].x as f64, src[i].y as f64, src[i].z as f64) - mu_s;
        let dc = Vec3F64::new(dst[i].x as f64, dst[i].y as f64, dst[i].z as f64) - mu_d;

        // Outer product H += dc * sc^T
        h += Mat3F64::from_cols(dc * sc.x, dc * sc.y, dc * sc.z);
    }

    h *= 1.0 / n;

    // 3. Internal f64 SVD path.
    let svd = svd3_f64(&h);

    let u = *svd.u();
    let v = *svd.v();

    // Enforce the rotation matrix to belong to the Special Orthogonal group SO(3).
    // SVD can yield a valid decomposition but result in a reflection matrix (det(R) = -1)
    // when we directly form R = U * V^T. In that case, we use a legacy convention of
    // flipping the third column of R to change the determinant to +1, yielding a proper
    // rotation. This operates on R itself (not on U or V) and preserves the existing API.
    let mut r = u * v.transpose();
    if r.determinant() < 0.0 {
        r.z_axis = -r.z_axis;
    }

    // Translation: t = mu_d - R * mu_s
    let t = mu_d - (r * mu_s);
    // Cast back to f32 for the output
    let r_cols = [
        r.x_axis.x as f32,
        r.x_axis.y as f32,
        r.x_axis.z as f32, // Col 0
        r.y_axis.x as f32,
        r.y_axis.y as f32,
        r.y_axis.z as f32, // Col 1
        r.z_axis.x as f32,
        r.z_axis.y as f32,
        r.z_axis.z as f32, // Col 2
    ];

    Ok((
        Mat3AF32::from_cols_array(&r_cols),
        Vec3AF32::new(t.x as f32, t.y as f32, t.z as f32),
        1.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Custom helper to bypass the `approx` crate dependency in CI
    fn assert_vec3_approx(a: &Vec3AF32, b: &Vec3AF32, eps: f32) {
        assert!(
            (a.x - b.x).abs() <= eps,
            "Vec3 x mismatch: {} vs {}",
            a.x,
            b.x
        );
        assert!(
            (a.y - b.y).abs() <= eps,
            "Vec3 y mismatch: {} vs {}",
            a.y,
            b.y
        );
        assert!(
            (a.z - b.z).abs() <= eps,
            "Vec3 z mismatch: {} vs {}",
            a.z,
            b.z
        );
    }

    // Custom helper to bypass the `approx` crate dependency in CI
    fn assert_mat3_approx(a: &Mat3AF32, b: &Mat3AF32, eps: f32) {
        // x_axis
        assert!((a.x_axis.x - b.x_axis.x).abs() <= eps, "x_axis.x mismatch");
        assert!((a.x_axis.y - b.x_axis.y).abs() <= eps, "x_axis.y mismatch");
        assert!((a.x_axis.z - b.x_axis.z).abs() <= eps, "x_axis.z mismatch");
        // y_axis
        assert!((a.y_axis.x - b.y_axis.x).abs() <= eps, "y_axis.x mismatch");
        assert!((a.y_axis.y - b.y_axis.y).abs() <= eps, "y_axis.y mismatch");
        assert!((a.y_axis.z - b.y_axis.z).abs() <= eps, "y_axis.z mismatch");
        // z_axis
        assert!((a.z_axis.x - b.z_axis.x).abs() <= eps, "z_axis.x mismatch");
        assert!((a.z_axis.y - b.z_axis.y).abs() <= eps, "z_axis.y mismatch");
        assert!((a.z_axis.z - b.z_axis.z).abs() <= eps, "z_axis.z mismatch");
    }

    #[test]
    fn test_umeyama_translation_only() -> Result<(), UmeyamaError> {
        let src: [Vec3AF32; 4] = [
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
        ];
        // True transform: Identity rotation, only translation
        let r = Mat3AF32::IDENTITY;
        let t = Vec3AF32::new(10.5, -5.2, std::f32::consts::PI);

        let mut dst = [Vec3AF32::ZERO; 4];
        for i in 0..4 {
            dst[i] = r * src[i] + t;
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        assert_mat3_approx(&r_est, &r, 1e-5);
        assert_vec3_approx(&t_est, &t, 1e-5);
        Ok(())
    }

    #[test]
    fn test_umeyama_synthetic_z90() -> Result<(), UmeyamaError> {
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
        assert_mat3_approx(&r_est, &r, 1e-5);
        assert_vec3_approx(&t_est, &t, 1e-5);
        Ok(())
    }

    #[test]
    fn test_umeyama_complex_transform() -> Result<(), UmeyamaError> {
        // Arbitrary 3D point cloud
        let src: [Vec3AF32; 5] = [
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 2.0, 3.0),
            Vec3AF32::new(-1.0, 5.0, -2.0),
            Vec3AF32::new(4.0, -1.0, 7.0),
            Vec3AF32::new(2.5, 3.5, 1.5),
        ];

        // True transform: A valid orthonormal rotation matrix across all 3 axes
        let r =
            Mat3AF32::from_cols_array(&[0.36, 0.48, 0.80, -0.80, 0.60, 0.00, -0.48, -0.64, 0.60]);
        let t = Vec3AF32::new(-10.5, 20.2, -5.5);

        let mut dst = [Vec3AF32::ZERO; 5];
        for i in 0..5 {
            dst[i] = r * src[i] + t;
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        assert_mat3_approx(&r_est, &r, 1e-6);
        assert_vec3_approx(&t_est, &t, 1e-6);
        Ok(())
    }

    #[test]
    fn test_umeyama_errors() {
        let src = vec![Vec3AF32::ZERO, Vec3AF32::new(1.0, 0.0, 0.0)];
        let dst = vec![Vec3AF32::ZERO];

        // Test mismatched lengths
        assert!(matches!(
            umeyama(&src, &dst),
            Err(UmeyamaError::MismatchedInputLengths)
        ));

        // Test empty input
        let empty_src: Vec<Vec3AF32> = vec![];
        let empty_dst: Vec<Vec3AF32> = vec![];
        assert!(matches!(
            umeyama(&empty_src, &empty_dst),
            Err(UmeyamaError::EmptyInput)
        ));
    }

    #[test]
    fn test_umeyama_handles_reflection() {
        let src = vec![
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
        ];

        // Reflected across the Z axis (z * -1)
        let dst = vec![
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, -1.0, 0.0),
        ];

        let (r_est, _, _) = umeyama(&src, &dst).unwrap();

        // The determinant of the rotation part MUST be +1 (a valid rotation, not a reflection)
        // Convert to f64 to use the determinant method safely
        let rot = Mat3F64::from_cols(
            Vec3F64::new(
                r_est.x_axis.x as f64,
                r_est.x_axis.y as f64,
                r_est.x_axis.z as f64,
            ),
            Vec3F64::new(
                r_est.y_axis.x as f64,
                r_est.y_axis.y as f64,
                r_est.y_axis.z as f64,
            ),
            Vec3F64::new(
                r_est.z_axis.x as f64,
                r_est.z_axis.y as f64,
                r_est.z_axis.z as f64,
            ),
        );
        assert!(
            rot.determinant() > 0.0,
            "Umeyama failed to correct reflection to a valid rotation"
        );
    }

    #[test]
    fn test_umeyama_noisy_data() {
        let src = vec![
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
        ];

        // Translated by (1, 1, 1) and added a tiny bit of noise
        let noise = 1e-4;
        let dst = vec![
            Vec3AF32::new(1.0 + noise, 1.0 - noise, 1.0),
            Vec3AF32::new(2.0, 1.0 + noise, 1.0),
            Vec3AF32::new(1.0 - noise, 2.0, 1.0 - noise),
        ];

        let (_, t_est, _) = umeyama(&src, &dst).unwrap();

        // Verify the translation is approximately (1,1,1) despite the noise
        assert!((t_est.x - 1.0).abs() < 1e-2);
        assert!((t_est.y - 1.0).abs() < 1e-2);
        assert!((t_est.z - 1.0).abs() < 1e-2);
    }
}
