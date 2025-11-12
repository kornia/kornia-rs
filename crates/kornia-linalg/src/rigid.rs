//! Rigid alignment utilities (Kabsch / Umeyama)

use crate::svd::svd3;
use glam::{Mat3, Vec3};
use thiserror::Error;

/// Rotation (R), translation (t), and scale (s) output of Umeyama without scaling (s = 1).
pub type UmeyamaOutput = ([[f32; 3]; 3], [f32; 3], f32);

/// Error type for Umeyama rigid alignment operations.
#[derive(Debug, Error)]
pub enum UmeyamaError {
    /// Source and destination arrays must have the same length
    #[error("Source and destination arrays must have the same length")]
    MismatchedInputLengths,
}

/// Result type alias for Umeyama.
pub type UmeyamaResult = Result<UmeyamaOutput, UmeyamaError>;

/// Umeyama/Kabsch algorithm without scale.
/// Returns (R, t, s) where s == 1.0.
pub fn umeyama(src: &[Vec3], dst: &[Vec3]) -> UmeyamaResult {
    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }
    let n = src.len() as f32;

    // Centroids
    let mut mu_s = Vec3::ZERO;
    let mut mu_d = Vec3::ZERO;
    for i in 0..src.len() {
        mu_s += src[i];
        mu_d += dst[i];
    }
    mu_s /= n;
    mu_d /= n;

    // Covariance matrix H
    let mut h = Mat3::ZERO;
    for i in 0..src.len() {
        let sc = src[i] - mu_s;
        let dc = dst[i] - mu_d;
        // H = sum(dc * sc^T)
        h += Mat3::from_cols(dc * sc.x, dc * sc.y, dc * sc.z);
    }
    h /= n;

    // Call the internal svd3 function
    let svd_result = svd3(&h);

    // Clone to get owned Mat3 regardless of whether svd3 returned &Mat3 or Mat3.
    // If svd3 returns a Result or other wrapper you will need to adapt this.
    let mut u_mat = svd_result.u().clone();
    let v_mat = svd_result.v().clone();

    // Canonical Umeyama sign correction: if det(U)*det(V) < 0, flip the last column of U.
    if (u_mat.determinant() * v_mat.determinant()) < 0.0 {
        u_mat.z_axis = -u_mat.z_axis;
    }

    // Compute rotation R = U * V^T
    let r_glam = u_mat * v_mat.transpose();

    // Calculate translation t = mu_d - R * mu_s
    let t_vec: Vec3 = mu_d - (r_glam * mu_s);
    let t: [f32; 3] = t_vec.into();

    // Convert glam::Mat3 (column-major) to a row-major array
    let r_arr = [
        [r_glam.x_axis.x, r_glam.y_axis.x, r_glam.z_axis.x], // Row 0
        [r_glam.x_axis.y, r_glam.y_axis.y, r_glam.z_axis.y], // Row 1
        [r_glam.x_axis.z, r_glam.y_axis.z, r_glam.z_axis.z], // Row 2
    ];

    Ok((r_arr, t, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use glam::Quat;

    fn apply_rt(r: &[[f32; 3]; 3], t: &[f32; 3], p: &[f32; 3]) -> Vec3 {
        let x = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
        let y = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
        let z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];
        Vec3::new(x, y, z)
    }

    #[test]
    fn test_umeyama_synthetic_z90() -> Result<(), UmeyamaError> {
        // Source points (square in XY plane, z=0)
        let src: [Vec3; 4] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        // True transform: 90° about Z, plus translation
        let r: [[f32; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let t = [0.5, -0.3, 2.0];
        // Create dst
        let mut dst = [Vec3::new(0.0, 0.0, 0.0); 4];
        for i in 0..4 {
            let src_array = [src[i].x, src[i].y, src[i].z];
            dst[i] = apply_rt(&r, &t, &src_array);
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_est[i][j], r[i][j], epsilon = 1e-6);
            }
        }
        for k in 0..3 {
            assert_relative_eq!(t_est[k], t[k], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_umeyama_identity() -> Result<(), UmeyamaError> {
        // Source points
        let src: [Vec3; 4] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        // Dst points are identical to Src
        let dst = src;

        // Expected R is identity, t is zero
        let r_expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t_expected = [0.0, 0.0, 0.0];

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_est[i][j], r_expected[i][j], epsilon = 1e-6);
            }
        }
        for k in 0..3 {
            assert_relative_eq!(t_est[k], t_expected[k], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_umeyama_translation_only() -> Result<(), UmeyamaError> {
        // Source points
        let src: [Vec3; 4] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        // Dst is Src + a translation
        let t_expected = [5.0, -2.0, 3.0];
        let mut dst = [Vec3::new(0.0, 0.0, 0.0); 4];
        for i in 0..4 {
            dst[i] = src[i] + Vec3::new(t_expected[0], t_expected[1], t_expected[2]);
        }

        // Expected R is identity
        let r_expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_est[i][j], r_expected[i][j], epsilon = 1e-6);
            }
        }
        for k in 0..3 {
            assert_relative_eq!(t_est[k], t_expected[k], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_umeyama_reflection() -> Result<(), UmeyamaError> {
        // Source points
        let src: [Vec3; 4] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        // Dst is a mirrored version of src (reflection across X)
        let mut dst = [Vec3::new(0.0, 0.0, 0.0); 4];
        for i in 0..4 {
            dst[i] = Vec3::new(-src[i].x, src[i].y, src[i].z);
        }

        let (r_est, _t_est, _s) = umeyama(&src, &dst)?;

        // Convert returned row-major r_est back into a Mat3 (columns)
        let r_mat = Mat3::from_cols(
            Vec3::new(r_est[0][0], r_est[1][0], r_est[2][0]),
            Vec3::new(r_est[0][1], r_est[1][1], r_est[2][1]),
            Vec3::new(r_est[0][2], r_est[1][2], r_est[2][2]),
        );

        // The algorithm should correct the reflection so determinant is +1
        assert!(r_mat.determinant() > 0.0);
        Ok(())
    }

    // New test for a more complex rotation
    #[test]
    fn test_umeyama_complex_rotation() -> Result<(), UmeyamaError> {
        let src: [Vec3; 4] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        // True transform: 45° about Y-axis
        let angle = std::f32::consts::FRAC_PI_4; // 45 degrees
        let r_quat = Quat::from_rotation_y(angle);
        let r_mat = Mat3::from_quat(r_quat);

        // Build row-major r from r_mat (glam is column-major)
        let r: [[f32; 3]; 3] = [
            [r_mat.x_axis.x, r_mat.y_axis.x, r_mat.z_axis.x],
            [r_mat.x_axis.y, r_mat.y_axis.y, r_mat.z_axis.y],
            [r_mat.x_axis.z, r_mat.y_axis.z, r_mat.z_axis.z],
        ];

        let t = [0.0, 0.0, 0.0]; // No translation

        // Create dst
        let mut dst = [Vec3::new(0.0, 0.0, 0.0); 4];
        for i in 0..4 {
            let src_array = [src[i].x, src[i].y, src[i].z];
            dst[i] = apply_rt(&r, &t, &src_array);
        }

        let (r_est, t_est, _s) = umeyama(&src, &dst)?;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_est[i][j], r[i][j], epsilon = 1e-6);
            }
        }
        for k in 0..3 {
            assert_relative_eq!(t_est[k], t[k], epsilon = 1e-6);
        }
        Ok(())
    }
}
