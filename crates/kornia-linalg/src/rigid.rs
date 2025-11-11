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

    // Convert H from [[f32; 3]; 3] (row-major) to glam::Mat3 (column-major)
    // This is the corrected block
    let h_glam = Mat3::from_cols(
        Vec3::new(h[0][0], h[1][0], h[2][0]), // Column 0
        Vec3::new(h[0][1], h[1][1], h[2][1]), // Column 1
        Vec3::new(h[0][2], h[1][2], h[2][2]), // Column 2
    );

    // Call the internal svd3 function
    let svd_result = svd3(&h_glam);
    let u = svd_result.u();
    let v = svd_result.v();
    let v_t = v.transpose(); // We need v_t, so we transpose v

    // Calculate rotation matrix R
    let mut r_glam = *u * v_t;
    if r_glam.determinant() < 0.0 {
        // glam's z_axis is the 3rd column
        r_glam.z_axis *= -1.0;
    }

    // Convert glam::Mat3 (column-major) to a row-major array
    let r_arr = [
        [r_glam.x_axis.x, r_glam.y_axis.x, r_glam.z_axis.x], // Row 0
        [r_glam.x_axis.y, r_glam.y_axis.y, r_glam.z_axis.y], // Row 1
        [r_glam.x_axis.z, r_glam.y_axis.z, r_glam.z_axis.z], // Row 2
    ];

    let mut t = [0.0; 3];
    for r in 0..3 {
        t[r] = mu_d[r] - (r_arr[r][0] * mu_s[0] + r_arr[r][1] * mu_s[1] + r_arr[r][2] * mu_s[2]);
    }

    Ok((r_arr, t, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
}
