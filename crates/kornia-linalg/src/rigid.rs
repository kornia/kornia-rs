//! Rigid alignment utilities (Kabsch / Umeyama)

// TODO: Make this work with kornia-linalg SVD(encountered some issues with precision)
use glam::Vec3;
use nalgebra::{Matrix3, SVD};

/// Umeyama/Kabsch algorithm without scale.
/// Returns (R, t, s) where s == 1.0.
pub fn umeyama(src: &[Vec3], dst: &[Vec3]) -> ([[f32; 3]; 3], [f32; 3], f32) {
    assert_eq!(src.len(), dst.len());
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
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

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

    (r_arr, t, 1.0)
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
    fn test_umeyama_synthetic_z90() {
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

        let (r_est, t_est, _s) = umeyama(&src, &dst);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_est[i][j], r[i][j], epsilon = 1e-6);
            }
        }
        for k in 0..3 {
            assert_relative_eq!(t_est[k], t[k], epsilon = 1e-6);
        }
    }
}
