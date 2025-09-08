use kornia_linalg::svd::svd3;

/// Estimate the fundamental matrix using the normalized 8-point algorithm.
///
/// - `x1`: points in image 1 as `&[[f64; 2]]` (length >= 8)
/// - `x2`: corresponding points in image 2 as `&[[f64; 2]]` (same length)
///
/// Returns `Ok([[f64; 3]; 3])` on success or an error if inputs are invalid.
pub fn fundamental_8point(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
) -> Result<[[f64; 3]; 3], Box<dyn std::error::Error>> {
    if x1.len() != x2.len() || x1.len() < 8 {
        return Err("Need at least 8 correspondences and equal lengths".into());
    }

    // Normalize points with similarity transforms T1, T2 to have zero mean and avg sqrt(2) distance
    let (x1n, t1) = normalize_points_2d(x1);
    let (x2n, t2) = normalize_points_2d(x2);

    // Build design matrix A (N x 9) for x2' * F * x1 = 0
    let n = x1n.len();
    let mut a = faer::Mat::<f64>::zeros(n, 9);
    for i in 0..n {
        let (x, y) = (x1n[i][0], x1n[i][1]);
        let (xp, yp) = (x2n[i][0], x2n[i][1]);
        unsafe {
            a.write_unchecked(i, 0, xp * x);
            a.write_unchecked(i, 1, xp * y);
            a.write_unchecked(i, 2, xp);
            a.write_unchecked(i, 3, yp * x);
            a.write_unchecked(i, 4, yp * y);
            a.write_unchecked(i, 5, yp);
            a.write_unchecked(i, 6, x);
            a.write_unchecked(i, 7, y);
            a.write_unchecked(i, 8, 1.0);
        }
    }

    // Solve Af = 0 via SVD: take last column of V
    let svd = a.svd();
    let fvec = svd.v().col(8);
    let mut f = [[0.0; 3]; 3];
    f[0] = [fvec[0], fvec[1], fvec[2]];
    f[1] = [fvec[3], fvec[4], fvec[5]];
    f[2] = [fvec[6], fvec[7], fvec[8]];

    // Enforce rank-2 constraint on F via SVD(F) and zero smallest singular value
    // Use kornia-linalg's SVD to enforce rank-2 with glam Mat3 for simplicity
    let f_glam = glam::Mat3::from_cols_array(&[
        f[0][0] as f32,
        f[1][0] as f32,
        f[2][0] as f32,
        f[0][1] as f32,
        f[1][1] as f32,
        f[2][1] as f32,
        f[0][2] as f32,
        f[1][2] as f32,
        f[2][2] as f32,
    ]);
    let svd = svd3(&f_glam);
    let u = *svd.u();
    let v = *svd.v();
    let mut s = *svd.s();
    s.z_axis.z = 0.0; // zero smallest singular value
    let f_rank2_glam = u * s * v.transpose();

    // Denormalize: F = T2^T * F * T1
    // Denormalize with faer matrices: F = T2^T * F * T1
    let t2t = t2.transpose();
    let f_rank2 = faer::mat![
        [f_rank2_glam.x_axis.x as f64, f_rank2_glam.y_axis.x as f64, f_rank2_glam.z_axis.x as f64],
        [f_rank2_glam.x_axis.y as f64, f_rank2_glam.y_axis.y as f64, f_rank2_glam.z_axis.y as f64],
        [f_rank2_glam.x_axis.z as f64, f_rank2_glam.y_axis.z as f64, f_rank2_glam.z_axis.z as f64],
    ];
    let f_denorm = t2t * f_rank2 * t1;
    Ok(from_faer_mat3(&f_denorm))
}

fn normalize_points_2d(x: &[[f64; 2]]) -> (Vec<[f64; 2]>, faer::Mat<f64>) {
    let n = x.len();
    let (mut mx, mut my) = (0.0, 0.0);
    for p in x {
        mx += p[0];
        my += p[1];
    }
    mx /= n as f64;
    my /= n as f64;
    let mut mean_dist = 0.0;
    for p in x {
        let dx = p[0] - mx;
        let dy = p[1] - my;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n as f64;
    let scale = if mean_dist > 0.0 { (2.0f64).sqrt() / mean_dist } else { 1.0 };

    let mut xn = Vec::with_capacity(n);
    for p in x {
        xn.push([(p[0] - mx) * scale, (p[1] - my) * scale]);
    }

    // Similarity transform matrix T = [[s,0,-s*mx],[0,s,-s*my],[0,0,1]]
    let t = faer::mat![
        [scale, 0.0, -scale * mx],
        [0.0, scale, -scale * my],
        [0.0, 0.0, 1.0]
    ];
    (xn, t)
}

fn to_faer_mat3(m: &[[f64; 3]; 3]) -> faer::Mat<f64> {
    faer::mat![
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]]
    ]
}

fn from_faer_mat3(m: &faer::Mat<f64>) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m.read(i, j);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Check x2'^T F x1 ~ 0 for perfect correspondences from a known F
    #[test]
    fn test_fundamental_8point_epipolar_constraint() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple synthetic example via random correspondences transformed by a known F
        // Use a simple F with rank 2
        let f_true = [
            [0.0, -0.001, 0.01],
            [0.0015, 0.0, -0.02],
            [-0.01, 0.02, 1.0],
        ];
        // Generate points x1 and compute epipolar lines l2 = F x1, then sample x2 on those lines
        let x1 = vec![
            [10.0, 20.0],
            [30.0, -5.0],
            [-15.0, 12.0],
            [7.0, 8.0],
            [100.0, 50.0],
            [-40.0, 70.0],
            [60.0, -30.0],
            [15.0, 15.0],
        ];
        let mut x2 = Vec::new();
        for p in &x1 {
            let x: [f64; 3] = [p[0], p[1], 1.0f64];
            let l = [
                f_true[0][0] * x[0] + f_true[0][1] * x[1] + f_true[0][2] * x[2],
                f_true[1][0] * x[0] + f_true[1][1] * x[1] + f_true[1][2] * x[2],
                f_true[2][0] * x[0] + f_true[2][1] * x[1] + f_true[2][2] * x[2],
            ];
            // pick x2 with y=0 => solve l0*x + l2 = 0
            let xp = if l[0].abs() > 1e-12f64 { -l[2] / l[0] } else { 0.0f64 };
            x2.push([xp, 0.0f64]);
        }

        let f_est = fundamental_8point(&x1, &x2)?;

        // Check epipolar constraint
        for i in 0..x1.len() {
            let x: [f64; 3] = [x1[i][0], x1[i][1], 1.0f64];
            let xp: [f64; 3] = [x2[i][0], x2[i][1], 1.0f64];
            let fx = [
                f_est[0][0] * x[0] + f_est[0][1] * x[1] + f_est[0][2] * x[2],
                f_est[1][0] * x[0] + f_est[1][1] * x[1] + f_est[1][2] * x[2],
                f_est[2][0] * x[0] + f_est[2][1] * x[1] + f_est[2][2] * x[2],
            ];
            let val = fx[0] * xp[0] + fx[1] * xp[1] + fx[2] * xp[2];
            assert!(val.abs() < 1e-6);
        }

        Ok(())
    }
}


