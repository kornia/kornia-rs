use kornia_algebra::{linalg::svd::svd3_f64, Mat3F64, Vec2F64, Vec3F64};

/// Error type for fundamental matrix estimation.
#[derive(thiserror::Error, Debug)]
pub enum FundamentalError {
    /// Input correspondences are invalid or insufficient.
    #[error("Need at least 8 correspondences and equal lengths")]
    InvalidInput,
    /// SVD failed or produced an invalid result.
    #[error("SVD failed to produce a valid fundamental matrix")]
    SvdFailure,
}

/// Estimate the fundamental matrix using the normalized 8-point algorithm.
///
/// - `x1`: points in image 1 as `&[Vec2F64]` (length >= 8)
/// - `x2`: corresponding points in image 2 as `&[Vec2F64]` (same length)
pub fn fundamental_8point(x1: &[Vec2F64], x2: &[Vec2F64]) -> Result<Mat3F64, FundamentalError> {
    if x1.len() != x2.len() || x1.len() < 8 {
        return Err(FundamentalError::InvalidInput);
    }

    let (x1n, t1) = normalize_points_2d(x1);
    let (x2n, t2) = normalize_points_2d(x2);

    let n = x1n.len();
    let mut a = faer::Mat::<f64>::zeros(n, 9);
    for i in 0..n {
        let (x, y) = (x1n[i].x, x1n[i].y);
        let (xp, yp) = (x2n[i].x, x2n[i].y);
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

    let svd = a.svd();
    let v = svd.v();
    if v.ncols() < 9 {
        return Err(FundamentalError::SvdFailure);
    }
    let fvec = v.col(8);
    let f = Mat3F64::from_cols(
        Vec3F64::new(fvec[0], fvec[1], fvec[2]),
        Vec3F64::new(fvec[3], fvec[4], fvec[5]),
        Vec3F64::new(fvec[6], fvec[7], fvec[8]),
    );

    let f_rank2 = enforce_rank2(&f)?;

    // Denormalize: F = T2^T * F * T1
    let f_denorm = t2.transpose() * f_rank2 * t1;
    Ok(f_denorm)
}

/// Compute the Sampson distance for a correspondence.
pub fn sampson_distance(f: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
    let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
    let x2h = Vec3F64::new(x2.x, x2.y, 1.0);

    let fx1 = *f * x1h;
    let ftx2 = f.transpose() * x2h;

    let err = x2h.dot(fx1);
    let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
    if denom <= 1e-12 {
        return err * err;
    }
    (err * err) / denom
}

fn normalize_points_2d(x: &[Vec2F64]) -> (Vec<Vec2F64>, Mat3F64) {
    let n = x.len() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for p in x {
        mx += p.x;
        my += p.y;
    }
    mx /= n;
    my /= n;

    let mut mean_dist = 0.0;
    for p in x {
        let dx = p.x - mx;
        let dy = p.y - my;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n;

    let scale = if mean_dist > 0.0 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let mut xn = Vec::with_capacity(x.len());
    for p in x {
        xn.push(Vec2F64::new((p.x - mx) * scale, (p.y - my) * scale));
    }

    let t = Mat3F64::from_cols(
        Vec3F64::new(scale, 0.0, 0.0),
        Vec3F64::new(0.0, scale, 0.0),
        Vec3F64::new(-scale * mx, -scale * my, 1.0),
    );

    (xn, t)
}

fn enforce_rank2(f: &Mat3F64) -> Result<Mat3F64, FundamentalError> {
    let svd = svd3_f64(f);
    let mut s = *svd.s();
    s.z_axis.z = 0.0;
    let f_rank2 = *svd.u() * s * svd.v().transpose();
    Ok(f_rank2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fundamental_8point_epipolar_constraint() {
        let f_true = Mat3F64::from_cols(
            Vec3F64::new(0.0, -0.001, 0.01),
            Vec3F64::new(0.0015, 0.0, -0.02),
            Vec3F64::new(-0.01, 0.02, 1.0),
        );
        let x1 = vec![
            Vec2F64::new(10.0, 20.0),
            Vec2F64::new(30.0, -5.0),
            Vec2F64::new(-15.0, 12.0),
            Vec2F64::new(7.0, 8.0),
            Vec2F64::new(100.0, 50.0),
            Vec2F64::new(-40.0, 70.0),
            Vec2F64::new(60.0, -30.0),
            Vec2F64::new(15.0, 15.0),
        ];
        let mut x2 = Vec::new();
        for p in &x1 {
            let x = Vec3F64::new(p.x, p.y, 1.0);
            let l = f_true * x;
            let xp = if l.x.abs() > 1e-12f64 {
                -l.z / l.x
            } else {
                0.0
            };
            x2.push(Vec2F64::new(xp, 0.0));
        }

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        let svd = svd3_f64(&f_est);
        let s = svd.s();
        assert!(s.z_axis.z.abs() < 1e-2);
        assert!(s.z_axis.z < 0.1 * s.x_axis.x);
    }

    #[test]
    fn test_sampson_distance_zero_on_epipolar_line() {
        let f_true = Mat3F64::from_cols(
            Vec3F64::new(0.0, -0.001, 0.01),
            Vec3F64::new(0.0015, 0.0, -0.02),
            Vec3F64::new(-0.01, 0.02, 1.0),
        );
        let x1 = Vec2F64::new(12.0, -3.0);
        let x = Vec3F64::new(x1.x, x1.y, 1.0);
        let l = f_true * x;
        let x2 = if l.x.abs() > 1e-12f64 {
            Vec2F64::new(-l.z / l.x, 0.0f64)
        } else {
            Vec2F64::new(0.0f64, -l.z / l.y)
        };

        let d = sampson_distance(&f_true, &x1, &x2);
        assert!(d.abs() < 1e-8);
    }
}
