//! # Fundamental Matrix
//!
//! The fundamental matrix F encodes epipolar geometry in **pixel space**: for corresponding
//! points x1, x2 across two views, `x2ᵀ F x1 = 0`.
//!
//! - 7 DOF: 3×3 matrix (9) minus scale (1) minus rank-2 constraint (1).
//! - Singular values: (σ₁, σ₂, 0). The zero enforces rank 2.
//! - Related to the essential matrix by camera intrinsics: `E = K2ᵀ F K1`,
//!   equivalently `F = K2⁻ᵀ E K1⁻¹`.
//!
//! ## 8-point algorithm
//!
//! Build a design matrix A from ≥8 correspondences, take the null vector of A (last
//! column of V from SVD), reshape into 3×3, enforce rank 2 by zeroing the smallest
//! singular value. Hartley normalization (translate centroid to origin, scale mean
//! distance to √2) is critical for numerical stability.
//!
//! ## Relationship to the essential manifold
//!
//! F and E encode the same geometry at different levels. F works in pixel space (depends
//! on camera intrinsics K), E works in metric space (intrinsics removed). The conversion
//! `E = K2ᵀ F K1` is invertible when K is known.
//!
//! F has 7 DOF; E has only 5 DOF because the essential manifold is a constrained subset
//! (the (σ,σ,0) singular value structure) — those 2 extra DOF in F encode the intrinsics.
//! The essential manifold is itself a quotient of SE(3), losing translation scale and sign
//! (see the [`essential`](super::essential) module).
//!
//! ## Pitfalls
//!
//! - **Transpose bugs are silent**: F and Fᵀ both satisfy rank-2, both have the same
//!   singular values. Only asymmetric tests (epipolar constraint direction, Sampson
//!   distance) catch them.
//! - **Column-major storage**: the SVD null vector is in row-major order. When using
//!   `from_cols`, consecutive 3 elements give columns, not rows — grouping wrong gives Fᵀ.

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
    let fvec = v.col(v.ncols() - 1);
    let f = Mat3F64::from_cols(
        Vec3F64::new(fvec[0], fvec[3], fvec[6]),
        Vec3F64::new(fvec[1], fvec[4], fvec[7]),
        Vec3F64::new(fvec[2], fvec[5], fvec[8]),
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

    /// Shared test setup: well-conditioned correspondences from a known two-camera geometry.
    /// Returns (x1, x2, F_true) where x2^T * F_true * x1 = 0 for all pairs.
    fn make_test_correspondences() -> (Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64) {
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );
        let k_inv = k.inverse();

        // Rotation ~5.7 deg around Y axis
        let angle = 0.1_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(1.0, 0.0, 0.2);
        let t_len = t.length();
        let t_unit = Vec3F64::new(t.x / t_len, t.y / t_len, t.z / t_len);

        // E = [t]_x * R, F = K^{-T} E K^{-1}
        let tx = Mat3F64::from_cols(
            Vec3F64::new(0.0, t_unit.z, -t_unit.y),
            Vec3F64::new(-t_unit.z, 0.0, t_unit.x),
            Vec3F64::new(t_unit.y, -t_unit.x, 0.0),
        );
        let e = tx * r;
        let f_true = k_inv.transpose() * e * k_inv;

        // 3D points well-spread in front of both cameras
        let pts = [
            Vec3F64::new(-0.5, -0.3, 4.0),
            Vec3F64::new(0.4, -0.2, 3.5),
            Vec3F64::new(-0.3, 0.5, 5.0),
            Vec3F64::new(0.6, 0.4, 4.5),
            Vec3F64::new(-0.1, -0.6, 3.0),
            Vec3F64::new(0.2, 0.3, 6.0),
            Vec3F64::new(-0.4, 0.1, 3.8),
            Vec3F64::new(0.5, -0.5, 4.2),
            Vec3F64::new(0.0, 0.0, 5.5),
            Vec3F64::new(-0.2, 0.4, 4.8),
            Vec3F64::new(0.3, -0.1, 3.2),
            Vec3F64::new(-0.6, 0.2, 5.8),
        ];

        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for p in &pts {
            let p1 = k * *p;
            x1.push(Vec2F64::new(p1.x / p1.z, p1.y / p1.z));
            let p2_cam = r * *p + t;
            let p2 = k * p2_cam;
            x2.push(Vec2F64::new(p2.x / p2.z, p2.y / p2.z));
        }

        (x1, x2, f_true)
    }

    #[test]
    fn test_fundamental_8point_epipolar_constraint() {
        let (x1, x2, _) = make_test_correspondences();

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Check rank-2 property
        let svd = svd3_f64(&f_est);
        let s = svd.s();
        assert!(
            s.z_axis.z.abs() < 1e-6,
            "smallest singular value should be near zero (rank-2 constraint), got: {}",
            s.z_axis.z
        );

        // Check epipolar constraint: x2^T * F_est * x1 ≈ 0 for all correspondences.
        // This would FAIL with large errors if F were transposed.
        for i in 0..x1.len() {
            let x1h = Vec3F64::new(x1[i].x, x1[i].y, 1.0);
            let x2h = Vec3F64::new(x2[i].x, x2[i].y, 1.0);
            let err = x2h.dot(f_est * x1h);
            assert!(
                err.abs() < 1e-8,
                "epipolar constraint violated for point {i}: err = {err}"
            );
        }
    }

    #[test]
    fn test_fundamental_8point_sampson_distance_near_zero() {
        let (x1, x2, _) = make_test_correspondences();

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Sampson distance (first-order geometric error) should be near zero
        // for all correspondences when F is correctly oriented.
        for i in 0..x1.len() {
            let d = sampson_distance(&f_est, &x1[i], &x2[i]);
            assert!(
                d < 1e-10,
                "sampson distance too large for point {i}: d = {d}"
            );
        }
    }

    #[test]
    fn test_fundamental_8point_proportional_to_true() {
        let (x1, x2, f_true) = make_test_correspondences();

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Normalize both matrices by Frobenius norm, then check element-wise proportionality.
        // This catches a transpose: F^T normalized differs from F normalized.
        let f_true_arr: [f64; 9] = f_true.into();
        let f_est_arr: [f64; 9] = f_est.into();

        let norm_true: f64 = f_true_arr.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_est: f64 = f_est_arr.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Find scale factor from a large element
        let mut scale = 0.0;
        for i in 0..9 {
            let a = f_true_arr[i] / norm_true;
            if a.abs() > 0.05 {
                scale = (f_est_arr[i] / norm_est) / a;
                break;
            }
        }
        assert!(scale.abs() > 1e-6, "could not determine scale factor");

        for i in 0..9 {
            let a = f_true_arr[i] / norm_true;
            let b = f_est_arr[i] / norm_est;
            let diff = (b - scale * a).abs();
            assert!(
                diff < 1e-4,
                "F_est not proportional to F_true at element {i}: expected {}, got {}, diff = {diff}",
                scale * a, b
            );
        }
    }

    #[test]
    fn test_sampson_distance_zero_on_epipolar_line() {
        let (_, _, f_true) = make_test_correspondences();

        // Pick an arbitrary point and find its match on the epipolar line
        let x1 = Vec2F64::new(350.0, 200.0);
        let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
        let l = f_true * x1h;
        // Find x2 on line l: l.x * u + l.y * v + l.z = 0
        // Choose u = 300, solve for v
        let u = 300.0;
        let v = -(l.x * u + l.z) / l.y;
        let x2 = Vec2F64::new(u, v);

        let d = sampson_distance(&f_true, &x1, &x2);
        assert!(d < 1e-10, "sampson distance = {d}");
    }
}
