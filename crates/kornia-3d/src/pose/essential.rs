use kornia_algebra::{Mat3F64, Vec3F64};

/// Build an essential matrix from a fundamental matrix and camera intrinsics.
///
/// E = K2^T * F * K1
pub fn essential_from_fundamental(f: &Mat3F64, k1: &Mat3F64, k2: &Mat3F64) -> Mat3F64 {
    k2.transpose() * *f * *k1
}

/// Perform SVD of a 3x3 matrix using faer, returning (U, singular_values, V).
/// TODO: temporary workaround until svd3_f64 is fixed to handle repeated
/// singular values: https://github.com/kornia/kornia-rs/issues/696
fn svd3_robust(m: &Mat3F64) -> (Mat3F64, Vec3F64, Mat3F64) {
    let arr: [f64; 9] = (*m).into();
    let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| arr[j * 3 + i]);
    let svd = a.svd();

    let u_f = svd.u();
    let v_f = svd.v();
    let s_f = svd.s_diagonal();

    let u = Mat3F64::from_cols(
        Vec3F64::new(u_f[(0, 0)], u_f[(1, 0)], u_f[(2, 0)]),
        Vec3F64::new(u_f[(0, 1)], u_f[(1, 1)], u_f[(2, 1)]),
        Vec3F64::new(u_f[(0, 2)], u_f[(1, 2)], u_f[(2, 2)]),
    );
    let s = Vec3F64::new(s_f[0], s_f[1], s_f[2]);
    let v = Mat3F64::from_cols(
        Vec3F64::new(v_f[(0, 0)], v_f[(1, 0)], v_f[(2, 0)]),
        Vec3F64::new(v_f[(0, 1)], v_f[(1, 1)], v_f[(2, 1)]),
        Vec3F64::new(v_f[(0, 2)], v_f[(1, 2)], v_f[(2, 2)]),
    );
    (u, s, v)
}

/// Enforce the (1,1,0) singular value constraint on an essential matrix.
pub fn enforce_essential_constraints(e: &Mat3F64) -> Mat3F64 {
    let (u, _s, v) = svd3_robust(e);
    let s_mat = Mat3F64::from_cols(
        Vec3F64::new(1.0, 0.0, 0.0),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(0.0, 0.0, 0.0),
    );
    u * s_mat * v.transpose()
}

/// Decompose an essential matrix into four possible (R, t) solutions.
///
/// Returns an array of candidate poses where R is 3x3 and t is a unit 3-vector.
pub fn decompose_essential(e: &Mat3F64) -> [(Mat3F64, Vec3F64); 4] {
    let (mut u, _s, mut v) = svd3_robust(e);

    if u.determinant() < 0.0 {
        u.z_axis = -u.z_axis;
    }
    if v.determinant() < 0.0 {
        v.z_axis = -v.z_axis;
    }

    let w = Mat3F64::from_cols(
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(-1.0, 0.0, 0.0),
        Vec3F64::new(0.0, 0.0, 1.0),
    );
    let vt = v.transpose();

    let r1 = u * w * vt;
    let r2 = u * w.transpose() * vt;

    let t = u.z_axis();
    let t_neg = -t;

    [(r1, t), (r1, t_neg), (r2, t), (r2, t_neg)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec2F64;

    fn skew(t: Vec3F64) -> Mat3F64 {
        Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        )
    }

    #[test]
    fn test_decompose_essential_identity_rotation() {
        let r = Mat3F64::IDENTITY;
        let t = Vec3F64::new(1.0, 0.0, 0.0);
        let e = skew(t) * r;

        let candidates = decompose_essential(&e);
        assert_eq!(candidates.len(), 4);

        let mut found = false;
        for (rc, tc) in candidates {
            let det = rc.determinant();
            assert!((det - 1.0).abs() < 1e-3);

            let dot = (tc.x * t.x + tc.y * t.y + tc.z * t.z).abs();
            if dot > 0.9 {
                let mut diff = 0.0;
                let ra: [f64; 9] = rc.into();
                let rb: [f64; 9] = r.into();
                for i in 0..9 {
                    diff += (ra[i] - rb[i]).abs();
                }
                if diff < 1e-2 {
                    found = true;
                    break;
                }
            }
        }

        assert!(found);
    }

    /// End-to-end test: known (R, t, K) → generate correspondences → F → E → decompose → verify R, t.
    #[test]
    fn test_fundamental_to_essential_to_pose_round_trip() {
        use crate::pose::fundamental::{fundamental_8point, sampson_distance};

        // Known camera intrinsics
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );
        let k_inv = k.inverse();

        // Known rotation (small rotation around Y axis) and translation
        let angle = 0.1_f64; // ~5.7 degrees
        let r_true = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t_true = Vec3F64::new(1.0, 0.0, 0.2);
        let t_true_norm = t_true.length();
        let t_true_unit = Vec3F64::new(
            t_true.x / t_true_norm,
            t_true.y / t_true_norm,
            t_true.z / t_true_norm,
        );

        // E = [t]_x * R
        let e_true = skew(t_true_unit) * r_true;

        // F = K^{-T} * E * K^{-1}
        let f_true = k.inverse().transpose() * e_true * k_inv;

        // Generate 3D points in front of both cameras
        let points_3d = vec![
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
        ];

        // Project into camera 1 (P1 = K[I|0]) and camera 2 (P2 = K[R|t])
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for p in &points_3d {
            // Camera 1: project [I|0]
            let p1 = k * *p;
            x1.push(Vec2F64::new(p1.x / p1.z, p1.y / p1.z));

            // Camera 2: project [R|t]
            let p2_cam = r_true * *p + t_true;
            let p2 = k * p2_cam;
            x2.push(Vec2F64::new(p2.x / p2.z, p2.y / p2.z));
        }

        // Verify the generated data satisfies x2^T F x1 = 0
        for i in 0..x1.len() {
            let d = sampson_distance(&f_true, &x1[i], &x2[i]);
            assert!(d < 1e-6, "ground truth Sampson distance {i}: {d}");
        }

        // Estimate F from correspondences
        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Verify epipolar constraint with estimated F
        for i in 0..x1.len() {
            let d = sampson_distance(&f_est, &x1[i], &x2[i]);
            assert!(d < 1e-4, "estimated Sampson distance {i}: {d}");
        }

        // F → E → decompose
        let e = essential_from_fundamental(&f_est, &k, &k);
        let e = enforce_essential_constraints(&e);
        let candidates = decompose_essential(&e);

        // One of the 4 candidates should match the known R and t
        let rt_arr: [f64; 9] = r_true.into();
        let mut found = false;
        for (rc, tc) in &candidates {
            // Check rotation: R_candidate ≈ R_true
            let rc_arr: [f64; 9] = (*rc).into();
            let rot_diff: f64 = rc_arr
                .iter()
                .zip(rt_arr.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            // Check translation direction: |t_candidate . t_true_unit| ≈ 1
            let t_dot = (tc.x * t_true_unit.x + tc.y * t_true_unit.y + tc.z * t_true_unit.z).abs();

            if rot_diff < 0.1 && t_dot > 0.9 {
                found = true;
                break;
            }
        }

        assert!(
            found,
            "no decomposed (R, t) candidate matched the ground truth"
        );
    }

    #[test]
    fn test_enforce_essential_constraints_rank2() {
        let e = Mat3F64::from_cols(
            Vec3F64::new(0.1, 0.2, -0.3),
            Vec3F64::new(0.4, -0.1, 0.2),
            Vec3F64::new(-0.2, 0.5, 0.3),
        );
        let e_fixed = enforce_essential_constraints(&e);
        let (_u, s, _v) = svd3_robust(&e_fixed);
        assert!(s.z.abs() < 1e-6);
        assert!((s.x - s.y).abs() < 1e-6);
    }
}
