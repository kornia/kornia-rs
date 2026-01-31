use crate::pose::utils::{mat3f32_to_mat3f64, mat3f64_to_mat3f32};
use kornia_algebra::{linalg::svd::svd3, Mat3F32, Mat3F64, Vec3F32, Vec3F64};

/// Build an essential matrix from a fundamental matrix and camera intrinsics.
///
/// E = K2^T * F * K1
pub fn essential_from_fundamental(f: &Mat3F64, k1: &Mat3F64, k2: &Mat3F64) -> Mat3F64 {
    k2.transpose() * *f * *k1
}

/// Enforce the (1,1,0) singular value constraint on an essential matrix.
pub fn enforce_essential_constraints(e: &Mat3F64) -> Mat3F64 {
    let e32 = mat3f64_to_mat3f32(e);
    let svd = svd3(&e32);
    let mut s = *svd.s();
    s.x_axis.x = 1.0;
    s.y_axis.y = 1.0;
    s.z_axis.z = 0.0;
    let e_fixed = *svd.u() * s * svd.v().transpose();
    mat3f32_to_mat3f64(&e_fixed)
}

/// Decompose an essential matrix into four possible (R, t) solutions.
///
/// Returns a vector of candidate poses where R is 3x3 and t is a unit 3-vector.
pub fn decompose_essential(e: &Mat3F64) -> Vec<(Mat3F64, Vec3F64)> {
    let e32 = mat3f64_to_mat3f32(e);
    let svd = svd3(&e32);
    let mut u = *svd.u();
    let mut v = *svd.v();

    if u.determinant() < 0.0 {
        u.z_axis = -u.z_axis;
    }
    if v.determinant() < 0.0 {
        v.z_axis = -v.z_axis;
    }

    let w = Mat3F32::from_cols(
        Vec3F32::new(0.0, 1.0, 0.0),
        Vec3F32::new(-1.0, 0.0, 0.0),
        Vec3F32::new(0.0, 0.0, 1.0),
    );
    let wt = w.transpose();

    let r1 = u * w * v.transpose();
    let r2 = u * wt * v.transpose();

    let t = Vec3F64::new(u.z_axis.x as f64, u.z_axis.y as f64, u.z_axis.z as f64);
    let t_neg = Vec3F64::new(-t.x, -t.y, -t.z);

    vec![
        (mat3f32_to_mat3f64(&r1), t),
        (mat3f32_to_mat3f64(&r1), t_neg),
        (mat3f32_to_mat3f64(&r2), t),
        (mat3f32_to_mat3f64(&r2), t_neg),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_enforce_essential_constraints_rank2() {
        let e = Mat3F64::from_cols(
            Vec3F64::new(0.1, 0.2, -0.3),
            Vec3F64::new(0.4, -0.1, 0.2),
            Vec3F64::new(-0.2, 0.5, 0.3),
        );
        let e_fixed = enforce_essential_constraints(&e);
        let e32 = mat3f64_to_mat3f32(&e_fixed);
        let svd = svd3(&e32);
        let s = svd.s();
        assert!(s.z_axis.z.abs() < 1e-3);
        assert!((s.x_axis.x - s.y_axis.y).abs() < 1e-1);
    }
}
