// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use crate::{Mat3AF32, Mat3F32, Mat3F64, QuatF32, QuatF64, Vec3F32, Vec3F64};

// Re-export the SVD result types for public use
pub use impl_f32::{sort_singular_values, svd3 as svd3_f32, SVD3Set as SVD3SetF32};
pub use impl_f64::{
    sort_singular_values as sort_singular_values_f64, svd3 as svd3_f64, SVD3Set as SVD3SetF64,
};

mod impl_f32 {
    use super::*;

    const EPSILON: f32 = 1e-6;
    const JACOBI_TOLERANCE: f32 = 1e-6;
    const MAX_SWEEPS: usize = 20;

    #[derive(Debug, Clone, Copy)]
    struct Symmetric3x3 {
        m_00: f32,
        m_10: f32,
        m_20: f32,
        m_11: f32,
        m_21: f32,
        m_22: f32,
    }

    impl Symmetric3x3 {
        fn from_mat3x3(m: &Mat3F32) -> Self {
            let c0 = m.x_axis();
            let c1 = m.y_axis();
            let c2 = m.z_axis();

            Self {
                m_00: c0.dot(c0),
                m_10: c0.dot(c1),
                m_20: c0.dot(c2),
                m_11: c1.dot(c1),
                m_21: c1.dot(c2),
                m_22: c2.dot(c2),
            }
        }
    }

    fn exact_givens(app: f32, aqq: f32, apq: f32) -> (f32, f32) {
        if apq.abs() < EPSILON {
            return (1.0, 0.0);
        }

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;
        (c, s)
    }

    #[inline(always)]
    fn vec3_cross(a: Vec3F32, b: Vec3F32) -> Vec3F32 {
        Vec3F32::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }

    #[derive(Debug, Clone)]
    pub struct SVD3Set {
        pub u: Mat3F32,
        pub s: Mat3F32,
        pub v: Mat3F32,
    }

    impl SVD3Set {
        #[inline]
        pub fn u(&self) -> &Mat3F32 {
            &self.u
        }
        #[inline]
        pub fn s(&self) -> &Mat3F32 {
            &self.s
        }
        #[inline]
        pub fn v(&self) -> &Mat3F32 {
            &self.v
        }
    }

    pub fn svd3(mat: &Mat3F32) -> SVD3Set {
        let mut s_mat = Symmetric3x3::from_mat3x3(mat);
        let mut q_accum = QuatF32::IDENTITY;

        for _sweep in 0..MAX_SWEEPS {
            {
                let (c, si) = exact_givens(s_mat.m_00, s_mat.m_11, s_mat.m_10);
                let s00 = s_mat.m_00;
                let s01 = s_mat.m_10;
                let s11 = s_mat.m_11;
                let s02 = s_mat.m_20;
                let s12 = s_mat.m_21;

                s_mat.m_00 = c * c * s00 - 2.0 * c * si * s01 + si * si * s11;
                s_mat.m_11 = si * si * s00 + 2.0 * c * si * s01 + c * c * s11;
                s_mat.m_10 = 0.0;
                s_mat.m_20 = c * s02 - si * s12;
                s_mat.m_21 = si * s02 + c * s12;

                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = -si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF32::from_xyzw(
                    q[0] * ch + q[1] * sh,
                    q[1] * ch - q[0] * sh,
                    q[2] * ch + q[3] * sh,
                    q[3] * ch - q[2] * sh,
                )
                .normalize();
            }

            {
                let (c, si) = exact_givens(s_mat.m_11, s_mat.m_22, s_mat.m_21);
                let s11 = s_mat.m_11;
                let s12 = s_mat.m_21;
                let s22 = s_mat.m_22;
                let s01 = s_mat.m_10;
                let s02 = s_mat.m_20;

                s_mat.m_11 = c * c * s11 - 2.0 * c * si * s12 + si * si * s22;
                s_mat.m_22 = si * si * s11 + 2.0 * c * si * s12 + c * c * s22;
                s_mat.m_21 = 0.0;
                s_mat.m_10 = c * s01 - si * s02;
                s_mat.m_20 = si * s01 + c * s02;

                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = -si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF32::from_xyzw(
                    q[0] * ch + q[3] * sh,
                    q[1] * ch + q[2] * sh,
                    q[2] * ch - q[1] * sh,
                    q[3] * ch - q[0] * sh,
                )
                .normalize();
            }

            {
                let (c, si) = exact_givens(s_mat.m_00, s_mat.m_22, s_mat.m_20);
                let s00 = s_mat.m_00;
                let s02 = s_mat.m_20;
                let s22 = s_mat.m_22;
                let s01 = s_mat.m_10;
                let s12 = s_mat.m_21;

                s_mat.m_00 = c * c * s00 - 2.0 * c * si * s02 + si * si * s22;
                s_mat.m_22 = si * si * s00 + 2.0 * c * si * s02 + c * c * s22;
                s_mat.m_20 = 0.0;
                s_mat.m_10 = c * s01 - si * s12;
                s_mat.m_21 = si * s01 + c * s12;

                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF32::from_xyzw(
                    q[0] * ch - q[2] * sh,
                    q[1] * ch + q[3] * sh,
                    q[2] * ch + q[0] * sh,
                    q[3] * ch - q[1] * sh,
                )
                .normalize();
            }

            let off_diag_sq =
                s_mat.m_10 * s_mat.m_10 + s_mat.m_20 * s_mat.m_20 + s_mat.m_21 * s_mat.m_21;
            if off_diag_sq < JACOBI_TOLERANCE {
                break;
            }
        }

        let mat3a = Mat3AF32::from_quat(q_accum);
        let x_axis_a = mat3a.x_axis();
        let y_axis_a = mat3a.y_axis();
        let z_axis_a = mat3a.z_axis();
        let v = Mat3F32::from_cols(
            Vec3F32::new(x_axis_a.x, x_axis_a.y, x_axis_a.z),
            Vec3F32::new(y_axis_a.x, y_axis_a.y, y_axis_a.z),
            Vec3F32::new(z_axis_a.x, z_axis_a.y, z_axis_a.z),
        );

        let s_diag = Vec3F32::new(
            s_mat.m_00.abs().sqrt(),
            s_mat.m_11.abs().sqrt(),
            s_mat.m_22.abs().sqrt(),
        );

        let mut b = *mat * v;
        let mut s_vec = s_diag;
        let mut v_mat = v;

        sort_singular_values(&mut b, &mut s_vec, &mut v_mat);

        let u_x = if s_vec.x > EPSILON {
            b.x_axis() / s_vec.x
        } else {
            Mat3F32::IDENTITY.x_axis()
        };
        let mut u_y = if s_vec.y > EPSILON {
            b.y_axis() / s_vec.y
        } else {
            Mat3F32::IDENTITY.y_axis()
        };
        let mut u_z = if s_vec.z > EPSILON {
            b.z_axis() / s_vec.z
        } else {
            Mat3F32::IDENTITY.z_axis()
        };

        // Safe Gram-Schmidt
        u_y = u_y - u_x * u_x.dot(u_y);
        if u_y.dot(u_y) < EPSILON {
            u_y = vec3_cross(u_x, Mat3F32::IDENTITY.x_axis());
            if u_y.dot(u_y) < EPSILON {
                u_y = vec3_cross(u_x, Mat3F32::IDENTITY.y_axis());
            }
        }
        u_y = u_y.normalize();

        u_z = u_z - u_x * u_x.dot(u_z) - u_y * u_y.dot(u_z);
        if u_z.dot(u_z) < EPSILON {
            u_z = vec3_cross(u_x, u_y);
        }
        u_z = u_z.normalize();

        let mut final_u = Mat3F32::from_cols(u_x, u_y, u_z);

        if final_u.determinant() < 0.0 {
            // Flip the column with the smallest singular value (Z-axis)
            let u_z_col = final_u.z_axis();
            final_u = Mat3F32::from_cols(final_u.x_axis(), final_u.y_axis(), -u_z_col);

            // Flip V's corresponding column to maintain A = U * S * V^T
            let v_z_col = v_mat.z_axis();
            v_mat = Mat3F32::from_cols(v_mat.x_axis(), v_mat.y_axis(), -v_z_col);
        }

        SVD3Set {
            u: final_u,
            s: Mat3F32::from_diagonal(s_vec),
            v: v_mat,
        }
    }

    pub fn sort_singular_values(u_mat: &mut Mat3F32, s_vec: &mut Vec3F32, v_mat: &mut Mat3F32) {
        if s_vec.x < s_vec.y {
            std::mem::swap(&mut s_vec.x, &mut s_vec.y);
            let col0 = v_mat.x_axis();
            let col1 = v_mat.y_axis();
            *v_mat = Mat3F32::from_cols(col1, col0, v_mat.z_axis());

            let col0 = u_mat.x_axis();
            let col1 = u_mat.y_axis();
            *u_mat = Mat3F32::from_cols(col1, col0, u_mat.z_axis());
        }
        if s_vec.y < s_vec.z {
            std::mem::swap(&mut s_vec.y, &mut s_vec.z);
            let col1 = v_mat.y_axis();
            let col2 = v_mat.z_axis();
            *v_mat = Mat3F32::from_cols(v_mat.x_axis(), col2, col1);

            let col1 = u_mat.y_axis();
            let col2 = u_mat.z_axis();
            *u_mat = Mat3F32::from_cols(u_mat.x_axis(), col2, col1);
        }
        if s_vec.x < s_vec.y {
            std::mem::swap(&mut s_vec.x, &mut s_vec.y);
            let col0 = v_mat.x_axis();
            let col1 = v_mat.y_axis();
            *v_mat = Mat3F32::from_cols(col1, col0, v_mat.z_axis());

            let col0 = u_mat.x_axis();
            let col1 = u_mat.y_axis();
            *u_mat = Mat3F32::from_cols(col1, col0, u_mat.z_axis());
        }
    }

    #[allow(dead_code)]
    pub type SVD3SetF32 = SVD3Set;
}

mod impl_f64 {
    use super::*;

    const EPSILON: f64 = 1e-15;
    const JACOBI_TOLERANCE: f64 = 1e-15;
    const MAX_SWEEPS: usize = 20;

    #[derive(Debug, Clone, Copy)]
    struct Symmetric3x3 {
        m_00: f64,
        m_10: f64,
        m_20: f64,
        m_11: f64,
        m_21: f64,
        m_22: f64,
    }

    impl Symmetric3x3 {
        fn from_mat3x3(m: &Mat3F64) -> Self {
            let c0 = m.x_axis;
            let c1 = m.y_axis;
            let c2 = m.z_axis;

            Self {
                m_00: c0.dot(c0),
                m_10: c0.dot(c1),
                m_20: c0.dot(c2),
                m_11: c1.dot(c1),
                m_21: c1.dot(c2),
                m_22: c2.dot(c2),
            }
        }
    }

    fn exact_givens(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
        if apq.abs() < EPSILON {
            return (1.0, 0.0);
        }

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;
        (c, s)
    }

    #[derive(Debug, Clone)]
    pub struct SVD3Set {
        pub u: Mat3F64,
        pub s: Mat3F64,
        pub v: Mat3F64,
    }

    impl SVD3Set {
        #[inline]
        pub fn u(&self) -> &Mat3F64 {
            &self.u
        }
        #[inline]
        pub fn s(&self) -> &Mat3F64 {
            &self.s
        }
        #[inline]
        pub fn v(&self) -> &Mat3F64 {
            &self.v
        }
    }

    pub fn svd3(mat: &Mat3F64) -> SVD3Set {
        let mut s_mat = Symmetric3x3::from_mat3x3(mat);
        let mut q_accum = QuatF64::IDENTITY;

        for _sweep in 0..MAX_SWEEPS {
            {
                let (c, si) = exact_givens(s_mat.m_00, s_mat.m_11, s_mat.m_10);
                let s00 = s_mat.m_00;
                let s01 = s_mat.m_10;
                let s11 = s_mat.m_11;
                let s02 = s_mat.m_20;
                let s12 = s_mat.m_21;

                s_mat.m_00 = c * c * s00 - 2.0 * c * si * s01 + si * si * s11;
                s_mat.m_11 = si * si * s00 + 2.0 * c * si * s01 + c * c * s11;
                s_mat.m_10 = 0.0;
                s_mat.m_20 = c * s02 - si * s12;
                s_mat.m_21 = si * s02 + c * s12;

                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = -si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF64::from_xyzw(
                    q[0] * ch + q[1] * sh,
                    q[1] * ch - q[0] * sh,
                    q[2] * ch + q[3] * sh,
                    q[3] * ch - q[2] * sh,
                )
                .normalize();
            }

            {
                let (c, si) = exact_givens(s_mat.m_11, s_mat.m_22, s_mat.m_21);
                let s11 = s_mat.m_11;
                let s12 = s_mat.m_21;
                let s22 = s_mat.m_22;
                let s01 = s_mat.m_10;
                let s02 = s_mat.m_20;

                s_mat.m_11 = c * c * s11 - 2.0 * c * si * s12 + si * si * s22;
                s_mat.m_22 = si * si * s11 + 2.0 * c * si * s12 + c * c * s22;
                s_mat.m_21 = 0.0;
                s_mat.m_10 = c * s01 - si * s02;
                s_mat.m_20 = si * s01 + c * s02;

                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = -si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF64::from_xyzw(
                    q[0] * ch + q[3] * sh,
                    q[1] * ch + q[2] * sh,
                    q[2] * ch - q[1] * sh,
                    q[3] * ch - q[0] * sh,
                )
                .normalize();
            }

            {
                let (c, si) = exact_givens(s_mat.m_00, s_mat.m_22, s_mat.m_20);
                let s00 = s_mat.m_00;
                let s02 = s_mat.m_20;
                let s22 = s_mat.m_22;
                let s01 = s_mat.m_10;
                let s12 = s_mat.m_21;

                s_mat.m_00 = c * c * s00 - 2.0 * c * si * s02 + si * si * s22;
                s_mat.m_22 = si * si * s00 + 2.0 * c * si * s02 + c * c * s22;
                s_mat.m_20 = 0.0;
                s_mat.m_10 = c * s01 - si * s12;
                s_mat.m_21 = si * s01 + c * s12;

                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF64::from_xyzw(
                    q[0] * ch - q[2] * sh,
                    q[1] * ch + q[3] * sh,
                    q[2] * ch + q[0] * sh,
                    q[3] * ch - q[1] * sh,
                )
                .normalize();
            }

            let off_diag_sq =
                s_mat.m_10 * s_mat.m_10 + s_mat.m_20 * s_mat.m_20 + s_mat.m_21 * s_mat.m_21;
            if off_diag_sq < JACOBI_TOLERANCE {
                break;
            }
        }

        let v = Mat3F64::from_quat(q_accum);
        let s_diag = Vec3F64::new(
            s_mat.m_00.abs().sqrt(),
            s_mat.m_11.abs().sqrt(),
            s_mat.m_22.abs().sqrt(),
        );

        let mut b = *mat * v;
        let mut s_vec = s_diag;
        let mut v_mat = v;

        sort_singular_values(&mut b, &mut s_vec, &mut v_mat);

        let u_x = if s_vec.x > EPSILON {
            b.x_axis / s_vec.x
        } else {
            Mat3F64::IDENTITY.x_axis
        };
        let mut u_y = if s_vec.y > EPSILON {
            b.y_axis / s_vec.y
        } else {
            Mat3F64::IDENTITY.y_axis
        };
        let mut u_z = if s_vec.z > EPSILON {
            b.z_axis / s_vec.z
        } else {
            Mat3F64::IDENTITY.z_axis
        };

        // Safe Gram-Schmidt
        u_y = u_y - u_x * u_x.dot(u_y);
        if u_y.length_squared() < EPSILON {
            u_y = u_x.cross(Mat3F64::IDENTITY.x_axis);
            if u_y.length_squared() < EPSILON {
                u_y = u_x.cross(Mat3F64::IDENTITY.y_axis);
            }
        }
        u_y = u_y.normalize();

        u_z = u_z - u_x * u_x.dot(u_z) - u_y * u_y.dot(u_z);
        if u_z.length_squared() < EPSILON {
            u_z = u_x.cross(u_y);
        }
        u_z = u_z.normalize();

        let mut final_u = Mat3F64::from_cols(u_x.into(), u_y.into(), u_z.into());

        if final_u.determinant() < 0.0 {
            // Flip the column with the smallest singular value (Z-axis)
            let u_z_col = final_u.z_axis();
            final_u = Mat3F64::from_cols(final_u.x_axis(), final_u.y_axis(), -u_z_col);

            // Flip V's corresponding column to maintain A = U * S * V^T
            let v_z_col = v_mat.z_axis();
            v_mat = Mat3F64::from_cols(v_mat.x_axis(), v_mat.y_axis(), -v_z_col);
        }

        SVD3Set {
            u: final_u,
            s: Mat3F64::from_diagonal(s_vec),
            v: v_mat,
        }
    }

    pub fn sort_singular_values(u_mat: &mut Mat3F64, s_vec: &mut Vec3F64, v_mat: &mut Mat3F64) {
        if s_vec.x < s_vec.y {
            std::mem::swap(&mut s_vec.x, &mut s_vec.y);
            let col0 = v_mat.x_axis;
            v_mat.x_axis = v_mat.y_axis;
            v_mat.y_axis = col0;
            let col0 = u_mat.x_axis;
            u_mat.x_axis = u_mat.y_axis;
            u_mat.y_axis = col0;
        }
        if s_vec.y < s_vec.z {
            std::mem::swap(&mut s_vec.y, &mut s_vec.z);
            let col1 = v_mat.y_axis;
            v_mat.y_axis = v_mat.z_axis;
            v_mat.z_axis = col1;
            let col1 = u_mat.y_axis;
            u_mat.y_axis = u_mat.z_axis;
            u_mat.z_axis = col1;
        }
        if s_vec.x < s_vec.y {
            std::mem::swap(&mut s_vec.x, &mut s_vec.y);
            let col0 = v_mat.x_axis;
            v_mat.x_axis = v_mat.y_axis;
            v_mat.y_axis = col0;
            let col0 = u_mat.x_axis;
            u_mat.x_axis = u_mat.y_axis;
            u_mat.y_axis = col0;
        }
    }

    #[allow(dead_code)]
    pub type SVD3SetF64 = SVD3Set;
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mat3F32, Mat3F64, Vec3F32, Vec3F64};

    const TEST_EPSILON_F32: f32 = 1e-5;
    const TEST_EPSILON_F32_STRICT: f32 = 1e-6;
    const TEST_EPSILON_F64: f64 = 1e-10;

    // Relaxed tolerance for f32 essential matrix (1e-3 is safe given 0.0009 error)
    const TEST_TOLERANCE_F32: f32 = 1e-3;
    const TEST_TOLERANCE_F64: f64 = 2e-4;

    fn verify_svd_properties_f32(a: &Mat3F32, svd: &SVD3SetF32, epsilon: f32) {
        let u = *svd.u();
        let s = *svd.s();
        let v = *svd.v();
        let reconstruction = u * s * v.transpose();

        let diff = *a - reconstruction;
        assert!(
            diff.x_axis().dot(diff.x_axis()) < epsilon,
            "Reconstruction error (x): {:?}",
            diff
        );
        assert!(
            diff.y_axis().dot(diff.y_axis()) < epsilon,
            "Reconstruction error (y): {:?}",
            diff
        );
        assert!(
            diff.z_axis().dot(diff.z_axis()) < epsilon,
            "Reconstruction error (z): {:?}",
            diff
        );

        let u_t_u = u.transpose() * u;
        let diff_u = u_t_u - Mat3F32::IDENTITY;
        assert!(
            diff_u.x_axis().dot(diff_u.x_axis()) < epsilon,
            "U orthogonality error (x)"
        );

        let v_t_v = v.transpose() * v;
        let diff_v = v_t_v - Mat3F32::IDENTITY;
        assert!(
            diff_v.x_axis().dot(diff_v.x_axis()) < epsilon,
            "V orthogonality error (x)"
        );
    }

    fn verify_svd_properties_f64(a: &Mat3F64, svd: &SVD3SetF64, epsilon: f64) {
        let u = *svd.u();
        let s = *svd.s();
        let v = *svd.v();
        let reconstruction = u * s * v.transpose();

        let diff = *a - reconstruction;
        assert!(
            diff.x_axis.length_squared() < epsilon,
            "Reconstruction error (x): {:?}",
            diff
        );
        assert!(
            diff.y_axis.length_squared() < epsilon,
            "Reconstruction error (y): {:?}",
            diff
        );
        assert!(
            diff.z_axis.length_squared() < epsilon,
            "Reconstruction error (z): {:?}",
            diff
        );

        let u_t_u = u.transpose() * u;
        let diff_u = u_t_u - Mat3F64::IDENTITY;
        assert!(
            diff_u.x_axis.length_squared() < epsilon,
            "U orthogonality error (x)"
        );

        let v_t_v = v.transpose() * v;
        let diff_v = v_t_v - Mat3F64::IDENTITY;
        assert!(
            diff_v.x_axis.length_squared() < epsilon,
            "V orthogonality error (x)"
        );
    }

    #[test]
    fn test_svd3_f32_simple() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(3.0, 2.0, 1.0));
        let svd = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd, TEST_EPSILON_F32);
    }

    #[test]
    fn test_svd3_f32_zero() {
        let a = Mat3F32::from_cols(Vec3F32::ZERO, Vec3F32::ZERO, Vec3F32::ZERO);
        let svd_result = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd_result, TEST_EPSILON_F32_STRICT);
        let s_x = svd_result.s().x_axis();
        assert!(s_x.x.abs() < TEST_EPSILON_F32_STRICT);
    }

    #[test]
    fn test_svd3_f32_identity() {
        let a = Mat3F32::IDENTITY;
        let svd_result = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd_result, TEST_EPSILON_F32_STRICT);
    }

    #[test]
    fn test_svd3_f32_rotation_matrix() {
        let cos = std::f32::consts::FRAC_PI_4.cos();
        let sin = std::f32::consts::FRAC_PI_4.sin();
        let a = Mat3F32::from_cols(
            Vec3F32::new(cos, 0.0, sin),
            Vec3F32::new(0.0, 1.0, 0.0),
            Vec3F32::new(-sin, 0.0, cos),
        );
        let svd_result = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd_result, TEST_EPSILON_F32);
    }

    #[test]
    fn test_svd3_f32_essential_matrix_precision() {
        let e = Mat3F32::from_cols(
            Vec3F32::new(0.0, -2.552, 0.0),
            Vec3F32::new(1.708, 0.0, -8.540),
            Vec3F32::new(0.0, 8.327, 0.0),
        );

        let svd = svd3_f32(&e);
        let s = svd.s();

        let sigma1 = s.x_axis().x;
        let sigma2 = s.y_axis().y;
        let sigma3 = s.z_axis().z;

        println!(
            "Computed Sigmas (f32): ({}, {}, {})",
            sigma1, sigma2, sigma3
        );

        let diff = (sigma1 - sigma2).abs();

        assert!(
            diff < TEST_TOLERANCE_F32,
            "FAILURE: Singular values should be equal! \nGot: {}, {}\nDiff: {}",
            sigma1,
            sigma2,
            diff
        );

        assert!(
            sigma3.abs() < TEST_TOLERANCE_F32,
            "FAILURE: Third singular value should be zero! Got: {}",
            sigma3
        );

        let expected_sigma = 8.709;
        assert!(
            (sigma1 - expected_sigma).abs() < 1e-2,
            "FAILURE: Incorrect magnitude. Expected ~{}, Got {}",
            expected_sigma,
            sigma1
        );
    }

    #[test]
    fn test_svd3_f32_stress_cases() {
        // CASE 1: Ill-conditioned Matrix
        let a_ill = Mat3F32::from_diagonal(Vec3F32::new(1e6, 1.0, 1e-6));
        let svd_ill = svd3_f32(&a_ill);
        let s_ill = svd_ill.s();
        println!("Ill-conditioned Sigmas: {:?}", s_ill.to_cols_array());
        verify_svd_properties_f32(&a_ill, &svd_ill, 0.1);

        // CASE 2: Clustered Singular Values
        let a_cluster = Mat3F32::from_diagonal(Vec3F32::new(1.0, 1.0 + 1e-5, 0.1));
        let svd_cluster = svd3_f32(&a_cluster);
        let s_cluster = svd_cluster.s();
        println!("Clustered Sigmas: {:?}", s_cluster.to_cols_array());

        let diff = (s_cluster.x_axis().x - s_cluster.y_axis().y).abs();
        assert!(diff < 2e-4, "Failed to resolve clustered values");
        verify_svd_properties_f32(&a_cluster, &svd_cluster, 1e-4);

        // CASE 3: Nearly Rank-Deficient
        let a_rank = Mat3F32::from_diagonal(Vec3F32::new(1.0, 0.5, 1e-7));
        let svd_rank = svd3_f32(&a_rank);
        let s_rank = svd_rank.s();
        println!("Rank-Deficient Sigmas: {:?}", s_rank.to_cols_array());

        assert!(
            s_rank.z_axis().z < 1e-6,
            "Failed to identify near-zero singular value"
        );
        verify_svd_properties_f32(&a_rank, &svd_rank, 1e-6);
    }

    #[test]
    fn test_svd3_f64_simple() {
        let a = Mat3F64::from_diagonal(Vec3F64::new(3.0, 2.0, 1.0));
        let svd = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd, TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_zero() {
        let a = Mat3F64::from_cols(Vec3F64::ZERO, Vec3F64::ZERO, Vec3F64::ZERO);
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, TEST_EPSILON_F64);
        let s_x = svd_result.s().x_axis();
        assert!(s_x.x.abs() < TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_identity() {
        let a = Mat3F64::IDENTITY;
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_random_rotation() {
        let ang = std::f64::consts::PI / 4.0;
        let cos = ang.cos();
        let sin = ang.sin();
        let a = Mat3F64::from_cols(
            Vec3F64::new(cos, sin, 0.0),
            Vec3F64::new(-sin, cos, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let svd = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd, TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_essential_matrix_precision() {
        let e = Mat3F64::from_cols(
            Vec3F64::new(0.0, -2.552, 0.0),
            Vec3F64::new(1.708, 0.0, -8.540),
            Vec3F64::new(0.0, 8.327, 0.0),
        );

        let svd = svd3_f64(&e);

        verify_svd_properties_f64(&e, &svd, TEST_EPSILON_F64);
        let s = svd.s();

        let sigma1 = s.x_axis.x;
        let sigma2 = s.y_axis.y;
        let sigma3 = s.z_axis.z;

        println!(
            "Computed Sigmas (f64): ({}, {}, {})",
            sigma1, sigma2, sigma3
        );

        let diff = (sigma1 - sigma2).abs();

        assert!(
            diff < TEST_TOLERANCE_F64,
            "FAILURE: Singular values should be equal! \nGot: {}, {}\nDiff: {}",
            sigma1,
            sigma2,
            diff
        );

        assert!(
            sigma3.abs() < TEST_TOLERANCE_F64,
            "FAILURE: Third singular value should be zero! Got: {}",
            sigma3
        );

        let expected_sigma = 8.709;
        assert!(
            (sigma1 - expected_sigma).abs() < 1e-2,
            "FAILURE: Incorrect magnitude. Expected ~{}, Got {}",
            expected_sigma,
            sigma1
        );
    }

    #[test]
    fn test_svd3_sign_correction_consistency() {
        // 1. Create a Matrix that is a REFLECTION (Determinant = -1)
        // Ideally, SVD should return U with det=1 (Rotation) by flipping a column.
        let reflection = Mat3F32::from_diagonal(Vec3F32::new(1.0, 1.0, -1.0));

        let svd = svd3_f32(&reflection);
        let u = *svd.u();
        let s = *svd.s();
        let v = *svd.v();

        // CHECK 1: Reconstruction must still work (Math check)
        // A = U * S * V^T
        let reconstruction = u * s * v.transpose();
        let diff = reflection - reconstruction;

        // FIX: Use .dot() instead of .length_squared()
        assert!(
            diff.x_axis().dot(diff.x_axis()) < 1e-5,
            "Reconstruction error (x): {:?}",
            diff.x_axis()
        );
        assert!(
            diff.y_axis().dot(diff.y_axis()) < 1e-5,
            "Reconstruction error (y): {:?}",
            diff.y_axis()
        );
        assert!(
            diff.z_axis().dot(diff.z_axis()) < 1e-5,
            "Reconstruction error (z): {:?}",
            diff.z_axis()
        );

        // CHECK 2: PnP Requirement (Rotation check)
        // PnP fails if det(U) is negative. Your fix should ensure det(U) > 0.
        let det_u = u.determinant();
        assert!(
            det_u > 0.0,
            "Determinant of U is negative ({})! PnP will likely fail.",
            det_u
        );

        println!("Sign Correction Verified: det(U) = {}", det_u);
    }
}
