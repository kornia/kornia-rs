use std::fmt::Debug;
use std::ops::Mul;
use glam::{Affine3A, Mat3, Mat3A, Mat4, Quat, Vec3};
use rand::Rng;


#[derive(Debug, Clone, Copy)]
pub struct SO3 {
    pub q: Quat,
}

impl SO3 {
    pub const IDENTITY: Self = Self { q: Quat::IDENTITY };

    pub fn from_quaternion(quat: &Quat) -> Self {
        Self { q: Quat::from_array([quat.x, quat.y, quat.z, quat.w]) }
    }

    pub fn from_matrix(mat: &Mat4) -> Self {
        Self { q: Quat::from_mat4(mat)}
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();
        let r3: f32 = rng.random();

        let w = (1.0 - r1).sqrt() * (2.0 * std::f32::consts::PI * r2).sin();
        let x = (1.0 - r1).sqrt() * (2.0 * std::f32::consts::PI * r2).cos();
        let y = r1.sqrt() * (2.0 * std::f32::consts::PI * r3).sin();
        let z = r1.sqrt() * (2.0 * std::f32::consts::PI * r3).cos();

        Self {
            q: Quat::from_xyzw(x, y, z, w)
        }
    }

    pub fn to_matrix(&self) -> Mat3A {
        Affine3A::from_quat(self.q).matrix3
    }

    pub fn adjoint(&self) -> Mat3A {
        self.to_matrix()
    }

    pub fn inverse(&self) -> Self {
        Self { q: self.q.inverse() }
    }

    /// Lie algebra -> Lie group
    pub fn exp(v: Vec3) -> Self {
        let theta = v.dot(v).sqrt();
        let theta_half = theta / 2.0;

        let (w,b) = if theta != 0.0 {
            (theta_half.cos(), theta_half.sin()/theta)
        } else {
            (1.0,0.0)
        };
        let xyz = b*v;

        Self {
            q: Quat::from_xyzw(xyz.x, xyz.y, xyz.z, w),
        }
    }

    /// Lie group -> Lie algebra
    pub fn log(&self) -> Vec3 {
        let real = self.q.w;
        let vec = Vec3::new(self.q.x, self.q.y, self.q.z);

        let theta = vec.dot(vec).sqrt();
        let omega = if theta != 0.0 {
            vec*2.0 * real.acos() / theta
        } else {
            vec*2.0 / real
        };

        omega
    }

    /// Vector space -> Lie algebra
    pub fn hat(v: Vec3) -> Mat3 {
        Mat3::from_cols_array(&[
             0.0, -v.z, v.y,
             v.z,  0.0,-v.x,
            -v.y,  v.x, 0.0,
        ])
    }

    /// Lie algebra -> vector space
    pub fn vee(omega: Mat3) -> Vec3 {
        Vec3::from_array([omega.col(2)[1], omega.col(0)[2], omega.col(1)[0]])
    }

    pub fn left_jacobian(v: Vec3) -> Mat3 {
        let skew = Self::hat(v);
        let theta = v.dot(v).sqrt();
        let ident = Mat3::IDENTITY;

        ident+((1.0-theta.cos())/theta.powi(2))*skew + ((theta-theta.sin()) / theta.powi(3)) * (skew*skew)
    }

    pub fn right_jacobian(v: Vec3) -> Mat3 {
        let skew = Self::hat(v);
        let theta = v.dot(v).sqrt();
        let ident = Mat3::IDENTITY;

        ident-((1.0-theta.cos())/theta.powi(2))*skew + ((theta-theta.sin()) / theta.powi(3)) * (skew*skew)
    }
}

impl Mul for SO3 {
    type Output = SO3;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let s = SO3::identity();
        assert_eq!(s.quaternion, Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_from_quaternion() {
        let q = Quat::from_xyzw(4.0, -2.0, 1.0, 3.5);
        let s = SO3::from_quaternion(&q);
        assert_eq!(s.q, q);
    }

    #[test]
    fn test_from_matrix() {
        let mat = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.6, 0.8, 0.0,
            0.0,-0.8, 0.6, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let s = SO3::from_matrix(&mat);
        assert!((s.q - Quat::from_xyzw(0.5, 0.0, 0.0, 1.0).normalize()).length() < 1e-5);
    }

    #[test]
    fn test_log() {
        {
            let so3 = SO3::from_quaternion(&Quat::from_xyzw(1.0, 1.0, 1.0, 1.0));
            let log = so3.log();
            assert!((log - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-5);
        }

        {
            let so3 = SO3::exp(Vec3::new(1.0, 0.0, 0.0));
            let log = so3.log();
            assert!((log - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        }
    }

    #[test]
    fn test_exp() {
        let v = Vec3::from_array([0.0, 0.0, 0.0]);
        let s = SO3::exp(v);
        assert_eq!(s.q, Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_hat() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let hat_v = SO3::hat(v);
        assert_eq!(hat_v.x_axis.y, -3.0);
        assert_eq!(hat_v.x_axis.z, 2.0);
        assert_eq!(hat_v.y_axis.x, 3.0);
        assert_eq!(hat_v.y_axis.z, -1.0);
        assert_eq!(hat_v.z_axis.x, -2.0);
        assert_eq!(hat_v.z_axis.y, 1.0);
    }

    #[test]
    fn test_vee() {
        {
            let omega = SO3::hat(Vec3::new(1.0, 2.0, 3.0));
            let v = SO3::vee(omega);
            assert!((v - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        }
    }

    #[test]
    fn test_adjoint() {
        let so3 = SO3::exp(Vec3::new(0.1, 0.2, 0.3));
        let adjoint = so3.adjoint();
        assert!(adjoint.is_finite());
    }

    #[test]
    fn test_inverse() {
        let so3 = SO3::exp(Vec3::new(0.5, -0.2, 0.1));
        let inv = so3.inverse();
        let identity = so3.to_matrix() * inv.to_matrix();
        
        let max_diff = (identity-Mat3A::IDENTITY).to_cols_array().iter().map(|&x| x.abs()).fold(0.0, f32::max);

        assert!(max_diff < 1e-5);
    }

    #[test]
    fn test_left_jacobian() {
        let v = Vec3::new(0.1, 0.2, 0.3);
        let left_jacobian = SO3::left_jacobian(v);
        assert!(left_jacobian.is_finite());
    }

    #[test]
    fn test_right_jacobian() {
        let v = Vec3::new(-0.1, 0.3, 0.2);
        let right_jacobian = SO3::right_jacobian(v);
        assert!(right_jacobian.is_finite());
    }
}
