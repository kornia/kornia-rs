use std::fmt::Debug;
use std::ops::Mul;
use glam::{Affine3A, Mat3, Mat3A, Mat4, Quat, Vec3};
use rand::Rng;


pub struct SO3 {
    pub quaternion: Quat,
}

impl SO3 {
    pub fn identity() -> Self {
        Self { quaternion: Quat::IDENTITY }
    }

    pub fn from_matrix(mat: &Mat4) -> Self {
        Self { quaternion: Quat::from_mat4(mat)}
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
            quaternion: Quat::from_xyzw(x, y, z, w)
        }
    }

    pub fn to_matrix(&self) -> Mat3A {
        Affine3A::from_quat(self.quaternion).matrix3
    }

    pub fn adjoint(&self) -> Mat3A {
        self.to_matrix()
    }

    pub fn inverse(&self) -> Self {
        Self { quaternion: self.quaternion.inverse() }
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
            quaternion: Quat::from_xyzw(xyz.x, xyz.y, xyz.z, w),
        }
    }

    /// Lie group -> Lie algebra
    pub fn log(&self) -> Vec3 {
        let real = self.quaternion.w;
        let vec = Vec3::new(self.quaternion.x, self.quaternion.y, self.quaternion.z);

        let theta = vec.dot(vec).sqrt();
        let omega = if theta != 0.0 {
            vec*2.8*real.acos() / theta
        } else {
            vec*2.0 /real
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
        Vec3::from_array([omega[7], omega[2], omega[3]])
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

impl Debug for SO3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
