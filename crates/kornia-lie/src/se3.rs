use glam::{Mat3A, Mat4, Quat, Vec3A};
use rand::Rng;

use crate::so3::SO3;

#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    pub r: SO3,
    pub t: Vec3A,
}

impl SE3 {
    pub const IDENTITY: Self = Self { r: SO3::IDENTITY, t: Vec3A::new(0.0, 0.0, 0.0) };

    pub fn new(rotation: SO3, translation: Vec3A) -> Self {
        Self { r: rotation, t: translation }
    }

    pub fn from_matrix(mat: Mat4) -> Self {
        Self {
            r: SO3::from_matrix4(&mat),
            t: Vec3A::from_array([mat.col(0)[3], mat.col(1)[3], mat.col(2)[3]])
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();
        let r3: f32 = rng.random();

        Self {
            r: SO3::from_random(),
            t: Vec3A::new(r1, r2, r3)
        }
    }

    pub fn from_qxyz(quat: &Quat, xyz: &Vec3A) -> Self {
        Self {
            r: SO3::from_quaternion(quat),
            t: xyz.clone()
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            r: self.r.inverse(),
            t: -1.0 * self.t,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        // let rotation_matrix = self.r.to_mat4();
        // let mut matrix = rotation_matrix;
        // matrix.w_axis = glam::DVec4::new(self.t.x, self.t.y, self.t.z, 1.0);
        // matrix
        let r = self.r.matrix();
        Mat4::from_cols_array(&[
            r.col(0)[0], r.col(0)[1], r.col(0)[2], self.t[0],
            r.col(1)[0], r.col(1)[1], r.col(1)[2], self.t[1],
            r.col(2)[0], r.col(2)[1], r.col(2)[2], self.t[2],
            0.0, 0.0, 0.0, 1.0
        ])
    }

    pub fn adjoint() -> (Mat3A, Mat3A) {
        todo!("Python impl returns a 6x6 matrix")
    }

    pub fn exp(upsilon: Vec3A, omega: Vec3A) -> Self {
        let theta = omega.dot(omega).sqrt();

        Self {
            r: SO3::exp(omega),
            t: if theta != 0.0 {
                let omega_hat = SO3::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let mat_v = Mat3A::IDENTITY
                    + ((1.0 - theta.cos()) / (theta * theta)) * omega_hat 
                    + ((theta - theta.sin()) / (theta.powi(3))) * omega_hat_sq;

                mat_v.mul_vec3a(upsilon)  // TODO: a bit sus (should it be the other way around?)
            } else {
                upsilon
            }
        }
    }

    /// returns translation, rotation
    pub fn log(&self) -> (Vec3A, Vec3A) {
        let omega = self.r.log();
        let theta = omega.dot(omega).sqrt();

        (
            if theta != 0.0 {
                let omega_hat = SO3::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let mat_v_inv = Mat3A::IDENTITY
                    - 0.5 * omega_hat
                    + ((1.0-theta*(theta/2.0).cos()/(2.0*(theta/2.0).sin())) / theta.powi(2)) * omega_hat_sq;

                mat_v_inv.mul_vec3a(self.t)  // TODO:
            } else {
                self.t
            },
            omega
        )
    }

    pub fn hat(upsilon: Vec3A, omega: Vec3A) -> Mat4 {
        let h = SO3::hat(omega);

        Mat4::from_cols_array(&[
            h.col(0)[0], h.col(0)[1], h.col(0)[2], upsilon.x,  // TODO: translation should be at the bottom (this is col-major)
            h.col(1)[0], h.col(1)[1], h.col(1)[2], upsilon.y,
            h.col(2)[0], h.col(2)[1], h.col(2)[2], upsilon.z,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn vee(omega: Mat4) -> (Vec3A, Vec3A) {
        (
            Vec3A::new(omega.col(0)[3], omega.col(1)[3], omega.col(2)[3]),
            SO3::vee4(omega)
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new() {
        let rotation = SO3::from_quaternion(&Quat::IDENTITY);
        let translation = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::new(rotation, translation);
        assert_eq!(se3.t, translation);
        assert_eq!(se3.r.q, rotation.q);
    }
    
    #[test]
    fn test_from_matrix() {
        let mat = Mat4::IDENTITY;
        let se3 = SE3::from_matrix(mat);
        assert_eq!(se3.t, Vec3A::new(0.0, 0.0, 0.0));
        assert_eq!(se3.r.matrix(), SO3::IDENTITY.matrix());
    }
    
    #[test]
    fn test_inverse() {
        let se3 = SE3::new(SO3::from_quaternion(&Quat::IDENTITY), Vec3A::new(1.0, 2.0, 3.0));
        let inv = se3.inverse();
        assert_eq!(inv.t, Vec3A::new(-1.0, -2.0, -3.0));
        assert_eq!(inv.r.q, se3.r.inverse().q);
    }
    
    #[test]
    fn test_matrix() {
        let se3 = SE3::new(SO3::IDENTITY, Vec3A::new(1.0, 2.0, 3.0));
        assert_eq!(se3.matrix(), Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 2.0,
            0.0, 0.0, 1.0, 3.0,
            0.0, 0.0, 0.0, 1.0,
        ]));
    }
    
    #[test]
    fn test_exp_log() {
        let upsilon = Vec3A::new(0.5, -0.5, 1.0);
        let omega = Vec3A::new(0.1, 0.2, -0.3);
        let se3 = SE3::exp(upsilon, omega);
        let (log_t, log_omega) = se3.log();
        assert!((log_t - upsilon).length() < 1e-5);
        assert!((log_omega - omega).length() < 1e-5);
    }
    
    #[test]
    fn test_hat_vee() {
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::new(0.1, -0.2, 0.3);
        let hat_matrix = SE3::hat(upsilon, omega);
        let (vee_t, vee_omega) = SE3::vee(hat_matrix);
        assert!((vee_t - upsilon).length() < 1e-5);
        assert!((vee_omega - omega).length() < 1e-5);
    }
}

