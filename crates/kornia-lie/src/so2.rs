use std::ops::Mul;

use glam::{Mat2, Mat3A, Vec2};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SO2 {
    /// representing complex number [real, imaginary]
    pub z: Vec2,
}

impl SO2 {
    pub const IDENTITY: Self = Self { z: Vec2 { x: 1.0, y: 0.0 } };

    pub fn new(z: Vec2) -> Self {
        Self { z }
    }

    pub fn from_matrix(mat: Mat2) -> Self {
        Self { z: Vec2 { x: mat.col(0)[0], y: mat.col(1)[0] } }
    }

    pub fn from_matrix3a(mat: Mat3A) -> Self {
        Self { z: Vec2 { x: mat.col(0)[0], y: mat.col(1)[0] } }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();

        Self { z: Vec2 { x: r1, y: r2 }}
    }

    pub fn matrix(&self) -> Mat2 {
        Mat2::from_cols_array(&[
             self.z[0],-self.z[1],
             self.z[1], self.z[0],
        ])
    }

    /// inverting the complex number z (represented as a 2D vector)
    pub fn inverse(&self) -> Self {
        let c: f32 = self.z.dot(self.z);
        Self { z: Vec2::new(self.z[0]/c, -self.z[1]/c) }
    }

    pub fn adjoint(&self) -> Mat2 {
        Self::IDENTITY.matrix()
    }

    pub fn exp(theta: f32) -> Self {
        Self { z: Vec2 { x: theta.cos(), y: theta.sin() } }
    }

    pub fn log(&self) -> f32 {
        self.z[1].atan2(self.z[0])
    }

    pub fn hat(theta: f32) -> Mat2 {
        Mat2::from_cols_array(&[
              0.0,theta,
            theta,  0.0,
        ])
    }

    pub fn vee(omega: Mat2) -> f32 {
        omega.col(0)[1]
    }
}


impl Mul<Vec2> for SO2 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.z[0]*rhs.x - self.z[1]*rhs.y, self.z[1]*rhs.x + self.z[0]*rhs.y)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let identity = SO2::IDENTITY;
        assert_eq!(identity.z, Vec2::new(1.0, 0.0));
    }
    
    #[test]
    fn test_new() {
        let z = Vec2::new(0.5, 0.5);
        let so2 = SO2::new(z);
        assert_eq!(so2.z, z);
    }
    
    #[test]
    fn test_from_matrix() {
        {
            let mat = Mat2::from_cols_array(&Mat2::IDENTITY.to_cols_array());
            let so2 = SO2::from_matrix(mat);
            assert_eq!(so2.z, Vec2::new(1.0, 0.0));
        }
        {
            let mat = Mat2::from_cols_array(&[0.6, -0.8, 0.8, 0.6]);
            let so2 = SO2::from_matrix(mat);
            assert_eq!(so2.z, Vec2::new(0.6, 0.8));
        }
    }
    
    #[test]
    fn test_as_matrix() {
        let so2 = SO2::new(Vec2::new(0.6, 0.8));
        let expected = Mat2::from_cols_array(&[0.6, -0.8, 0.8, 0.6]);
        assert_eq!(so2.matrix(), expected);
    }
    
    #[test]
    fn test_inverse() {
        {
            let so2 = SO2::IDENTITY;
            let inv = so2.inverse();
            let expected = Vec2::new(1.0, 0.0);
            assert!((inv.z - expected).length() < 1e-10);
        }
        {
            let so2 = SO2::new(Vec2::new(0.6, 0.8));
            let inv = so2.inverse();
            let expected = Vec2::new(0.6, -0.8);
            assert!((inv.z - expected).length() < 1e-10);
        }
    }
    
    #[test]
    fn test_adjoint() {
        let so2 = SO2::IDENTITY;
        assert_eq!(so2.adjoint(), so2.matrix());
    }
    
    #[test]
    fn test_exp() {
        let theta = std::f32::consts::PI / 4.0;
        let so2 = SO2::exp(theta);
        assert!((so2.z.x - theta.cos()).abs() < 1e-10);
        assert!((so2.z.y - theta.sin()).abs() < 1e-10);
    }
    
    #[test]
    fn test_log() {
        let so2 = SO2::new(Vec2::new(0.6, 0.8));
        let theta = so2.log();
        assert!((theta - 0.9273).abs() < 1e-4);
    }
    
    #[test]
    fn test_hat() {
        let theta = 0.5;
        let expected = Mat2::from_cols_array(&[0.0, 0.5, 0.5, 0.0]);
        assert_eq!(SO2::hat(theta), expected);
    }
    
    #[test]
    fn test_vee() {
        let omega = Mat2::from_cols_array(&[0.0, 0.5, 0.5, 0.0]);
        let theta = SO2::vee(omega);
        assert_eq!(theta, 0.5);
    }
}

