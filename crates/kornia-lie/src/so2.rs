use glam::{DMat2, DVec2};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SO2 {
    /// representing complex number [real, imaginary]
    pub z: DVec2,
}

impl SO2 {
    pub const IDENTITY: Self = Self { z: DVec2 { x: 1.0, y: 0.0 } };

    pub fn new(z: DVec2) -> Self {
        Self { z }
    }

    pub fn from_matrix(mat: DMat2) -> Self {
        Self { z: DVec2 { x: mat.col(0)[0], y: mat.col(1)[0] } }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        Self { z: DVec2 { x: r1, y: r2 }}
    }

    pub fn as_matrix(&self) -> DMat2 {
        DMat2::from_cols_array(&[
             self.z[0],-self.z[1],
             self.z[1], self.z[0],
        ])
    }

    /// inverting the complex number z (represented as a 2D vector)
    pub fn inverse(&self) -> Self {
        let c: f64 = self.z.dot(self.z);
        Self { z: DVec2::new(self.z[0]/c, -self.z[1]/c) }
    }

    pub fn adjoint(&self) -> DMat2 {
        Self::IDENTITY.as_matrix()
    }

    pub fn exp(theta: f64) -> Self {
        Self { z: DVec2 { x: theta.cos(), y: theta.sin() } }
    }

    pub fn log(&self) -> f64 {
        self.z[1].atan2(self.z[0])
    }

    pub fn hat(theta: f64) -> DMat2 {
        DMat2::from_cols_array(&[
              0.0,theta,
            theta,  0.0,
        ])
    }

    pub fn vee(omega: DMat2) -> f64 {
        omega.col(0)[1]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let identity = SO2::IDENTITY;
        assert_eq!(identity.z, DVec2::new(1.0, 0.0));
    }
    
    #[test]
    fn test_new() {
        let z = DVec2::new(0.5, 0.5);
        let so2 = SO2::new(z);
        assert_eq!(so2.z, z);
    }
    
    #[test]
    fn test_from_matrix() {
        {
            let mat = DMat2::from_cols_array(&DMat2::IDENTITY.to_cols_array());
            let so2 = SO2::from_matrix(mat);
            assert_eq!(so2.z, DVec2::new(1.0, 0.0));
        }
        {
            let mat = DMat2::from_cols_array(&[0.6, -0.8, 0.8, 0.6]);
            let so2 = SO2::from_matrix(mat);
            assert_eq!(so2.z, DVec2::new(0.6, 0.8));
        }
    }
    
    #[test]
    fn test_as_matrix() {
        let so2 = SO2::new(DVec2::new(0.6, 0.8));
        let expected = DMat2::from_cols_array(&[0.6, -0.8, 0.8, 0.6]);
        assert_eq!(so2.as_matrix(), expected);
    }
    
    #[test]
    fn test_inverse() {
        {
            let so2 = SO2::IDENTITY;
            let inv = so2.inverse();
            let expected = DVec2::new(1.0, 0.0);
            assert!((inv.z - expected).length() < 1e-10);
        }
        {
            let so2 = SO2::new(DVec2::new(0.6, 0.8));
            let inv = so2.inverse();
            let expected = DVec2::new(0.6, -0.8);
            assert!((inv.z - expected).length() < 1e-10);
        }
    }
    
    #[test]
    fn test_adjoint() {
        let so2 = SO2::IDENTITY;
        assert_eq!(so2.adjoint(), so2.as_matrix());
    }
    
    #[test]
    fn test_exp() {
        let theta = std::f64::consts::PI / 4.0;
        let so2 = SO2::exp(theta);
        assert!((so2.z.x - theta.cos()).abs() < 1e-10);
        assert!((so2.z.y - theta.sin()).abs() < 1e-10);
    }
    
    #[test]
    fn test_log() {
        let so2 = SO2::new(DVec2::new(0.6, 0.8));
        let theta = so2.log();
        assert!((theta - 0.9273).abs() < 1e-4);
    }
    
    #[test]
    fn test_hat() {
        let theta = 0.5;
        let expected = DMat2::from_cols_array(&[0.0, 0.5, 0.5, 0.0]);
        assert_eq!(SO2::hat(theta), expected);
    }
    
    #[test]
    fn test_vee() {
        let omega = DMat2::from_cols_array(&[0.0, 0.5, 0.5, 0.0]);
        let theta = SO2::vee(omega);
        assert_eq!(theta, 0.5);
    }
}

