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
        Self { z: DVec2 { x: mat.col(0)[0], y: mat.col(0)[1] } }
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

    pub fn inverse(&self) -> Self {
        Self { z: DVec2 { x: 1.0, y: 1.0 }/self.z }
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
    }
}

