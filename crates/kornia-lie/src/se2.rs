use glam::{Mat2, Mat3A, Vec2, Vec3A};
use rand::Rng;

use crate::so2::SO2;

#[derive(Debug, Clone, Copy)]
pub struct SE2 {
    pub r: SO2,
    pub t: Vec2,
}

impl SE2 {
    pub const IDENTITY: Self = Self { r: SO2::IDENTITY, t: Vec2::from_array([0.0, 0.0]) };

    pub fn new(r: SO2, t: Vec2) -> Self {
        Self { r, t }
    }

    pub fn from_matrix(mat: Mat3A) -> Self {
        Self { r: SO2::from_matrix3a(mat), t: Vec2::new(mat.col(0)[2], mat.col(1)[2]) }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();

        Self { r: SO2::from_random(), t: Vec2::new(r1, r2) }
    }

    pub fn matrix(&self) -> Mat3A {
        let r = self.r.matrix();
        Mat3A::from_cols_array(&[
            r.x_axis.x, r.y_axis.x, self.t.x,
            r.x_axis.y, r.y_axis.y, self.t.y,
            0.0, 0.0, 1.0,
        ])
    }

    pub fn inverse(&self) -> Self {
        Self { r: self.r.inverse(), t: self.r.inverse() * (-self.t) }
    }

    pub fn adjoint(&self) -> Mat3A {
        let mut mat = self.matrix();
        mat.col_mut(0)[2] = self.t[1];
        mat.col_mut(1)[2] =-self.t[0];

        mat
    }

    pub fn exp(upsilon: Vec2, theta: f32) -> Self {
        let so2 = SO2::exp(theta);
        
        Self {
            r: so2,
            t: {
                let (a,b) = if theta != 0.0 {
                    (so2.z[1]/theta, (1.0-so2.z[0])/theta)  
                } else {
                    (0.0, 0.0)
                };
                Vec2::new(a*upsilon[0]-b*upsilon[1], b*upsilon[0]+a*upsilon[1])
            }
        }
    }

    pub fn log(&self) -> (Vec2, f32) {
        let theta = self.r.log();
        let half_theta = 0.5*theta;
        let denom = self.r.z[0]-1.0;
        let a = if denom != 0.0 {
            -(half_theta*self.r.z[1]) / denom
        } else {
            0.0
        };
        let mat_v_inv = Mat2::from_cols_array(&[
             a,-half_theta,
             half_theta, a
        ]);
        let upsilon = mat_v_inv * self.t;

        (upsilon, theta)
    }

    pub fn hat(upsilon: Vec2, theta: f32) -> Mat3A {
        let hat_theta = SO2::hat(theta);
        
        let col0 = Mat3A::from_cols(
            hat_theta.col(0).extend(upsilon.x).into(),
            hat_theta.col(1).extend(upsilon.y).into(),
            Vec3A::ZERO, // Padding the last column with zeroes
        );
        
        col0
    }

    pub fn vee(omega: Mat3A) -> (Vec2, f32) {
        (
            Vec2::new(omega.col(0)[2], omega.col(1)[2]),  // TODO: why is the translation/upsilon at the bottom???
            SO2::vee(Mat2::from_cols_array(&[
                omega.col(0)[0], omega.col(0)[1],
                omega.col(1)[0], omega.col(1)[1],
            ]))
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let rotation = SO2::exp(0.5);
        let translation = Vec2::new(1.0, 2.0);
        let se2 = SE2::new(rotation, translation);
        assert_eq!(se2.t, translation);
        assert_eq!(se2.r.z, rotation.z);
    }
    
    #[test]
    fn test_from_matrix() {
        let mat = Mat3A::IDENTITY;
        let se2 = SE2::from_matrix(mat);
        assert_eq!(se2.t, Vec2::new(0.0, 0.0));
        assert_eq!(se2.r.matrix(), SO2::IDENTITY.matrix());
    }
    
    #[test]
    fn test_inverse() {
        let se2 = SE2::new(SO2::exp(0.5), Vec2::new(1.0, 2.0));
        let inv = se2.inverse();
        assert_eq!(inv.t, -(se2.r.inverse() * se2.t));
        assert_eq!(inv.r.z, se2.r.inverse().z);
    }
    
    #[test]
    fn test_matrix() {
        let se2 = SE2::new(SO2::exp(0.5), Vec2::new(1.0, 2.0));
        let mat = se2.matrix();
        assert_eq!(mat.col(0)[2], 1.0);
        assert_eq!(mat.col(1)[2], 2.0);
        assert_eq!(mat.col(2)[2], 1.0);
    }
    
    #[test]
    fn test_exp() {
        let upsilon = Vec2::new(1.0, 1.0);
        let theta = 1.0;

        let se2 = SE2::exp(upsilon, theta);

        println!("{:?}", se2);
        
        assert!((se2.r.z - Vec2::new(0.5403, 0.8415)).length() < 1e-3);
        assert!((se2.t - Vec2::new(0.3818, 1.3012)).length() < 1e-3);
    }
    
    #[test]
    fn test_log() {
        {
            let upsilon = Vec2::new(1.0, 1.0);
            let theta = 1.0;

            let se2 = SE2::exp(upsilon, theta);

            let (log_t, log_theta) = se2.log();
            
            assert!((log_t - upsilon).length() < 1e-3);
            assert!((log_theta - theta).abs() < 1e-3);
        }
        {
            let upsilon = Vec2::new(0.5/0.7071067812, -0.5/0.7071067812);
            let theta = 0.3;

            let se2 = SE2::exp(upsilon, theta);

            let (log_t, log_theta) = se2.log();
            
            assert!((log_t - upsilon).length() < 1e-5);
            assert!((log_theta - theta).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_hat_vee() {
        let upsilon = Vec2::new(1.0/2.236067977, 2./2.236067977);
        let theta = 0.3;
        let hat_matrix = SE2::hat(upsilon, theta);
        let (vee_t, vee_theta) = SE2::vee(hat_matrix);

        // println!("{:?}", upsilon);
        // println!("{:?}", vee_t);
        // println!("{:?}", theta);
        // println!("{:?}", vee_theta);

        assert!((vee_t - upsilon).length() < 1e-5);
        assert!((vee_theta - theta).abs() < 1e-5);
    }
}
