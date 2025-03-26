use glam::{DMat2, DMat3, DVec2, Mat3, Mat3A, Vec2, Vec3A};

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

        todo!()
    }

    pub fn from_random() -> Self {

        todo!()
    }

    pub fn as_matrix(&self) -> Mat3A {
        let r = self.r.as_matrix();
        Mat3A::from_cols_array(&[
            r.x_axis.x, r.y_axis.x, self.t.x,
            r.x_axis.y, r.y_axis.y, self.t.y,
            0.0, 0.0, 1.0,
        ])
    }

    pub fn inverse(&self) -> Self {
        // let inv_rot = self.r.inverse();
        // let inv_trans = -(inv_rot.rotation * self.t);
        // Self { r: inv_rot, t: inv_trans }
    
        todo!()
    }

    pub fn adjoint(&self) -> Mat3A {
        todo!()
    }

    pub fn exp(upsilon: Vec2, theta: f32) -> Self {
        todo!("Takes in vector of shape 6")
    }

    pub fn log(&self) -> (Vec2, f32) {
        todo!("Takes in vector of shape 6")
    }

    pub fn hat(upsilon: Vec2, theta: f32) -> Mat3A {
        todo!("Takes in vector of shape 6")
    }

    pub fn vee(omega: Mat3A) -> (Vec2, f32) {
        todo!("Takes in vector of shape 6")
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
    }
}
