use glam::{DMat4, DQuat, DVec3, Mat3A, Mat4, Quat, Vec3A};

use crate::so3::SO3;

#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    pub r: SO3,
    pub t: Vec3A,
}

impl SE3 {
    pub const IDENTITY: Self = Self { r: SO3::IDENTITY, t: Vec3A::from_array([0.0, 0.0, 0.0]) };

    pub fn new(rotation: SO3, translation: Vec3A) -> Self {
        Self { r: rotation, t: translation }
    }

    pub fn from_matrix(mat: Mat4) -> Self {

        todo!()
    }

    pub fn from_random() -> Self {

        todo!()
    }

    pub fn from_quaternion(quat: &Quat) -> Self {
        todo!()
    }

    pub fn inverse(&self) -> Self {
        // let inv_rot = self.r.inverse();
        // let inv_trans = -(inv_rot * self.t);
        // Self { r: inv_rot, t: inv_trans }
        
        todo!()
    }

    pub fn as_matrix(&self) -> Mat4 {
        // let rotation_matrix = self.r.to_mat4();
        // let mut matrix = rotation_matrix;
        // matrix.w_axis = glam::DVec4::new(self.t.x, self.t.y, self.t.z, 1.0);
        // matrix
        todo!()
    }

    pub fn adjoint() -> (Mat3A, Mat3A) {
        todo!("Python impl returns a 6x6 matrix")
    }

    pub fn exp(upsilon: Vec3A, omega: Vec3A) -> Self {
        todo!("Takes in vector of shape 6")
    }

    pub fn log(&self) -> (Vec3A, Vec3A) {
        todo!("Takes in vector of shape 6")
    }

    pub fn hat(upsilon: Vec3A, omega: Vec3A) -> Mat4 {
        todo!("Takes in vector of shape 6")
    }

    pub fn vee(omega: Mat4) -> ! {
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

