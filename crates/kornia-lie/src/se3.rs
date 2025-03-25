use glam::{DMat4, DQuat, DVec3};

use crate::so3::SO3;

#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    pub r: SO3,
    pub t: DVec3,
}

impl SE3 {
    pub const IDENTITY: Self = Self { r: SO3::IDENTITY, t: DVec3::from_array([0.0, 0.0, 0.0]) };

    pub fn new(rotation: SO3, translation: DVec3) -> Self {
        Self { r: rotation, t: translation }
    }

    pub fn from_matrix(m: DMat4) -> Self {

        todo!()
    }

    pub fn from_random() -> Self {

        todo!()
    }

    pub fn inverse(&self) -> Self {
        // let inv_rot = self.r.inverse();
        // let inv_trans = -(inv_rot * self.t);
        // Self { r: inv_rot, t: inv_trans }
        
        todo!()
    }

    pub fn as_matrix(&self) -> DMat4 {
        // let rotation_matrix = self.r.to_mat4();
        // let mut matrix = rotation_matrix;
        // matrix.w_axis = glam::DVec4::new(self.t.x, self.t.y, self.t.z, 1.0);
        // matrix
        todo!()
    }

    pub fn adjoint() -> ! {
        todo!("Python impl returns a 6x6 matrix")
    }

    pub fn exp() -> ! {
        todo!("Takes in vector of shape 6")
    }

    pub fn log() -> ! {
        todo!("Takes in vector of shape 6")
    }

    pub fn hat() -> ! {
        todo!("Takes in vector of shape 6")
    }

    pub fn vee() -> ! {
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

