use glam::{DMat4, DQuat, DVec3};

use crate::so3::SO3;

#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    rotation: SO3,
    translation: DVec3,
}

impl SE3 {
    pub fn new(rotation: SO3, translation: DVec3) -> Self {
        Self { rotation, translation }
    }

    pub fn from_axis_angle(axis: DVec3, angle: f64, translation: DVec3) -> Self {
        let rotation = DQuat::from_axis_angle(axis.normalize(), angle);
        Self { rotation, translation }
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.inverse();
        let inv_trans = -(inv_rot * self.translation);
        Self { rotation: inv_rot, translation: inv_trans }
    }

    pub fn compose(&self, other: &SE3) -> Self {
        let new_rotation = self.rotation * other.rotation;
        let new_translation = self.translation + (self.rotation * other.translation);
        Self { rotation: new_rotation, translation: new_translation }
    }

    pub fn as_matrix(&self) -> DMat4 {
        let rotation_matrix = self.rotation.to_mat4();
        let mut matrix = rotation_matrix;
        matrix.w_axis = glam::DVec4::new(self.translation.x, self.translation.y, self.translation.z, 1.0);
        matrix
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
    }
}

