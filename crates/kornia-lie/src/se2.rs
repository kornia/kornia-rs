use crate::so2::SO2;

#[derive(Debug, Clone, Copy)]
pub struct SE2 {
    rotation: SO2,
    translation: DVec2,
}

impl SE2 {
    pub fn new(rotation: SO2, translation: DVec2) -> Self {
        Self { rotation, translation }
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.inverse();
        let inv_trans = -(inv_rot.rotation * self.translation);
        Self { rotation: inv_rot, translation: inv_trans }
    }

    pub fn compose(&self, other: &SE2) -> Self {
        let new_rotation = self.rotation.compose(&other.rotation);
        let new_translation = self.translation + (self.rotation.rotation * other.translation);
        Self { rotation: new_rotation, translation: new_translation }
    }

    pub fn as_matrix(&self) -> DMat3 {
        let r = self.rotation.as_matrix();
        DMat3::from_cols_array(&[
            r.x_axis.x, r.y_axis.x, self.translation.x,
            r.x_axis.y, r.y_axis.y, self.translation.y,
            0.0, 0.0, 1.0,
        ])
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
    }
}
