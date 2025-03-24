use glam::{DMat3, DVec2};

use crate::so2::SO2;

#[derive(Debug, Clone, Copy)]
pub struct SE2 {
    pub r: SO2,
    pub t: DVec2,
}

impl SE2 {
    pub fn new(r: SO2, t: DVec2) -> Self {
        Self { r, t }
    }

    pub fn from_matrix() -> Self {

    }

    pub fn from_random() -> Self {

    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.r.inverse();
        let inv_trans = -(inv_rot.rotation * self.t);
        Self { r: inv_rot, t: inv_trans }
    }

    pub fn compose(&self, other: &SE2) -> Self {
        let new_rotation = self.r.compose(&other.r);
        let new_translation = self.t + (self.r.rotation * other.t);
        Self { r: new_rotation, t: new_translation }
    }

    pub fn as_matrix(&self) -> DMat3 {
        let r = self.r.as_matrix();
        DMat3::from_cols_array(&[
            r.x_axis.x, r.y_axis.x, self.t.x,
            r.x_axis.y, r.y_axis.y, self.t.y,
            0.0, 0.0, 1.0,
        ])
    }

    pub fn adjoint(&self) -> DMat2 {
        
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
