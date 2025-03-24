use glam::DMat2;

#[derive(Debug, Clone, Copy)]
pub struct SO2 {
    rotation: DMat2,
}

impl SO2 {
    pub fn from_angle(theta: f64) -> Self {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let rotation = DMat2::from_cols_array(&[cos_theta, -sin_theta, sin_theta, cos_theta]);
        Self { rotation }
    }

    pub fn angle(&self) -> f64 {
        self.rotation.x_axis.x.acos()
    }

    pub fn inverse(&self) -> Self {
        Self { rotation: self.rotation.transpose() }
    }

    pub fn compose(&self, other: &SO2) -> Self {
        Self { rotation: self.rotation * other.rotation }
    }

    pub fn as_matrix(&self) -> DMat2 {
        self.rotation
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
    }
}

