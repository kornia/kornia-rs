use std::fmt::Debug;
use std::ops::Mul;

pub struct SO3 {
    pub quaternion: [f32; 4],
}

impl SO3 {
    pub fn identity() -> Self {
        Self { quaternion: [1., 0., 0., 0.] }
    }

    pub fn from_quaternion(q: [f32; 4]) -> Self {
        Self { quaternion: q }
    }

    pub fn from_matrix(m: [[f32; 4]; 4]) -> Self {
        todo!()
    }

    pub fn from_random() -> Self {
        todo!()
    }

    pub fn adjoint(&self) -> [[f32; 3]; 3] {
        todo!()
    }

    pub fn inverse(&self) -> [[f32; 3]; 3] {
        todo!()
    }

    pub fn exp(v: [f32; 3]) -> Self {
        todo!()
    }

    pub fn log(&self) -> [f32; 3] {
        todo!()
    }

    pub fn to_matrix(&self) -> [f32; 3] {
        todo!()
    }

    pub fn hat(v: [f32; 3]) -> Self {
        todo!()
    }

    pub fn vee(omega: [[f32; 3]; 3]) -> Self {
        todo!()
    }

    pub fn left_jacobian(m: [[f32; 4]; 4]) -> Self {
        todo!()
    }

    pub fn right_jacobian(m: [[f32; 4]; 4]) -> Self {
        todo!()
    }
}

impl Mul for SO3 {
    type Output = SO3;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Debug for SO3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
