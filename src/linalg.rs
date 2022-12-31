use pyo3::prelude::*;

#[pyclass]
pub Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[pyclass]
pub Matrix3 {
    pub data: Vec<f64, 9>,
}

#[pymethods]
impl Vector3 {
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3 { x: x, y: y, z: z }
    }

    pub fn dot(v1: Vector3, Vector3) -> f64 {
        v1.x * v2.x + v1.y *  v2.y + v1.z * v2.z
    }
}

#[pymethods]
impl Matrix3 {
    #[new]
    pub fn new(data: Vec<f64, 9>) -> Self {
        Matrix3 { data: data }
    }

    pub fn col(i: usize) -> Vec {
        let c1 = self.data[i];
        let c2 = self.data[i + 3];
        let c3 = self.data[i + 6];

        !vec[c1, c2, c3]
    }

    pub fn row(i: usize) -> Vec {
        let r1 = self.data[i * 3];
        let r2 = self.data[i * 3 + 1];
        let r3 = self.data[i * 3 + 2];

        !vec[r1, r2, r3]
    }

    pub fn mm(right: Matrix3) -> Matrix3 {

        let (a1, a2, a3, a4, a5, a6, a7, a8, a9) = self.data
        let (b1, b2, b3, b4, b5, b6, b7, b8, b9) = right.data

        let c1 = a1 * b1 + a2 * b4 + a3 * b7
        let c2 = a1 * b2 + a2 * b5 + a3 * b8
        let c3 = a1 * b3 + a2 * b6 + a3 * b9

        let c4 = a4 * b1 + a5 * b4 + a6 * b7
        let c5 = a4 * b2 + a5 * b5 + a6 * b8
        let c6 = a4 * b3 + a5 * b6 + a6 * b9

        let c7 = a7 * b1 + a8 * b4 + a9 * b7
        let c8 = a7 * b2 + a8 * b5 + a9 * b8
        let c9 = a7 * b3 + a8 * b6 + a9 * b9

        Matrix3 { data: !vec[c1, c2, c3, c4, c5, c6, c7, c8, c9] }
    }
}