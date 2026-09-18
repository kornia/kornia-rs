use kornia_algebra::{Vec3F32, Vec3F64};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python binding for `kornia_algebra::Vec3F32`.
#[pyclass(name = "Vec3F32", module = "kornia_rs", frozen)]
pub struct PyVec3F32(Vec3F32);

#[pymethods]
impl PyVec3F32 {
    #[new]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3F32::new(x, y, z))
    }

    #[staticmethod]
    fn from_numpy(values: PyReadonlyArray1<'_, f32>) -> PyResult<Self> {
        let values = values.as_array();

        if values.len() != 3 {
            return Err(PyValueError::new_err("expected an array with shape (3,)"));
        }

        Ok(Self(Vec3F32::new(values[0], values[1], values[2])))
    }

    #[getter]
    fn x(&self) -> f32 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f32 {
        self.0.y
    }

    #[getter]
    fn z(&self) -> f32 {
        self.0.z
    }

    fn as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        [self.0.x, self.0.y, self.0.z].to_pyarray(py)
    }
}

/// Python binding for `kornia_algebra::Vec3F64`.
#[pyclass(name = "Vec3F64", module = "kornia_rs", frozen)]
pub struct PyVec3F64(Vec3F64);

#[pymethods]
impl PyVec3F64 {
    #[new]
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self(Vec3F64::new(x, y, z))
    }

    #[staticmethod]
    fn from_numpy(values: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let values = values.as_array();

        if values.len() != 3 {
            return Err(PyValueError::new_err("expected an array with shape (3,)"));
        }

        Ok(Self(Vec3F64::new(values[0], values[1], values[2])))
    }

    #[getter]
    fn x(&self) -> f64 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.0.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.0.z
    }

    fn as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        [self.0.x, self.0.y, self.0.z].to_pyarray(py)
    }
}
