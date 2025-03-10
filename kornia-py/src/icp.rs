use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use kornia_3d::pointcloud::PointCloud;
use kornia_icp::{icp_vanilla as icp_vanilla_fn, ICPConvergenceCriteria, ICPResult};

use crate::pointcloud::{FromPyPointCloud, PyPointCloud};

#[pyclass(name = "ICPConvergenceCriteria")]
#[derive(Clone)]
pub struct PyICPConvergenceCriteria(ICPConvergenceCriteria);

#[pymethods]
impl PyICPConvergenceCriteria {
    #[new]
    pub fn new(max_iterations: usize, tolerance: f64) -> PyResult<PyICPConvergenceCriteria> {
        Ok(PyICPConvergenceCriteria(ICPConvergenceCriteria {
            max_iterations,
            tolerance,
        }))
    }

    #[getter]
    pub fn max_iterations(&self) -> usize {
        self.0.max_iterations
    }

    #[getter]
    pub fn tolerance(&self) -> f64 {
        self.0.tolerance
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "ICPConvergenceCriteria(max_iterations: {}, tolerance: {})",
            self.0.max_iterations, self.0.tolerance
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ICPConvergenceCriteria(max_iterations: {}, tolerance: {})",
            self.0.max_iterations, self.0.tolerance
        ))
    }
}

#[pyclass(name = "ICPResult")]
pub struct PyICPResult(ICPResult);

#[pymethods]
impl PyICPResult {
    #[new]
    pub fn new() -> PyResult<PyICPResult> {
        Ok(PyICPResult(ICPResult {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            num_iterations: 0,
            rmse: 0.0,
        }))
    }

    #[getter]
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        self.0.rotation
    }

    #[getter]
    pub fn translation(&self) -> [f64; 3] {
        self.0.translation
    }

    #[getter]
    pub fn num_iterations(&self) -> usize {
        self.0.num_iterations
    }

    #[getter]
    pub fn rmse(&self) -> f64 {
        self.0.rmse
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "ICPResult(rotation: {:?}, translation: {:?}, num_iterations: {}, rmse: {})",
            self.0.rotation, self.0.translation, self.0.num_iterations, self.0.rmse
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ICPResult(rotation: {:?}, translation: {:?}, num_iterations: {}, rmse: {})",
            self.0.rotation, self.0.translation, self.0.num_iterations, self.0.rmse
        ))
    }
}

#[pyfunction]
pub fn icp_vanilla(
    source: PyPointCloud,
    target: PyPointCloud,
    initial_rot: Py<PyArray2<f64>>,
    initial_trans: Py<PyArray1<f64>>,
    criteria: PyICPConvergenceCriteria,
) -> PyResult<PyICPResult> {
    let source = PointCloud::from_pypointcloud(source)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let target = PointCloud::from_pypointcloud(target)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    // convert the initial rotation and translation to a vector
    let initial_rot = Python::with_gil(|py| {
        let array = initial_rot.bind(py);
        let data_slice =
            unsafe { array.as_slice() }.expect("Failed to convert initial rotation to vector");
        [
            [data_slice[0], data_slice[1], data_slice[2]],
            [data_slice[3], data_slice[4], data_slice[5]],
            [data_slice[6], data_slice[7], data_slice[8]],
        ]
    });

    let initial_trans = Python::with_gil(|py| {
        let array = initial_trans.bind(py);
        let data_slice =
            unsafe { array.as_slice() }.expect("Failed to convert initial translation to vector");
        [data_slice[0], data_slice[1], data_slice[2]]
    });

    // run the icp algorithm

    let result = icp_vanilla_fn(&source, &target, initial_rot, initial_trans, criteria.0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    Ok(PyICPResult(result))
}
