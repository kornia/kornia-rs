use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use kornia_3d::pointcloud::PointCloud;
use kornia_3d::registration::{icp_vanilla as icp_vanilla_fn, ICPConvergenceCriteria, ICPResult};

use crate::pointcloud::{FromPyPointCloud, PyPointCloud};

#[pyclass(name = "ICPConvergenceCriteria", frozen, from_py_object)]
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

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ICPConvergenceCriteria(max_iterations: {}, tolerance: {})",
            self.0.max_iterations, self.0.tolerance
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

#[pyclass(name = "ICPResult", frozen)]
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

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ICPResult(rotation: {:?}, translation: {:?}, num_iterations: {}, rmse: {})",
            self.0.rotation, self.0.translation, self.0.num_iterations, self.0.rmse
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
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
    let initial_rot = Python::attach(|py| -> PyResult<[[f64; 3]; 3]> {
        let array = initial_rot.bind(py);
        let d = crate::pyutils::c_slice(array, "initial_rot")?;
        if d.len() != 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_rot must be a (3, 3) float64 array",
            ));
        }
        Ok([[d[0], d[1], d[2]], [d[3], d[4], d[5]], [d[6], d[7], d[8]]])
    })?;

    let initial_trans = Python::attach(|py| -> PyResult<[f64; 3]> {
        let array = initial_trans.bind(py);
        let d = crate::pyutils::c_slice(array, "initial_trans")?;
        if d.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_trans must be a (3,) float64 array",
            ));
        }
        Ok([d[0], d[1], d[2]])
    })?;

    let result = icp_vanilla_fn(&source, &target, initial_rot, initial_trans, criteria.0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    Ok(PyICPResult(result))
}
