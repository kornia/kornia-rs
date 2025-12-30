use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use kornia_3d::pointcloud::PointCloud;
use kornia_icp::{icp_vanilla as icp_vanilla_fn, ICPConvergenceCriteria, ICPResult};

use crate::pointcloud::{FromPyPointCloud, PyPointCloud};

#[pyclass(name = "ICPConvergenceCriteria", frozen)]
#[derive(Clone)]
pub struct PyICPConvergenceCriteria(pub ICPConvergenceCriteria);

#[pymethods]
impl PyICPConvergenceCriteria {
    #[new]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self(ICPConvergenceCriteria {
            max_iterations,
            tolerance,
        })
    }

    #[getter]
    pub fn max_iterations(&self) -> usize {
        self.0.max_iterations
    }

    #[getter]
    pub fn tolerance(&self) -> f64 {
        self.0.tolerance
    }

    fn __repr__(&self) -> String {
        format!(
            "ICPConvergenceCriteria(max_iterations={}, tolerance={})",
            self.0.max_iterations, self.0.tolerance
        )
    }
}

#[pyclass(name = "ICPResult", frozen)]
pub struct PyICPResult(pub ICPResult);

#[pymethods]
impl PyICPResult {
    #[new]
    pub fn new() -> Self {
        Self(ICPResult {
            rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            translation: [0.0, 0.0, 0.0],
            num_iterations: 0,
            rmse: 0.0,
        })
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

    fn __repr__(&self) -> String {
        format!(
            "ICPResult(num_iterations={}, rmse={:.6}, rotation={:?}, translation={:?})",
            self.0.num_iterations,
            self.0.rmse,
            self.0.rotation,
            self.0.translation
        )
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
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;

    let target = PointCloud::from_pypointcloud(target)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;

    Python::attach(|py| {
        let array_rot = initial_rot.bind(py);
        let array_trans = initial_trans.bind(py);

        let data_rot = unsafe { array_rot.as_slice() }
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;

        let data_trans = unsafe { array_trans.as_slice() }
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;

        let rot = [
            [data_rot[0], data_rot[1], data_rot[2]],
            [data_rot[3], data_rot[4], data_rot[5]],
            [data_rot[6], data_rot[7], data_rot[8]],
        ];

        let trans = [data_trans[0], data_trans[1], data_trans[2]];

        let result = icp_vanilla_fn(&source, &target, rot, trans, criteria.0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;

        Ok(PyICPResult(result))
    })
}