use super::dlpack::{cvtensor_to_dlpack, cvtensor_to_dltensor};
use kornia_rs::tensor::Tensor;
use pyo3::prelude::*;

#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(shape: Vec<i64>, data: Vec<u8>) -> PyResult<PyTensor> {
        let inner = Tensor::new(shape, data);
        Ok(PyTensor { inner })
    }

    #[pyo3(name = "__dlpack__")]
    pub fn dlpack_py(&self, py: Python) -> PyResult<PyObject> {
        cvtensor_to_dlpack(&self.inner, py)
    }

    #[pyo3(name = "__dlpack_device__")]
    pub fn dlpack_device_py(&self) -> (i32, i32) {
        let dl_tensor = cvtensor_to_dltensor(&self.inner);
        (
            dl_tensor.device.device_type as i32,
            dl_tensor.device.device_id,
        )
    }

    #[getter]
    pub fn shape(&self) -> Vec<i64> {
        self.inner.shape.clone()
    }

    #[getter]
    pub fn data(&self) -> Vec<u8> {
        self.inner.data.clone()
    }

    #[getter]
    pub fn strides(&self) -> Vec<i64> {
        self.inner.strides.clone()
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "Tensor(shape: {:?}, strides: {:?})",
            self.inner.shape, self.inner.strides
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Tensor(shape: {:?}, strides: {:?})",
            self.inner.shape, self.inner.strides
        ))
    }
}

impl From<Tensor> for PyTensor {
    fn from(tensor: Tensor) -> Self {
        PyTensor { inner: tensor }
    }
}

impl From<PyTensor> for Tensor {
    fn from(py_tensor: PyTensor) -> Self {
        py_tensor.inner
    }
}
