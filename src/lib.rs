// TODO(carlos): enable
//#![feature(test)]
//extern crate test;
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub mod dlpack;
pub mod dlpack_py;
pub mod io;
#[allow(dead_code)]
pub mod tensor;
pub mod viz;

use crate::dlpack_py::cvtensor_to_dlpack;
use crate::io::read_image_jpeg;
use crate::io::read_image_rs;
use crate::viz::show_image_from_file;
use crate::viz::show_image_from_tensor;

use pyo3::prelude::*;

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_image_rs, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(cvtensor_to_dlpack, m)?)?;
    m.add_class::<tensor::cv::Tensor>()?;
    // TODO(edgar): support later
    // m.add_class::<viz::VizManager>()?;
    Ok(())
}
