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

use crate::dlpack_py::cvtensor_to_dlpack;
use crate::io::read_image_rs;

#[cfg(feature = "libjpeg-turbo")]
use crate::io::read_image_jpeg;

#[cfg(feature = "viz")]
pub mod viz;
#[cfg(feature = "viz")]
use crate::viz::show_image_from_file;
#[cfg(feature = "viz")]
use crate::viz::show_image_from_tensor;

use pyo3::prelude::*;

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_image_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cvtensor_to_dlpack, m)?)?;
    m.add_class::<tensor::cv::Tensor>()?;

    #[cfg(feature = "libjpeg-turbo")]
    m.add_function(wrap_pyfunction!(read_image_jpeg, m)?)?;

    #[cfg(feature = "viz")]
    {
        m.add_function(wrap_pyfunction!(show_image_from_file, m)?)?;
        m.add_function(wrap_pyfunction!(show_image_from_tensor, m)?)?;
    }

    Ok(())
}
