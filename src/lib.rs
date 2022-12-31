// TODO(carlos): enable
//#![feature(test)]
//extern crate test;
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub mod dlpack_py;
pub mod io;
#[allow(dead_code)]
pub mod tensor;

use crate::io::read_image_jpeg;
use crate::io::read_image_rs;
use crate::io::write_image_jpeg;
use crate::io::ImageDecoder;
use crate::io::ImageEncoder;
use crate::io::ImageSize;

#[cfg(feature = "viz")]
pub mod viz;
#[cfg(feature = "viz")]
use crate::viz::show_image_from_file;
#[cfg(feature = "viz")]
use crate::viz::show_image_from_tensor;

use pyo3::prelude::*;

pub fn get_version() -> String {
    let version = env!("CARGO_PKG_VERSION").to_string();
    // cargo uses "1.0-alpha1" etc. while python uses "1.0.0a1", this is not full compatibility,
    // but it's good enough for now
    // see https://docs.rs/semver/1.0.9/semver/struct.Version.html#method.parse for rust spec
    // see https://peps.python.org/pep-0440/ for python spec
    // it seems the dot after "alpha/beta" e.g. "-alpha.1" is not necessary, hence why this works
    version.replace("-alpha", "a").replace("-beta", "b")
}

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", get_version())?;
    m.add_function(wrap_pyfunction!(read_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(write_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_rs, m)?)?;
    m.add_class::<tensor::cv::Tensor>()?;
    m.add_class::<ImageSize>()?;
    m.add_class::<ImageDecoder>()?;
    m.add_class::<ImageEncoder>()?;

    #[cfg(feature = "viz")]
    {
        m.add_function(wrap_pyfunction!(show_image_from_file, m)?)?;
        m.add_function(wrap_pyfunction!(show_image_from_tensor, m)?)?;
    }

    Ok(())
}
