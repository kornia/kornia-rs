mod image;
mod io;
mod resize;

use crate::image::PyImageSize;
use crate::io::functional::{read_image_any, read_image_jpeg, write_image_jpeg};
use crate::io::jpeg::{PyImageDecoder, PyImageEncoder};
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
    m.add_function(wrap_pyfunction!(read_image_any, m)?)?;
    m.add_function(wrap_pyfunction!(resize::resize, m)?)?;
    m.add_class::<PyImageSize>()?;
    m.add_class::<PyImageDecoder>()?;
    m.add_class::<PyImageEncoder>()?;
    Ok(())
}
