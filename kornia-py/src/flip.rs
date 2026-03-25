use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_imgproc::flip;

#[pyfunction]
pub fn horizontal_flip(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| flip::horizontal_flip(&src, &mut dst)).map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn vertical_flip(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| flip::vertical_flip(&src, &mut dst)).map_err(to_pyerr)?;
    Ok(out)
}
