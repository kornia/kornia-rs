use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_image::ImageSize;
use kornia_imgproc::crop::crop_image;

#[pyfunction]
pub fn crop(
    py: Python<'_>,
    image: PyImage,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, ImageSize { width, height })? };
    py.detach(|| crop_image(&src, &mut dst, x, y))
        .map_err(to_pyerr)?;
    Ok(out)
}
