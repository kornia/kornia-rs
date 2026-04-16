use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, parse_interpolation, to_pyerr, PyImage};
use kornia_image::ImageSize;
use kornia_imgproc::resize::resize_fast_rgb_aa;

#[pyfunction]
#[pyo3(signature = (image, new_size, interpolation, antialias=true))]
pub fn resize(
    py: Python<'_>,
    image: PyImage,
    new_size: (usize, usize),
    interpolation: &str,
    antialias: bool,
) -> PyResult<PyImage> {
    let interpolation = parse_interpolation(interpolation)?;
    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, new_size)? };
    py.detach(|| resize_fast_rgb_aa(&src, &mut dst, interpolation, antialias))
        .map_err(to_pyerr)?;
    Ok(out)
}
