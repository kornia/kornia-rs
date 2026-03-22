use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::crop::crop_image;

#[pyfunction]
pub fn crop(image: PyImage, x: usize, y: usize, width: usize, height: usize) -> PyResult<PyImage> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let crop_size = ImageSize { width, height };

    let mut dst = Image::from_size_val(crop_size, 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    crop_image(&image, &mut dst, x, y)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}
