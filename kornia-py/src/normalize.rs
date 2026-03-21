use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, PyImageF32, ToPyImageF32};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::normalize;

#[pyfunction]
pub fn normalize_mean_std(image: PyImage, mean: [f32; 3], std: [f32; 3]) -> PyResult<PyImageF32> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let image = image
        .cast::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut dst: Image<f32, 3, _> = Image::from_size_val(image.size(), 0.0f32, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    normalize::normalize_mean_std(&image, &mut dst, &mean, &std)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage_f32().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}
