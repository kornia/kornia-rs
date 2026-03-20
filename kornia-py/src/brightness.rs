use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::enhance;

#[pyfunction]
#[pyo3(name = "adjust_brightness")]
pub fn adjust_brightness_py(image: PyImage, factor: f32) -> PyResult<PyImage> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let image = image
        .cast::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    // Normalize to [0,1] range
    let mut normalized: Image<f32, 3, _> =
        Image::from_size_val(image.size(), 0.0f32, CpuAllocator)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    normalized
        .as_slice_mut()
        .iter_mut()
        .zip(image.as_slice().iter())
        .for_each(|(dst, &src)| {
            *dst = src / 255.0;
        });

    // Apply brightness adjustment (clamped to [0,1])
    let mut dst: Image<f32, 3, _> =
        Image::from_size_val(normalized.size(), 0.0f32, CpuAllocator)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    enhance::adjust_brightness(&normalized, &mut dst, factor, true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    // Scale back to [0,255]
    dst.as_slice_mut().iter_mut().for_each(|v| {
        *v *= 255.0;
    });

    let dst = dst
        .cast::<u8>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}
