use pyo3::prelude::*;

use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, numpy_as_image, numpy_as_image_f32, to_pyerr,
    PyImage, PyImageF32,
};
use kornia_imgproc::color;

#[pyfunction]
pub fn rgb_from_gray(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<1>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_gray(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn bgr_from_rgb(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::bgr_from_rgb(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

/// RGB f32 → grayscale f32, zero-copy in and out.
///
/// Accepts a (H, W, 3) numpy float32 array; returns a (H, W, 1) numpy float32 array.
/// GIL is released for the NEON/AVX2/scalar kernel invocation.
#[pyfunction]
pub fn gray_from_rgb_f32(py: Python<'_>, image: PyImageF32) -> PyResult<PyImageF32> {
    let src = unsafe { numpy_as_image_f32::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<1>(py, src.size())? };
    py.detach(|| color::gray_from_rgb_f32(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn gray_from_rgb(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
    py.detach(|| color::gray_from_rgb_u8(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_rgba(
    py: Python<'_>,
    image: PyImage,
    background: Option<[u8; 3]>,
) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<4>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_rgba(&src, &mut dst, background))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn apply_colormap(py: Python<'_>, image: PyImage, colormap: &str) -> PyResult<PyImage> {
    // Validate name before touching the image array — fail fast on bad input.
    let cm = color::ColormapType::from_name(colormap).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown colormap '{colormap}'; valid names: autumn, bone, jet, winter, rainbow, \
             ocean, summer, spring, cool, hsv, pink, hot, parula, magma, inferno, plasma, \
             viridis, cividis, twilight, turbo, deepgreen"
        ))
    })?;
    let src = unsafe { numpy_as_image::<1>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::apply_colormap(&src, &mut dst, cm))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_bgra(
    py: Python<'_>,
    image: PyImage,
    background: Option<[u8; 3]>,
) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<4>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_bgra(&src, &mut dst, background))
        .map_err(to_pyerr)?;
    Ok(out)
}
