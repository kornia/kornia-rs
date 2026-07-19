use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::dispatch::cpu_op;
use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, numpy_as_image, numpy_as_image_f32, to_pyerr,
};
use kornia_imgproc::filter;

/// Gaussian blur.
///
/// Residency-dispatched: a device `Image` (u8 1/3/4-channel or f32
/// 1/3-channel) runs the CUDA separable kernels — byte-exact (u8) /
/// bit-exact (f32) with the CPU paths — and a numpy u8 array runs the CPU
/// path.
#[pyfunction]
pub fn gaussian_blur(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::filter::gaussian_blur(&img, kernel_size, sigma)?.into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<3>(py, &arr)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| filter::gaussian_blur_u8(&src, &mut dst, kernel_size, sigma))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

/// Box blur.
///
/// Residency-dispatched like [`gaussian_blur`].
#[pyfunction]
pub fn box_blur(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    kernel_size: (usize, usize),
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::filter::box_blur(&img, kernel_size)?.into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<3>(py, &arr)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| filter::box_blur_u8(&src, &mut dst, kernel_size))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

/// Sobel gradient magnitude (f32).
///
/// Residency-dispatched: an f32 device `Image` (1/3-channel) runs the CUDA
/// separable kernels + magnitude fold — bit-exact with the CPU path — and a
/// numpy f32 array runs the CPU path.
#[pyfunction]
#[pyo3(signature = (image, kernel_size=3))]
pub fn sobel(py: Python<'_>, image: &Bound<'_, PyAny>, kernel_size: usize) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::filter::sobel(&img, kernel_size)?.into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<f32>>| {
        let c = arr.bind(py).shape()[2];
        match c {
            1 => {
                let src = unsafe { numpy_as_image_f32::<1>(py, &arr)? };
                let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<1>(py, src.size())? };
                py.detach(|| filter::sobel(&src, &mut dst, kernel_size))
                    .map_err(to_pyerr)?;
                Ok(out)
            }
            3 => {
                let src = unsafe { numpy_as_image_f32::<3>(py, &arr)? };
                let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, src.size())? };
                py.detach(|| filter::sobel(&src, &mut dst, kernel_size))
                    .map_err(to_pyerr)?;
                Ok(out)
            }
            c => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sobel supports 1 or 3 channels; got {c}"
            ))),
        }
    })
}

/// Median blur — byte-for-byte with `cv2.medianBlur` and VPI's CUDA
/// `MedianFilter`. `kernel_size` must be 3 or 5; borders replicate.
///
/// Residency-dispatched: a u8 device `Image` (1/3/4-channel) runs the CUDA
/// sorting-network kernel (byte-identical to the CPU path); a numpy u8
/// array of shape (H, W, 1|3) runs the CPU path.
#[pyfunction]
#[pyo3(signature = (image, kernel_size=3))]
pub fn median_blur(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    kernel_size: usize,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::filter::median_blur(&img, kernel_size)?.into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let c = arr.bind(py).shape()[2];
        match c {
            1 => {
                let src = unsafe { numpy_as_image::<1>(py, &arr)? };
                let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
                py.detach(|| filter::median_blur(&src, &mut dst, kernel_size))
                    .map_err(to_pyerr)?;
                Ok(out)
            }
            3 => {
                let src = unsafe { numpy_as_image::<3>(py, &arr)? };
                let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
                py.detach(|| filter::median_blur(&src, &mut dst, kernel_size))
                    .map_err(to_pyerr)?;
                Ok(out)
            }
            c => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "median_blur supports 1 or 3 channels; got {c}"
            ))),
        }
    })
}

/// Bilateral filter — byte-for-byte with
/// `cv2.bilateralFilter(src, d, sigma_color, sigma_space)` for u8
/// single-channel images (VPI's bilateral uses a different formula and is
/// NOT byte-comparable).
///
/// Residency-dispatched: a u8 single-channel device `Image` runs the CUDA
/// kernel (byte-identical to the CPU path); a numpy u8 array of shape
/// (H, W, 1) runs the CPU path.
#[pyfunction]
#[pyo3(signature = (image, d=5, sigma_color=50.0, sigma_space=50.0))]
pub fn bilateral_filter(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::filter::bilateral_filter(&img, d, sigma_color, sigma_space)?
                .into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
        py.detach(|| filter::bilateral_filter(&src, &mut dst, d, sigma_color, sigma_space))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}
