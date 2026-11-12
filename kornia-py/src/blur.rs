use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::dispatch::cpu_op;
use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_t, numpy_as_image, numpy_as_image_t,
    numpy_as_out_image, to_pyerr, UNINIT,
};
use kornia_imgproc::filter;

/// Gaussian blur.
///
/// Residency-dispatched: a device `Image` (u8 1/3/4-channel or f32
/// 1/3-channel) runs the CUDA separable kernels — byte-exact (u8) /
/// bit-exact (f32) with the CPU paths — and a numpy u8 array of shape
/// (H, W, 1|3|4) runs the CPU path, matching what the device path accepts.
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
        fn run<const C: usize>(
            py: Python<'_>,
            arr: &Py<numpy::PyArray3<u8>>,
            kernel_size: (usize, usize),
            sigma: (f32, f32),
        ) -> PyResult<crate::image::PyImage> {
            let src = unsafe { numpy_as_image::<C>(py, arr)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray::<C>(py, src.size())? };
            py.detach(|| filter::gaussian_blur_u8(&src, &mut dst, kernel_size, sigma))
                .map_err(to_pyerr)?;
            Ok(out)
        }
        let c = arr.bind(py).shape()[2];
        match c {
            1 => run::<1>(py, &arr, kernel_size, sigma),
            3 => run::<3>(py, &arr, kernel_size, sigma),
            4 => run::<4>(py, &arr, kernel_size, sigma),
            c => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "gaussian_blur: host path supports 1, 3, or 4 channels (u8); got {c}"
            ))),
        }
    })
}

/// Box blur.
///
/// Residency-dispatched like [`gaussian_blur`]; the host path accepts the
/// same 1/3/4-channel u8 shapes as the device path.
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
        fn run<const C: usize>(
            py: Python<'_>,
            arr: &Py<numpy::PyArray3<u8>>,
            kernel_size: (usize, usize),
        ) -> PyResult<crate::image::PyImage> {
            let src = unsafe { numpy_as_image::<C>(py, arr)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray::<C>(py, src.size())? };
            py.detach(|| filter::box_blur_u8(&src, &mut dst, kernel_size))
                .map_err(to_pyerr)?;
            Ok(out)
        }
        let c = arr.bind(py).shape()[2];
        match c {
            1 => run::<1>(py, &arr, kernel_size),
            3 => run::<3>(py, &arr, kernel_size),
            4 => run::<4>(py, &arr, kernel_size),
            c => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "box_blur: host path supports 1, 3, or 4 channels (u8); got {c}"
            ))),
        }
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
                // SAFETY: the view borrows the numpy array, which the caller keeps alive and does
                // not mutate for the duration of this call.
                let src = unsafe { numpy_as_image_t::<f32, 1>(py, &arr)? };
                // SAFETY: `dst` aliases the fresh array `out`, which stays alive and is only
                // handed to Python after the kernel has written every element through `dst`.
                let (mut dst, out) =
                    unsafe { alloc_output_pyarray_t::<f32, 1, UNINIT>(py, src.size())? };
                py.detach(|| filter::sobel(&src, &mut dst, kernel_size))
                    .map_err(to_pyerr)?;
                Ok(out)
            }
            3 => {
                // SAFETY: the view borrows the numpy array, which the caller keeps alive and does
                // not mutate for the duration of this call.
                let src = unsafe { numpy_as_image_t::<f32, 3>(py, &arr)? };
                // SAFETY: `dst` aliases the fresh array `out`, which stays alive and is only
                // handed to Python after the kernel has written every element through `dst`.
                let (mut dst, out) =
                    unsafe { alloc_output_pyarray_t::<f32, 3, UNINIT>(py, src.size())? };
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

/// Resolve `out=` (validated: shape, writeable, contiguous, and not aliasing
/// `src`) or allocate a fresh output array. Shared by the median/bilateral CPU
/// paths.
fn resolve_out<const C: usize>(
    py: Python<'_>,
    op: &str,
    src: &kornia_image::Image<u8, C>,
    out: Option<crate::image::PyImage>,
) -> PyResult<(kornia_image::Image<u8, C>, crate::image::PyImage)> {
    let (rows, cols) = (src.rows(), src.cols());
    match out {
        Some(out_pyarr) => {
            // SAFETY: `out_pyarr` is returned alongside the Image and kept alive
            // by the caller for as long as the Image is used.
            let img = unsafe {
                numpy_as_out_image::<C>(py, op, &out_pyarr, [rows, cols, C], src.as_slice())?
            };
            Ok((img, out_pyarr))
        }
        None => unsafe {
            alloc_output_pyarray::<C>(
                py,
                kornia_image::ImageSize {
                    width: cols,
                    height: rows,
                },
            )
        },
    }
}

/// Median blur — byte-for-byte with `cv2.medianBlur` and VPI's CUDA
/// `MedianFilter`. `kernel_size` must be 3 or 5; borders replicate.
///
/// Residency-dispatched: a u8 device `Image` (1/3/4-channel) runs the CUDA
/// sorting-network kernel (byte-identical to the CPU path); a numpy u8
/// array of shape (H, W, 1|3) runs the CPU path.
#[pyfunction]
#[pyo3(signature = (image, kernel_size=3, out=None))]
pub fn median_blur(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    kernel_size: usize,
    out: Option<crate::image::PyImage>,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            if out.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "median_blur: out= is only supported on the CPU (numpy) path",
                ));
            }
            return crate::cuda_ext::filter::median_blur(&img, kernel_size)?.into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        fn run<const C: usize>(
            py: Python<'_>,
            arr: &Py<numpy::PyArray3<u8>>,
            kernel_size: usize,
            out: Option<crate::image::PyImage>,
        ) -> PyResult<crate::image::PyImage> {
            let src = unsafe { numpy_as_image::<C>(py, arr)? };
            let (mut dst, out_arr) = resolve_out::<C>(py, "median_blur", &src, out)?;
            py.detach(|| filter::median_blur(&src, &mut dst, kernel_size))
                .map_err(to_pyerr)?;
            Ok(out_arr)
        }
        let c = arr.bind(py).shape()[2];
        match c {
            1 => run::<1>(py, &arr, kernel_size, out),
            3 => run::<3>(py, &arr, kernel_size, out),
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
#[pyo3(signature = (image, d=5, sigma_color=50.0, sigma_space=50.0, out=None))]
pub fn bilateral_filter(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    out: Option<crate::image::PyImage>,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            if out.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "bilateral_filter: out= is only supported on the CPU (numpy) path",
                ));
            }
            return crate::cuda_ext::filter::bilateral_filter(&img, d, sigma_color, sigma_space)?
                .into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
        let (mut dst, out_arr) = resolve_out::<1>(py, "bilateral_filter", &src, out)?;
        py.detach(|| filter::bilateral_filter(&src, &mut dst, d, sigma_color, sigma_space))
            .map_err(to_pyerr)?;
        Ok(out_arr)
    })
}
