use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_imgproc as imgproc;

#[pyfunction]
pub fn compute_histogram(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    num_bins: usize,
) -> PyResult<Vec<usize>> {
    let mut histogram = vec![0; num_bins];

    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            crate::cuda_ext::histogram::compute_histogram(py, &img, &mut histogram, num_bins)?;
            return Ok(histogram);
        }
    }

    let arr: Py<numpy::PyArray3<u8>> = if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        crate::dispatch::no_gpu_kernel_if_device(&api.borrow())?;
        let view = api.call_method0("numpy")?;
        view.extract()?
    } else {
        image.extract()?
    };
    let img = unsafe { numpy_as_image::<1>(py, &arr)? };
    py.detach(|| imgproc::histogram::compute_histogram(&img, &mut histogram, num_bins))
        .map_err(to_pyerr)?;
    Ok(histogram)
}

/// Histogram-equalize an 8-bit single-channel image — byte-for-byte with
/// `cv2.equalizeHist`.
///
/// Residency-dispatched: a u8 device `Image` runs the CUDA
/// histogram → LUT → apply chain (byte-identical to the CPU path); a numpy
/// u8 array of shape (H, W, 1) runs the CPU path.
#[pyfunction]
pub fn equalize_hist(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::histogram::equalize_hist(py, &img)?.into_py(py);
        }
    }

    crate::dispatch::cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
        py.detach(|| imgproc::histogram::equalize_hist(&src, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

/// Contrast-Limited Adaptive Histogram Equalization for 8-bit
/// single-channel images — byte-for-byte with
/// `cv2.createCLAHE(clip_limit, grid_size).apply(src)`.
///
/// `grid_size` is `(tiles_x, tiles_y)`; `clip_limit <= 0` disables clipping.
/// Residency-dispatched: a u8 device `Image` runs the CUDA kernels
/// (byte-identical to the CPU path); a numpy u8 array of shape (H, W, 1)
/// runs the CPU path.
#[pyfunction]
#[pyo3(signature = (image, clip_limit=40.0, grid_size=(8, 8), out=None))]
pub fn clahe(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    clip_limit: f64,
    grid_size: (usize, usize),
    out: Option<PyImage>,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            if out.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "clahe: out= is only supported on the CPU (numpy) path",
                ));
            }
            return crate::cuda_ext::clahe_dev::clahe(py, &img, clip_limit, grid_size)?.into_py(py);
        }
    }

    crate::dispatch::cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        use numpy::PyUntypedArrayMethods;
        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
        let (mut dst, out_arr) = match out {
            Some(out_pyarr) => {
                let shape: Vec<usize> = out_pyarr.bind(py).shape().to_vec();
                if shape != [src.rows(), src.cols(), 1] {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "clahe: out shape {:?} must match the source ({}, {}, 1)",
                        shape,
                        src.rows(),
                        src.cols()
                    )));
                }
                let img = unsafe { numpy_as_image::<1>(py, &out_pyarr)? };
                (img, out_pyarr)
            }
            None => unsafe { alloc_output_pyarray::<1>(py, src.size())? },
        };
        py.detach(|| imgproc::clahe::clahe(&src, &mut dst, clip_limit, grid_size))
            .map_err(to_pyerr)?;
        Ok(out_arr)
    })
}
