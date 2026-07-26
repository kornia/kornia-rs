use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::dispatch::cpu_op;
use crate::image::{alloc_output_pyarray, numpy_as_image, parse_interpolation, to_pyerr, PyImage};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::warp;

fn warp_dispatch<const C: usize, F>(
    py: Python<'_>,
    image: PyImage,
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<PyImage>,
    op: F,
) -> PyResult<PyImage>
where
    F: Send + FnOnce(&Image<u8, C>, &mut Image<u8, C>) -> Result<(), ImageError>,
{
    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    let _ = parse_interpolation(interpolation)?;
    let src_u8 = unsafe { numpy_as_image::<C>(py, &image)? };
    let (mut dst_u8, out_arr) = match out {
        Some(out_pyarr) => {
            let shape = out_pyarr.bind(py).shape();
            if shape[0] != new_size.height || shape[1] != new_size.width || shape[2] != C {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "out shape ({}, {}, {}) must match new_size ({}, {}, {})",
                    shape[0], shape[1], shape[2], new_size.height, new_size.width, C
                )));
            }
            let img = unsafe { numpy_as_image::<C>(py, &out_pyarr)? };
            (img, out_pyarr)
        }
        None => unsafe { alloc_output_pyarray::<C>(py, new_size)? },
    };

    py.detach(|| op(&src_u8, &mut dst_u8)).map_err(to_pyerr)?;

    Ok(out_arr)
}

fn src_channels(py: Python<'_>, image: &PyImage) -> PyResult<usize> {
    Ok(image.bind(py).shape()[2])
}

fn unsupported_channels(c: usize) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "warp supports 1, 3, or 4 channels; got {}",
        c
    ))
}

/// Warp an image with a 2×3 affine transform.
///
/// Residency-dispatched: a device `Image` (f32, 3-channel) runs the CUDA
/// kernels, bit-identical to the CPU f32 path (`out=` is not supported on the
/// device path); a host `Image` or numpy u8 array runs the CPU fast path.
#[pyfunction]
#[pyo3(signature = (image, m, new_size, interpolation, out=None))]
pub fn warp_affine(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    m: [f32; 6],
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            let dev_out = match out {
                Some(o) => Some(o.extract::<Py<PyAny>>(py)?),
                None => None,
            };
            return crate::cuda_ext::geometry::warp_affine(
                py,
                &img,
                m,
                new_size,
                interpolation,
                dev_out,
            )?
            .into_py(py);
        }
    }
    let np_out: Option<PyImage> = match out {
        Some(o) => Some(o.extract::<PyImage>(py).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "warp_affine: out= must be a numpy u8 array on the CPU path",
            )
        })?),
        None => None,
    };
    let interp = interpolation.to_string();
    cpu_op(
        py,
        image,
        move |py, arr: Py<numpy::PyArray3<u8>>| match src_channels(py, &arr)? {
            1 => warp_dispatch::<1, _>(py, arr, new_size, &interp, np_out, |s, d| {
                warp::warp_affine_u8(s, d, &m)
            }),
            3 => warp_dispatch::<3, _>(py, arr, new_size, &interp, np_out, |s, d| {
                warp::warp_affine_u8(s, d, &m)
            }),
            4 => warp_dispatch::<4, _>(py, arr, new_size, &interp, np_out, |s, d| {
                warp::warp_affine_u8(s, d, &m)
            }),
            c => Err(unsupported_channels(c)),
        },
    )
}

/// Warp an image with a 3×3 homography.
///
/// Residency-dispatched — see [`warp_affine`].
#[pyfunction]
#[pyo3(signature = (image, m, new_size, interpolation, out=None))]
pub fn warp_perspective(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    m: [f32; 9],
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            let dev_out = match out {
                Some(o) => Some(o.extract::<Py<PyAny>>(py)?),
                None => None,
            };
            return crate::cuda_ext::geometry::warp_perspective(
                py,
                &img,
                m,
                new_size,
                interpolation,
                dev_out,
            )?
            .into_py(py);
        }
    }
    let np_out: Option<PyImage> = match out {
        Some(o) => Some(o.extract::<PyImage>(py).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "warp_perspective: out= must be a numpy u8 array on the CPU path",
            )
        })?),
        None => None,
    };
    let interp = interpolation.to_string();
    cpu_op(
        py,
        image,
        move |py, arr: Py<numpy::PyArray3<u8>>| match src_channels(py, &arr)? {
            1 => warp_dispatch::<1, _>(py, arr, new_size, &interp, np_out, |s, d| {
                warp::warp_perspective_u8(s, d, &m)
            }),
            3 => warp_dispatch::<3, _>(py, arr, new_size, &interp, np_out, |s, d| {
                warp::warp_perspective_u8(s, d, &m)
            }),
            4 => warp_dispatch::<4, _>(py, arr, new_size, &interp, np_out, |s, d| {
                warp::warp_perspective_u8(s, d, &m)
            }),
            c => Err(unsupported_channels(c)),
        },
    )
}
