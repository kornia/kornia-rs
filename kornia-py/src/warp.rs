use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, parse_interpolation, to_pyerr, PyImage};
use kornia_image::{allocator::ForeignAllocator, Image, ImageError, ImageSize};
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
    F: Send
        + FnOnce(
            &Image<u8, C, ForeignAllocator>,
            &mut Image<u8, C, ForeignAllocator>,
        ) -> Result<(), ImageError>,
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

#[pyfunction]
#[pyo3(signature = (image, m, new_size, interpolation, out=None))]
pub fn warp_affine(
    py: Python<'_>,
    image: PyImage,
    m: [f32; 6],
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<PyImage>,
) -> PyResult<PyImage> {
    match src_channels(py, &image)? {
        1 => warp_dispatch::<1, _>(py, image, new_size, interpolation, out, |s, d| {
            warp::warp_affine_u8(s, d, &m)
        }),
        3 => warp_dispatch::<3, _>(py, image, new_size, interpolation, out, |s, d| {
            warp::warp_affine_u8(s, d, &m)
        }),
        4 => warp_dispatch::<4, _>(py, image, new_size, interpolation, out, |s, d| {
            warp::warp_affine_u8(s, d, &m)
        }),
        c => Err(unsupported_channels(c)),
    }
}

#[pyfunction]
#[pyo3(signature = (image, m, new_size, interpolation, out=None))]
pub fn warp_perspective(
    py: Python<'_>,
    image: PyImage,
    m: [f32; 9],
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<PyImage>,
) -> PyResult<PyImage> {
    match src_channels(py, &image)? {
        1 => warp_dispatch::<1, _>(py, image, new_size, interpolation, out, |s, d| {
            warp::warp_perspective_u8(s, d, &m)
        }),
        3 => warp_dispatch::<3, _>(py, image, new_size, interpolation, out, |s, d| {
            warp::warp_perspective_u8(s, d, &m)
        }),
        4 => warp_dispatch::<4, _>(py, image, new_size, interpolation, out, |s, d| {
            warp::warp_perspective_u8(s, d, &m)
        }),
        c => Err(unsupported_channels(c)),
    }
}
