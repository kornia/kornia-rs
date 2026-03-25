use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, numpy_to_f32_image, to_pyerr, PyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageError};
use kornia_imgproc::color;

#[pyfunction]
pub fn rgb_from_gray(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<1>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_gray(&src, &mut dst)).map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn bgr_from_rgb(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::bgr_from_rgb(&src, &mut dst)).map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn gray_from_rgb(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let size = src_f32.size();
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<1>(py, size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(size, 0f32, CpuAllocator)?;
        color::gray_from_rgb(&src_f32, &mut dst_f32)?;
        dst_u8.as_slice_mut().iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s as u8);
        Ok(())
    }).map_err(to_pyerr)?;

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_rgba(py: Python<'_>, image: PyImage, background: Option<[u8; 3]>) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<4>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_rgba(&src, &mut dst, background)).map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_bgra(py: Python<'_>, image: PyImage, background: Option<[u8; 3]>) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<4>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_bgra(&src, &mut dst, background)).map_err(to_pyerr)?;
    Ok(out)
}
