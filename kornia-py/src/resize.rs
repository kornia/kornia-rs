use pyo3::prelude::*;

use crate::dispatch::{cpu_op, try_dispatch_device};
use crate::image::{alloc_output_pyarray, numpy_as_image, parse_interpolation, to_pyerr};
use kornia_image::ImageSize;
use kornia_imgproc::resize::resize_fast_rgb_aa;

/// Resize an image.
///
/// Residency-dispatched like the color ops: a device `Image` (f32, 3-channel)
/// runs the CUDA kernels — bit-identical to the CPU f32 path; a host `Image`
/// or numpy u8 array runs the CPU fast path. `antialias` applies only to the
/// u8 CPU path (the f32 paths, CPU and GPU alike, have never antialiased).
#[pyfunction]
#[pyo3(signature = (image, new_size, interpolation, antialias=true))]
pub fn resize(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    new_size: (usize, usize),
    interpolation: &str,
    antialias: bool,
) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, |api| crate::cuda_ext::geometry::resize(
        api,
        new_size,
        interpolation
    ));
    let interpolation = parse_interpolation(interpolation)?;
    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    cpu_op(py, image, |py, image: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<3>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, new_size)? };
        py.detach(|| resize_fast_rgb_aa(&src, &mut dst, interpolation, antialias))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}
