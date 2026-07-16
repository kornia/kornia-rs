use pyo3::prelude::*;

use crate::dispatch::cpu_op;
use crate::image::{
    alloc_output_pyarray, numpy_as_image, parse_interpolation, to_pyerr, PyImageApi,
};
use kornia_image::ImageSize;
use kornia_imgproc::resize::resize_fast_rgb_aa;

/// Resize an image.
///
/// Residency-dispatched like the color ops: a device `Image` (f32, 3-channel)
/// runs the CUDA kernels — bit-identical to the CPU f32 path — and accepts a
/// preallocated device `out=` (torch-style) so frame loops allocate nothing;
/// a host `Image` or numpy u8 array runs the CPU fast path. `antialias`
/// applies only to the u8 CPU path (the f32 paths, CPU and GPU alike, have
/// never antialiased).
#[pyfunction]
#[pyo3(signature = (image, new_size, interpolation, antialias=true, out=None))]
pub fn resize(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    new_size: (usize, usize),
    interpolation: &str,
    antialias: bool,
    out: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::geometry::resize(py, &img, new_size, interpolation, out)?
                .into_py(py);
        }
    }
    if out.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resize: out= is supported for device images only",
        ));
    }
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
