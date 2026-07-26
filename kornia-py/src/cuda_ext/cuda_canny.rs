//! Device-resident GPU Canny (`kornia_rs.imgproc.canny` device path).

use super::cuda_geometry::PyOut;
use super::*;

pub(crate) fn canny(
    _py: Python<'_>,
    img: &PyImageApi,
    low_threshold: f64,
    high_threshold: f64,
    l2_gradient: bool,
) -> PyResult<PyOut> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "canny: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    let Inner::U8C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "canny: the GPU path supports u8 single-channel device images, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    let stream = source_stream(src)?;
    // SAFETY: canny_finalize writes every output pixel (bounds-guarded
    // grid), so the uninitialized destination is fully overwritten.
    let mut dst = unsafe { Image::<u8, 1>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    kornia_imgproc::canny::canny(src, &mut dst, low_threshold, high_threshold, l2_gradient)
        .map_err(err)?;
    Ok(PyOut::New(PyImageApi::from_device(
        Inner::U8C1(dst),
        img.color_space,
        device_mode::<u8>(1),
    )))
}
