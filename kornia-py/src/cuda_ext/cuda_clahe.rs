//! Device-resident GPU CLAHE (`kornia_rs.imgproc.clahe` device path).

use super::cuda_geometry::PyOut;
use super::*;

pub(crate) fn clahe(
    _py: Python<'_>,
    img: &PyImageApi,
    clip_limit: f64,
    grid: (usize, usize),
) -> PyResult<PyOut> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "clahe: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    let Inner::U8C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "clahe: the GPU path supports u8 single-channel device images, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    let stream = source_stream(src)?;
    // SAFETY: clahe_apply_u8 writes every output pixel (bounds-guarded
    // grid), so the uninitialized destination is fully overwritten.
    let mut dst = unsafe { Image::<u8, 1>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    kornia_imgproc::clahe::clahe(src, &mut dst, clip_limit, grid).map_err(err)?;
    Ok(PyOut::New(PyImageApi::from_device(
        Inner::U8C1(dst),
        img.color_space,
        device_mode::<u8>(1),
    )))
}
