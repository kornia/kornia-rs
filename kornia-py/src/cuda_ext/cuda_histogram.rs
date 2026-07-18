//! Device-resident GPU histogram equalization
//! (`kornia_rs.imgproc.equalize_hist` device path). Same shape as
//! `cuda_morphology`: allocate the destination on the source's stream, call
//! the public residency-dispatched op.

use super::cuda_geometry::PyOut;
use super::*;

pub(crate) fn equalize_hist(_py: Python<'_>, img: &PyImageApi) -> PyResult<PyOut> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "equalize_hist: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    let Inner::U8C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "equalize_hist: the GPU path supports u8 single-channel device images, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    let stream = source_stream(src)?;
    // SAFETY: apply_lut_u8 writes every output pixel (bounds-guarded grid),
    // so the uninitialized destination is fully overwritten.
    let mut dst = unsafe { Image::<u8, 1>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    kornia_imgproc::histogram::equalize_hist(src, &mut dst).map_err(err)?;
    Ok(PyOut::New(PyImageApi::from_device(
        Inner::U8C1(dst),
        img.color_space,
        device_mode::<u8>(1),
    )))
}

pub(crate) fn compute_histogram(
    py: Python<'_>,
    img: &PyImageApi,
    hist: &mut [usize],
    num_bins: usize,
) -> PyResult<()> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "compute_histogram: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    let Inner::U8C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "compute_histogram: the GPU path supports u8 single-channel device images, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    py.detach(|| kornia_imgproc::histogram::compute_histogram(src, hist, num_bins))
        .map_err(err)
}
