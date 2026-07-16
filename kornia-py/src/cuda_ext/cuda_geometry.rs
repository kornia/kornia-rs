//! Device-resident GPU geometry ops (`kornia_rs.resize` / `warp_*` device path).
//!
//! Each op takes a device-resident unified [`PyImageApi`], allocates the
//! destination on the source's own stream, and calls the **public**
//! `kornia_imgproc` function — whose residency dispatch routes the device pair
//! to the CUDA kernels. Kernel choice therefore lives in exactly one place
//! (the Rust core), and output is bit-identical to the CPU path by the
//! byte-exact contract.

use super::*;
use pyo3::types::PyAnyMethods;

/// Borrow a device `Image<f32, 3>` out of a unified image, or raise the
/// uniform "no GPU kernel for this dtype/channels" error.
fn device_f32c3<'a>(img: &'a PyImageApi, op: &str) -> PyResult<&'a Image<f32, 3>> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(format!(
            "{op}: expected a device Image (on a CUDA device); for a host image \
             pass its numpy array, or move it to a device with .to_cuda(stream)"
        ))
    })?;
    match dev {
        Inner::F32C3(src) => Ok(src),
        other => Err(PyValueError::new_err(format!(
            "{op}: the GPU path supports 3-channel f32 device images, got \
             {:?} with {} channel(s); convert on the host or move the image \
             back with .cpu()",
            other.dtype_enum(),
            other.channels()
        ))),
    }
}

/// A geometry-op result: a freshly allocated device image, or the caller's
/// `out=` object handed back (torch-style) so frame loops allocate nothing.
pub(crate) enum PyOut {
    New(PyImageApi),
    Reused(Py<PyAny>),
}

impl PyOut {
    pub(crate) fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            PyOut::New(img) => Ok(Py::new(py, img)?.into_any()),
            PyOut::Reused(obj) => Ok(obj),
        }
    }
}

/// Borrow a caller-provided `out=` device image mutably, validating dtype,
/// exclusivity (see `as_device_mut`), size, and that it is not the source.
fn with_out<R>(
    py: Python<'_>,
    out: &Py<PyAny>,
    src: &Image<f32, 3>,
    dst_size: kornia_image::ImageSize,
    op: &str,
    f: impl FnOnce(&mut Image<f32, 3>) -> PyResult<R>,
) -> PyResult<R> {
    let bound = out.bind(py);
    let cell = bound.cast::<PyImageApi>().map_err(|_| {
        PyValueError::new_err(format!(
            "{op}: out= must be a device Image (f32, 3-channel)"
        ))
    })?;
    // try_borrow_mut: the input image is already borrowed by the caller, so
    // out=input surfaces here as a borrow conflict — turn it into the same
    // aliasing error the pointer check below gives for distinct objects that
    // share storage.
    let mut api = cell.try_borrow_mut().map_err(|_| {
        PyValueError::new_err(format!(
            "{op}: out= must not alias the input (in-place warps race)"
        ))
    })?;
    let dev = api
        .as_device_mut()?
        .ok_or_else(|| PyValueError::new_err(format!("{op}: out= must be a device Image")))?;
    let Inner::F32C3(dst) = dev else {
        return Err(PyValueError::new_err(format!(
            "{op}: out= must be a 3-channel f32 device Image"
        )));
    };
    if dst.size() != dst_size {
        return Err(PyValueError::new_err(format!(
            "{op}: out= size {}x{} does not match new_size {}x{}",
            dst.cols(),
            dst.rows(),
            dst_size.width,
            dst_size.height
        )));
    }
    // Pointer identity via the storage pointer (device address; as_slice()
    // would be a host access of device memory).
    if std::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        return Err(PyValueError::new_err(format!(
            "{op}: out= must not alias the input (in-place warps race)"
        )));
    }
    f(dst)
}

/// Shared tail: allocate the destination on the source's stream, run `f`
/// (the public residency-dispatched op), wrap the result.
fn run_geometry(
    src: &Image<f32, 3>,
    cs: kornia_image::ColorSpace,
    dst_size: kornia_image::ImageSize,
    f: impl FnOnce(&mut Image<f32, 3>) -> Result<(), kornia_image::ImageError>,
) -> PyResult<PyImageApi> {
    let stream = source_stream(src)?;
    // SAFETY: every geometry kernel writes each output pixel exactly once
    // (bounds-guarded grid; warp border cases write 0 rather than skipping),
    // so the uninitialized destination is fully overwritten.
    let mut dst = unsafe { Image::<f32, 3>::uninit_cuda(dst_size, &stream) }.map_err(err)?;
    f(&mut dst).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::F32C3(dst),
        cs,
        device_mode::<f32>(3),
    ))
}

/// Device f32 resize on the half-pixel grid — bit-identical to the CPU
/// `resize` (see the parity tests in `kornia-imgproc`).
pub(crate) fn resize(
    py: Python<'_>,
    img: &PyImageApi,
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<Py<PyAny>>,
) -> PyResult<PyOut> {
    let mode = crate::image::parse_interpolation(interpolation)?;
    let src = device_f32c3(img, "resize")?;
    let dst_size = kornia_image::ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    if let Some(out) = out {
        with_out(py, &out, src, dst_size, "resize", |dst| {
            kornia_imgproc::resize::resize(src, dst, mode).map_err(err)
        })?;
        return Ok(PyOut::Reused(out));
    }
    run_geometry(src, img.color_space, dst_size, |dst| {
        kornia_imgproc::resize::resize(src, dst, mode)
    })
    .map(PyOut::New)
}

/// Device f32 warp-affine (forward 2×3 matrix, inverted internally like CPU).
pub(crate) fn warp_affine(
    py: Python<'_>,
    img: &PyImageApi,
    m: [f32; 6],
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<Py<PyAny>>,
) -> PyResult<PyOut> {
    let mode = crate::image::parse_interpolation(interpolation)?;
    let src = device_f32c3(img, "warp_affine")?;
    let dst_size = kornia_image::ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    if let Some(out) = out {
        with_out(py, &out, src, dst_size, "warp_affine", |dst| {
            kornia_imgproc::warp::warp_affine(src, dst, &m, mode).map_err(err)
        })?;
        return Ok(PyOut::Reused(out));
    }
    run_geometry(src, img.color_space, dst_size, |dst| {
        kornia_imgproc::warp::warp_affine(src, dst, &m, mode)
    })
    .map(PyOut::New)
}

/// Device f32 warp-perspective (forward 3×3 homography).
pub(crate) fn warp_perspective(
    py: Python<'_>,
    img: &PyImageApi,
    m: [f32; 9],
    new_size: (usize, usize),
    interpolation: &str,
    out: Option<Py<PyAny>>,
) -> PyResult<PyOut> {
    let mode = crate::image::parse_interpolation(interpolation)?;
    let src = device_f32c3(img, "warp_perspective")?;
    let dst_size = kornia_image::ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    if let Some(out) = out {
        with_out(py, &out, src, dst_size, "warp_perspective", |dst| {
            kornia_imgproc::warp::warp_perspective(src, dst, &m, mode).map_err(err)
        })?;
        return Ok(PyOut::Reused(out));
    }
    run_geometry(src, img.color_space, dst_size, |dst| {
        kornia_imgproc::warp::warp_perspective(src, dst, &m, mode)
    })
    .map(PyOut::New)
}
