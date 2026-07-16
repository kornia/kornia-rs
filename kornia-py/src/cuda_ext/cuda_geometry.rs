//! Device-resident GPU geometry ops (`kornia_rs.resize` / `warp_*` device path).
//!
//! Each op takes a device-resident unified [`PyImageApi`], allocates the
//! destination on the source's own stream, and calls the **public**
//! `kornia_imgproc` function — whose residency dispatch routes the device pair
//! to the CUDA kernels. Kernel choice therefore lives in exactly one place
//! (the Rust core), and output is bit-identical to the CPU path by the
//! byte-exact contract.

use super::*;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use pyo3::types::PyAnyMethods;

/// Element types the geometry helpers can allocate device outputs for.
trait GeomElem: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static {}
impl GeomElem for f32 {}
impl GeomElem for u8 {}

fn no_device_err(op: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{op}: expected a device Image (on a CUDA device); for a host image \
         pass its numpy array, or move it to a device with .to_cuda(stream)"
    ))
}

/// Borrow a device `Image<f32, 3>` out of a unified image, or raise the
/// uniform "no GPU kernel for this dtype/channels" error.
fn device_f32c3<'a>(img: &'a PyImageApi, op: &str) -> PyResult<&'a Image<f32, 3>> {
    let dev = img.as_device().ok_or_else(|| no_device_err(op))?;
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

/// Borrow a caller-provided `out=` device image mutably, validating dtype
/// (via `unwrap`, which pattern-matches the expected [`Inner`] variant),
/// exclusivity (see `as_device_mut`), size, and that it is not the source.
fn with_out_t<T: GeomElem, const C: usize, R>(
    py: Python<'_>,
    out: &Py<PyAny>,
    src: &Image<T, C>,
    dst_size: kornia_image::ImageSize,
    op: &str,
    expect: &str,
    unwrap: fn(&mut Inner) -> Option<&mut Image<T, C>>,
    f: impl FnOnce(&mut Image<T, C>) -> PyResult<R>,
) -> PyResult<R> {
    let bound = out.bind(py);
    let cell = bound.cast::<PyImageApi>().map_err(|_| {
        PyValueError::new_err(format!("{op}: out= must be a device Image ({expect})"))
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
    let Some(dst) = unwrap(dev) else {
        return Err(PyValueError::new_err(format!(
            "{op}: out= must be a device Image matching the input ({expect})"
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

fn with_out<R>(
    py: Python<'_>,
    out: &Py<PyAny>,
    src: &Image<f32, 3>,
    dst_size: kornia_image::ImageSize,
    op: &str,
    f: impl FnOnce(&mut Image<f32, 3>) -> PyResult<R>,
) -> PyResult<R> {
    with_out_t(py, out, src, dst_size, op, "f32, 3-channel", unwrap_f32c3, f)
}

fn unwrap_f32c3(i: &mut Inner) -> Option<&mut Image<f32, 3>> {
    match i {
        Inner::F32C3(x) => Some(x),
        _ => None,
    }
}
fn unwrap_u8c1(i: &mut Inner) -> Option<&mut Image<u8, 1>> {
    match i {
        Inner::U8C1(x) => Some(x),
        _ => None,
    }
}
fn unwrap_u8c3(i: &mut Inner) -> Option<&mut Image<u8, 3>> {
    match i {
        Inner::U8C3(x) => Some(x),
        _ => None,
    }
}
fn unwrap_u8c4(i: &mut Inner) -> Option<&mut Image<u8, 4>> {
    match i {
        Inner::U8C4(x) => Some(x),
        _ => None,
    }
}

/// Shared tail: allocate the destination on the source's stream, run `f`
/// (the public residency-dispatched op), wrap the result.
fn run_geometry_t<T: GeomElem, const C: usize>(
    src: &Image<T, C>,
    cs: kornia_image::ColorSpace,
    dst_size: kornia_image::ImageSize,
    wrap: fn(Image<T, C>) -> Inner,
    f: impl FnOnce(&mut Image<T, C>) -> Result<(), kornia_image::ImageError>,
) -> PyResult<PyImageApi> {
    let stream = source_stream(src)?;
    // SAFETY: every geometry kernel writes each output pixel exactly once
    // (bounds-guarded grid; warp border cases write 0 rather than skipping),
    // so the uninitialized destination is fully overwritten.
    let mut dst = unsafe { Image::<T, C>::uninit_cuda(dst_size, &stream) }.map_err(err)?;
    f(&mut dst).map_err(err)?;
    Ok(PyImageApi::from_device(wrap(dst), cs, device_mode::<T>(C)))
}

fn run_geometry(
    src: &Image<f32, 3>,
    cs: kornia_image::ColorSpace,
    dst_size: kornia_image::ImageSize,
    f: impl FnOnce(&mut Image<f32, 3>) -> Result<(), kornia_image::ImageError>,
) -> PyResult<PyImageApi> {
    run_geometry_t(src, cs, dst_size, Inner::F32C3, f)
}

/// Device resize — f32 C3 on the half-pixel grid, or u8 C1/C3/C4 through the
/// integer u8 kernel cascade. Both bit-identical to their CPU twins (see the
/// parity tests in `kornia-imgproc`). `antialias` matches the CPU semantics:
/// it shapes the u8 bicubic/lanczos kernels and is a no-op for f32 and for
/// u8 nearest/bilinear.
pub(crate) fn resize(
    py: Python<'_>,
    img: &PyImageApi,
    new_size: (usize, usize),
    interpolation: &str,
    antialias: bool,
    out: Option<Py<PyAny>>,
) -> PyResult<PyOut> {
    let mode = crate::image::parse_interpolation(interpolation)?;
    let dst_size = kornia_image::ImageSize {
        height: new_size.0,
        width: new_size.1,
    };

    /// One u8 channel-count arm: `out=` path or fresh allocation, both
    /// through the public residency-dispatched `resize_fast_u8_aa`.
    fn resize_u8<const C: usize>(
        py: Python<'_>,
        img: &PyImageApi,
        src: &Image<u8, C>,
        dst_size: kornia_image::ImageSize,
        mode: kornia_imgproc::interpolation::InterpolationMode,
        antialias: bool,
        out: Option<Py<PyAny>>,
        unwrap: fn(&mut Inner) -> Option<&mut Image<u8, C>>,
        wrap: fn(Image<u8, C>) -> Inner,
    ) -> PyResult<PyOut> {
        if let Some(out) = out {
            with_out_t(
                py,
                &out,
                src,
                dst_size,
                "resize",
                "u8, same channel count as the input",
                unwrap,
                |dst| {
                    kornia_imgproc::resize::resize_fast_u8_aa(src, dst, mode, antialias)
                        .map_err(err)
                },
            )?;
            return Ok(PyOut::Reused(out));
        }
        run_geometry_t(src, img.color_space, dst_size, wrap, |dst| {
            kornia_imgproc::resize::resize_fast_u8_aa(src, dst, mode, antialias)
        })
        .map(PyOut::New)
    }

    let dev = img.as_device().ok_or_else(|| no_device_err("resize"))?;
    match dev {
        Inner::F32C3(src) => {
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
        Inner::U8C1(src) => resize_u8::<1>(
            py, img, src, dst_size, mode, antialias, out, unwrap_u8c1, Inner::U8C1,
        ),
        Inner::U8C3(src) => resize_u8::<3>(
            py, img, src, dst_size, mode, antialias, out, unwrap_u8c3, Inner::U8C3,
        ),
        Inner::U8C4(src) => resize_u8::<4>(
            py, img, src, dst_size, mode, antialias, out, unwrap_u8c4, Inner::U8C4,
        ),
        other => Err(PyValueError::new_err(format!(
            "resize: the GPU path supports f32 3-channel and u8 1/3/4-channel \
             device images, got {:?} with {} channel(s); convert on the host \
             or move the image back with .cpu()",
            other.dtype_enum(),
            other.channels()
        ))),
    }
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
