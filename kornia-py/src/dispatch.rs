//! Shared residency dispatch for the unified-Image pyfunctions.
//!
//! `color`, `resize`, and `warp` all follow the same shape: a device `Image`
//! early-returns through its GPU op, a host `Image` or numpy array falls
//! through to the CPU path, and a device image reaching a CPU-only op is a
//! typed error (never an implicit download).

use pyo3::prelude::*;

use crate::image::PyImageApi;

/// Device half of the residency dispatch: if `image` is a **device** `Image`,
/// run the GPU conversion `dev_op` and return the device `Image`. Returns `None`
/// for a numpy array or a host `Image` (the caller's CPU path handles those).
#[cfg(feature = "cuda")]
pub(crate) fn dispatch_device<F>(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    dev_op: F,
) -> PyResult<Option<Py<PyAny>>>
where
    F: FnOnce(&PyImageApi) -> PyResult<PyImageApi>,
{
    if let Ok(api) = image.cast::<PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            let out = dev_op(&img)?;
            return Ok(Some(Py::new(py, out)?.into_any()));
        }
    }
    Ok(None)
}

/// Early-return the GPU result when `$image` is a device `Image`; otherwise
/// fall through to the caller's CPU path below. Centralizes the per-op device
/// prologue that every residency-dispatching color `#[pyfunction]` repeats.
/// `$dev` (a GPU op fn or closure) is referenced only under the `cuda` feature,
/// so a CPU-only build never needs it to exist; expands to nothing there.
#[macro_export]
#[doc(hidden)]
macro_rules! __try_dispatch_device {
    ($py:expr, $image:expr, $dev:expr) => {
        #[cfg(feature = "cuda")]
        if let Some(dev) = $crate::dispatch::dispatch_device($py, $image, $dev)? {
            return Ok(dev);
        }
    };
}

/// Run a CPU color op (element type `E` — `u8` for [`PyImage`], `f32` for
/// [`PyImageF32`]) that accepts numpy **or** a host `Image`, returning the same
/// kind: numpy → numpy, host `Image` → host `Image`. (Device images are handled
/// by [`dispatch_device`] before this is reached.) `E` is pinned by the
/// concrete `Py<PyArray3<E>>` type the caller's closure body operates on
/// (`numpy_as_image`/`numpy_as_image_f32` etc. each fix one element type), so
/// call sites need no turbofish.
pub(crate) fn cpu_op<E, C>(py: Python<'_>, image: &Bound<'_, PyAny>, cpu: C) -> PyResult<Py<PyAny>>
where
    E: numpy::Element,
    C: FnOnce(Python<'_>, Py<numpy::PyArray3<E>>) -> PyResult<Py<numpy::PyArray3<E>>>,
{
    if let Ok(api) = image.cast::<PyImageApi>() {
        no_gpu_kernel_if_device(&api.borrow())?;
        // Host Image: run on its numpy view, wrap the numpy result as a host Image.
        let view = api.call_method0("numpy")?;
        let arr: Py<numpy::PyArray3<E>> = view.extract()?;
        let out = cpu(py, arr)?;
        let img = PyImageApi::from_numpy_borrow(py, out.bind(py).as_any(), None, None, false)?;
        return Ok(Py::new(py, img)?.into_any());
    }
    let arr: Py<numpy::PyArray3<E>> = image.extract()?;
    Ok(cpu(py, arr)?.into_any())
}

/// A device `Image` reaching a CPU-only op (no GPU kernel) is an error — we do
/// not silently D2H-download. Host images fall through.
pub(crate) fn no_gpu_kernel_if_device(api: &PyImageApi) -> PyResult<()> {
    if api.is_device() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "this color op has no GPU kernel for a device Image; move it to the host \
             with .cpu() first (or pass a numpy array)",
        ));
    }
    Ok(())
}

/// Early-return the GPU result when `$image` is a device `Image`.
pub(crate) use crate::__try_dispatch_device as try_dispatch_device;

/// Helper to handle host-path boilerplate: given a Python object that may be a
/// numpy array or a host `Image`, return its numpy view and the name of its
/// dtype (e.g. "uint8", "float32"). If it is a device `Image`, returns an error
/// demanding `.cpu()` first.
pub(crate) fn host_view_and_dtype<'py>(
    image: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, String)> {
    let view = if let Ok(api) = image.cast::<PyImageApi>() {
        no_gpu_kernel_if_device(&api.borrow())?;
        api.call_method0("numpy")?
    } else {
        image.clone()
    };
    use pyo3::types::PyAnyMethods;
    let dtype = view
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    Ok((view, dtype))
}

/// Raise a clear error naming the actual dtype if a non-f32 numpy array or
/// host `Image` reaches an f32-only CPU color conversion, instead of letting
/// pyo3's generic array-downcast error surface (`'ndarray' object is not an
/// instance of 'ndarray'`).
pub(crate) fn require_f32_host(image: &Bound<'_, PyAny>, op: &str) -> PyResult<()> {
    // If it's a device Image, we don't throw here; try_dispatch_device! already
    // handled the fast path, or if we got here it means there's no GPU kernel,
    // in which case host_view_and_dtype will correctly emit the "no GPU kernel"
    // error. Wait, we want to allow device image if try_dispatch_device was used!
    // But require_f32_host is called *after* try_dispatch_device!. If we get here
    // with a device image, it's either an error anyway (no GPU kernel), or it shouldn't
    // happen.
    // Actually, host_view_and_dtype throws if device image. So it's perfect.
    let (_, dtype) = host_view_and_dtype(image)?;
    if dtype != "float32" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{op}: host path supports float32 only (got dtype={dtype}); \
             there is currently no uint8 kernel for this conversion in kornia-imgproc"
        )));
    }
    Ok(())
}
