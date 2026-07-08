//! `Tensor`/`Preprocessor` (CPU/GPU) plus, on a `cuda`-feature build, GPU color
//! conversions and the fused camera-preprocessing device kernels.
//!
//! `Tensor` and `Preprocessor` (`device=None`) work on every build. The `cuda`
//! cargo feature adds device residency: `Preprocessor(device=<ordinal>)`, the
//! `kornia_rs.cuda` submodule (`Stream`, `is_available`, `mem_get_info`), and
//! the device color-conversion kernels behind `kornia_rs.imgproc.*`. cudarc
//! dynamic-loading means the `cuda` feature itself compiles everywhere; at
//! runtime [`is_available`] probes for a usable driver and everything degrades
//! gracefully without one — so most builds should just enable it. On a build
//! without it, `is_available()` is always `False` and device-only calls raise
//! a clear "CUDA support is not compiled in" error.
//!
//! Design: device pixels live in the unified `kornia_rs.image.Image` (create one
//! with `Image.from_numpy(a).to_cuda(stream)`, or allocate directly with
//! `Image.zeros(..., stream=stream)`); the `kornia_rs.imgproc` color-conversion
//! ops dispatch to these device functions when given a device `Image`, returning
//! a device `Image` in turn. Model input (CHW) becomes a [`Tensor`] via
//! [`Preprocessor`]. Everything exports zero-copy to torch / cupy /
//! cuda-python via `__dlpack__` and `__cuda_array_interface__`.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};
use numpy::{PyArray1, PyArrayMethods};
#[cfg(feature = "cuda")]
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use kornia_image::color_spaces::{Bgr8, Bgra8, Gray8, Hsvf32, Labf32, Rgb8, Rgba8, Rgbf32, YCbCr8};
#[cfg(feature = "cuda")]
use kornia_image::Image;
#[cfg(feature = "cuda")]
use kornia_imgproc::color::{self, ConvertColor, ConvertColorWithBackground};
use kornia_imgproc::preprocess::{Normalize, Preprocessor, ResizeMode, SourceFormat};
use kornia_tensor::Tensor;

#[cfg(not(feature = "cuda"))]
use crate::image::cuda_not_compiled;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Widen an f16 slice to an owned f32 `Vec` (element-wise, unavoidable copy).
fn widen_f16_to_f32(data: &[half::f16]) -> Vec<f32> {
    data.iter().map(|v| v.to_f32()).collect()
}

/// Default-stream handle for CUDA device `ordinal` (created lazily, cached per
/// device for the process). The stream's context is the device selector — the
/// residency of any image produced on it (`to_cuda`/`zeros_cuda`) is that
/// ordinal, mirroring Rust's `CudaContext::new(ordinal)`.
#[cfg(feature = "cuda")]
pub(crate) fn default_stream_for(ordinal: i32) -> PyResult<Arc<CudaStream>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<i32, Result<Arc<CudaStream>, String>>>> = OnceLock::new();
    if ordinal < 0 {
        return Err(PyRuntimeError::new_err(format!(
            "invalid CUDA device ordinal {ordinal}"
        )));
    }
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("stream cache poisoned");
    map.entry(ordinal)
        .or_insert_with(|| {
            CudaContext::new(ordinal as usize)
                .map(|ctx| ctx.default_stream())
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(PyRuntimeError::new_err)
}

/// Default-stream handle for device 0.
#[cfg(feature = "cuda")]
pub(crate) fn default_stream() -> PyResult<Arc<CudaStream>> {
    default_stream_for(0)
}

/// Best-effort CUDA device ordinal that a raw foreign `CUstream` handle belongs
/// to, so an adopted foreign stream (`Stream.from_handle`) from a non-zero GPU
/// causes kornia to launch its work on that same GPU rather than always
/// device 0. Returns `0` (the historical default) for a default-stream sentinel
/// handle or if the driver can't resolve it — never errors, so a foreign stream
/// that can't be probed still works (on device 0) exactly as before.
#[cfg(feature = "cuda")]
pub(crate) fn foreign_stream_device(handle: usize) -> i32 {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    // A stream's owning device is immutable for its lifetime, so memoize
    // handle -> ordinal (same cache shape as `default_stream_for`): a serving
    // loop that re-adopts one TensorRT stream every frame then skips the 4-call
    // driver probe below after the first lookup.
    static CACHE: OnceLock<Mutex<HashMap<usize, i32>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&ord) = cache.lock().expect("stream-device cache poisoned").get(&handle) {
        return ord;
    }
    let ord = probe_foreign_stream_device(handle);
    cache
        .lock()
        .expect("stream-device cache poisoned")
        .insert(handle, ord);
    ord
}

/// The uncached driver probe behind [`foreign_stream_device`].
#[cfg(feature = "cuda")]
fn probe_foreign_stream_device(handle: usize) -> i32 {
    use cudarc::driver::sys;
    // The default-stream sentinels (0 = legacy/null, 1 = legacy default,
    // 2 = per-thread default) have no distinct owning context to query.
    if handle <= 2 {
        return 0;
    }
    // SAFETY: raw driver queries. `handle` is a `CUstream` the caller vouched
    // for via `Stream.from_handle`/`from_cuda_stream`. We push the stream's
    // context, read its device, and ALWAYS pop it back so the thread's context
    // stack is restored on every path; any driver error falls back to device 0
    // instead of propagating.
    unsafe {
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        if sys::cuStreamGetCtx(handle as sys::CUstream, &mut ctx)
            .result()
            .is_err()
            || ctx.is_null()
        {
            return 0;
        }
        if sys::cuCtxPushCurrent_v2(ctx).result().is_err() {
            return 0;
        }
        let mut dev: sys::CUdevice = 0;
        let dev_ok = sys::cuCtxGetDevice(&mut dev).result().is_ok();
        // Restore the previous context regardless of the query result.
        let mut popped: sys::CUcontext = std::ptr::null_mut();
        let _ = sys::cuCtxPopCurrent_v2(&mut popped);
        if dev_ok {
            dev
        } else {
            0
        }
    }
}

/// True if a CUDA driver and device 0 are usable in this process. Always
/// `False` on a build without the `cuda` feature.
#[pyfunction]
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        default_stream().is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Free and total device-0 global memory in bytes, as `(free, total)`.
///
/// Wraps `cuMemGetInfo`; use it to bracket a loop and assert the free byte
/// count returns to its baseline — the primitive behind the memory-leak
/// integration tests. Synchronizes the default stream first so all pending
/// frees/allocs are reflected in the reading.
#[pyfunction]
pub fn mem_get_info(py: Python<'_>) -> PyResult<(usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        let stream = default_stream()?;
        py.detach(|| {
            stream.synchronize().map_err(err)?;
            stream.context().mem_get_info().map_err(err)
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = py;
        Err(cuda_not_compiled())
    }
}

/// Order a DLPack consumer's stream after the producer's `launch` stream per the
/// array-API `__dlpack__` (CUDA) convention — **without blocking the host**
/// whenever the protocol allows. Shared by both device DLPack exporters
/// (`Tensor` and the unified `Image`) so they can't diverge.
///
/// Consumer stream: `-1` = no sync (skip); `0` = the null/legacy default stream,
/// `1` = legacy default, `2` = per-thread default, any other positive = a raw
/// `CUstream` handle — each is event-fenced against `launch` (a foreign or
/// default stream is ordered after the producer without a host block); `None` =
/// a stricter-than-spec host sync of `launch` for bare `__dlpack__()` callers.
/// Negatives other than `-1` are invalid and rejected. Fencing against the
/// actual producing `launch` stream (rather than assuming the legacy default)
/// keeps it correct even when the producer ran on a custom stream.
#[cfg(feature = "cuda")]
pub(crate) fn dlpack_fence_consumer(launch: &Arc<CudaStream>, consumer: Option<isize>) -> PyResult<()> {
    match consumer {
        Some(-1) => Ok(()),
        Some(h) if h < -1 => Err(PyValueError::new_err(format!(
            "__dlpack__: invalid stream handle {h}; expected -1 (no sync), 0, 1, 2, \
             or a valid CUDA stream address"
        ))),
        // 0/1/2 are valid default-stream sentinels; >2 is a real handle. All are
        // legal CUstream values for cuStreamWaitEvent.
        Some(h) => fence_stream_into(launch, Some(h as usize)),
        None => launch.synchronize().map_err(err),
    }
}

// ── GPU color conversions (operate on device-resident unified `Image`) ────────

/// Device-resident pixels, monomorphized per supported (dtype, channels).
///
/// Shared with the unified `Image` (`Backing::Device`); see [`crate::device`].
#[cfg(feature = "cuda")]
use crate::device::DeviceImage as Inner;
#[cfg(feature = "cuda")]
use crate::image::PyImageApi;
#[cfg(feature = "cuda")]
use kornia_image::ColorSpace;

/// View a plain `Image` as its `#[repr(transparent)]` color-space newtype.
///
/// SAFETY: every `define_color_space!` newtype is `#[repr(transparent)]` over
/// `Image<T, C>`, so the pointer casts are layout-sound.
#[cfg(feature = "cuda")]
macro_rules! as_newtype {
    ($img:expr, $nt:ty) => {
        unsafe { &*($img as *const _ as *const $nt) }
    };
    (mut $img:expr, $nt:ty) => {
        unsafe { &mut *($img as *mut _ as *mut $nt) }
    };
}

/// Run `src.convert(&mut dst)` through the repr(transparent) newtypes.
#[cfg(feature = "cuda")]
macro_rules! convert_pair {
    ($src:expr, $snt:ty, $dst:expr, $dnt:ty) => {{
        let s = as_newtype!($src, $snt);
        let d = as_newtype!(mut $dst, $dnt);
        s.convert(d).map_err(err)
    }};
}

/// Run `src.convert_with_bg(&mut dst, bg)` through the repr(transparent)
/// newtypes — the alpha-compositing sibling of [`convert_pair!`].
#[cfg(feature = "cuda")]
macro_rules! convert_pair_bg {
    ($src:expr, $snt:ty, $dst:expr, $dnt:ty, $bg:expr) => {{
        let s = as_newtype!($src, $snt);
        let d = as_newtype!(mut $dst, $dnt);
        s.convert_with_bg(d, $bg).map_err(err)
    }};
}


/// Build the `TensorInfo` (device + C-contiguous shape + dtype) that both
/// capsule packers below share. Pure data construction, no `unsafe`.
fn build_dl_tensor_info<T, const N: usize>(
    t: &Tensor<T, N>,
    dtype: dlpack_rs::ffi::DLDataType,
    dl_device: (i32, i32),
) -> dlpack_rs::safe::TensorInfo {
    let device = dlpack_rs::ffi::DLDevice {
        device_type: dl_device.0 as u32,
        device_id: dl_device.1,
    };
    let shape: Vec<i64> = t.shape.iter().map(|&s| s as i64).collect();
    dlpack_rs::safe::TensorInfo::contiguous(t.as_ptr() as *mut std::ffi::c_void, device, dtype, shape)
}

/// Wrap an already-packed `DLManagedTensor*` in a named PyCapsule with `deleter`.
///
/// # Safety
/// `managed` must be a live pointer from `safe::pack`/`pack_versioned` and
/// `deleter` the matching destructor for `name` (see `dlpack_capsule_destructor!`).
unsafe fn wrap_dlpack_capsule(
    py: Python<'_>,
    managed: *mut std::ffi::c_void,
    name: &std::ffi::CStr,
    deleter: unsafe extern "C" fn(*mut pyo3::ffi::PyObject),
) -> PyResult<Py<PyAny>> {
    let capsule = unsafe { pyo3::ffi::PyCapsule_New(managed, name.as_ptr(), Some(deleter)) };
    if capsule.is_null() {
        return Err(PyRuntimeError::new_err("failed to create DLPack capsule"));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

/// Generate a capsule destructor: run the producer's deleter iff the consumer
/// never renamed the capsule (i.e. never claimed it). The two DLPack variants
/// differ only in capsule name + managed struct type, so they share this shape.
macro_rules! dlpack_capsule_destructor {
    ($name:ident, $cstr:literal, $managed:ty) => {
        unsafe extern "C" fn $name(capsule: *mut pyo3::ffi::PyObject) {
            unsafe {
                if pyo3::ffi::PyCapsule_IsValid(capsule, $cstr.as_ptr()) == 1 {
                    let managed = pyo3::ffi::PyCapsule_GetPointer(capsule, $cstr.as_ptr())
                        as *mut $managed;
                    if !managed.is_null() {
                        if let Some(deleter) = (*managed).deleter {
                            deleter(managed);
                        }
                    }
                }
            }
        }
    };
}
dlpack_capsule_destructor!(
    dlpack_capsule_destructor,
    c"dltensor",
    dlpack_rs::ffi::DLManagedTensor
);
dlpack_capsule_destructor!(
    dlpack_capsule_destructor_versioned,
    c"dltensor_versioned",
    dlpack_rs::ffi::DLManagedTensorVersioned
);

/// Pack a DLPack capsule whose keepalive is an `Arc` clone of the owner —
/// export without consuming, buffer freed when both Python sides drop.
fn arc_dlpack_capsule<T, K, const N: usize>(
    py: Python<'_>,
    keepalive: Arc<K>,
    t: &Tensor<T, N>,
    dtype: dlpack_rs::ffi::DLDataType,
    dl_device: (i32, i32),
) -> PyResult<Py<PyAny>>
where
    K: Send + Sync + 'static,
{
    let info = build_dl_tensor_info(t, dtype, dl_device);
    // SAFETY: the Arc keepalive owns (a reference to) the device buffer and is
    // dropped by the capsule deleter; `dlpack_capsule_destructor` matches the
    // "dltensor" name and `DLManagedTensor` type packed here.
    let managed = dlpack_rs::safe::pack(keepalive, info);
    unsafe {
        wrap_dlpack_capsule(
            py,
            managed as *mut std::ffi::c_void,
            c"dltensor",
            dlpack_capsule_destructor,
        )
    }
}

/// DLPack 1.0 (versioned) sibling of [`arc_dlpack_capsule`]: emits a
/// `"dltensor_versioned"` capsule for consumers that negotiate `max_version >=
/// (1, 0)` (modern torch / NumPy 2 / TensorRT tooling). `flags` carries the
/// DLPack bitmask (e.g. read-only); device tensors here pass `0`.
fn arc_dlpack_capsule_versioned<T, K, const N: usize>(
    py: Python<'_>,
    keepalive: Arc<K>,
    t: &Tensor<T, N>,
    dtype: dlpack_rs::ffi::DLDataType,
    flags: u64,
    dl_device: (i32, i32),
) -> PyResult<Py<PyAny>>
where
    K: Send + Sync + 'static,
{
    let info = build_dl_tensor_info(t, dtype, dl_device);
    // SAFETY: as in `arc_dlpack_capsule`, with the versioned name/type pair.
    let managed = dlpack_rs::safe::pack_versioned(keepalive, info, flags);
    unsafe {
        wrap_dlpack_capsule(
            py,
            managed as *mut std::ffi::c_void,
            c"dltensor_versioned",
            dlpack_capsule_destructor_versioned,
        )
    }
}


/// Keeps the DLPack producer's Python object alive for a zero-copy import.
///
/// The DLPack consumer may drop the tensor off-GIL (e.g. from a worker
/// thread), so the handle is released under a re-acquired GIL — same
/// discipline as `dlpack::ImageExport`. During interpreter finalization the
/// handle is forgotten instead (CPython reclaims everything anyway and
/// `Python::attach` would panic).
#[cfg(feature = "cuda")]
struct PyKeepalive(std::mem::ManuallyDrop<Py<PyAny>>);

#[cfg(feature = "cuda")]
impl PyKeepalive {
    fn new(obj: Py<PyAny>) -> Self {
        Self(std::mem::ManuallyDrop::new(obj))
    }
}

#[cfg(feature = "cuda")]
impl Drop for PyKeepalive {
    fn drop(&mut self) {
        // SAFETY: we own the handle inside ManuallyDrop and drop it exactly once.
        let keepalive = unsafe { std::mem::ManuallyDrop::take(&mut self.0) };
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|_py| drop(keepalive));
        } else {
            std::mem::forget(keepalive);
        }
    }
}

/// Import a device-resident DLPack tensor (torch / cupy) into a shared
/// [`DeviceImage`] handle — the core behind `Image.from_dlpack` (device
/// inference) and `Image.cuda.from_dlpack`.
///
/// Accepts a 3-D C-contiguous `(H, W, C)` CUDA tensor — uint8 `C∈{1,3,4}` or
/// float32 `C∈{1,3}`. `copy=True` (default): device-to-device into an owned
/// buffer, synchronized before return. `copy=False`: zero-copy alias that keeps
/// `obj` alive for the image's lifetime.
#[cfg(feature = "cuda")]
pub(crate) fn dlpack_to_device_arc(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<Arc<Inner>> {
    use dlpack_rs::ffi::{DLManagedTensor, DLManagedTensorVersioned};
    use pyo3::types::{PyCapsule, PyCapsuleMethods, PyDict};
    use std::ffi::CStr;

    // Probe the protocol's device query first: host tensors get the helpful
    // redirect before any `__dlpack__` call (some producers, e.g. torch-CPU,
    // reject the `stream` kwarg with errors other than TypeError).
    if let Ok(dev) = obj.call_method0("__dlpack_device__") {
        let (ty, _id): (u32, i32) = dev.extract()?;
        if ty != dlpack_rs::ffi::DLDeviceType::kDLCUDA {
            return Err(PyValueError::new_err(
                "from_dlpack: tensor is not on a CUDA device; \
                 for host tensors use Image.from_numpy or Image.from_dlpack",
            ));
        }
    }

    // Per the DLPack protocol `stream=1` is CUDA's legacy default stream — the
    // one this module launches on — so a compliant producer (torch, cupy)
    // makes the data stream-ordered against our copy below. Fall back for
    // producers that reject the newer keywords.
    let capsule_obj = {
        let kwargs = PyDict::new(py);
        kwargs.set_item("stream", 1i64)?;
        kwargs.set_item("max_version", (1u32, 0u32))?;
        // Retry with fewer keywords only on TypeError (pre-spec producer
        // rejecting the kwarg); any other error is the producer's real
        // failure and is surfaced as-is.
        obj.call_method("__dlpack__", (), Some(&kwargs))
            .or_else(|e| {
                if !e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    return Err(e);
                }
                let kwargs = PyDict::new(py);
                kwargs.set_item("stream", 1i64)?;
                obj.call_method("__dlpack__", (), Some(&kwargs))
            })
            .or_else(|e| {
                if !e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    return Err(e);
                }
                obj.call_method0("__dlpack__")
            })?
    };
    let capsule: Bound<'_, PyCapsule> = capsule_obj.cast_into()?;

    const NAME_DL: &CStr = c"dltensor";
    const NAME_DLV: &CStr = c"dltensor_versioned";
    let cap_name = capsule.name()?;
    let name_cstr: &CStr = match &cap_name {
        Some(n) => unsafe { n.as_cstr() },
        None => {
            return Err(PyValueError::new_err(
                "from_dlpack: DLPack capsule has no name",
            ))
        }
    };
    // Borrow the DLTensor; the capsule stays alive for this whole function and
    // its destructor (which runs the producer's deleter) fires on normal GC —
    // correct here because we copy rather than take ownership.
    let t: &dlpack_rs::ffi::DLTensor = if name_cstr == NAME_DL {
        let nn = capsule.pointer_checked(Some(NAME_DL))?;
        unsafe { &(*(nn.as_ptr() as *const DLManagedTensor)).dl_tensor }
    } else if name_cstr == NAME_DLV {
        let nn = capsule.pointer_checked(Some(NAME_DLV))?;
        unsafe { &(*(nn.as_ptr() as *const DLManagedTensorVersioned)).dl_tensor }
    } else {
        return Err(PyValueError::new_err(format!(
            "from_dlpack: unexpected capsule name {name_cstr:?}"
        )));
    };

    if t.device.device_type != dlpack_rs::ffi::DLDeviceType::kDLCUDA {
        return Err(PyValueError::new_err(
            "from_dlpack: tensor is not on a CUDA device (device_type != kDLCUDA); \
             for host tensors use Image.from_numpy or Image.from_dlpack",
        ));
    }
    // Resolve the stream from the TENSOR's own device, not a hardcoded device 0
    // — a multi-GPU producer's data must be imported (and, for copy=false,
    // later operated on) through its actual device's context, or CUDA ops
    // against it are undefined / target the wrong GPU.
    let stream = default_stream_for(t.device.device_id)?;
    crate::dlpack::validate_dlpack_rank(t.ndim, t.shape)?;
    let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
    if shape.len() != 3 || shape.iter().any(|&d| d <= 0) {
        return Err(PyValueError::new_err(format!(
            "from_dlpack: expected a 3-D (H, W, C) tensor with positive dims, got {shape:?}"
        )));
    }
    if !t.strides.is_null() {
        let strides = unsafe { std::slice::from_raw_parts(t.strides, 3) };
        let expect = [shape[1] * shape[2], shape[2], 1];
        if strides != expect {
            return Err(PyValueError::new_err(
                "from_dlpack: tensor is not C-contiguous; call .contiguous() first",
            ));
        }
    }
    let (h, w, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    // Reject a shape whose byte extent would overflow `usize` up front, so no
    // downstream `h * w * C * size_of` (device alloc here, or the D2H buffer in
    // `device::dl_owned`) can silently wrap to an undersized allocation. Guard
    // with the largest supported itemsize (f32 = 4B) so it holds for any dtype.
    crate::backing::byte_len(h, w, c, crate::backing::Dtype::F32)?;
    let ptr = t.data as u64 + t.byte_offset;

    /// Materialize the producer's `h*w*C` elements at `ptr` as a device image:
    /// an owned copy, or (copy=false) a zero-copy alias kept valid by `obj`.
    fn dl_image<T, const C: usize>(
        stream: &Arc<CudaStream>,
        ptr: u64,
        h: usize,
        w: usize,
        copy: bool,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<Image<T, C>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static,
    {
        let n = h * w * C;
        // SAFETY: ptr/len come from the live DLPack tensor validated above.
        let src = unsafe { stream.upgrade_device_ptr::<T>(ptr, n) };
        if !copy {
            // Zero-copy: the foreign tensor leaks the aliasing slice on drop
            // and releases the producer handle instead of freeing.
            return Ok(Image(Tensor::from_foreign_cudaslice(
                src,
                [h, w, C],
                stream.clone(),
                Box::new(PyKeepalive::new(obj.clone().unbind())),
            )));
        }
        // Owned copy into an uninitialized buffer (fully overwritten by the
        // copy); the borrowed alias is `leak()`ed so the producer's
        // allocation is never freed by us.
        let owned = unsafe { stream.alloc::<T>(n) }
            .map_err(err)
            .and_then(|mut owned| {
                stream.memcpy_dtod(&src, &mut owned).map_err(err)?;
                Ok(owned)
            });
        src.leak();
        let owned = owned?;
        // The producer may free its buffer as soon as we return.
        stream.synchronize().map_err(err)?;
        Ok(Image(Tensor::from_cudaslice(
            owned,
            [h, w, C],
            stream.clone(),
        )))
    }

    use dlpack_rs::ffi::{K_DL_FLOAT, K_DL_UINT};
    let inner = match (t.dtype.code, t.dtype.bits, t.dtype.lanes, c) {
        (code, 8, 1, 1) if code == K_DL_UINT => {
            Inner::U8C1(dl_image(&stream, ptr, h, w, copy, obj)?)
        }
        (code, 8, 1, 3) if code == K_DL_UINT => {
            Inner::U8C3(dl_image(&stream, ptr, h, w, copy, obj)?)
        }
        (code, 8, 1, 4) if code == K_DL_UINT => {
            Inner::U8C4(dl_image(&stream, ptr, h, w, copy, obj)?)
        }
        (code, 32, 1, 1) if code == K_DL_FLOAT => {
            Inner::F32C1(dl_image(&stream, ptr, h, w, copy, obj)?)
        }
        (code, 32, 1, 3) if code == K_DL_FLOAT => {
            Inner::F32C3(dl_image(&stream, ptr, h, w, copy, obj)?)
        }
        (code, bits, lanes, c) => {
            return Err(PyValueError::new_err(format!(
                "from_dlpack: unsupported dtype (code {code}, {bits} bits, {lanes} lanes) \
                 with {c} channels — expected uint8 C∈{{1,3,4}} or float32 C∈{{1,3}}"
            )))
        }
    };
    Ok(Arc::new(inner))
}

// ── Color conversions (device-resident unified `Image` in and out) ────────────
//
// Genuinely device-only — wrapped in its own module so the whole span can be
// `#[cfg(feature = "cuda")]`-gated with one attribute instead of one per item,
// then re-exported so `crate::cuda_ext::gray_from_rgb` etc. (used by
// `crate::color`'s dispatch) keeps resolving unchanged.
#[cfg(feature = "cuda")]
mod cuda_color {
    use super::*;

/// PIL-style mode string for a device output of channel count `channels`.
fn device_mode<T: 'static>(channels: usize) -> String {
    let dt = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        crate::backing::Dtype::F32
    } else {
        crate::backing::Dtype::U8
    };
    crate::image::mode_for_dtype(dt, channels)
}

/// Borrow a `&PyImageApi` as its device source variant, or raise a clear error.
macro_rules! device_src {
    ($img:expr, $pyname:expr, $srcvar:ident) => {{
        let dev = $img.as_device().ok_or_else(|| {
            PyValueError::new_err(concat!(
                $pyname,
                ": expected a device Image (on a CUDA device); for a host image pass \
                 its numpy array, or move it to a device with .to_cuda(stream)"
            ))
        })?;
        let Inner::$srcvar(src) = dev else {
            return Err(PyValueError::new_err(concat!(
                $pyname,
                ": wrong input dtype/channels for this conversion"
            )));
        };
        src
    }};
}

/// The stream a device source image lives on, for allocating a same-device
/// destination. Falls back to device 0's default stream if the backing carries
/// no stream (shouldn't normally happen for a typed device image).
fn source_stream<T, const C: usize>(src: &Image<T, C>) -> PyResult<Arc<CudaStream>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static,
{
    match src.cuda_stream() {
        Some(s) => Ok(s.clone()),
        None => default_stream(),
    }
}

/// Allocate a device destination and run one `ConvertColor` pair, returning a
/// device-resident unified `Image` tagged with the output color space.
macro_rules! conv_fn {
    ($(#[$meta:meta])* $pyname:ident, $srcvar:ident, $snt:ty, $t:ty, $dc:literal, $dvar:ident, $dnt:ty, $dcs:expr) => {
        $(#[$meta])*
        #[pyfunction]
        pub fn $pyname(img: &PyImageApi) -> PyResult<PyImageApi> {
            let src = device_src!(img, stringify!($pyname), $srcvar);
            // Allocate the destination on the SOURCE's own stream/device, not a
            // hardcoded device 0: a source resident on a non-default GPU would
            // otherwise get a device-0 dst and the downstream residency check
            // (`pair_residency`) would reject the mismatched pair with
            // `DeviceMismatch` on multi-GPU systems.
            let stream = source_stream(src)?;
            // SAFETY: the ConvertColor kernel writes every output pixel (one
            // thread per pixel, bounds-guarded), so the uninitialized destination
            // is fully overwritten before any read.
            let mut dst = unsafe { Image::<$t, $dc>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
            convert_pair!(src, $snt, &mut dst, $dnt)?;
            Ok(PyImageApi::from_device(
                Inner::$dvar(dst),
                $dcs,
                device_mode::<$t>($dc),
            ))
        }
    };
}

conv_fn!(
    /// RGB8 → Gray8 (BT.601, bit-exact vs the CPU path).
    gray_from_rgb, U8C3, Rgb8, u8, 1, U8C1, Gray8, ColorSpace::Gray
);
conv_fn!(
    /// Gray8 → RGB8 broadcast.
    rgb_from_gray, U8C1, Gray8, u8, 3, U8C3, Rgb8, ColorSpace::Rgb
);
conv_fn!(
    /// RGB8 → BGR8 channel swap (symmetric).
    bgr_from_rgb, U8C3, Rgb8, u8, 3, U8C3, Bgr8, ColorSpace::Bgr
);
conv_fn!(
    /// RGB8 → RGBA8 (opaque alpha).
    rgba_from_rgb, U8C3, Rgb8, u8, 4, U8C4, Rgba8, ColorSpace::Rgba
);
// RGBA8/BGRA8 → RGB8 are provided by `rgb_from_rgba_bg`/`rgb_from_bgra_bg`
// (below) instead of `conv_fn!`, so the device path can honor the `background`
// alpha-composite argument like the CPU path rather than only dropping alpha.
conv_fn!(
    /// RGB8 → YCbCr8 (full-range Q14, bit-exact vs the CPU path).
    ycbcr_from_rgb, U8C3, Rgb8, u8, 3, U8C3, YCbCr8, ColorSpace::YCbCr
);
conv_fn!(
    /// YCbCr8 → RGB8.
    rgb_from_ycbcr, U8C3, YCbCr8, u8, 3, U8C3, Rgb8, ColorSpace::Rgb
);
conv_fn!(
    /// RGB f32 → HSV f32 (kornia conventions, [0,255] scale).
    hsv_from_rgb, F32C3, Rgbf32, f32, 3, F32C3, Hsvf32, ColorSpace::Hsv
);
conv_fn!(
    /// HSV f32 → RGB f32.
    rgb_from_hsv, F32C3, Hsvf32, f32, 3, F32C3, Rgbf32, ColorSpace::Rgb
);
conv_fn!(
    /// RGB f32 → CIE Lab f32 (RGB in [0,1], L in [0,100]).
    lab_from_rgb, F32C3, Rgbf32, f32, 3, F32C3, Labf32, ColorSpace::Lab
);
conv_fn!(
    /// Lab f32 → RGB f32.
    rgb_from_lab, F32C3, Labf32, f32, 3, F32C3, Rgbf32, ColorSpace::Rgb
);

/// Sepia tone on RGB8 (Q8 fixed point, bit-exact vs the CPU path).
#[pyfunction]
pub fn sepia_from_rgb(img: &PyImageApi) -> PyResult<PyImageApi> {
    let src = device_src!(img, "sepia_from_rgb", U8C3);
    // Allocate the destination on the source's own stream/device (see `conv_fn!`).
    let stream = source_stream(src)?;
    // SAFETY: the sepia kernel writes every output pixel, so the uninitialized
    // destination is fully overwritten before any read.
    let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    color::sepia_from_rgb_u8(src, &mut dst).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::U8C3(dst),
        ColorSpace::Rgb,
        device_mode::<u8>(3),
    ))
}

/// Apply one of the 21 OpenCV colormaps to a Gray8 image (name as in
/// `kornia_rs.imgproc.apply_colormap`).
#[pyfunction]
pub fn apply_colormap(img: &PyImageApi, colormap: &str) -> PyResult<PyImageApi> {
    let src = device_src!(img, "apply_colormap", U8C1);
    let cmap = color::ColormapType::from_name(colormap)
        .ok_or_else(|| PyValueError::new_err(format!("unknown colormap '{colormap}'")))?;
    // Allocate the destination on the source's own stream/device (see `conv_fn!`).
    let stream = source_stream(src)?;
    // SAFETY: the colormap kernel writes every output pixel, so the uninitialized
    // destination is fully overwritten before any read.
    let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    color::apply_colormap(src, &mut dst, cmap).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::U8C3(dst),
        ColorSpace::Rgb,
        device_mode::<u8>(3),
    ))
}

/// Bayer demosaic (pattern: "rggb" | "bggr" | "grbg" | "gbrg").
#[pyfunction]
pub fn rgb_from_bayer(img: &PyImageApi, pattern: &str) -> PyResult<PyImageApi> {
    use kornia_image::color_spaces::BayerPattern;
    let src = device_src!(img, "rgb_from_bayer", U8C1);
    // Allocate the destination on the source's own stream/device (see `conv_fn!`).
    let stream = source_stream(src)?;
    let pat = match pattern.to_ascii_lowercase().as_str() {
        "rggb" => BayerPattern::Rggb,
        "bggr" => BayerPattern::Bggr,
        "grbg" => BayerPattern::Grbg,
        "gbrg" => BayerPattern::Gbrg,
        p => {
            return Err(PyValueError::new_err(format!(
                "unknown bayer pattern '{p}'"
            )))
        }
    };
    // SAFETY: the bayer demosaic kernel writes every output pixel (bounds-guarded,
    // replicate-border reads), so the uninitialized destination is fully
    // overwritten before any read.
    let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    color::rgb_from_bayer(src, pat, &mut dst).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::U8C3(dst),
        ColorSpace::Rgb,
        device_mode::<u8>(3),
    ))
}

/// `{Rgba8,Bgra8}8 → RGB8` on device with optional `background` alpha
/// compositing (mirrors the CPU path's `background=` — otherwise a device image
/// would silently drop alpha while the host path composited; `None` = opaque
/// alpha drop). The two only differ by source newtype, so share this body.
macro_rules! conv_bg_fn {
    ($(#[$meta:meta])* $pyname:ident, $snt:ty, $errname:literal) => {
        $(#[$meta])*
        pub fn $pyname(img: &PyImageApi, background: Option<[u8; 3]>) -> PyResult<PyImageApi> {
            let src = device_src!(img, $errname, U8C4);
            let stream = source_stream(src)?;
            // SAFETY: the blend/drop kernel writes every output pixel (one
            // thread per pixel, bounds-guarded), so the uninitialized
            // destination is fully overwritten before any read.
            let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
            convert_pair_bg!(src, $snt, &mut dst, Rgb8, background)?;
            Ok(PyImageApi::from_device(
                Inner::U8C3(dst),
                ColorSpace::Rgb,
                device_mode::<u8>(3),
            ))
        }
    };
}
conv_bg_fn!(
    /// RGBA8 → RGB8 on device, compositing over `background` when given.
    rgb_from_rgba_bg, Rgba8, "rgb_from_rgba"
);
conv_bg_fn!(
    /// BGRA8 → RGB8 on device, compositing over `background` when given.
    rgb_from_bgra_bg, Bgra8, "rgb_from_bgra"
);

} // mod cuda_color
#[cfg(feature = "cuda")]
pub(crate) use cuda_color::*;

// ── Tensor (model input) ──────────────────────────────────────────────────────
//
// Mirrors `kornia_tensor::Tensor<T, N>` as its own Python type instead of a
// bespoke "CudaTensor" — `Image` is for 2-D HWC pixel data (with color-space
// semantics); this is for the N-D device/host arrays (e.g. the preprocessor's
// `[N, C, H, W]` model input) that don't fit that shape. Currently scoped to
// the CUDA preprocessor's 4-D f32/f16 output; a fuller rank/dtype-generic
// binding can grow from here.

enum TensorInnerEnum {
    F32(Tensor<f32, 4>),
    F16(Tensor<half::f16, 4>),
}

/// Bind the inner typed `Tensor` of either dtype variant as `$t` and evaluate
/// `$body` — collapses the identical 2-arm `F32 | F16` match that every
/// type-agnostic accessor would otherwise spell out (mirrors `device.rs`'s
/// `dispatch!`). Only usable where `$body` type-checks for both element types.
macro_rules! tdispatch {
    ($self:expr, $t:ident => $body:expr) => {
        match &*$self.inner {
            TensorInnerEnum::F32($t) => $body,
            TensorInnerEnum::F16($t) => $body,
        }
    };
}

/// A device-resident `[N, C, H, W]` tensor — the preprocessor's output, i.e.
/// model input. Feed it to torch/TensorRT zero-copy via `__dlpack__`, or
/// `.numpy()` an f32 copy.
#[pyclass(name = "Tensor", frozen, module = "kornia_rs")]
pub struct PyTensor {
    inner: Arc<TensorInnerEnum>,
}

impl PyTensor {
    /// True when the tensor is device-resident. Cheap; the single residency
    /// source of truth every other method here checks.
    fn is_device(&self) -> bool {
        use kornia_tensor::MemoryDomain;
        let domain = tdispatch!(self, t => t.storage.domain());
        matches!(domain, MemoryDomain::Device { .. } | MemoryDomain::Unified { .. })
    }

    /// This tensor's real CUDA device ordinal (not always 0) when device- or
    /// unified-resident; `None` on a host tensor. Reads the residency straight
    /// from `storage` (like `DeviceImage::device_id`) rather than hopping
    /// through the stream, so it's correct even for a stream-less device/unified
    /// buffer. Single source of truth for `device()`/`__dlpack_device__`.
    #[cfg(feature = "cuda")]
    fn device_ordinal(&self) -> Option<i32> {
        use kornia_tensor::MemoryDomain;
        let (domain, device_id) =
            tdispatch!(self, t => (t.storage.domain(), t.storage.device_id()));
        matches!(
            domain,
            MemoryDomain::Device { .. } | MemoryDomain::Unified { .. }
        )
        .then_some(device_id)
    }
}

#[pymethods]
impl PyTensor {
    /// Tensor shape `(N, C, H, W)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let s = tdispatch!(self, t => t.shape);
        (s[0], s[1], s[2], s[3])
    }

    /// Element dtype: `"float32"` or `"float16"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        match &*self.inner {
            TensorInnerEnum::F32(_) => "float32",
            TensorInnerEnum::F16(_) => "float16",
        }
    }

    /// Device this tensor lives on: `"cpu"` or `"cuda:{id}"` (the ordinal of
    /// the `Preprocessor` that produced it, not always 0 — see its `device=`
    /// constructor argument).
    #[getter]
    fn device(&self) -> String {
        #[cfg(feature = "cuda")]
        if let Some(ordinal) = self.device_ordinal() {
            return format!("cuda:{ordinal}");
        }
        "cpu".to_string()
    }

    /// Raw device pointer (`CUdeviceptr` as an integer) to the contiguous
    /// `[N, C, H, W]` buffer. Hand it straight to `context.set_tensor_address()`
    /// for a zero-copy TensorRT input binding. Valid while this `Tensor` is
    /// alive; never dereference it on the host.
    #[getter]
    fn data_ptr(&self) -> usize {
        tdispatch!(self, t => t.as_ptr() as usize)
    }

    /// The [CUDA Array Interface] (v3) for zero-copy sharing with CuPy / Numba /
    /// nvidia `cuda-python` (and, via them, TensorRT). The `stream` entry carries
    /// the producing stream so a consumer can order its work after ours.
    ///
    /// Present on a device tensor only; raises `AttributeError` on a host one
    /// (matching `Image`).
    ///
    /// [CUDA Array Interface]: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    #[getter]
    fn __cuda_array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        #[cfg(not(feature = "cuda"))]
        let _ = py;
        if !self.is_device() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "__cuda_array_interface__ is only present on a device Tensor",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            use pyo3::types::PyDict;
            let (shape, typestr, ptr, stream) = match &*self.inner {
                TensorInnerEnum::F32(t) => (
                    t.shape,
                    "<f4",
                    t.as_ptr() as usize,
                    t.cuda_stream().map(|s| s.cu_stream() as usize),
                ),
                TensorInnerEnum::F16(t) => (
                    t.shape,
                    "<f2",
                    t.as_ptr() as usize,
                    t.cuda_stream().map(|s| s.cu_stream() as usize),
                ),
            };
            let d = PyDict::new(py);
            d.set_item("shape", (shape[0], shape[1], shape[2], shape[3]))?;
            d.set_item("typestr", typestr)?;
            d.set_item("data", (ptr, false))?;
            // C-contiguous NCHW — `strides = None` per the interface.
            d.set_item("strides", py.None())?;
            d.set_item("version", 3)?;
            d.set_item("stream", crate::image::cai_stream_value(py, stream))?;
            Ok(d.into_any().unbind())
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!("is_device() is always false without the cuda feature")
    }

    /// Numpy array of the buffer, as float32 (f16 tensors are widened). A
    /// device tensor is copied back to host (D2H) first; a host f32 tensor is
    /// exposed as a zero-copy view (this `Tensor` as the keep-alive base) — no
    /// copy of the raw bytes beyond the f16->f32 widen when applicable.
    fn numpy<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let me = slf.borrow();
        // The four (residency x dtype) cases, spelled out explicitly: only
        // host+F32 is a true zero-copy view, the other three all copy.
        if !me.is_device() {
            if let TensorInnerEnum::F32(t) = &*me.inner {
                let base: Py<PyAny> = slf.clone().into_any().unbind();
                // SAFETY: `t.as_ptr()` points to `t.shape.iter().product()` live
                // f32 elements for as long as `self.inner` (the base) is alive.
                return Ok(unsafe {
                    crate::numpy_view::view::<f32>(
                        py,
                        t.as_ptr() as *mut u8,
                        &t.shape,
                        base,
                        false,
                    )
                }?
                .unbind());
            }
        }
        let (data, shape): (Vec<f32>, [usize; 4]) = match &*me.inner {
            // Unreachable without `cuda`: `me.is_device()` is always false, so
            // F32 always took the zero-copy view above, and F16's `is_device()`
            // guard below never matches.
            #[cfg(feature = "cuda")]
            TensorInnerEnum::F32(t) => {
                let host = t.to_host_owned().map_err(err)?;
                (host.as_slice().to_vec(), t.shape)
            }
            #[cfg(not(feature = "cuda"))]
            TensorInnerEnum::F32(_) => unreachable!("host F32 already returned above"),
            #[cfg(feature = "cuda")]
            TensorInnerEnum::F16(t) if me.is_device() => {
                let host = t.to_host_owned().map_err(err)?;
                (widen_f16_to_f32(host.as_slice()), t.shape)
            }
            TensorInnerEnum::F16(t) => (widen_f16_to_f32(t.as_slice()), t.shape),
        };
        let arr = PyArray1::from_vec(py, data);
        let arr = arr.reshape(shape)?;
        Ok(arr.into_any().unbind())
    }

    /// DLPack device tuple: `(kDLCUDA, device_id)` or `(kDLCPU, 0)`. `device_id`
    /// is this tensor's real CUDA ordinal (not always 0) so a multi-GPU
    /// consumer (e.g. `torch.from_dlpack`) selects the right device.
    fn __dlpack_device__(&self) -> (i32, i32) {
        #[cfg(feature = "cuda")]
        if let Some(ordinal) = self.device_ordinal() {
            return (dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32, ordinal);
        }
        (dlpack_rs::ffi::DLDeviceType::kDLCPU as i32, 0)
    }

    /// Export as a DLPack capsule (zero-copy). Synchronizes the producing
    /// stream first so the consumer sees completed data on any stream.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<isize>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = dl_device;
        if copy == Some(true) {
            return Err(PyValueError::new_err("copy=True is not supported"));
        }
        // Fence the consumer against this tensor's own producing stream (same
        // policy as Image::__dlpack__). A host tensor has no device work to
        // order against — nothing to fence. `is_device()` is always false
        // without the `cuda` feature, so there's nothing to gate here besides
        // the otherwise-unused `stream` param.
        #[cfg(feature = "cuda")]
        if self.is_device() {
            let launch = tdispatch!(self, t => t.cuda_stream().cloned());
            match launch {
                Some(s) => dlpack_fence_consumer(&s, stream)?,
                None => dlpack_fence_consumer(&default_stream()?, stream)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = stream;
        let dl_device = self.__dlpack_device__();
        use kornia_tensor::dlpack::DlpackElem;
        // f16: kDLFloat (code 2), 16 bits — half::f16 is IEEE binary16.
        let f16_dtype = dlpack_rs::ffi::DLDataType {
            code: 2,
            bits: 16,
            lanes: 1,
        };
        // Consumers advertising DLPack v1.0+ get the versioned capsule; older
        // ones fall back to the unversioned "dltensor". Output is writable and
        // not a copy, so flags = 0.
        let versioned = max_version.is_some_and(|(maj, _)| maj >= 1);
        match &*self.inner {
            TensorInnerEnum::F32(t) => {
                let dt = f32::dl_dtype();
                if versioned {
                    arc_dlpack_capsule_versioned(py, self.inner.clone(), t, dt, 0, dl_device)
                } else {
                    arc_dlpack_capsule(py, self.inner.clone(), t, dt, dl_device)
                }
            }
            TensorInnerEnum::F16(t) => {
                if versioned {
                    arc_dlpack_capsule_versioned(
                        py,
                        self.inner.clone(),
                        t,
                        f16_dtype,
                        0,
                        dl_device,
                    )
                } else {
                    arc_dlpack_capsule(py, self.inner.clone(), t, f16_dtype, dl_device)
                }
            }
        }
    }
}

// ── Preprocessor ─────────────────────────────────────────────────────────────

/// The device-only half of [`PyPreprocessor`]'s state — absent for a CPU
/// (`device=None`) instance.
#[cfg(feature = "cuda")]
struct CudaBacking {
    stream: Arc<CudaStream>,
    /// Persistent upload staging: one page-locked host buffer + one device
    /// buffer per batch slot, grown on demand and reused across calls.
    /// Page-locking (`cuMemHostAlloc`) is a syscall far too expensive for a
    /// frame loop, and pinned memory turns the H2D copy into a straight DMA
    /// instead of a bounce through the driver's pageable staging buffer.
    staging: std::sync::Mutex<Staging>,
}

/// Copy a single frame into `cuda`'s shared pinned staging buffer, after
/// draining the previous call's in-flight H2D upload. Shared by the
/// single-frame and `out=`-in-place bodies of `run`. Returns the locked
/// staging guard (drop it, or take `&mut *guard`, before entering
/// `py.detach`) and the frame length.
#[cfg(feature = "cuda")]
fn stage_single_upload<'a>(
    cuda: &'a CudaBacking,
    frame: &numpy::PyReadonlyArray1<'_, u8>,
) -> PyResult<(std::sync::MutexGuard<'a, Staging>, usize)> {
    let frame_len = frame.len();
    let mut staging = cuda.staging.lock().expect("staging mutex poisoned");
    // Drain the prior call's H2D BEFORE ensure() — ensure() may free/realloc
    // the shared pinned buffer on a size increase, and that host free
    // (cuMemFreeHost) is not ordered against an in-flight upload reading it.
    // Waiting first also protects the copy_from_slice overwrite below.
    staging.wait_prev_upload()?;
    staging.ensure(&cuda.stream, 1, frame_len)?;
    staging.pinned.as_mut().expect("ensured").as_slice_mut()[..frame_len]
        .copy_from_slice(frame.as_slice()?);
    Ok((staging, frame_len))
}

/// Fused camera preprocessing: raw frame → normalized `[1, 3, H, W]` CHW
/// [`Tensor`] in one pass. `device=<ordinal>` (default 0) runs on the GPU —
/// every format (NV12/YUYV/RGB/BGR/RGBA/BGRA/Gray), fused color-decode +
/// resize + normalize in **one kernel launch**. `device=None` runs on the
/// CPU via the crate's SIMD resize+normalize kernel — interleaved formats
/// only (`rgb`/`bgr`/`rgba`/`bgra`; there is no CPU fused decoder for
/// `gray`/`nv12`/`yuyv`, which raise a clear error) and always f32 output
/// (`f16=True` casts the result afterward, since there is no CPU f16 kernel).
#[pyclass(name = "Preprocessor", frozen, module = "kornia_rs")]
pub struct PyPreprocessor {
    pre: Preprocessor,
    #[cfg(feature = "cuda")]
    cuda: Option<CudaBacking>,
    f16: bool,
    format: SourceFormat,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct Staging {
    /// Page-locked host buffer holding all frames of a call back-to-back.
    pinned: Option<Tensor<u8, 1>>,
    /// Device destination per batch slot.
    device: Vec<cudarc::driver::CudaSlice<u8>>,
    /// Event recorded right after the previous call's H2D upload. The next call
    /// must host-wait it before overwriting `pinned`, otherwise the plain host
    /// `copy_from_slice` (not stream-ordered) clobbers page-locked bytes while
    /// the prior async `memcpy_htod` is still draining — corrupting that frame's
    /// device input. Waiting only the upload keeps the kernel + consumer fence
    /// asynchronous.
    upload_done: Option<cudarc::driver::CudaEvent>,
}

#[cfg(feature = "cuda")]
impl Staging {
    /// Block the host until the previous call's H2D upload has completed, so the
    /// shared pinned buffer is safe to overwrite. No-op on the first call.
    fn wait_prev_upload(&mut self) -> PyResult<()> {
        if let Some(ev) = self.upload_done.take() {
            ev.synchronize().map_err(err)?;
        }
        Ok(())
    }

    /// Record an upload-complete event on `stream` (after the H2D copies) for the
    /// next call to wait on.
    fn mark_upload(&mut self, stream: &Arc<CudaStream>) -> PyResult<()> {
        self.upload_done = Some(stream.record_event(None).map_err(err)?);
        Ok(())
    }

    /// Grow (never shrink) to hold `slots` frames of `frame_len` bytes and
    /// return the pinned host slice covering all of them.
    fn ensure(&mut self, stream: &Arc<CudaStream>, slots: usize, frame_len: usize) -> PyResult<()> {
        let total = slots * frame_len;
        if self.pinned.as_ref().is_none_or(|p| p.numel() < total) {
            self.pinned =
                Some(kornia_tensor::zeros_pinned::<u8, 1>([total], stream.context()).map_err(err)?);
        }
        while self.device.len() < slots {
            self.device
                .push(stream.alloc_zeros::<u8>(frame_len).map_err(err)?);
        }
        for d in &mut self.device[..slots] {
            if d.len() < frame_len {
                *d = stream.alloc_zeros::<u8>(frame_len).map_err(err)?;
            }
        }
        Ok(())
    }
}

fn parse_format(s: &str) -> PyResult<SourceFormat> {
    SourceFormat::from_name(s)
        .ok_or_else(|| PyValueError::new_err(format!("unknown source format '{s}'")))
}

/// After work is enqueued on `launch`, make a consumer stream wait on its
/// completion (record an event, `cuStreamWaitEvent`) so the caller's subsequent
/// work — e.g. `execute_async_v3(their_stream)` — is ordered after the
/// preprocess without a host sync. No-op when `consumer` is `None`.
#[cfg(feature = "cuda")]
pub(crate) fn fence_stream_into(launch: &Arc<CudaStream>, consumer: Option<usize>) -> PyResult<()> {
    let Some(h) = consumer else { return Ok(()) };
    let ev = launch.record_event(None).map_err(err)?;
    // SAFETY: `h` is the caller's live `CUstream`; the wait is enqueued before
    // `ev` drops and CUDA keeps the event alive until the wait completes.
    unsafe {
        cudarc::driver::sys::cuStreamWaitEvent(h as cudarc::driver::sys::CUstream, ev.cu_event(), 0)
    }
    .result()
    .map_err(err)
}

impl PyPreprocessor {
    /// The `CudaBacking` or a clear error — for the paths that stay
    /// device-only for now (`alloc_output`, and `run`'s batch/`out=` cases;
    /// unlike the single-frame CPU case, these exist to avoid per-frame
    /// host/device allocation in a GPU serving loop, a concern that doesn't
    /// apply to a CPU preprocessor).
    #[cfg(feature = "cuda")]
    fn cuda_backing(&self, method: &str) -> PyResult<&CudaBacking> {
        self.cuda.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{method}: CPU Preprocessor (device=None) only supports run(); \
                 construct with device=<ordinal> for batching/serving-loop methods"
            ))
        })
    }

    /// CPU branch of [`run`](Self::run): interleaved formats only
    /// (`rgb`/`bgr`/`rgba`/`bgra` — there is no CPU fused decoder for
    /// `gray`/`nv12`/`yuyv`, so those raise here); always runs the crate's
    /// f32 SIMD kernel (`self.f16` widens/narrows the result afterward — no
    /// fused CPU f16 path exists).
    fn run_cpu(
        &self,
        py: Python<'_>,
        frame: numpy::PyReadonlyArray1<'_, u8>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
    ) -> PyResult<PyTensor> {
        let c = match self.format {
            SourceFormat::Rgb8 | SourceFormat::Bgr8 => 3,
            SourceFormat::Rgba8 | SourceFormat::Bgra8 => 4,
            fmt => {
                return Err(PyValueError::new_err(format!(
                    "run: CPU Preprocessor (device=None) has no kernel for format \
                     {fmt:?}; pass device=<ordinal> for GPU, or construct with an \
                     rgb/bgr/rgba/bgra format"
                )))
            }
        };
        let frame_slice = frame.as_slice()?;
        // Checked (like `backing::byte_len`): an unchecked `width * height * c`
        // could wrap for adversarial dims and pass the too-small guard below,
        // then `Image::from_raw_parts` (which does NOT validate len vs shape)
        // would sample far past the buffer. u8 itemsize is 1, so this byte
        // length is exactly the required element count.
        let expected = crate::backing::byte_len(height, width, c, crate::backing::Dtype::U8)?;
        if frame_slice.len() < expected {
            return Err(PyValueError::new_err(format!(
                "run: frame too small: got {} bytes, need >= {expected} ({height}x{width}x{c})",
                frame_slice.len(),
            )));
        }
        // Raw pointers aren't Send; carry it across the detach boundary as an
        // address and cast back inside (the numpy borrow — and thus the bytes
        // at this address — outlives the whole call).
        let ptr_addr = frame_slice.as_ptr() as usize;
        py.detach(|| -> PyResult<PyTensor> {
            macro_rules! run_typed {
                ($c:literal) => {{
                    // SAFETY: `ptr_addr`/`expected` describe exactly
                    // `height*width*C` live host bytes for the duration of this
                    // call (the numpy borrow that produced `frame_slice`
                    // outlives it).
                    let img = unsafe {
                        kornia_image::Image::<u8, $c>::from_raw_parts(
                            kornia_image::ImageSize { width, height },
                            ptr_addr as *const u8,
                            expected,
                            kornia_image::allocator::host_alloc(),
                        )
                    }
                    .map_err(err)?;
                    let mut dst = kornia_tensor::Tensor::<f32, 4>::zeros([1, 3, out_height, out_width]);
                    self.pre.run::<$c>(&img, &mut dst).map_err(err)?;
                    dst
                }};
            }
            let dst = match c {
                3 => run_typed!(3),
                4 => run_typed!(4),
                _ => unreachable!("c is only ever 3 or 4, checked above"),
            };
            let inner = if self.f16 {
                let data: Vec<half::f16> =
                    dst.as_slice().iter().map(|&v| half::f16::from_f32(v)).collect();
                let dst16 = kornia_tensor::Tensor::<half::f16, 4>::from_shape_vec(dst.shape, data)
                    .map_err(err)?;
                TensorInnerEnum::F16(dst16)
            } else {
                TensorInnerEnum::F32(dst)
            };
            Ok(PyTensor {
                inner: Arc::new(inner),
            })
        })
    }
}

#[pymethods]
impl PyPreprocessor {
    /// Build a preprocessor. On a device build (`device=<ordinal>`, default 0)
    /// this compiles one CUDA kernel per output dtype at construction; the
    /// source format is a runtime kernel argument, so reuse the instance
    /// across frames (and formats need no recompilation).
    ///
    /// mode: "letterbox" | "stretch"; format: "rgb"|"bgr"|"rgba"|"bgra"|
    /// "gray"|"nv12"|"yuyv" (the last three need `device` — no CPU fused
    /// decoder); sampling: "bilinear"|"nearest"|"lanczos"; f16: half output
    /// (device only — a CPU build always produces f32; see `run`); mean/std:
    /// optional per-channel normalization; pad_value: letterbox padding byte
    /// (default 114); device: CUDA device ordinal (default 0), or `None` to
    /// build a CPU preprocessor.
    #[new]
    #[pyo3(signature = (mode = "letterbox", format = "rgb", sampling = "bilinear", f16 = false, mean = None, std = None, pad_value = 114, device = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mode: &str,
        format: &str,
        sampling: &str,
        f16: bool,
        mean: Option<[f32; 3]>,
        std: Option<[f32; 3]>,
        pad_value: u8,
        device: Option<i32>,
    ) -> PyResult<Self> {
        let src_fmt = parse_format(format)?;
        let mut builder = Preprocessor::builder()
            .mode(match mode.to_ascii_lowercase().as_str() {
                "letterbox" => ResizeMode::Letterbox,
                "stretch" => ResizeMode::Stretch,
                m => return Err(PyValueError::new_err(format!("unknown mode '{m}'"))),
            })
            .source_format(src_fmt)
            .sampling(crate::image::parse_interpolation(sampling)?)
            .pad_value(pad_value);
        if mean.is_some() || std.is_some() {
            builder = builder.normalize(Normalize::MeanStd {
                mean: mean.unwrap_or([0.0; 3]),
                std: std.unwrap_or([1.0; 3]),
            });
        }
        match device {
            Some(dev) => {
                #[cfg(feature = "cuda")]
                {
                    let stream = default_stream_for(dev)?;
                    let pre = builder.build_cuda(stream.clone()).map_err(err)?;
                    Ok(Self {
                        pre,
                        cuda: Some(CudaBacking {
                            stream,
                            staging: std::sync::Mutex::new(Staging::default()),
                        }),
                        f16,
                        format: src_fmt,
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = dev;
                    Err(PyValueError::new_err(
                        "Preprocessor(device=<ordinal>): CUDA support is not compiled in \
                         (build kornia-rs with the 'cuda' feature); pass device=None for CPU",
                    ))
                }
            }
            None => {
                let pre = builder.build().map_err(err)?;
                #[cfg(feature = "cuda")]
                {
                    Ok(Self {
                        pre,
                        cuda: None,
                        f16,
                        format: src_fmt,
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(Self {
                        pre,
                        f16,
                        format: src_fmt,
                    })
                }
            }
        }
    }

    /// Preprocess one raw frame, or a batch of same-sized frames, into a
    /// `[N, 3, out_h, out_w]` [`Tensor`] (`N=1` for a single frame). The one
    /// entry point for this preprocessor:
    ///
    /// - `frame`: a flat 1-D `uint8` array (single frame, in the constructor's
    ///   format layout) **or** a list of them (batch — one fused kernel launch
    ///   per frame, all on the same stream, one sync for the whole batch;
    ///   device build only, no CPU batch path).
    /// - `out`: an existing [`Tensor`] (e.g. from a prior `run()`, or
    ///   [`alloc_output`](Self::alloc_output)) to write into **in place**
    ///   instead of allocating — for a fixed inference-engine input binding in
    ///   a serving loop, call this every frame with no per-call allocation.
    ///   Must be single-frame shape `[1, 3, H, W]` matching this
    ///   preprocessor's dtype; the *same* `Tensor` object is returned. The
    ///   write is asynchronous — do not read or free it until the work
    ///   completes (sync, or pass `stream` and order your consumer after it).
    ///   Device build only; incompatible with a batch `frame`.
    /// - `stream`: device build only — an optional consumer `Stream` (e.g.
    ///   your TensorRT execution stream, adopted via `Stream.from_handle`) to
    ///   fence the output into, so `execute_async_v3` on it is ordered after
    ///   this preprocess with no host sync.
    ///
    /// A CPU preprocessor (`device=None`) only supports the single-frame,
    /// fresh-output case (`out=None`); always produces `float32` output
    /// (`f16=True` casts it afterward — no CPU f16 kernel); and only the
    /// interleaved formats (`rgb`/`bgr`/`rgba`/`bgra` — no CPU fused decoder
    /// for `gray`/`nv12`/`yuyv`).
    #[pyo3(signature = (frame, width, height, out_height, out_width, out = None, stream = None))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        frame: &Bound<'_, PyAny>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
        out: Option<Py<PyTensor>>,
        stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<Py<PyTensor>> {
        if let Ok(single) = frame.extract::<numpy::PyReadonlyArray1<'_, u8>>() {
            if let Some(out_py) = out {
                #[cfg(feature = "cuda")]
                {
                    let out_ref = out_py.bind(py).borrow();
                    self.run_into_impl(
                        py, out_ref, single, width, height, out_height, out_width, stream,
                    )?;
                    return Ok(out_py);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (out_py, out_height, out_width);
                    return Err(cuda_not_compiled());
                }
            }
            let t = self.run_single(py, single, width, height, out_height, out_width, stream)?;
            return Py::new(py, t);
        }
        if out.is_some() {
            return Err(PyValueError::new_err(
                "run: out= only supports a single-frame frame argument, not a batch list",
            ));
        }
        #[cfg(feature = "cuda")]
        if let Ok(frames) = frame.extract::<Vec<numpy::PyReadonlyArray1<'_, u8>>>() {
            let t = self.run_batch_impl(py, frames, width, height, out_height, out_width, stream)?;
            return Py::new(py, t);
        }
        #[cfg(not(feature = "cuda"))]
        if frame.extract::<Vec<numpy::PyReadonlyArray1<'_, u8>>>().is_ok() {
            return Err(cuda_not_compiled());
        }
        Err(PyValueError::new_err(
            "run: frame must be a 1-D uint8 numpy array (single frame) or a list of them \
             (batch, needs the 'cuda' feature)",
        ))
    }

    /// Allocate a zero-initialized output [`Tensor`] of shape
    /// `[batch, 3, out_height, out_width]`, dtype following this preprocessor's
    /// `f16` flag. Preallocate one and reuse it across frames via `run(...,
    /// out=...)` for an allocation-free serving loop. Device build only.
    #[pyo3(signature = (out_height, out_width, batch = 1))]
    fn alloc_output(
        &self,
        py: Python<'_>,
        out_height: usize,
        out_width: usize,
        batch: usize,
    ) -> PyResult<PyTensor> {
        #[cfg(feature = "cuda")]
        {
            let cuda = self.cuda_backing("alloc_output")?;
            let shape = [batch, 3, out_height, out_width];
            let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
                if self.f16 {
                    Ok(TensorInnerEnum::F16(
                        kornia_tensor::zeros_cuda::<half::f16, 4>(shape, &cuda.stream).map_err(err)?,
                    ))
                } else {
                    Ok(TensorInnerEnum::F32(
                        kornia_tensor::zeros_cuda::<f32, 4>(shape, &cuda.stream).map_err(err)?,
                    ))
                }
            })?;
            Ok(PyTensor {
                inner: Arc::new(inner),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (py, out_height, out_width, batch);
            Err(cuda_not_compiled())
        }
    }
}

impl PyPreprocessor {
    /// Single-frame, fresh-output body of [`run`](Self::run).
    #[allow(clippy::too_many_arguments)]
    fn run_single(
        &self,
        py: Python<'_>,
        frame: numpy::PyReadonlyArray1<'_, u8>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
        stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<PyTensor> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            self.run_cpu(py, frame, width, height, out_height, out_width)
        }
        #[cfg(feature = "cuda")]
        {
        let Some(cuda) = &self.cuda else {
            // A CPU preprocessor has no device work to fence, so a consumer
            // `stream` is meaningless — reject it rather than silently ignoring
            // it and leaving the caller to believe their stream was ordered.
            if stream.is_some() {
                return Err(PyValueError::new_err(
                    "run: stream= is device-only; a CPU Preprocessor (device=None) \
                     has no stream to fence",
                ));
            }
            return self.run_cpu(py, frame, width, height, out_height, out_width);
        };
        let consumer = stream.map(|s| s.raw_handle());
        let (mut staging, frame_len) = stage_single_upload(cuda, &frame)?;
        let staging = &mut *staging;

        // Everything past the numpy borrow runs without the GIL: the pinned
        // H2D DMA, the fused kernel launch, and the output allocation.
        let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
            cuda.stream
                .memcpy_htod(
                    &staging.pinned.as_ref().expect("ensured").as_slice()[..frame_len],
                    &mut staging.device[0],
                )
                .map_err(err)?;
            // Record upload-complete so the next call can host-wait just this DMA.
            staging.mark_upload(&cuda.stream)?;
            let d_src = &staging.device[0];
            // SAFETY: the resize/normalize kernel writes every output element —
            // the sampled region and the letterbox pad border (one thread per
            // pixel, bounds-guarded) — so the uninitialized dst is fully
            // overwritten before it is read.
            let shape = [1, 3, out_height, out_width];
            if self.f16 {
                let mut dst = unsafe {
                    kornia_tensor::uninit_cuda::<half::f16, 4>(shape, &cuda.stream)
                }
                .map_err(err)?;
                self.pre
                    .run_raw_f16(d_src, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F16(dst))
            } else {
                let mut dst =
                    unsafe { kornia_tensor::uninit_cuda::<f32, 4>(shape, &cuda.stream) }
                        .map_err(err)?;
                self.pre
                    .run_raw(d_src, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F32(dst))
            }
        })?;
        fence_stream_into(&cuda.stream, consumer)?;
        Ok(PyTensor {
            inner: Arc::new(inner),
        })
        } // cfg(feature = "cuda")
    }

    /// Batch body of [`run`](Self::run): one fused kernel launch per frame,
    /// all on the same stream, one sync for the whole batch (multi-camera
    /// rigs, batched engines). Output dtype follows `self.f16`. Device only.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_batch_impl(
        &self,
        py: Python<'_>,
        frames: Vec<numpy::PyReadonlyArray1<'_, u8>>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
        stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<PyTensor> {
        if frames.is_empty() {
            return Err(PyValueError::new_err("run: batch needs at least one frame"));
        }
        let cuda = self.cuda_backing("run (batch)")?;
        let consumer = stream.map(|s| s.raw_handle());
        let n = frames.len();
        let frame_len = frames[0].len();
        let mut staging = cuda.staging.lock().expect("staging mutex poisoned");
        // Drain the prior call's H2D BEFORE ensure() (which may free/realloc the
        // shared pinned buffer) and before we overwrite it below.
        staging.wait_prev_upload()?;
        staging.ensure(&cuda.stream, n, frame_len)?;
        {
            let pinned = staging.pinned.as_mut().expect("ensured").as_slice_mut();
            for (i, f) in frames.iter().enumerate() {
                let src = f.as_slice()?;
                if src.len() != frame_len {
                    return Err(PyValueError::new_err(
                        "run: batch frames must all have the same length",
                    ));
                }
                pinned[i * frame_len..(i + 1) * frame_len].copy_from_slice(src);
            }
        }
        let staging = &mut *staging;

        let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
            // Enqueue every frame's H2D. If one fails mid-loop, earlier copies are
            // already in-flight DMAs reading `pinned`, so we still record the
            // upload event (below) before propagating — otherwise the next call
            // would skip its wait and could free/reuse `pinned` under those DMAs.
            let copies = {
                let pinned = staging.pinned.as_ref().expect("ensured").as_slice();
                let mut res = Ok(());
                for (i, d) in staging.device[..n].iter_mut().enumerate() {
                    if let Err(e) = cuda
                        .stream
                        .memcpy_htod(&pinned[i * frame_len..(i + 1) * frame_len], d)
                    {
                        res = Err(err(e));
                        break;
                    }
                }
                res
            };
            // Record upload-complete (gates whatever DMAs were enqueued) BEFORE
            // surfacing a partial-copy error, so the next call always waits them.
            staging.mark_upload(&cuda.stream)?;
            copies?;
            let refs: Vec<_> = staging.device[..n].iter().collect();
            let shape = [n, 3, out_height, out_width];
            // SAFETY: run_raw_batch writes every element of all N output planes
            // (per-pixel, bounds-guarded, pad border included), so the
            // uninitialized dst is fully overwritten before it is read.
            if self.f16 {
                let mut dst = unsafe { kornia_tensor::uninit_cuda::<half::f16, 4>(shape, &cuda.stream) }
                    .map_err(err)?;
                self.pre
                    .run_raw_batch_f16(&refs, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F16(dst))
            } else {
                let mut dst = unsafe { kornia_tensor::uninit_cuda::<f32, 4>(shape, &cuda.stream) }
                    .map_err(err)?;
                self.pre
                    .run_raw_batch(&refs, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F32(dst))
            }
        })?;
        fence_stream_into(&cuda.stream, consumer)?;
        Ok(PyTensor {
            inner: Arc::new(inner),
        })
    }

    /// `out=`-in-place body of [`run`](Self::run). Device only.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_into_impl(
        &self,
        py: Python<'_>,
        out: PyRef<'_, PyTensor>,
        frame: numpy::PyReadonlyArray1<'_, u8>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
        stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<()> {
        // Validate the destination matches this preprocessor's dtype & shape.
        let out_shape = match &*out.inner {
            TensorInnerEnum::F32(t) => {
                if self.f16 {
                    return Err(PyValueError::new_err(
                        "run: out= is float32 but this preprocessor is f16",
                    ));
                }
                t.shape
            }
            TensorInnerEnum::F16(t) => {
                if !self.f16 {
                    return Err(PyValueError::new_err(
                        "run: out= is float16 but this preprocessor is float32",
                    ));
                }
                t.shape
            }
        };
        if out_shape[0] != 1 || out_shape[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "run: out= must be a single-frame [1, 3, H, W] tensor, got {out_shape:?} \
                 (pass a list of frames without out= for a batch)"
            )));
        }
        let (out_h, out_w) = (out_shape[2], out_shape[3]);
        // The requested output size must match `out`'s shape — otherwise the
        // `out_height`/`out_width` args would be silently ignored in favor of
        // whatever `out` was allocated as, producing wrong-resolution output.
        if (out_h, out_w) != (out_height, out_width) {
            return Err(PyValueError::new_err(format!(
                "run: out_height/out_width ({out_height}x{out_width}) do not match the out= \
                 tensor's shape ({out_h}x{out_w}); they must agree"
            )));
        }
        let cuda = self.cuda_backing("run (out=)")?;
        // `out.data_ptr()` is written by the fused CUDA kernel as a raw device
        // pointer, so `out` MUST be device-resident and on this preprocessor's
        // device — a host tensor's address (or a different GPU's) would launch
        // a kernel through an invalid pointer (undefined behavior; on unified
        // memory a host VA can even alias real device memory). dtype/shape
        // matching alone does not establish residency.
        let pre_ordinal = cuda.stream.context().ordinal() as i32;
        match out.device_ordinal() {
            Some(ord) if ord == pre_ordinal => {}
            Some(ord) => {
                return Err(PyValueError::new_err(format!(
                    "run: out= is on cuda:{ord} but this preprocessor runs on \
                     cuda:{pre_ordinal}; allocate out with alloc_output() on the same device"
                )));
            }
            None => {
                return Err(PyValueError::new_err(
                    "run: out= must be a device Tensor (e.g. from alloc_output()), got a host tensor",
                ));
            }
        }
        let out_ptr = out.data_ptr() as u64;
        let consumer = stream.map(|s| s.raw_handle());

        let (mut staging, frame_len) = stage_single_upload(cuda, &frame)?;
        let staging = &mut *staging;

        py.detach(|| -> PyResult<()> {
            cuda.stream
                .memcpy_htod(
                    &staging.pinned.as_ref().expect("ensured").as_slice()[..frame_len],
                    &mut staging.device[0],
                )
                .map_err(err)?;
            // Record upload-complete so the next call can host-wait just this DMA.
            staging.mark_upload(&cuda.stream)?;
            let d_src = &staging.device[0];
            let n_elem = 3 * out_h * out_w;
            let shape = [1, 3, out_h, out_w];
            // Wrap the caller's device buffer as a non-owning destination tensor:
            // on drop the aliasing slice is leaked (never freed) — `out` keeps
            // ownership and frees it.
            // SAFETY: `out_ptr`/`n_elem` come from the live `out` Tensor of
            // exactly this shape & dtype, kept alive for this whole call.
            macro_rules! run_into_foreign {
                ($ty:ty, $run:ident) => {{
                    let slice =
                        unsafe { cuda.stream.upgrade_device_ptr::<$ty>(out_ptr, n_elem) };
                    let mut dst = Tensor::from_foreign_cudaslice(
                        slice,
                        shape,
                        cuda.stream.clone(),
                        Box::new(()),
                    );
                    self.pre.$run(d_src, width, height, &mut dst).map_err(err)?;
                }};
            }
            if self.f16 {
                run_into_foreign!(half::f16, run_raw_f16);
            } else {
                run_into_foreign!(f32, run_raw);
            }
            Ok(())
        })?;
        fence_stream_into(&cuda.stream, consumer)?;
        Ok(())
    }
}

// ── module registration ─────────────────────────────────────────────────────

/// Register `kornia_rs.cuda`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "cuda")?;
    m.add_function(wrap_pyfunction!(is_available, &m)?)?;
    m.add_function(wrap_pyfunction!(mem_get_info, &m)?)?;
    // Color conversions are no longer exposed here — they live behind the
    // residency-dispatching `kornia_rs.imgproc.*` ops (which route a device
    // `Image` to these same device kernels). The functions below stay
    // `pub(crate)` and are called by `crate::color`.
    crate::add_imagenet_consts(&m)?;
    m.add_class::<crate::image::PyStream>()?;
    parent.add_submodule(&m)?;
    // Make `import kornia_rs.cuda` work (mirror of the other submodules).
    py.import("sys")?
        .getattr("modules")?
        .set_item("kornia_rs.cuda", &m)?;
    Ok(())
}
