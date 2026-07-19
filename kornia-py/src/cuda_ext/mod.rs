//! `Tensor`/`Preprocessor` (CPU/GPU) plus, on a `cuda`-feature build, GPU color
//! conversions and the fused camera-preprocessing device kernels.
//!
//! `Tensor` and `Preprocessor` (`stream=None`, CPU) work on every build. The
//! `cuda` cargo feature adds device residency: `Preprocessor(stream=<Stream>)`, the
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
#[cfg(feature = "cuda")]
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use kornia_image::color_spaces::{
    Bgr8, Bgra8, Gray8, Grayf32, Hlsf32, Hsvf32, Labf32, LinearRgbf32, Luvf32, Rgb8, Rgba8, Rgbf32,
    Xyzf32, YCbCr8, YCbCrf32, Yuv8, Yuvf32,
};
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
    if let Some(&ord) = cache
        .lock()
        .expect("stream-device cache poisoned")
        .get(&handle)
    {
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
pub(crate) fn dlpack_fence_consumer(
    launch: &Arc<CudaStream>,
    consumer: Option<isize>,
) -> PyResult<()> {
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

// ── DLPack capsule pack/export + device import ────────────────────────────────
// `arc_dlpack_capsule*` are always compiled (host Tensor export via DLPack);
// `dlpack_to_device_arc` is `cuda`-gated (device import).
mod capsule;
#[cfg(feature = "cuda")]
pub(crate) use capsule::dlpack_to_device_arc;
use capsule::{arc_dlpack_capsule, arc_dlpack_capsule_versioned};

// ── GPU color conversions (device-resident `Image` in and out) ────────────────
// Self-contained device-only submodule; re-exported so `crate::cuda_ext::<op>`
// (used by `crate::color`'s residency dispatch) keeps resolving unchanged.
/// PIL-style mode string for a device output of channel count `channels`.
#[cfg(feature = "cuda")]
pub(crate) fn device_mode<T: 'static>(channels: usize) -> String {
    let dt = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        crate::backing::Dtype::F32
    } else {
        crate::backing::Dtype::U8
    };
    crate::image::mode_for_dtype(dt, channels)
}

/// The stream a device source image lives on, for allocating a same-device
/// destination. Falls back to device 0's default stream if the backing carries
/// no stream (shouldn't normally happen for a typed device image).
#[cfg(feature = "cuda")]
pub(crate) fn source_stream<T, const C: usize>(src: &Image<T, C>) -> PyResult<Arc<CudaStream>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static,
{
    match src.cuda_stream() {
        Some(s) => Ok(s.clone()),
        None => default_stream(),
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda_canny;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_ccl;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_clahe;
#[cfg(feature = "cuda")]
mod cuda_color;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_filter;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_geometry;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_histogram;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_morphology;
#[cfg(feature = "cuda")]
pub(crate) use cuda_canny as canny_dev;
#[cfg(feature = "cuda")]
pub(crate) use cuda_ccl as ccl_dev;
#[cfg(feature = "cuda")]
pub(crate) use cuda_clahe as clahe_dev;
#[cfg(feature = "cuda")]
pub(crate) use cuda_color::*;
#[cfg(feature = "cuda")]
pub(crate) use cuda_filter as filter;
#[cfg(feature = "cuda")]
pub(crate) use cuda_geometry as geometry;
#[cfg(feature = "cuda")]
pub(crate) use cuda_histogram as histogram;
#[cfg(feature = "cuda")]
pub(crate) use cuda_morphology as morphology;

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
        matches!(
            domain,
            MemoryDomain::Device { .. } | MemoryDomain::Unified { .. }
        )
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
                    crate::numpy_view::view::<f32>(py, t.as_ptr() as *mut u8, &t.shape, base, false)
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
                    arc_dlpack_capsule_versioned(py, self.inner.clone(), t, f16_dtype, 0, dl_device)
                } else {
                    arc_dlpack_capsule(py, self.inner.clone(), t, f16_dtype, dl_device)
                }
            }
        }
    }
}

// ── Preprocessor ─────────────────────────────────────────────────────────────

/// The device-only half of [`PyPreprocessor`]'s state — absent for a CPU
/// (`stream=None`, CPU) instance.
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
    frame: &[u8],
) -> PyResult<(std::sync::MutexGuard<'a, Staging>, usize)> {
    let frame_len = frame.len();
    let mut staging = cuda.staging.lock().expect("staging mutex poisoned");
    // Drain the prior call's H2D BEFORE ensure() — ensure() may free/realloc
    // the shared pinned buffer on a size increase, and that host free
    // (cuMemFreeHost) is not ordered against an in-flight upload reading it.
    // Waiting first also protects the copy_from_slice overwrite below.
    staging.wait_prev_upload()?;
    staging.ensure(&cuda.stream, 1, frame_len)?;
    staging.pinned.as_mut().expect("ensured").as_slice_mut()[..frame_len].copy_from_slice(frame);
    Ok((staging, frame_len))
}

/// Fused camera preprocessing: raw frame → normalized `[1, 3, H, W]` CHW
/// [`Tensor`] in one pass. `stream=<Stream>` runs on the GPU (that stream's device) —
/// every format (NV12/YUYV/RGB/BGR/RGBA/BGRA/Gray), fused color-decode +
/// resize + normalize in **one kernel launch**. `stream=None` runs on the
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
                "{method}: CPU Preprocessor (stream=None) only supports run(); \
                 construct with stream=<Stream> for batching/serving-loop methods"
            ))
        })
    }

    /// Host-`Image` branch of [`run`](Self::run) on **non-CUDA builds**: with no
    /// device residency, every `Image` is host-backed, so validate it against
    /// the passed dims and run the CPU kernel. (On CUDA builds this case is
    /// handled inside `run_image`'s host branch, which is compiled out here
    /// along with the rest of the device path.)
    #[cfg(not(feature = "cuda"))]
    fn run_host_image_cpu(
        &self,
        py: Python<'_>,
        img: &crate::image::PyImageApi,
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
                    "run: an Image input is only supported for rgb/bgr/rgba/bgra formats, not \
                     {fmt:?}; pass raw bytes for gray/nv12/yuyv"
                )))
            }
        };
        if img.dtype != crate::backing::Dtype::U8 {
            return Err(PyValueError::new_err(
                "run: host Image must be uint8 for the fused preprocessor input",
            ));
        }
        let (ih, iw, ic) = img.shape_hwc();
        if (iw, ih, ic) != (width, height, c) {
            return Err(PyValueError::new_err(format!(
                "run: Image is {ih}x{iw}x{ic} but run(width={width}, height={height}) with format \
                 {:?} expects {height}x{width}x{c}",
                self.format
            )));
        }
        self.run_cpu(py, img.u8_elems(), width, height, out_height, out_width)
    }

    /// CPU branch of [`run`](Self::run): interleaved formats only
    /// (`rgb`/`bgr`/`rgba`/`bgra` — there is no CPU fused decoder for
    /// `gray`/`nv12`/`yuyv`, so those raise here); always runs the crate's
    /// f32 SIMD kernel (`self.f16` widens/narrows the result afterward — no
    /// fused CPU f16 path exists).
    fn run_cpu(
        &self,
        py: Python<'_>,
        frame_slice: &[u8],
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
                    "run: CPU Preprocessor (stream=None) has no kernel for format \
                     {fmt:?}; construct with stream=<Stream> for GPU, or an \
                     rgb/bgr/rgba/bgra format"
                )))
            }
        };
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
                    let mut dst =
                        kornia_tensor::Tensor::<f32, 4>::zeros([1, 3, out_height, out_width]);
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
                let data: Vec<half::f16> = dst
                    .as_slice()
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
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
    /// Build a preprocessor. `stream` is the device selector, exactly like the
    /// rest of the API (`Image.to_cuda(stream)`, `Image.zeros(stream=)`) and
    /// mirroring Rust's `PreprocessorBuilder::build_cuda(stream)`:
    ///
    /// - `stream=None` → **CPU** preprocessor (mirrors Rust `build()`).
    /// - `stream=<Stream>` → **GPU** on that stream's device; the fused CUDA
    ///   kernel is compiled once here (the source format is a runtime kernel
    ///   argument, so reuse the instance across frames and formats).
    ///
    /// mode: "letterbox" | "stretch"; format: "rgb"|"bgr"|"rgba"|"bgra"|
    /// "gray"|"nv12"|"yuyv" (the last three need a GPU `stream` — no CPU fused
    /// decoder); sampling: "bilinear"|"nearest"|"lanczos"; f16: half output
    /// (GPU only — a CPU preprocessor always produces f32; see `run`); mean/std:
    /// optional per-channel normalization; pad_value: letterbox padding byte
    /// (default 114).
    #[new]
    #[pyo3(signature = (mode = "letterbox", format = "rgb", sampling = "bilinear", f16 = false, mean = None, std = None, pad_value = 114, stream = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mode: &str,
        format: &str,
        sampling: &str,
        f16: bool,
        mean: Option<[f32; 3]>,
        std: Option<[f32; 3]>,
        pad_value: u8,
        stream: Option<PyRef<'_, crate::image::PyStream>>,
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
        match stream {
            // GPU: the stream selects the device (mirrors Rust build_cuda(stream)).
            Some(s) => {
                #[cfg(feature = "cuda")]
                {
                    let launch = crate::image::resolve_stream(Some(s))?.launch;
                    let pre = builder.build_cuda(launch.clone()).map_err(err)?;
                    Ok(Self {
                        pre,
                        cuda: Some(CudaBacking {
                            stream: launch,
                            staging: std::sync::Mutex::new(Staging::default()),
                        }),
                        f16,
                        format: src_fmt,
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = s;
                    Err(PyValueError::new_err(
                        "Preprocessor(stream=<Stream>): CUDA support is not compiled in \
                         (build kornia-rs with the 'cuda' feature); pass stream=None for CPU",
                    ))
                }
            }
            // CPU (mirrors Rust build()).
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
    /// - `consumer_stream`: GPU only — an optional consumer `Stream` (e.g. your
    ///   TensorRT execution stream, adopted via `Stream.from_handle`) to fence
    ///   the output into, so `execute_async_v3` on it is ordered after this
    ///   preprocess with no host sync. (Distinct from the constructor `stream`,
    ///   which selects the *device* this preprocessor launches on.)
    ///
    /// `frame` also accepts an already-decoded interleaved `Image`
    /// (`rgb`/`bgr`/`rgba`/`bgra`) instead of raw bytes: a **device** `Image` on
    /// this preprocessor's device feeds the fused kernel its device buffer
    /// directly (zero-copy input, no H2D), a **host** `Image` is uploaded like
    /// raw bytes. Its `(width, height)` must match the ones passed.
    ///
    /// A CPU preprocessor (`stream=None`) only supports the single-frame,
    /// fresh-output case (`out=None`); always produces `float32` output
    /// (`f16=True` casts it afterward — no CPU f16 kernel); and only the
    /// interleaved formats (`rgb`/`bgr`/`rgba`/`bgra` — no CPU fused decoder
    /// for `gray`/`nv12`/`yuyv`).
    #[pyo3(signature = (frame, width, height, out_height, out_width, out = None, consumer_stream = None))]
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
        consumer_stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<Py<PyTensor>> {
        // An already-decoded interleaved `Image` (device or host) — dispatch on
        // its residency, feeding a same-device buffer to the kernel zero-copy.
        if let Ok(img) = frame.extract::<PyRef<'_, crate::image::PyImageApi>>() {
            #[cfg(feature = "cuda")]
            {
                let t = self.run_image(
                    py,
                    &img,
                    width,
                    height,
                    out_height,
                    out_width,
                    out,
                    consumer_stream,
                )?;
                return Ok(t);
            }
            // Non-CUDA build: no device residency exists, so every Image is
            // host-backed — run its bytes on the CPU (mirrors run_image's host
            // branch, which is compiled out with the rest of the device path).
            #[cfg(not(feature = "cuda"))]
            {
                let t = self.run_host_image_cpu(py, &img, width, height, out_height, out_width)?;
                return Py::new(py, t);
            }
        }
        if let Ok(single) = frame.extract::<numpy::PyReadonlyArray1<'_, u8>>() {
            if let Some(out_py) = out {
                #[cfg(feature = "cuda")]
                {
                    let out_ref = out_py.bind(py).borrow();
                    self.run_into_impl(
                        py,
                        out_ref,
                        single,
                        width,
                        height,
                        out_height,
                        out_width,
                        consumer_stream,
                    )?;
                    return Ok(out_py);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (out_py, out_height, out_width);
                    return Err(cuda_not_compiled());
                }
            }
            let t = self.run_single(
                py,
                single,
                width,
                height,
                out_height,
                out_width,
                consumer_stream,
            )?;
            return Py::new(py, t);
        }
        if out.is_some() {
            return Err(PyValueError::new_err(
                "run: out= only supports a single-frame frame argument, not a batch list",
            ));
        }
        #[cfg(feature = "cuda")]
        if let Ok(frames) = frame.extract::<Vec<numpy::PyReadonlyArray1<'_, u8>>>() {
            let t = self.run_batch_impl(
                py,
                frames,
                width,
                height,
                out_height,
                out_width,
                consumer_stream,
            )?;
            return Py::new(py, t);
        }
        #[cfg(not(feature = "cuda"))]
        if frame
            .extract::<Vec<numpy::PyReadonlyArray1<'_, u8>>>()
            .is_ok()
        {
            return Err(cuda_not_compiled());
        }
        Err(PyValueError::new_err(
            "run: frame must be a 1-D uint8 numpy array (single frame), a list of them \
             (batch), or an interleaved Image",
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
                        kornia_tensor::zeros_cuda::<half::f16, 4>(shape, &cuda.stream)
                            .map_err(err)?,
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
            self.run_cpu(py, frame.as_slice()?, width, height, out_height, out_width)
        }
        #[cfg(feature = "cuda")]
        {
            let Some(cuda) = &self.cuda else {
                // A CPU preprocessor has no device work to fence, so a consumer
                // `stream` is meaningless — reject it rather than silently ignoring
                // it and leaving the caller to believe their stream was ordered.
                if stream.is_some() {
                    return Err(PyValueError::new_err(
                        "run: consumer_stream= is device-only; a CPU Preprocessor \
                     (stream=None) has no stream to fence",
                    ));
                }
                return self.run_cpu(py, frame.as_slice()?, width, height, out_height, out_width);
            };
            let consumer = stream.map(|s| s.raw_handle());
            let (mut staging, frame_len) = stage_single_upload(cuda, frame.as_slice()?)?;
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
                    let mut dst =
                        unsafe { kornia_tensor::uninit_cuda::<half::f16, 4>(shape, &cuda.stream) }
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
                let mut dst =
                    unsafe { kornia_tensor::uninit_cuda::<half::f16, 4>(shape, &cuda.stream) }
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

        let (mut staging, frame_len) = stage_single_upload(cuda, frame.as_slice()?)?;
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
                    let slice = unsafe { cuda.stream.upgrade_device_ptr::<$ty>(out_ptr, n_elem) };
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

    /// `run` body for an already-decoded interleaved `Image` (see [`run`](Self::run)):
    /// a **device** `Image` on this preprocessor's device feeds the fused kernel
    /// its device buffer directly (zero-copy input, no H2D); a **host** `Image`
    /// is uploaded/run like raw bytes. Single-frame, fresh output only.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_image(
        &self,
        py: Python<'_>,
        img: &crate::image::PyImageApi,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
        out: Option<Py<PyTensor>>,
        consumer_stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<Py<PyTensor>> {
        if out.is_some() {
            return Err(PyValueError::new_err(
                "run: out= is not supported with an Image input; pass raw bytes for the \
                 in-place serving path",
            ));
        }
        // Interleaved u8 formats only; the Image's dims must match the passed ones.
        let c = match self.format {
            SourceFormat::Rgb8 | SourceFormat::Bgr8 => 3,
            SourceFormat::Rgba8 | SourceFormat::Bgra8 => 4,
            fmt => {
                return Err(PyValueError::new_err(format!(
                    "run: an Image input is only supported for rgb/bgr/rgba/bgra formats, \
                     not {fmt:?}; pass raw bytes for gray/nv12/yuyv"
                )))
            }
        };
        let (ih, iw, ic) = img.shape_hwc();
        if (iw, ih, ic) != (width, height, c) {
            return Err(PyValueError::new_err(format!(
                "run: Image is {ih}x{iw}x{ic} but run(width={width}, height={height}) with \
                 format {:?} expects {height}x{width}x{c}",
                self.format
            )));
        }

        match img.as_device() {
            // DEVICE Image: feed its device buffer straight to the fused kernel.
            Some(dev) => {
                let cuda = self.cuda_backing("run (device Image)")?;
                let pre_ord = cuda.stream.context().ordinal() as i32;
                if dev.device_id() != pre_ord {
                    return Err(PyValueError::new_err(format!(
                        "run: device Image is on cuda:{} but this preprocessor runs on \
                         cuda:{pre_ord}",
                        dev.device_id()
                    )));
                }
                if dev.dtype_enum() != crate::backing::Dtype::U8 {
                    return Err(PyValueError::new_err(
                        "run: device Image must be uint8 for the fused preprocessor input",
                    ));
                }
                // Cross-stream fence: if the Image was produced on a different
                // stream than the preprocessor's, make our launch stream wait on
                // the producer's pending writes before the kernel reads the
                // buffer (same event-fence discipline as imgproc's
                // cuda_dispatch). Same stream => already ordered, skip.
                if let Some(src_stream) = dev.cuda_stream() {
                    if !Arc::ptr_eq(src_stream, &cuda.stream) {
                        let ev = src_stream.record_event(None).map_err(err)?;
                        cuda.stream.wait(&ev).map_err(err)?;
                    }
                }
                let consumer = consumer_stream.map(|s| s.raw_handle());
                let src_ptr = dev.as_ptr() as u64;
                let n = width * height * c;
                let shape = [1, 3, out_height, out_width];
                let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
                    // SAFETY: `src_ptr`/`n` describe the device Image's live buffer.
                    // The slice is `leak()`ed (never frees the Image's buffer), and
                    // we synchronize below before returning so the kernel finishes
                    // reading it before the caller can drop the Image — the
                    // zero-copy input has no staging buffer keeping it alive.
                    let src = unsafe { cuda.stream.upgrade_device_ptr::<u8>(src_ptr, n) };
                    let run = || -> PyResult<TensorInnerEnum> {
                        if self.f16 {
                            let mut dst = unsafe {
                                kornia_tensor::uninit_cuda::<half::f16, 4>(shape, &cuda.stream)
                            }
                            .map_err(err)?;
                            self.pre
                                .run_raw_f16(&src, width, height, &mut dst)
                                .map_err(err)?;
                            Ok(TensorInnerEnum::F16(dst))
                        } else {
                            let mut dst = unsafe {
                                kornia_tensor::uninit_cuda::<f32, 4>(shape, &cuda.stream)
                            }
                            .map_err(err)?;
                            self.pre
                                .run_raw(&src, width, height, &mut dst)
                                .map_err(err)?;
                            Ok(TensorInnerEnum::F32(dst))
                        }
                    };
                    let result = run();
                    src.leak();
                    cuda.stream.synchronize().map_err(err)?;
                    result
                })?;
                fence_stream_into(&cuda.stream, consumer)?;
                Py::new(
                    py,
                    PyTensor {
                        inner: Arc::new(inner),
                    },
                )
            }
            // HOST Image: run its bytes on the CPU, or upload + GPU kernel.
            None => {
                if img.dtype != crate::backing::Dtype::U8 {
                    return Err(PyValueError::new_err(
                        "run: host Image must be uint8 for the fused preprocessor input",
                    ));
                }
                let bytes = img.u8_elems();
                let Some(cuda) = &self.cuda else {
                    if consumer_stream.is_some() {
                        return Err(PyValueError::new_err(
                            "run: consumer_stream= is device-only; a CPU Preprocessor has no \
                             stream to fence",
                        ));
                    }
                    let t = self.run_cpu(py, bytes, width, height, out_height, out_width)?;
                    return Py::new(py, t);
                };
                let consumer = consumer_stream.map(|s| s.raw_handle());
                let (mut staging, frame_len) = stage_single_upload(cuda, bytes)?;
                let staging = &mut *staging;
                let shape = [1, 3, out_height, out_width];
                let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
                    cuda.stream
                        .memcpy_htod(
                            &staging.pinned.as_ref().expect("ensured").as_slice()[..frame_len],
                            &mut staging.device[0],
                        )
                        .map_err(err)?;
                    staging.mark_upload(&cuda.stream)?;
                    let d_src = &staging.device[0];
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
                Py::new(
                    py,
                    PyTensor {
                        inner: Arc::new(inner),
                    },
                )
            }
        }
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
    #[cfg(feature = "cuda")]
    m.add_class::<PyGraph>()?;
    parent.add_submodule(&m)?;
    // Make `import kornia_rs.cuda` work (mirror of the other submodules).
    py.import("sys")?
        .getattr("modules")?
        .set_item("kornia_rs.cuda", &m)?;
    Ok(())
}

// ── CUDA Graph capture / replay ───────────────────────────────────────────────

/// A captured CUDA graph: record a fixed sequence of device ops once, replay
/// it per frame with microsecond launch overhead.
///
/// Capture requires the recorded ops to be **allocation-free** — pass
/// preallocated `out=` device images to the geometry ops (allocations inside
/// a capture become graph-owned memory nodes and typically fail with the
/// stream-ordered allocator). The captured topology is fixed: same shapes,
/// same source/destination buffers every replay; update the *contents* of the
/// input image between replays (e.g. `img.copy_from_numpy(...)` or an
/// upstream graph node), not the objects.
#[cfg(feature = "cuda")]
#[pyclass(name = "Graph", module = "kornia_rs.cuda", frozen)]
pub struct PyGraph {
    graph: cudarc::driver::CudaGraph,
    /// Keep-alives: every Python object the captured kernels read or write.
    /// The graph holds raw device pointers; dropping the images would free
    /// the memory under the next replay.
    #[allow(dead_code)]
    retained: Vec<Py<PyAny>>,
}

// SAFETY: CUgraph/CUgraphExec are opaque driver handles; the CUDA driver API
// is thread-safe for graph launch/destroy, and cudarc's CudaGraph carries the
// owning Arc<CudaStream> (itself Send+Sync). The raw pointers are only handed
// back to driver calls, never dereferenced on the host.
#[cfg(feature = "cuda")]
unsafe impl Send for PyGraph {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PyGraph {}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyGraph {
    /// Capture `f()` — a callable that enqueues device ops on `stream` (the
    /// default stream if `None`) — and return a replayable `Graph`.
    ///
    /// `retain` must list every device `Image` the callable reads from or
    /// writes into; the graph keeps them alive for its own lifetime.
    #[staticmethod]
    #[pyo3(signature = (f, retain, stream=None))]
    fn capture(
        py: Python<'_>,
        f: Py<PyAny>,
        retain: Vec<Py<PyAny>>,
        stream: Option<PyRef<'_, crate::image::PyStream>>,
    ) -> PyResult<Self> {
        let arc = match stream {
            Some(s) => s.owned_arc()?,
            None => default_stream()?,
        };
        // cudarc's automatic multi-stream fencing waits on legacy CudaEvents
        // recorded before the capture began, and cuStreamWaitEvent on a
        // pre-capture legacy event invalidates a capture. The check happens at
        // kernel-arg time, so suspending event tracking for the capture window
        // keeps the recorded graph clean. Cross-stream correctness inside
        // kornia ops is unaffected — the residency dispatch does its own
        // explicit event fencing — and drop-safety holds because frees are
        // stream-ordered (cuMemFreeAsync).
        //
        // SAFETY (disable/enable_event_tracking): scoped to the capture; the
        // flag is context-global, so concurrent slice *creation* on other
        // threads during this window would skip tracking — capture is a
        // setup-time operation, documented as such.
        let ctx = arc.context();
        let was_tracking = ctx.is_event_tracking();
        if was_tracking {
            unsafe { ctx.disable_event_tracking() };
        }
        let restore = |on: bool| {
            if on {
                unsafe { ctx.enable_event_tracking() };
            }
        };
        if let Err(e) = arc.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        ) {
            restore(was_tracking);
            return Err(err(e));
        }
        // Run the recording callable. Always end the capture (even on error)
        // so the stream is usable again, then surface the original error.
        let call_result = f.call0(py);
        let graph = arc.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
        );
        restore(was_tracking);
        let graph = graph.map_err(err)?;
        call_result?;
        let graph = graph.ok_or_else(|| {
            PyValueError::new_err(
                "Graph.capture: nothing was captured (the callable enqueued no \
                 device work on this stream)",
            )
        })?;
        Ok(Self {
            graph,
            retained: retain,
        })
    }

    /// Enqueue one replay of the captured work on the capture stream.
    /// Asynchronous — pair with `Stream.synchronize()` (or another replay).
    fn replay(&self) -> PyResult<()> {
        self.graph.launch().map_err(err)
    }
}
