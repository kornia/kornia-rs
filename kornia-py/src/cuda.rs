//! `kornia_rs.cuda` — GPU color conversions and fused camera preprocessing.
//!
//! Enabled by the `cuda` cargo feature. cudarc dynamic-loading means this
//! module compiles everywhere; at runtime [`is_available`] probes for a
//! usable driver and everything degrades gracefully without one.
//!
//! Design: data stays a **[`CudaImage`]** while it is pixels (HWC, typed
//! channels) and becomes a **[`CudaTensor`]** only when it turns into model
//! input (the preprocessor's CHW output). Both export zero-copy to
//! torch/cupy via `__dlpack__`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use kornia_image::color_spaces::{Bgr8, Bgra8, Gray8, Hsvf32, Labf32, Rgb8, Rgba8, Rgbf32, YCbCr8};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::{self, ConvertColor};
use kornia_imgproc::preprocess::{Normalize, Preprocessor, ResizeMode, SourceFormat};
use kornia_tensor::Tensor;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Default-stream handle for device 0 (created lazily, shared per process).
fn default_stream() -> PyResult<Arc<CudaStream>> {
    use std::sync::OnceLock;
    static STREAM: OnceLock<Result<Arc<CudaStream>, String>> = OnceLock::new();
    STREAM
        .get_or_init(|| {
            CudaContext::new(0)
                .map(|ctx| ctx.default_stream())
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(PyRuntimeError::new_err)
}

/// True if a CUDA driver and device 0 are usable in this process.
#[pyfunction]
pub fn is_available() -> bool {
    default_stream().is_ok()
}

/// Make the producer's pending work visible to the DLPack consumer's stream
/// **without blocking the host** whenever the protocol allows.
///
/// Per the array-API `__dlpack__` spec (CUDA): `-1` = consumer wants no
/// sync; `1`/`2` = legacy / per-thread default stream; any other value = a
/// raw `CUstream` handle. This module launches everything on the legacy
/// default stream, so `1`/`2` are already ordered and a foreign stream gets
/// an event fence — modern consumers (torch, cupy) never pay a host block.
/// `None` (spec: producer may assume the legacy default stream) keeps a
/// stricter-than-spec host sync by choice — it only reaches legacy consumers
/// calling `__dlpack__()` bare, where safety beats a blocked host.
fn dlpack_sync_for_consumer(stream: Option<isize>) -> PyResult<()> {
    match stream {
        Some(-1) | Some(1) | Some(2) => Ok(()),
        Some(h) if h > 2 => {
            let ev = default_stream()?.record_event(None).map_err(err)?;
            // SAFETY: `h` is the consumer's live CUstream per the protocol;
            // the wait is enqueued before the event is dropped (legal —
            // CUDA keeps the event alive until the enqueued wait completes).
            unsafe {
                cudarc::driver::sys::cuStreamWaitEvent(
                    h as cudarc::driver::sys::CUstream,
                    ev.cu_event(),
                    0,
                )
            }
            .result()
            .map_err(err)
        }
        _ => default_stream()?.synchronize().map_err(err),
    }
}

// ── CudaImage ────────────────────────────────────────────────────────────────

/// Device-resident pixels, one variant per supported (dtype, channels).
enum Inner {
    U8C1(Image<u8, 1>),
    U8C3(Image<u8, 3>),
    U8C4(Image<u8, 4>),
    F32C1(Image<f32, 1>),
    F32C3(Image<f32, 3>),
}

impl Inner {
    fn size(&self) -> ImageSize {
        match self {
            Inner::U8C1(i) => i.size(),
            Inner::U8C3(i) => i.size(),
            Inner::U8C4(i) => i.size(),
            Inner::F32C1(i) => i.size(),
            Inner::F32C3(i) => i.size(),
        }
    }

    fn channels(&self) -> usize {
        match self {
            Inner::U8C1(_) | Inner::F32C1(_) => 1,
            Inner::U8C3(_) | Inner::F32C3(_) => 3,
            Inner::U8C4(_) => 4,
        }
    }

    fn dtype(&self) -> &'static str {
        match self {
            Inner::U8C1(_) | Inner::U8C3(_) | Inner::U8C4(_) => "uint8",
            Inner::F32C1(_) | Inner::F32C3(_) => "float32",
        }
    }
}

/// A device-resident image (HWC). The device twin of `kornia_rs.image.Image`:
/// created with [`upload`], consumed by the `kornia_rs.cuda` color
/// conversions, brought back with `download()`, or handed zero-copy to
/// torch/cupy via `__dlpack__`.
#[pyclass(name = "CudaImage", frozen, module = "kornia_rs.cuda")]
pub struct PyCudaImage {
    inner: Arc<Inner>,
}

/// View a plain `Image` as its `#[repr(transparent)]` color-space newtype.
///
/// SAFETY: every `define_color_space!` newtype is `#[repr(transparent)]` over
/// `Image<T, C>`, so the pointer casts are layout-sound.
macro_rules! as_newtype {
    ($img:expr, $nt:ty) => {
        unsafe { &*($img as *const _ as *const $nt) }
    };
    (mut $img:expr, $nt:ty) => {
        unsafe { &mut *($img as *mut _ as *mut $nt) }
    };
}

/// Run `src.convert(&mut dst)` through the repr(transparent) newtypes.
macro_rules! convert_pair {
    ($src:expr, $snt:ty, $dst:expr, $dnt:ty) => {{
        let s = as_newtype!($src, $snt);
        let d = as_newtype!(mut $dst, $dnt);
        s.convert(d).map_err(err)
    }};
}

#[pymethods]
impl PyCudaImage {
    /// Image width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.inner.size().width
    }

    /// Image height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.inner.size().height
    }

    /// Channel count (1, 3, or 4).
    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Element dtype: `"uint8"` or `"float32"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype()
    }

    /// Copy the image back to host as an (H, W, C) numpy array.
    fn download<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        fn dl<'py, T>(py: Python<'py>, img: &Image<T, 3>) -> PyResult<Py<PyAny>>
        where
            T: numpy::Element
                + cudarc::driver::DeviceRepr
                + cudarc::driver::ValidAsZeroBits
                + Copy
                + Default
                + 'static,
        {
            let host = img.download().map_err(err)?;
            let (h, w) = (host.height(), host.width());
            let arr = PyArray1::from_slice(py, host.as_slice());
            let arr = arr.reshape([h, w, 3])?;
            Ok(arr.into_any().unbind())
        }
        // 1/4-channel variants share the same shape logic with C != 3.
        macro_rules! dl_c {
            ($img:expr, $c:literal, $t:ty) => {{
                let host = $img.download().map_err(err)?;
                let (h, w) = (host.height(), host.width());
                let arr = PyArray1::from_slice(py, host.as_slice());
                let arr = arr.reshape([h, w, $c])?;
                Ok(arr.into_any().unbind())
            }};
        }
        match &*self.inner {
            Inner::U8C1(i) => dl_c!(i, 1, u8),
            Inner::U8C3(i) => dl(py, i),
            Inner::U8C4(i) => dl_c!(i, 4, u8),
            Inner::F32C1(i) => dl_c!(i, 1, f32),
            Inner::F32C3(i) => dl(py, i),
        }
    }

    /// DLPack device tuple: `(kDLCUDA, device_id)`.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32, 0)
    }

    /// Export as a DLPack capsule (zero-copy; the capsule keeps this image's
    /// buffer alive). `stream`: for safety this synchronizes the producing
    /// stream before export, so the consumer sees completed pixels on any
    /// stream.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<isize>,
        max_version: Option<Py<PyAny>>,
        dl_device: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (max_version, dl_device);
        if copy == Some(true) {
            return Err(PyValueError::new_err("copy=True is not supported"));
        }
        dlpack_sync_for_consumer(stream)?;
        use kornia_tensor::dlpack::DlpackElem;
        match &*self.inner {
            Inner::U8C1(i) => arc_dlpack_capsule(py, self.inner.clone(), &i.0, u8::dl_dtype()),
            Inner::U8C3(i) => arc_dlpack_capsule(py, self.inner.clone(), &i.0, u8::dl_dtype()),
            Inner::U8C4(i) => arc_dlpack_capsule(py, self.inner.clone(), &i.0, u8::dl_dtype()),
            Inner::F32C1(i) => arc_dlpack_capsule(py, self.inner.clone(), &i.0, f32::dl_dtype()),
            Inner::F32C3(i) => arc_dlpack_capsule(py, self.inner.clone(), &i.0, f32::dl_dtype()),
        }
    }
}

/// Pack a DLPack capsule whose keepalive is an `Arc` clone of the owner —
/// export without consuming, buffer freed when both Python sides drop.
fn arc_dlpack_capsule<T, K, const N: usize>(
    py: Python<'_>,
    keepalive: Arc<K>,
    t: &Tensor<T, N>,
    dtype: dlpack_rs::ffi::DLDataType,
) -> PyResult<Py<PyAny>>
where
    K: Send + Sync + 'static,
{
    use dlpack_rs::safe::{self, TensorInfo};
    let device = dlpack_rs::ffi::DLDevice {
        device_type: dlpack_rs::ffi::DLDeviceType::kDLCUDA,
        device_id: 0,
    };
    let shape: Vec<i64> = t.shape.iter().map(|&s| s as i64).collect();
    let info = TensorInfo::contiguous(t.as_ptr() as *mut std::ffi::c_void, device, dtype, shape);
    // SAFETY: the Arc keepalive owns (a reference to) the device buffer and is
    // dropped by the capsule deleter.
    let managed = safe::pack(keepalive, info);
    unsafe {
        let capsule = pyo3::ffi::PyCapsule_New(
            managed as *mut std::ffi::c_void,
            c"dltensor".as_ptr(),
            Some(dlpack_capsule_destructor),
        );
        if capsule.is_null() {
            return Err(PyRuntimeError::new_err("failed to create DLPack capsule"));
        }
        Ok(Bound::from_owned_ptr(py, capsule).unbind())
    }
}

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // Only delete if the consumer never claimed it (name still "dltensor").
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) == 1 {
            let managed = pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr())
                as *mut dlpack_rs::ffi::DLManagedTensor;
            if !managed.is_null() {
                if let Some(deleter) = (*managed).deleter {
                    deleter(managed);
                }
            }
        }
    }
}

/// Upload an (H, W, C) numpy array (uint8 C∈{1,3,4} or float32 C∈{1,3}) to
/// the GPU as a [`CudaImage`].
#[pyfunction]
pub fn upload(py: Python<'_>, array: Py<PyAny>) -> PyResult<PyCudaImage> {
    let stream = default_stream()?;

    // Try u8 first, then f32.
    if let Ok(arr) = array.extract::<Bound<'_, PyArray3<u8>>>(py) {
        let arr = arr.try_readonly()?;
        let shape = arr.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let size = ImageSize {
            width: w,
            height: h,
        };
        let data = arr.as_slice()?;
        let inner = match c {
            1 => Inner::U8C1(host_to_cuda::<u8, 1>(size, data, &stream)?),
            3 => Inner::U8C3(host_to_cuda::<u8, 3>(size, data, &stream)?),
            4 => Inner::U8C4(host_to_cuda::<u8, 4>(size, data, &stream)?),
            c => {
                return Err(PyValueError::new_err(format!(
                    "unsupported channel count {c} (expected 1, 3, or 4)"
                )))
            }
        };
        return Ok(PyCudaImage {
            inner: Arc::new(inner),
        });
    }
    if let Ok(arr) = array.extract::<Bound<'_, PyArray3<f32>>>(py) {
        let arr = arr.try_readonly()?;
        let shape = arr.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let size = ImageSize {
            width: w,
            height: h,
        };
        let data = arr.as_slice()?;
        let inner = match c {
            1 => Inner::F32C1(host_to_cuda::<f32, 1>(size, data, &stream)?),
            3 => Inner::F32C3(host_to_cuda::<f32, 3>(size, data, &stream)?),
            c => {
                return Err(PyValueError::new_err(format!(
                    "unsupported float32 channel count {c} (expected 1 or 3)"
                )))
            }
        };
        return Ok(PyCudaImage {
            inner: Arc::new(inner),
        });
    }
    Err(PyValueError::new_err(
        "expected a contiguous (H, W, C) uint8 or float32 numpy array",
    ))
}

fn host_to_cuda<T, const C: usize>(
    size: ImageSize,
    data: &[T],
    stream: &Arc<CudaStream>,
) -> PyResult<Image<T, C>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone + Default + 'static,
{
    let host = Image::<T, C>::new(size, data.to_vec()).map_err(err)?;
    host.to_cuda_image(stream).map_err(err)
}

/// Import a device-resident DLPack tensor (torch / cupy) as a [`CudaImage`].
///
/// Accepts a 3-D C-contiguous `(H, W, C)` tensor on a CUDA device — uint8 with
/// C∈{1,3,4} or float32 with C∈{1,3}.
///
/// `copy=True` (default): the pixels are copied device-to-device into a buffer
/// this image owns; the copy is synchronized before returning, so the producer
/// tensor may be freed immediately after.
///
/// `copy=False`: **zero-copy** — the image aliases the producer's memory and
/// holds a reference to `obj` for as long as the image (or anything derived
/// from its buffer, e.g. a re-export) lives. Mutating the producer mutates
/// the image. The producer's work is stream-ordered against this module's
/// launches via the `stream=1` protocol handshake.
/// Keeps the DLPack producer's Python object alive for a zero-copy import.
///
/// The DLPack consumer may drop the tensor off-GIL (e.g. from a worker
/// thread), so the handle is released under a re-acquired GIL — same
/// discipline as `dlpack::ImageExport`. During interpreter finalization the
/// handle is forgotten instead (CPython reclaims everything anyway and
/// `Python::attach` would panic).
struct PyKeepalive(std::mem::ManuallyDrop<Py<PyAny>>);

impl PyKeepalive {
    fn new(obj: Py<PyAny>) -> Self {
        Self(std::mem::ManuallyDrop::new(obj))
    }
}

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

#[pyfunction]
#[pyo3(signature = (obj, copy = true))]
pub fn from_dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>, copy: bool) -> PyResult<PyCudaImage> {
    use dlpack_rs::ffi::{DLManagedTensor, DLManagedTensorVersioned};
    use pyo3::types::{PyCapsule, PyCapsuleMethods, PyDict};
    use std::ffi::CStr;

    let stream = default_stream()?;

    // Probe the protocol's device query first: host tensors get the helpful
    // redirect before any `__dlpack__` call (some producers, e.g. torch-CPU,
    // reject the `stream` kwarg with errors other than TypeError).
    if let Ok(dev) = obj.call_method0("__dlpack_device__") {
        let (ty, _id): (u32, i32) = dev.extract()?;
        if ty != dlpack_rs::ffi::DLDeviceType::kDLCUDA {
            return Err(PyValueError::new_err(
                "from_dlpack: tensor is not on a CUDA device; \
                 for host tensors use kornia_rs.cuda.upload",
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
             for host tensors use kornia_rs.cuda.upload",
        ));
    }
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
    Ok(PyCudaImage {
        inner: Arc::new(inner),
    })
}

// ── Color conversions ────────────────────────────────────────────────────────

/// Allocate a device destination and run one `ConvertColor` pair.
macro_rules! conv_fn {
    ($(#[$meta:meta])* $pyname:ident, $srcvar:ident, $snt:ty, $t:ty, $dc:literal, $dvar:ident, $dnt:ty) => {
        $(#[$meta])*
        #[pyfunction]
        pub fn $pyname(img: &PyCudaImage) -> PyResult<PyCudaImage> {
            let stream = default_stream()?;
            let Inner::$srcvar(src) = &*img.inner else {
                return Err(PyValueError::new_err(concat!(
                    stringify!($pyname),
                    ": wrong input dtype/channels for this conversion"
                )));
            };
            let mut dst = Image::<$t, $dc>::zeros_cuda(src.size(), &stream).map_err(err)?;
            convert_pair!(src, $snt, &mut dst, $dnt)?;
            Ok(PyCudaImage {
                inner: Arc::new(Inner::$dvar(dst)),
            })
        }
    };
}

conv_fn!(
    /// RGB8 → Gray8 (BT.601, bit-exact vs the CPU path).
    gray_from_rgb, U8C3, Rgb8, u8, 1, U8C1, Gray8
);
conv_fn!(
    /// Gray8 → RGB8 broadcast.
    rgb_from_gray, U8C1, Gray8, u8, 3, U8C3, Rgb8
);
conv_fn!(
    /// RGB8 → BGR8 channel swap (symmetric).
    bgr_from_rgb, U8C3, Rgb8, u8, 3, U8C3, Bgr8
);
conv_fn!(
    /// RGB8 → RGBA8 (opaque alpha).
    rgba_from_rgb, U8C3, Rgb8, u8, 4, U8C4, Rgba8
);
conv_fn!(
    /// RGBA8 → RGB8 (alpha dropped).
    rgb_from_rgba, U8C4, Rgba8, u8, 3, U8C3, Rgb8
);
conv_fn!(
    /// BGRA8 → RGB8.
    rgb_from_bgra, U8C4, Bgra8, u8, 3, U8C3, Rgb8
);
conv_fn!(
    /// RGB8 → YCbCr8 (full-range Q14, bit-exact vs the CPU path).
    ycbcr_from_rgb, U8C3, Rgb8, u8, 3, U8C3, YCbCr8
);
conv_fn!(
    /// YCbCr8 → RGB8.
    rgb_from_ycbcr, U8C3, YCbCr8, u8, 3, U8C3, Rgb8
);
conv_fn!(
    /// RGB f32 → HSV f32 (kornia conventions, [0,255] scale).
    hsv_from_rgb, F32C3, Rgbf32, f32, 3, F32C3, Hsvf32
);
conv_fn!(
    /// HSV f32 → RGB f32.
    rgb_from_hsv, F32C3, Hsvf32, f32, 3, F32C3, Rgbf32
);
conv_fn!(
    /// RGB f32 → CIE Lab f32 (RGB in [0,1], L in [0,100]).
    lab_from_rgb, F32C3, Rgbf32, f32, 3, F32C3, Labf32
);
conv_fn!(
    /// Lab f32 → RGB f32.
    rgb_from_lab, F32C3, Labf32, f32, 3, F32C3, Rgbf32
);

/// Sepia tone on RGB8 (Q8 fixed point, bit-exact vs the CPU path).
#[pyfunction]
pub fn sepia_from_rgb(img: &PyCudaImage) -> PyResult<PyCudaImage> {
    let stream = default_stream()?;
    let Inner::U8C3(src) = &*img.inner else {
        return Err(PyValueError::new_err("sepia_from_rgb expects uint8 HxWx3"));
    };
    let mut dst = Image::<u8, 3>::zeros_cuda(src.size(), &stream).map_err(err)?;
    color::sepia_from_rgb_u8(src, &mut dst).map_err(err)?;
    Ok(PyCudaImage {
        inner: Arc::new(Inner::U8C3(dst)),
    })
}

/// Apply one of the 21 OpenCV colormaps to a Gray8 image (name as in
/// `kornia_rs.imgproc.apply_colormap`).
#[pyfunction]
pub fn apply_colormap(img: &PyCudaImage, colormap: &str) -> PyResult<PyCudaImage> {
    let stream = default_stream()?;
    let Inner::U8C1(src) = &*img.inner else {
        return Err(PyValueError::new_err("apply_colormap expects uint8 HxWx1"));
    };
    let cmap = color::ColormapType::from_name(colormap)
        .ok_or_else(|| PyValueError::new_err(format!("unknown colormap '{colormap}'")))?;
    let mut dst = Image::<u8, 3>::zeros_cuda(src.size(), &stream).map_err(err)?;
    color::apply_colormap(src, &mut dst, cmap).map_err(err)?;
    Ok(PyCudaImage {
        inner: Arc::new(Inner::U8C3(dst)),
    })
}

/// Bayer demosaic (pattern: "rggb" | "bggr" | "grbg" | "gbrg").
#[pyfunction]
pub fn rgb_from_bayer(img: &PyCudaImage, pattern: &str) -> PyResult<PyCudaImage> {
    use kornia_image::color_spaces::BayerPattern;
    let stream = default_stream()?;
    let Inner::U8C1(src) = &*img.inner else {
        return Err(PyValueError::new_err("rgb_from_bayer expects uint8 HxWx1"));
    };
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
    let mut dst = Image::<u8, 3>::zeros_cuda(src.size(), &stream).map_err(err)?;
    color::rgb_from_bayer(src, pat, &mut dst).map_err(err)?;
    Ok(PyCudaImage {
        inner: Arc::new(Inner::U8C3(dst)),
    })
}

// ── CudaTensor (model input) ─────────────────────────────────────────────────

enum TensorInnerEnum {
    F32(Tensor<f32, 4>),
    F16(Tensor<half::f16, 4>),
}

/// A device-resident `[N, C, H, W]` tensor — the preprocessor's output, i.e.
/// model input. Feed it to torch/TensorRT zero-copy via `__dlpack__`, or
/// `download()` an f32 numpy copy.
#[pyclass(name = "CudaTensor", frozen, module = "kornia_rs.cuda")]
pub struct PyCudaTensor {
    inner: Arc<TensorInnerEnum>,
}

#[pymethods]
impl PyCudaTensor {
    /// Tensor shape `(N, C, H, W)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let s = match &*self.inner {
            TensorInnerEnum::F32(t) => t.shape,
            TensorInnerEnum::F16(t) => t.shape,
        };
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

    /// Copy to host as a float32 numpy array (f16 tensors are widened).
    fn download<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let (data, shape): (Vec<f32>, [usize; 4]) = match &*self.inner {
            TensorInnerEnum::F32(t) => {
                let host = t.download().map_err(err)?;
                (host.as_slice().to_vec(), t.shape)
            }
            TensorInnerEnum::F16(t) => {
                let host = t.download().map_err(err)?;
                (
                    host.as_slice().iter().map(|v| v.to_f32()).collect(),
                    t.shape,
                )
            }
        };
        let arr = PyArray1::from_vec(py, data);
        let arr = arr.reshape(shape)?;
        Ok(arr.into_any().unbind())
    }

    /// DLPack device tuple: `(kDLCUDA, device_id)`.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32, 0)
    }

    /// Export as a DLPack capsule (zero-copy). Synchronizes the producing
    /// stream first so the consumer sees completed data on any stream.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<isize>,
        max_version: Option<Py<PyAny>>,
        dl_device: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (max_version, dl_device);
        if copy == Some(true) {
            return Err(PyValueError::new_err("copy=True is not supported"));
        }
        dlpack_sync_for_consumer(stream)?;
        use kornia_tensor::dlpack::DlpackElem;
        // f16: kDLFloat (code 2), 16 bits — half::f16 is IEEE binary16.
        let f16_dtype = dlpack_rs::ffi::DLDataType {
            code: 2,
            bits: 16,
            lanes: 1,
        };
        match &*self.inner {
            TensorInnerEnum::F32(t) => {
                arc_dlpack_capsule(py, self.inner.clone(), t, f32::dl_dtype())
            }
            TensorInnerEnum::F16(t) => arc_dlpack_capsule(py, self.inner.clone(), t, f16_dtype),
        }
    }
}

// ── CudaPreprocessor ─────────────────────────────────────────────────────────

/// Fused camera preprocessing on the GPU: raw frame (NV12/YUYV/RGB/BGR/RGBA/
/// BGRA/Gray) → normalized `[1, 3, H, W]` CHW tensor in **one kernel launch**.
#[pyclass(name = "CudaPreprocessor", frozen, module = "kornia_rs.cuda")]
pub struct PyCudaPreprocessor {
    pre: Preprocessor,
    stream: Arc<CudaStream>,
    f16: bool,
    /// Persistent upload staging: one page-locked host buffer + one device
    /// buffer per batch slot, grown on demand and reused across calls.
    /// Page-locking (`cuMemHostAlloc`) is a syscall far too expensive for a
    /// frame loop, and pinned memory turns the H2D copy into a straight DMA
    /// instead of a bounce through the driver's pageable staging buffer.
    staging: std::sync::Mutex<Staging>,
}

#[derive(Default)]
struct Staging {
    /// Page-locked host buffer holding all frames of a call back-to-back.
    pinned: Option<Tensor<u8, 1>>,
    /// Device destination per batch slot.
    device: Vec<cudarc::driver::CudaSlice<u8>>,
}

impl Staging {
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

#[pymethods]
impl PyCudaPreprocessor {
    /// Build a preprocessor. Compiles one CUDA kernel per output dtype at
    /// construction; the source format is a runtime kernel argument, so reuse
    /// the instance across frames (and formats need no recompilation).
    ///
    /// mode: "letterbox" | "stretch"; format: "rgb"|"bgr"|"rgba"|"bgra"|
    /// "gray"|"nv12"|"yuyv"; sampling: "bilinear"|"nearest"|"lanczos";
    /// f16: half output; mean/std: optional per-channel normalization fused
    /// into the kernel; pad_value: letterbox padding byte (default 114).
    #[new]
    #[pyo3(signature = (mode = "letterbox", format = "rgb", sampling = "bilinear", f16 = false, mean = None, std = None, pad_value = 114))]
    fn new(
        mode: &str,
        format: &str,
        sampling: &str,
        f16: bool,
        mean: Option<[f32; 3]>,
        std: Option<[f32; 3]>,
        pad_value: u8,
    ) -> PyResult<Self> {
        let stream = default_stream()?;
        let mut builder = Preprocessor::builder()
            .mode(match mode.to_ascii_lowercase().as_str() {
                "letterbox" => ResizeMode::Letterbox,
                "stretch" => ResizeMode::Stretch,
                m => return Err(PyValueError::new_err(format!("unknown mode '{m}'"))),
            })
            .source_format(parse_format(format)?)
            .sampling(crate::image::parse_interpolation(sampling)?)
            .pad_value(pad_value);
        if mean.is_some() || std.is_some() {
            builder = builder.normalize(Normalize::MeanStd {
                mean: mean.unwrap_or([0.0; 3]),
                std: std.unwrap_or([1.0; 3]),
            });
        }
        let pre = builder.build_cuda(stream.clone()).map_err(err)?;
        Ok(Self {
            pre,
            stream,
            f16,
            staging: std::sync::Mutex::new(Staging::default()),
        })
    }

    /// Preprocess one raw frame (flat bytes in the constructor's format
    /// layout) into a fresh `[1, 3, out_h, out_w]` [`CudaTensor`].
    #[pyo3(signature = (frame, width, height, out_height, out_width))]
    fn run(
        &self,
        py: Python<'_>,
        frame: numpy::PyReadonlyArray1<'_, u8>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
    ) -> PyResult<PyCudaTensor> {
        let frame_len = frame.len();
        let mut staging = self.staging.lock().expect("staging mutex poisoned");
        staging.ensure(&self.stream, 1, frame_len)?;
        staging.pinned.as_mut().expect("ensured").as_slice_mut()[..frame_len]
            .copy_from_slice(frame.as_slice()?);
        let staging = &mut *staging;

        // Everything past the numpy borrow runs without the GIL: the pinned
        // H2D DMA, the fused kernel launch, and the output allocation.
        let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
            let pinned = staging.pinned.as_ref().expect("ensured");
            let d_src = &mut staging.device[0];
            self.stream
                .memcpy_htod(&pinned.as_slice()[..frame_len], d_src)
                .map_err(err)?;
            if self.f16 {
                let mut dst = kornia_tensor::zeros_cuda::<half::f16, 4>(
                    [1, 3, out_height, out_width],
                    &self.stream,
                )
                .map_err(err)?;
                self.pre
                    .run_raw_f16(d_src, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F16(dst))
            } else {
                let mut dst = kornia_tensor::zeros_cuda::<f32, 4>(
                    [1, 3, out_height, out_width],
                    &self.stream,
                )
                .map_err(err)?;
                self.pre
                    .run_raw(d_src, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F32(dst))
            }
        })?;
        Ok(PyCudaTensor {
            inner: Arc::new(inner),
        })
    }

    /// Preprocess a **batch** of same-sized raw frames into one
    /// `[N, 3, out_h, out_w]` [`CudaTensor`] — one fused kernel launch per
    /// frame, all on the same stream, one sync for the whole batch
    /// (multi-camera rigs, batched engines). Output dtype follows the
    /// constructor's `f16` flag, like [`run`](Self::run).
    #[pyo3(signature = (frames, width, height, out_height, out_width))]
    fn run_batch(
        &self,
        py: Python<'_>,
        frames: Vec<numpy::PyReadonlyArray1<'_, u8>>,
        width: usize,
        height: usize,
        out_height: usize,
        out_width: usize,
    ) -> PyResult<PyCudaTensor> {
        if frames.is_empty() {
            return Err(PyValueError::new_err("run_batch needs at least one frame"));
        }
        let n = frames.len();
        let frame_len = frames[0].len();
        let mut staging = self.staging.lock().expect("staging mutex poisoned");
        staging.ensure(&self.stream, n, frame_len)?;
        {
            let pinned = staging.pinned.as_mut().expect("ensured").as_slice_mut();
            for (i, f) in frames.iter().enumerate() {
                let src = f.as_slice()?;
                if src.len() != frame_len {
                    return Err(PyValueError::new_err(
                        "run_batch frames must all have the same length",
                    ));
                }
                pinned[i * frame_len..(i + 1) * frame_len].copy_from_slice(src);
            }
        }
        let staging = &mut *staging;

        let inner = py.detach(|| -> PyResult<TensorInnerEnum> {
            let pinned = staging.pinned.as_ref().expect("ensured").as_slice();
            for (i, d) in staging.device[..n].iter_mut().enumerate() {
                self.stream
                    .memcpy_htod(&pinned[i * frame_len..(i + 1) * frame_len], d)
                    .map_err(err)?;
            }
            let refs: Vec<_> = staging.device[..n].iter().collect();
            let shape = [n, 3, out_height, out_width];
            if self.f16 {
                let mut dst =
                    kornia_tensor::zeros_cuda::<half::f16, 4>(shape, &self.stream).map_err(err)?;
                self.pre
                    .run_raw_batch_f16(&refs, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F16(dst))
            } else {
                let mut dst =
                    kornia_tensor::zeros_cuda::<f32, 4>(shape, &self.stream).map_err(err)?;
                self.pre
                    .run_raw_batch(&refs, width, height, &mut dst)
                    .map_err(err)?;
                Ok(TensorInnerEnum::F32(dst))
            }
        })?;
        Ok(PyCudaTensor {
            inner: Arc::new(inner),
        })
    }
}

// ── module registration ─────────────────────────────────────────────────────

/// Register `kornia_rs.cuda`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "cuda")?;
    m.add_function(wrap_pyfunction!(is_available, &m)?)?;
    m.add_function(wrap_pyfunction!(upload, &m)?)?;
    m.add_function(wrap_pyfunction!(from_dlpack, &m)?)?;
    m.add_function(wrap_pyfunction!(gray_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_gray, &m)?)?;
    m.add_function(wrap_pyfunction!(bgr_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(rgba_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_rgba, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_bgra, &m)?)?;
    m.add_function(wrap_pyfunction!(ycbcr_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_ycbcr, &m)?)?;
    m.add_function(wrap_pyfunction!(hsv_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_hsv, &m)?)?;
    m.add_function(wrap_pyfunction!(lab_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_lab, &m)?)?;
    m.add_function(wrap_pyfunction!(sepia_from_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(apply_colormap, &m)?)?;
    m.add_function(wrap_pyfunction!(rgb_from_bayer, &m)?)?;
    // Re-exports of the top-level constants (device-agnostic, but handy here).
    let [m0, m1, m2] = kornia_imgproc::preprocess::IMAGENET_MEAN;
    let [s0, s1, s2] = kornia_imgproc::preprocess::IMAGENET_STD;
    m.add("IMAGENET_MEAN", (m0, m1, m2))?;
    m.add("IMAGENET_STD", (s0, s1, s2))?;
    m.add_class::<PyCudaImage>()?;
    m.add_class::<PyCudaTensor>()?;
    m.add_class::<PyCudaPreprocessor>()?;
    parent.add_submodule(&m)?;
    // Make `import kornia_rs.cuda` work (mirror of the other submodules).
    py.import("sys")?
        .getattr("modules")?
        .set_item("kornia_rs.cuda", &m)?;
    Ok(())
}
