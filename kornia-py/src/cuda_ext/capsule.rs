//! DLPack capsule pack/export for `Tensor`, and device DLPack import.
//!
//! The capsule packers (`arc_dlpack_capsule*`) are always compiled (host
//! `Tensor`/`Image` export via DLPack too); the device import path
//! (`dlpack_to_device_arc`, `PyKeepalive`) is `cuda`-gated. Named `capsule` to
//! avoid clashing with the crate-level `crate::dlpack` (Image export helpers).

use super::*;

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
    dlpack_rs::safe::TensorInfo::contiguous(
        t.as_ptr() as *mut std::ffi::c_void,
        device,
        dtype,
        shape,
    )
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
                    let managed =
                        pyo3::ffi::PyCapsule_GetPointer(capsule, $cstr.as_ptr()) as *mut $managed;
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
pub(super) fn arc_dlpack_capsule<T, K, const N: usize>(
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
pub(super) fn arc_dlpack_capsule_versioned<T, K, const N: usize>(
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
/// inference).
///
/// Accepts a 3-D C-contiguous `(H, W, C)` CUDA tensor — uint8 `C∈{1,3,4}` or
/// float32 `C∈{1,3}`. Always a zero-copy alias (mirroring torch's zero-copy
/// DLPack): the returned image keeps `obj` alive for its lifetime. Callers who
/// want an independent buffer should copy it themselves afterward.
#[cfg(feature = "cuda")]
pub(crate) fn dlpack_to_device_arc(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arc<Inner>> {
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
    // Borrow the DLTensor from the managed struct (renaming below does not move
    // or free it, so `t` stays valid).
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

    // CONSUME the capsule (rename to "used_dltensor[_versioned]") so its C
    // destructor will NOT run the producer's deleter when the borrowed capsule
    // GCs at function end — same discipline as the host `Image::from_dlpack`.
    // Without this, a spec-conformant producer whose `__dlpack__` transfers
    // buffer ownership to the managed tensor (rather than keeping an
    // independent reference like torch/cupy do) would free the buffer at import
    // time, dangling our zero-copy alias (a use-after-free; torch/cupy happen to
    // be safe only because `obj` retains its own ref). The buffer then stays
    // alive via the `obj` keep-alive below.
    {
        use pyo3::ffi::PyCapsule_SetName;
        let consumed: &'static CStr = if name_cstr == NAME_DLV {
            c"used_dltensor_versioned"
        } else {
            c"used_dltensor"
        };
        // SAFETY: `capsule` is a valid PyCapsule (cast_into validated it) and
        // `consumed` is a 'static C string.
        if unsafe { PyCapsule_SetName(capsule.as_ptr(), consumed.as_ptr()) } != 0 {
            return Err(PyRuntimeError::new_err(
                "from_dlpack: failed to consume DLPack capsule (PyCapsule_SetName failed)",
            ));
        }
    }

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

    /// Zero-copy alias of the producer's `h*w*C` elements at `ptr` as a device
    /// image, kept valid by holding `obj` (the producer) alive.
    fn dl_image<T, const C: usize>(
        stream: &Arc<CudaStream>,
        ptr: u64,
        h: usize,
        w: usize,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<Image<T, C>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static,
    {
        let n = h * w * C;
        // SAFETY: ptr/len come from the live DLPack tensor validated above. The
        // foreign tensor leaks the aliasing slice on drop (never frees) and
        // releases the producer handle instead; the buffer is owned by `obj`.
        let src = unsafe { stream.upgrade_device_ptr::<T>(ptr, n) };
        Ok(Image(Tensor::from_foreign_cudaslice(
            src,
            [h, w, C],
            stream.clone(),
            Box::new(PyKeepalive::new(obj.clone().unbind())),
        )))
    }

    use dlpack_rs::ffi::{K_DL_FLOAT, K_DL_UINT};
    let inner = match (t.dtype.code, t.dtype.bits, t.dtype.lanes, c) {
        (code, 8, 1, 1) if code == K_DL_UINT => Inner::U8C1(dl_image(&stream, ptr, h, w, obj)?),
        (code, 8, 1, 3) if code == K_DL_UINT => Inner::U8C3(dl_image(&stream, ptr, h, w, obj)?),
        (code, 8, 1, 4) if code == K_DL_UINT => Inner::U8C4(dl_image(&stream, ptr, h, w, obj)?),
        (code, 32, 1, 1) if code == K_DL_FLOAT => Inner::F32C1(dl_image(&stream, ptr, h, w, obj)?),
        (code, 32, 1, 3) if code == K_DL_FLOAT => Inner::F32C3(dl_image(&stream, ptr, h, w, obj)?),
        (code, bits, lanes, c) => {
            return Err(PyValueError::new_err(format!(
                "from_dlpack: unsupported dtype (code {code}, {bits} bits, {lanes} lanes) \
                 with {c} channels — expected uint8 C∈{{1,3,4}} or float32 C∈{{1,3}}"
            )))
        }
    };
    Ok(Arc::new(inner))
}
