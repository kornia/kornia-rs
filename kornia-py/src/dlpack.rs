//! DLPack export glue for `PyImageApi`.
//!
//! **Export (`__dlpack__`)** — true zero-copy: a `Py<PyAny>` handle to the
//! Image object keeps the backing alive while the consumer holds the tensor.
//! No bytes are copied.
//!
//! **Import (`from_dlpack`)** — zero-copy via non-consuming capsule keep-alive.
//! See `image.rs::from_dlpack` for the import path.

use std::ffi::{c_void, CStr};

use dlpack_rs::{
    ffi::{DLDataType, DLDevice},
    pyo3_glue::IntoDLPack,
    safe::TensorInfo,
};
use pyo3::prelude::*;

use crate::backing::Dtype;

// ─────────────────────────────────────────────────────────────────────────────
// Export: keep-alive wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-copy DLPack export wrapper.
///
/// `keepalive` holds a `Py<PyAny>` handle to the exporting `PyImageApi`.
/// While the DLPack consumer retains the tensor (the capsule / the `ManagedContext`
/// allocated by `safe::pack`), `keepalive` is kept alive, and therefore
/// the `Backing` buffer is kept alive too.  When the consumer's deleter runs
/// it drops the `ManagedContext<ImageExport>`, which drops `keepalive`, which
/// decrements the Image's refcount (and potentially frees the buffer if nothing
/// else holds a reference).
///
/// # GIL safety
///
/// `keepalive` is wrapped in `ManuallyDrop` so that field-drop never calls
/// `Py_DECREF` implicitly (which would be UB if the DLPack deleter runs
/// off-GIL, e.g. from a PyTorch worker thread).  Our custom `Drop` impl
/// acquires the GIL before releasing the reference count.
pub struct ImageExport {
    /// Strong reference to the `PyImageApi` Python object — keeps Backing alive.
    /// Never "read" by Rust — this field exists to be *held*, not accessed.
    /// Wrapped in `ManuallyDrop` to prevent implicit off-GIL `Py_DECREF`.
    #[allow(dead_code)]
    pub keepalive: std::mem::ManuallyDrop<Py<PyAny>>,
    /// Raw pointer into the Image's backing buffer (NOT a copy).
    pub data: *mut c_void,
    /// HWC shape as `[H, W, C]` (i64 for DLPack).
    pub shape: Vec<i64>,
    /// DLPack data-type descriptor.
    pub dtype: DLDataType,
    /// DLPack device as `(device_type, device_id)`. Carries the image's own
    /// device so a zero-copy export reports the correct device (CPU or CUDA).
    pub device: (i32, i32),
}

/// Release a Python reference from a DLPack deleter / keep-alive `Drop`.
///
/// Uses the raw GIL-state API rather than `Python::attach`: the consumer may
/// release the GIL internally before invoking a deleter (PyTorch does while
/// tearing down a tensor) while pyo3's thread-local GIL count still says
/// "attached" from an outer `attach` further up the stack. `Python::attach`
/// would then skip re-acquiring and the object would be freed without an
/// attached thread state (a crash on CPython 3.12+). `PyGILState_Ensure`
/// checks the real thread state. During interpreter finalization the
/// reference is leaked instead (CPython reclaims everything anyway).
pub(crate) fn release_py_ref(obj: Py<PyAny>) {
    // SAFETY: plain query of interpreter state.
    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
        std::mem::forget(obj);
        return;
    }
    // SAFETY: the interpreter is initialized; Ensure/Release are paired and the
    // reference we decref is the one `obj` owns.
    unsafe {
        let gil = pyo3::ffi::PyGILState_Ensure();
        pyo3::ffi::Py_DecRef(obj.into_ptr());
        pyo3::ffi::PyGILState_Release(gil);
    }
}

impl Drop for ImageExport {
    fn drop(&mut self) {
        // The DLPack consumer's deleter may run off-GIL (e.g. from a PyTorch
        // worker thread), so the reference is released via `release_py_ref`.
        // SAFETY: we own this handle inside `ManuallyDrop` and release it once.
        let keepalive = unsafe { std::mem::ManuallyDrop::take(&mut self.keepalive) };
        release_py_ref(keepalive);
    }
}

// SAFETY: `data` points into `keepalive`'s backing.  `ManagedContext<ImageExport>`
// (heap-allocated by `safe::pack`) owns the `ImageExport`, which owns `keepalive`.
// The buffer therefore outlives the exported tensor.  `Py<PyAny>` is `Send` under
// the assumption that operations on the GIL-held Python object happen under the GIL.
unsafe impl Send for ImageExport {}

impl IntoDLPack for ImageExport {
    fn tensor_info(&self) -> TensorInfo {
        let device = DLDevice {
            device_type: self.device.0 as u32,
            device_id: self.device.1,
        };
        TensorInfo::contiguous(self.data, device, self.dtype, self.shape.clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Import: ownership of a consumed DLManagedTensor
// ─────────────────────────────────────────────────────────────────────────────

/// Raw pointer to a producer's managed tensor, legacy or versioned.
enum ManagedTensorPtr {
    Legacy(std::ptr::NonNull<dlpack_rs::ffi::DLManagedTensor>),
    Versioned(std::ptr::NonNull<dlpack_rs::ffi::DLManagedTensorVersioned>),
}

/// Owner of a DLPack managed tensor taken from a consumed capsule.
///
/// Per the DLPack protocol, a consumer that renames a capsule to
/// `used_dltensor[_versioned]` takes ownership of the `DLManagedTensor` and
/// MUST call its `deleter` exactly once when done with the data. Before this
/// type existed `from_dlpack` renamed the capsule but never called the deleter,
/// leaking the producer's manager context (for numpy: a strong reference to the
/// source array, i.e. the whole buffer) on every import.
///
/// Dropping the owner calls the deleter exactly once (it is not `Clone`, and
/// the capsule was renamed so the capsule destructor will not call it too).
pub struct DlManagedOwner {
    ptr: ManagedTensorPtr,
}

// SAFETY: the owner only holds a pointer whose sole use is the one-shot deleter
// call in `Drop`, which runs with the GIL attached. DLPack deleters are required
// to be callable from any thread.
unsafe impl Send for DlManagedOwner {}
// SAFETY: no `&self` method touches the pointee.
unsafe impl Sync for DlManagedOwner {}

impl DlManagedOwner {
    /// Consume a `dltensor` / `dltensor_versioned` capsule and take ownership of
    /// its managed tensor.
    ///
    /// Renames the capsule to `used_dltensor[_versioned]` (so the producer's
    /// capsule destructor will not free it) and returns the owner that will
    /// call the deleter on drop. Call this only once all fallible validation
    /// that should leave the capsule untouched has been done; any error after
    /// this call still frees the tensor correctly via the owner's `Drop`.
    ///
    /// # Errors
    ///
    /// `ValueError` if the capsule name is not a DLPack name or its pointer is
    /// null; `RuntimeError` if renaming fails.
    pub fn consume_capsule(
        capsule: &Bound<'_, pyo3::types::PyCapsule>,
    ) -> PyResult<DlManagedOwner> {
        use pyo3::types::PyCapsuleMethods;
        type Ctor = fn(std::ptr::NonNull<c_void>) -> ManagedTensorPtr;
        let name = capsule.name()?;
        // Branch on the capsule kind once: the name to check the pointer
        // against, the name to rename it to, and how to type the pointer.
        // SAFETY: the name pointer is valid for as long as the capsule is alive
        // and not renamed; we only compare it before renaming below.
        let (expected, consumed, ctor): (&'static CStr, &'static CStr, Ctor) = match &name {
            Some(n) if unsafe { n.as_cstr() } == c"dltensor_versioned" => {
                (c"dltensor_versioned", c"used_dltensor_versioned", |p| {
                    ManagedTensorPtr::Versioned(p.cast())
                })
            }
            Some(n) if unsafe { n.as_cstr() } == c"dltensor" => {
                (c"dltensor", c"used_dltensor", |p| {
                    ManagedTensorPtr::Legacy(p.cast())
                })
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "from_dlpack: capsule is not an unconsumed DLPack tensor",
                ))
            }
        };
        let ptr = ctor(capsule.pointer_checked(Some(expected))?);
        // SAFETY: `capsule` is a live PyCapsule and `consumed` is a 'static C
        // string, as PyCapsule_SetName requires.
        if unsafe { pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), consumed.as_ptr()) } != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "from_dlpack: failed to consume DLPack capsule (PyCapsule_SetName failed)",
            ));
        }
        Ok(DlManagedOwner { ptr })
    }
}

impl Drop for DlManagedOwner {
    fn drop(&mut self) {
        let call = || match self.ptr {
            ManagedTensorPtr::Legacy(p) => {
                // SAFETY: `p` came from a live, consumed `dltensor` capsule and is
                // owned exclusively by us; the deleter (if any) is called once.
                if let Some(deleter) = unsafe { (*p.as_ptr()).deleter } {
                    // SAFETY: DLPack contract — deleter(self) frees the tensor.
                    unsafe { deleter(p.as_ptr()) };
                }
            }
            ManagedTensorPtr::Versioned(p) => {
                // SAFETY: as above, for the versioned struct.
                if let Some(deleter) = unsafe { (*p.as_ptr()).deleter } {
                    // SAFETY: DLPack contract — deleter(self) frees the tensor.
                    unsafe { deleter(p.as_ptr()) };
                }
            }
        };
        // Producer deleters (numpy, torch) may touch Python objects; run them
        // with the GIL attached. During interpreter finalization there is no
        // interpreter to attach to — leak instead (the process is exiting).
        // SAFETY: plain query of interpreter state.
        // As in `ImageExport::drop`, use the raw GIL-state API: an outer
        // `Python::attach` may be on the stack while the GIL was released
        // underneath it, so pyo3's thread-local count cannot be trusted here.
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            // SAFETY: interpreter initialized; Ensure/Release are paired.
            unsafe {
                let gil = pyo3::ffi::PyGILState_Ensure();
                call();
                pyo3::ffi::PyGILState_Release(gil);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Dtype <-> DLDataType
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a `Dtype` to the corresponding `DLDataType`.
///
/// Delegates to [`Dtype::to_dldatatype`] — canonical mapping lives in `backing.rs`.
pub fn dtype_to_dl(dtype: Dtype) -> DLDataType {
    dtype.to_dldatatype()
}

/// Convert a `DLDataType` to `Dtype`, or return a `ValueError`.
///
/// Delegates to [`Dtype::from_dldatatype`] — canonical mapping lives in `backing.rs`.
pub fn dl_to_dtype(dt: DLDataType) -> PyResult<Dtype> {
    Dtype::from_dldatatype(dt)
}

/// Validate a producer-supplied DLPack rank before it is used as a slice length.
///
/// `ndim` comes from an untrusted `__dlpack__` producer. A negative value casts to
/// `usize::MAX` (so `slice::from_raw_parts(shape, ndim)` is instant UB) and an
/// oversized one reads out of bounds. Only 2D/3D images are supported, so we bound
/// `ndim` to `2..=3` (capping any slice to ≤3 elements) and reject a null `shape`
/// pointer, before any slice is constructed from `shape`/`strides`.
pub fn validate_dlpack_rank(ndim: i32, shape: *const i64) -> PyResult<()> {
    if !(2..=3).contains(&ndim) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "from_dlpack: expected a 2D or 3D tensor, got ndim={ndim}"
        )));
    }
    if shape.is_null() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "from_dlpack: null shape pointer",
        ));
    }
    Ok(())
}
