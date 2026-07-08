//! DLPack export glue for `PyImageApi`.
//!
//! **Export (`__dlpack__`)** — true zero-copy: a `Py<PyAny>` handle to the
//! Image object keeps the backing alive while the consumer holds the tensor.
//! No bytes are copied.
//!
//! **Import (`from_dlpack`)** — zero-copy: the capsule is consumed and its
//! managed tensor owned by a [`DlpackManaged`] keep-alive that runs the deleter
//! on drop. See `image.rs::from_dlpack` (host) and `cuda_ext::capsule` (device).

use std::ffi::c_void;

use dlpack_rs::{
    ffi::{DLDataType, DLDevice},
    pyo3_glue::IntoDLPack,
    safe::TensorInfo,
};
use pyo3::prelude::*;

use crate::backing::Dtype;

// ─────────────────────────────────────────────────────────────────────────────
// Import: consumed-managed-tensor owner
// ─────────────────────────────────────────────────────────────────────────────

/// Owns a *consumed* DLPack managed tensor and runs its deleter exactly once on
/// drop — the correct zero-copy import keep-alive.
///
/// When `from_dlpack` consumes a capsule (renames it to `used_dltensor*`), the
/// capsule's own C destructor will no longer call the producer's deleter, so
/// ownership of the `DLManagedTensor` transfers to us. Holding this and running
/// the deleter on drop releases the producer's storage reference (and frees the
/// manager struct) — versus the older "keep the producer object alive, never
/// call the deleter" model, which leaked the manager context for producers
/// (e.g. PyTorch) whose export takes an extra reference.
pub(crate) struct DlpackManaged {
    managed: *mut c_void,
    versioned: bool,
}

// SAFETY: `managed` is logically owned here and only dereferenced on drop; a
// DLPack deleter is required by the spec to be callable from any thread.
unsafe impl Send for DlpackManaged {}

impl DlpackManaged {
    /// # Safety
    /// `managed` must be a live `DLManagedTensor*` (`versioned == false`) or
    /// `DLManagedTensorVersioned*` (`versioned == true`) whose capsule has been
    /// consumed (renamed), so this owner is the *only* caller of its deleter.
    pub(crate) unsafe fn new(managed: *mut c_void, versioned: bool) -> Self {
        Self { managed, versioned }
    }
}

impl Drop for DlpackManaged {
    fn drop(&mut self) {
        use dlpack_rs::ffi::{DLManagedTensor, DLManagedTensorVersioned};
        if self.managed.is_null() {
            return;
        }
        let (managed, versioned) = (self.managed, self.versioned);
        // SAFETY: `managed` is the live, consumed managed tensor; its deleter is
        // called exactly once (here, on drop) — the capsule was renamed so its
        // own destructor will not also call it.
        let call = move || unsafe {
            if versioned {
                let m = managed as *mut DLManagedTensorVersioned;
                if let Some(del) = (*m).deleter {
                    del(m);
                }
            } else {
                let m = managed as *mut DLManagedTensor;
                if let Some(del) = (*m).deleter {
                    del(m);
                }
            }
        };
        // The deleter may `Py_DECREF` (numpy) or free device memory (torch-CUDA)
        // and can run off-GIL (e.g. a worker thread drops the tensor); re-acquire
        // the GIL when the interpreter is alive, mirroring `ImageExport::drop`.
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|_| call());
        } else {
            call();
        }
    }
}

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

impl Drop for ImageExport {
    fn drop(&mut self) {
        // SAFETY: We own this `Py<PyAny>` inside `ManuallyDrop`; we drop it
        // exactly once here, under the GIL, to prevent `Py_DECREF` off-GIL.
        // The DLPack consumer's deleter may run off-GIL (e.g. from a PyTorch
        // worker thread), so we must re-acquire it before touching the refcount.
        let keepalive = unsafe { std::mem::ManuallyDrop::take(&mut self.keepalive) };
        // During `Py_FinalizeEx`, torch may call our capsule destructor after
        // `Py_IsInitialized()` has returned 0. `Python::attach` asserts the
        // interpreter is alive, so it would panic. Instead, forget the handle:
        // CPython will reclaim everything during finalization regardless.
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|_py| drop(keepalive));
        } else {
            std::mem::forget(keepalive);
        }
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
