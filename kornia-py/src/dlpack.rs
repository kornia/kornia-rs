//! DLPack export glue for `PyImageApi`.
//!
//! **Export (`__dlpack__`)** — true zero-copy: a `Py<PyAny>` handle to the
//! Image object keeps the backing alive while the consumer holds the tensor.
//! No bytes are copied.
//!
//! **Import (`from_dlpack`)** — zero-copy via non-consuming capsule keep-alive.
//! See `image.rs::from_dlpack` for the import path.

use std::ffi::c_void;

use dlpack_rs::{
    ffi::{DLDataType, DLDevice, K_DL_CPU},
    pyo3_glue::IntoDLPack,
    safe::{cpu_device, TensorInfo},
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
        TensorInfo::contiguous(self.data, cpu_device(), self.dtype, self.shape.clone())
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

// ─────────────────────────────────────────────────────────────────────────────
// Helper: device validation
// ─────────────────────────────────────────────────────────────────────────────

/// Assert the device is CPU, or raise `NotImplementedError`.
pub fn require_cpu(device: DLDevice) -> PyResult<()> {
    if device.device_type != K_DL_CPU {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "from_dlpack: only CPU (device_type={K_DL_CPU}) tensors are supported; \
             got device_type={}. GPU/CUDA support is a future extension.",
            device.device_type,
        )))
    } else {
        Ok(())
    }
}
