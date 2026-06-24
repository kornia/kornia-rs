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
    ffi::{DLDataType, DLDevice, K_DL_CPU, K_DL_FLOAT, K_DL_UINT},
    pyo3_glue::IntoDLPack,
    safe::{cpu_device, dtype_f32, dtype_u16, dtype_u8, TensorInfo},
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
pub struct ImageExport {
    /// Strong reference to the `PyImageApi` Python object — keeps Backing alive.
    /// Never "read" by Rust — this field exists to be *held*, not accessed.
    #[allow(dead_code)]
    pub keepalive: Py<PyAny>,
    /// Raw pointer into the Image's backing buffer (NOT a copy).
    pub data: *mut c_void,
    /// HWC shape as `[H, W, C]` (i64 for DLPack).
    pub shape: Vec<i64>,
    /// DLPack data-type descriptor.
    pub dtype: DLDataType,
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
pub fn dtype_to_dl(dtype: Dtype) -> DLDataType {
    match dtype {
        Dtype::U8 => dtype_u8(),
        Dtype::U16 => dtype_u16(),
        Dtype::F32 => dtype_f32(),
    }
}

/// Convert a `DLDataType` to `Dtype`, or return a `ValueError`.
pub fn dl_to_dtype(dt: DLDataType) -> PyResult<Dtype> {
    match (dt.code, dt.bits, dt.lanes) {
        (c, 8, 1) if c == K_DL_UINT => Ok(Dtype::U8),
        (c, 16, 1) if c == K_DL_UINT => Ok(Dtype::U16),
        (c, 32, 1) if c == K_DL_FLOAT => Ok(Dtype::F32),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "from_dlpack: unsupported DLPack dtype \
             (code={code}, bits={bits}, lanes={lanes}); \
             expected uint8, uint16, or float32",
            code = dt.code,
            bits = dt.bits,
            lanes = dt.lanes,
        ))),
    }
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

