//! DLPack interoperability for [`Tensor`].
//!
//! Provides [`tensor_to_dlpack`] (export) and [`tensor_from_dlpack_raw`] (import).
//! Both CPU and CUDA device tensors are accepted at the container level;
//! host-slicing of device tensors already panics in [`TensorStorage`], which is
//! the correct safety barrier.

use std::sync::Arc;

use dlpack_rs::{
    ffi::{DLDevice, DLManagedTensor},
    safe::{self, TensorInfo},
};

use crate::{
    allocator::host_alloc,
    storage::MemoryDomain,
    tensor::{Tensor, TensorError},
};

// ── DlpackElem trait ─────────────────────────────────────────────────────────

/// Element types that can be represented as DLPack data types.
///
/// Implement this for any scalar type you want to export/import via DLPack.
pub trait DlpackElem {
    /// Returns the DLPack data-type descriptor for this element type.
    fn dl_dtype() -> dlpack_rs::ffi::DLDataType;
}

macro_rules! impl_dlpack_elem {
    ($ty:ty, $code:expr, $bits:expr, $lanes:expr) => {
        impl DlpackElem for $ty {
            #[inline]
            fn dl_dtype() -> dlpack_rs::ffi::DLDataType {
                dlpack_rs::ffi::DLDataType {
                    code: $code as u8,
                    bits: $bits,
                    lanes: $lanes,
                }
            }
        }
    };
}

// DLDataTypeCode values: kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4, kDLBool=6
impl_dlpack_elem!(u8, 1u32, 8, 1);
impl_dlpack_elem!(u16, 1u32, 16, 1);
impl_dlpack_elem!(i32, 0u32, 32, 1);
impl_dlpack_elem!(i64, 0u32, 64, 1);
impl_dlpack_elem!(f32, 2u32, 32, 1);
impl_dlpack_elem!(f64, 2u32, 64, 1);

// ── tensor_to_dlpack ─────────────────────────────────────────────────────────

/// Exports a [`Tensor`] to a heap-allocated `DLManagedTensor`.
///
/// The tensor itself is the keepalive: the consumer's deleter will drop it, freeing
/// the buffer. The export is zero-copy.
///
/// # Arguments
///
/// * `t` - The tensor to export; ownership is transferred into the `DLManagedTensor` keepalive.
///
/// # Returns
///
/// A raw pointer to a `DLManagedTensor` that the consumer owns.  The consumer
/// MUST call `managed.deleter(managed)` when done.
///
/// # Panics
///
/// Panics if `T` does not implement [`DlpackElem`] (compile-time, via trait bound).
pub fn tensor_to_dlpack<T, const N: usize>(t: Tensor<T, N>) -> *mut DLManagedTensor
where
    T: DlpackElem + 'static + Send,
{
    let dtype = T::dl_dtype();
    let device = match t.storage.domain() {
        MemoryDomain::Host => safe::cpu_device(),
        MemoryDomain::Device { id } => safe::cuda_device(id),
        MemoryDomain::Unified { id } => dlpack_rs::ffi::DLDevice {
            device_type: dlpack_rs::ffi::DLDeviceType::kDLCUDAManaged,
            device_id: id,
        },
    };
    let data_ptr = t.storage.as_ptr() as *mut std::ffi::c_void;
    let shape: Vec<i64> = t.shape.iter().map(|&s| s as i64).collect();

    let info = TensorInfo::contiguous(data_ptr, device, dtype, shape);
    // SAFETY: `t` (moved into Box as keepalive) owns the buffer and outlives the
    // DLManagedTensor. The deleter provided by `safe::pack` will drop the Box,
    // which drops `t` and its storage.
    safe::pack(t, info)
}

// ── tensor_from_dlpack_raw ────────────────────────────────────────────────────

/// Error variants specific to DLPack import.
#[derive(Debug)]
pub enum DlpackError {
    /// The ndim in the DLManagedTensor does not match the const generic `N`.
    NdimMismatch {
        /// The expected number of dimensions.
        expected: usize,
        /// The actual number of dimensions found in the DLManagedTensor.
        got: i32,
    },
    /// The element type in the DLManagedTensor does not match `T`.
    DtypeMismatch,
    /// The tensor is not C-contiguous (non-null strides).
    NotContiguous,
    /// The pointer in the DLManagedTensor is null.
    NullPointer,
    /// Integer overflow computing element count or byte length from the shape.
    Overflow,
    /// A shape dimension is zero or negative.
    NegativeDimension,
    /// Allocation error forwarded from the storage layer.
    TensorError(TensorError),
}

impl core::fmt::Display for DlpackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NdimMismatch { expected, got } => {
                write!(f, "ndim mismatch: expected {expected}, got {got}")
            }
            Self::DtypeMismatch => write!(f, "dtype mismatch"),
            Self::NotContiguous => write!(f, "tensor is not C-contiguous"),
            Self::NullPointer => write!(f, "null data pointer in DLManagedTensor"),
            Self::Overflow => write!(f, "integer overflow computing tensor size from shape"),
            Self::NegativeDimension => write!(f, "shape dimension must be positive"),
            Self::TensorError(e) => write!(f, "tensor error: {e}"),
        }
    }
}

impl std::error::Error for DlpackError {}

/// Imports a [`Tensor`] from a raw `DLManagedTensor` pointer without copying data.
///
/// Accepts both CPU (`kDLCPU`) and CUDA (`kDLCUDA`) tensors at the container level.
/// CUDA tensors will have `MemoryDomain::Device` — calling `as_slice` on them panics,
/// which is the correct behaviour (use device-transfer APIs).
///
/// `keepalive` keeps the source object alive for the lifetime of the returned tensor.
/// The caller must NOT call the DLManagedTensor's own deleter after passing it here;
/// that responsibility transfers to `keepalive`.
///
/// # Arguments
///
/// * `mt` - Raw pointer to a valid, initialised `DLManagedTensor`; must be non-null.
/// * `keepalive` - An `Arc` keepalive that keeps the source data alive for the lifetime
///   of the returned `Tensor`. The caller must NOT call the tensor's own deleter after
///   passing it here.
///
/// # Returns
///
/// A zero-copy [`Tensor`] borrowing the buffer pointed to by `mt`.
///
/// # Safety
///
/// - `mt` must be non-null and point to a valid, initialised `DLManagedTensor`.
/// - The `dl_tensor.data` pointer must be valid for at least the number of bytes
///   implied by `ndim`, `shape`, and `sizeof(T)` for the full lifetime of `keepalive`.
/// - Strides must be `null` (contiguous) or describe a C-contiguous layout (not validated).
/// - The dtype in `mt` must match `T`.
///
/// # Errors
///
/// Returns `Err(DlpackError::*)` for ndim mismatch, dtype mismatch (including `lanes != 1`),
/// non-contiguous strides, or null pointer.
pub unsafe fn tensor_from_dlpack_raw<T, const N: usize>(
    mt: *mut DLManagedTensor,
    keepalive: Arc<dyn core::any::Any + Send + Sync>,
) -> Result<Tensor<T, N>, DlpackError>
where
    T: DlpackElem + Clone,
{
    // SAFETY: caller guarantees mt is non-null and valid.
    let dl = unsafe { &(*mt).dl_tensor };

    // Validate ndim.
    if dl.ndim != N as i32 {
        return Err(DlpackError::NdimMismatch {
            expected: N,
            got: dl.ndim,
        });
    }

    // Validate dtype (code, bits, and lanes — lanes != 1 means a SIMD-packed type
    // that would yield the wrong element count and cause OOB access).
    let expected = T::dl_dtype();
    if dl.dtype.code != expected.code || dl.dtype.bits != expected.bits || dl.dtype.lanes != 1 {
        return Err(DlpackError::DtypeMismatch);
    }

    // Accept null strides (C-contiguous per DLPack spec) OR explicit strides that
    // equal the C-contiguous strides for this shape.  Reject genuinely non-contiguous
    // strides so we don't silently produce a Tensor with wrong element addressing.
    if !dl.strides.is_null() {
        // SAFETY: dl.strides is valid for dl.ndim elements (same guarantee as dl.shape).
        let strides_slice = unsafe { std::slice::from_raw_parts(dl.strides, N) };
        // Compute expected C-contiguous strides: stride[i] = product(shape[i+1..]).
        // Read shape_slice eagerly; it was not yet materialised at this point, so recompute
        // here using the raw pointer (shape_slice is re-read below after this block).
        let shape_raw = unsafe { std::slice::from_raw_parts(dl.shape, N) };
        let expected_contiguous = {
            let mut s = [0i64; N];
            let mut acc = 1i64;
            for i in (0..N).rev() {
                s[i] = acc;
                acc = acc.saturating_mul(shape_raw[i]);
            }
            s
        };
        if strides_slice != expected_contiguous {
            return Err(DlpackError::NotContiguous);
        }
    }

    // Validate data pointer.
    if dl.data.is_null() {
        return Err(DlpackError::NullPointer);
    }

    // Read shape.
    // SAFETY: dl.shape is valid for dl.ndim elements (caller contract + DLPack spec).
    let shape_slice = unsafe { std::slice::from_raw_parts(dl.shape, N) };
    for &d in shape_slice {
        if d <= 0 {
            return Err(DlpackError::NegativeDimension);
        }
    }
    let shape: [usize; N] = std::array::from_fn(|i| shape_slice[i] as usize);

    // Compute byte length with checked arithmetic to guard against hostile/corrupt shapes.
    let n_elems = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or(DlpackError::Overflow)?;
    let len_bytes = n_elems
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(DlpackError::Overflow)?;

    // Map device (the device id is encoded in the returned domain).
    let (domain, _) = map_dl_device(dl.device);

    // Convert byte_offset safely: u64 -> usize (fails on 32-bit if value > usize::MAX).
    let byte_offset = usize::try_from(dl.byte_offset).map_err(|_| DlpackError::Overflow)?;

    // Build storage (borrowed, no dealloc).
    // SAFETY: data pointer is valid for len_bytes (caller contract), keepalive keeps
    // the source alive, domain correctly reflects the DLDevice.
    let data_ptr = unsafe { (dl.data as *const u8).add(byte_offset) as *const T };
    let storage = unsafe {
        crate::storage::TensorStorage::from_borrowed(
            data_ptr,
            len_bytes,
            host_alloc(),
            domain,
            keepalive,
        )
    };

    // Build tensor.
    let strides = crate::tensor::get_strides_from_shape(shape);
    Ok(Tensor {
        storage,
        shape,
        strides,
    })
}

/// Maps a `DLDevice` to `(MemoryDomain, device_id)`.
fn map_dl_device(device: DLDevice) -> (MemoryDomain, i32) {
    use dlpack_rs::ffi::{DLDeviceType, K_DL_CPU};
    if device.device_type == K_DL_CPU {
        (MemoryDomain::Host, 0)
    } else if device.device_type == DLDeviceType::kDLCUDAManaged {
        // kDLCUDAManaged -> Unified (CUDA managed / unified memory)
        (
            MemoryDomain::Unified {
                id: device.device_id,
            },
            device.device_id,
        )
    } else {
        // kDLCUDA and any other device type -> Device domain.
        (
            MemoryDomain::Device {
                id: device.device_id,
            },
            device.device_id,
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // ── round-trip: owned CPU tensor ──────────────────────────────────────────

    #[test]
    fn test_tensor_to_dlpack_and_deleter() {
        // Drop counter: we build the tensor around a borrowed buffer whose keepalive
        // is a Guard. When the DLPack deleter fires it drops the tensor, which drops
        // the ForeignResource, which drops the keepalive Arc, which fires the Guard.
        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        // Build a borrowed tensor whose lifetime is tied to a DropCounter keepalive.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(DropCounter(counter.clone()));
        let tensor = unsafe {
            use crate::storage::TensorStorage;
            let storage = TensorStorage::<f32>::from_borrowed(
                data.as_ptr(),
                data.len() * std::mem::size_of::<f32>(),
                crate::allocator::host_alloc(),
                crate::storage::MemoryDomain::Host,
                keep,
            );
            let shape = [4usize];
            let strides = crate::tensor::get_strides_from_shape(shape);
            Tensor {
                storage,
                shape,
                strides,
            }
        };

        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let mt = tensor_to_dlpack(tensor);
        assert!(!mt.is_null());

        // invoke deleter — this drops the tensor (and the keepalive Arc)
        unsafe {
            let deleter = (*mt).deleter;
            if let Some(del) = deleter {
                del(mt);
            }
        }

        // drop counter must have fired exactly once
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "keepalive must drop exactly once"
        );
        // keep `data` alive until here so the borrowed pointer is valid above.
        drop(data);
    }

    // ── round-trip: from_borrowed with drop counter ────────────────────────────

    #[test]
    fn test_from_dlpack_raw_cpu_round_trip() {
        let counter = Arc::new(AtomicUsize::new(0));

        // Build a synthetic CPU DLManagedTensor over a heap buffer
        let data: Vec<f32> = vec![10.0, 20.0, 30.0];
        let shape_arr: Vec<i64> = vec![3i64];

        let data_ptr = data.as_ptr();

        struct Guard {
            _data: Vec<f32>,
            counter: Arc<AtomicUsize>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }
        let guard = Arc::new(Guard {
            _data: data,
            counter: counter.clone(),
        });

        // Build a minimal DLManagedTensor on the stack (strides=null = contiguous)
        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 1,
            dtype: f32::dl_dtype(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let tensor =
            unsafe { tensor_from_dlpack_raw::<f32, 1>(&mut managed as *mut _, guard as Arc<_>) }
                .unwrap();

        // Verify domain and device_id
        assert_eq!(tensor.storage.domain(), MemoryDomain::Host);
        assert_eq!(tensor.storage.device_id(), 0);

        // Verify data
        let slice = tensor.storage.as_slice();
        assert_eq!(slice, &[10.0f32, 20.0, 30.0]);

        drop(tensor); // drops keepalive Arc
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "guard must drop exactly once"
        );
    }

    // ── synthetic CUDA device import ──────────────────────────────────────────

    #[test]
    fn test_from_dlpack_raw_cuda_device_import() {
        // Prove CUDA-readiness without real CUDA hardware.
        // We create a DLManagedTensor claiming kDLCUDA device 2 and verify:
        // 1. Import succeeds (domain==Device, device_id==2)
        // 2. as_slice panics (correct safety barrier)

        use dlpack_rs::ffi::{DLDevice, K_DL_CUDA};

        // Use a real allocation as the fake "device" ptr (we won't dereference it via Rust).
        let data: Vec<f32> = vec![0.0f32; 4];
        let data_ptr = data.as_ptr();
        let shape_arr: Vec<i64> = vec![4i64];

        let device = DLDevice {
            device_type: K_DL_CUDA,
            device_id: 2,
        };

        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device,
            ndim: 1,
            dtype: f32::dl_dtype(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        // keepalive keeps `data` alive
        let keepalive: Arc<dyn core::any::Any + Send + Sync> = Arc::new(data);

        let tensor =
            unsafe { tensor_from_dlpack_raw::<f32, 1>(&mut managed as *mut _, keepalive) }.unwrap();

        // CUDA-readiness assertions
        assert!(
            matches!(tensor.storage.domain(), MemoryDomain::Device { .. }),
            "must be Device domain"
        );
        assert_eq!(tensor.storage.device_id(), 2, "device_id must be 2");
    }

    #[test]
    #[should_panic(expected = "non-host-accessible")]
    fn test_cuda_tensor_slice_panics() {
        use dlpack_rs::ffi::{DLDevice, K_DL_CUDA};

        let data: Vec<f32> = vec![0.0f32; 4];
        let data_ptr = data.as_ptr();
        let shape_arr: Vec<i64> = vec![4i64];

        let device = DLDevice {
            device_type: K_DL_CUDA,
            device_id: 0,
        };
        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device,
            ndim: 1,
            dtype: f32::dl_dtype(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn core::any::Any + Send + Sync> = Arc::new(data);
        let tensor =
            unsafe { tensor_from_dlpack_raw::<f32, 1>(&mut managed as *mut _, keepalive) }.unwrap();

        // This MUST panic:
        let _ = tensor.storage.as_slice();
    }

    // ── explicit C-contiguous strides ─────────────────────────────────────────

    /// A DLManagedTensor with explicit C-contiguous strides must import successfully.
    ///
    /// For shape [H, W, C] = [4, 3, 2] the C-contiguous strides are [6, 2, 1].
    #[test]
    fn test_from_dlpack_explicit_contiguous_strides_accepted() {
        // shape [4, 3, 2] → C-contiguous strides [6, 2, 1]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shape_arr: Vec<i64> = vec![4, 3, 2];
        let strides_arr: Vec<i64> = vec![6, 2, 1]; // explicit C-contiguous strides

        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: f32::dl_dtype(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: strides_arr.as_ptr() as *mut i64,
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn core::any::Any + Send + Sync> = Arc::new(data.clone());
        let tensor = unsafe { tensor_from_dlpack_raw::<f32, 3>(&mut managed as *mut _, keepalive) };
        assert!(
            tensor.is_ok(),
            "explicit C-contiguous strides must be accepted, got: {:?}",
            tensor.err().map(|e| e.to_string())
        );
        // Verify data is accessible and correct.
        let t = tensor.unwrap();
        assert_eq!(t.shape, [4, 3, 2]);
        let slice = t.storage.as_slice();
        assert_eq!(slice.len(), 24);
        assert_eq!(slice[0], 0.0f32);
        assert_eq!(slice[23], 23.0f32);
    }

    /// A DLManagedTensor with genuinely non-contiguous strides must be rejected.
    #[test]
    fn test_from_dlpack_non_contiguous_strides_rejected() {
        // shape [4, 3] with strides [4, 1] — NOT C-contiguous (should be [3, 1]).
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let shape_arr: Vec<i64> = vec![4, 3];
        let strides_arr: Vec<i64> = vec![4, 1]; // non-contiguous: C-contiguous would be [3, 1]

        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 2,
            dtype: f32::dl_dtype(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: strides_arr.as_ptr() as *mut i64,
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn core::any::Any + Send + Sync> = Arc::new(data);
        let result = unsafe { tensor_from_dlpack_raw::<f32, 2>(&mut managed as *mut _, keepalive) };
        assert!(
            matches!(result, Err(DlpackError::NotContiguous)),
            "non-contiguous strides must return NotContiguous, got: {:?}",
            result.map(|_| ())
        );
    }
}
