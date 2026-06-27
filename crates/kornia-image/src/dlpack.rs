//! DLPack interoperability for typed [`Image<T, C, A>`].
//!
//! Provides [`image_to_dlpack`] (export) and [`image_from_dlpack`] (import).
//! Both CPU and CUDA device images are accepted at the container level.

use std::{any::Any, sync::Arc};

use dlpack_rs::ffi::DLManagedTensor;
use kornia_tensor::{dlpack::tensor_to_dlpack, storage::TensorStorage, MemoryDomain};

use crate::{
    allocator::{ForeignAllocator, ImageAllocator},
    error::ImageError,
    image::Image,
};

/// Re-export [`DlpackElem`] so callers get it from one place.
pub use kornia_tensor::dlpack::DlpackElem;

// ── image_to_dlpack ──────────────────────────────────────────────────────────

/// Exports a typed [`Image<T, C, A>`] to a heap-allocated `DLManagedTensor`.
///
/// The image is moved into the DLManagedTensor's keepalive; the consumer MUST call
/// `managed.deleter(managed)` when done to free the buffer.  Export is zero-copy.
/// Shape is encoded as `[H, W, C]` (HWC layout); strides are contiguous C-order.
///
/// # Arguments
///
/// * `img` - The image to export; ownership is transferred into the `DLManagedTensor` keepalive.
///
/// # Returns
///
/// Raw pointer to a `DLManagedTensor` that the consumer owns.
///
/// # Errors
///
/// This function does not return an error. The allocation is infallible for heap-backed images.
pub fn image_to_dlpack<T, const C: usize, A>(img: Image<T, C, A>) -> *mut DLManagedTensor
where
    T: DlpackElem + 'static + Send,
    A: ImageAllocator + Send + 'static,
{
    // Image<T,C,A> is a newtype over Tensor<T,3,A>; delegate directly.
    tensor_to_dlpack(img.0)
}

// ── image_from_dlpack ────────────────────────────────────────────────────────

/// Imports a typed [`Image<T, C, ForeignAllocator>`] from a raw `DLManagedTensor` pointer.
///
/// Zero-copy: the `keepalive` keeps the source alive. The caller must NOT call the
/// tensor's own deleter after passing it here.
///
/// # Shape
/// The tensor must be 3-dimensional with layout `[H, W, C]`.
/// The const generic `C` is validated against the runtime channel count.
///
/// # Device support
/// Both CPU (`kDLCPU`) and CUDA (`kDLCUDA`) tensors are accepted at the container
/// level. CUDA images have `MemoryDomain::Device`; calling `as_slice()` on them will
/// panic — use device-transfer APIs instead.
///
/// # Arguments
///
/// * `mt` - Raw pointer to a valid, initialised `DLManagedTensor`; must be non-null (see `# Safety`).
/// * `keepalive` - An `Arc` keepalive that keeps the source data alive for the lifetime of the
///   returned `Image`. The caller must NOT call the tensor's own deleter after passing it here.
///
/// # Errors
/// - [`ImageError::DlpackShapeError`] if `T` does not match the tensor's dtype, or if
///   ndim != 3, or the tensor is non-contiguous, or the pointer is null, or overflow.
/// - [`ImageError::InvalidImageShape`] wraps a `TensorError` for storage-layer issues.
/// - [`ImageError::InvalidChannelShape`] if runtime channel count != `C`.
///
/// # Safety
/// - `mt` must be non-null and point to a valid, initialised `DLManagedTensor`.
/// - `dl_tensor.data` must be valid for at least `H*W*C*size_of::<T>()` bytes for
///   the full lifetime of `keepalive`.
/// - Strides must be null (C-contiguous per spec) or explicit C-contiguous `[W*C, C, 1]`.
pub unsafe fn image_from_dlpack<T, const C: usize>(
    mt: *mut DLManagedTensor,
    keepalive: Arc<dyn Any + Send + Sync>,
) -> Result<Image<T, C, ForeignAllocator>, ImageError>
where
    T: DlpackElem + Clone,
{
    use dlpack_rs::ffi::K_DL_CPU;

    // SAFETY: caller guarantees mt is non-null and valid.
    let dl = unsafe { &(*mt).dl_tensor };

    // 1. Validate ndim == 3.
    if dl.ndim != 3 {
        return Err(ImageError::DlpackShapeError(format!(
            "ndim mismatch: expected 3, got {}",
            dl.ndim
        )));
    }

    // 2. Validate dtype matches T.
    let expected = T::dl_dtype();
    if dl.dtype.code != expected.code || dl.dtype.bits != expected.bits {
        return Err(ImageError::DlpackShapeError(
            "dtype mismatch: requested T does not match tensor dtype".to_string(),
        ));
    }

    // 3. Validate data pointer.
    if dl.data.is_null() {
        return Err(ImageError::DlpackShapeError(
            "null data pointer in DLManagedTensor".to_string(),
        ));
    }

    // 4. Read shape [H, W, actual_C].
    // SAFETY: dl.shape is valid for dl.ndim (== 3) elements (caller contract + DLPack spec).
    let shape_slice = unsafe { std::slice::from_raw_parts(dl.shape, 3) };

    // 5. Validate that all shape dimensions are positive (before casting to usize).
    for (i, &dim) in shape_slice.iter().enumerate() {
        if dim <= 0 {
            return Err(ImageError::DlpackShapeError(format!(
                "dimension[{}] must be positive, got {}",
                i, dim
            )));
        }
    }

    let shape: [usize; 3] = [
        shape_slice[0] as usize,
        shape_slice[1] as usize,
        shape_slice[2] as usize,
    ];

    // 6. Validate channel count.
    let actual_c = shape[2];
    if actual_c != C {
        return Err(ImageError::InvalidChannelShape(actual_c, C));
    }

    // 7. Validate strides: null means C-contiguous per DLPack spec; non-null strides
    //    must equal the explicit C-contiguous layout [W*C, C, 1] for HWC tensors.
    if !dl.strides.is_null() {
        // SAFETY: dl.strides is valid for dl.ndim (== 3) elements per DLPack spec.
        let s = unsafe { std::slice::from_raw_parts(dl.strides, 3) };
        let expected_strides = [(shape[1] * shape[2]) as i64, shape[2] as i64, 1i64];
        if s != expected_strides {
            return Err(ImageError::DlpackShapeError(format!(
                "tensor is not C-contiguous: strides {:?}, expected {:?}",
                s, expected_strides
            )));
        }
    }

    // 8. Compute byte length with checked arithmetic.
    let n_elems = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| {
            ImageError::DlpackShapeError("integer overflow computing tensor size".to_string())
        })?;
    let len_bytes = n_elems
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            ImageError::DlpackShapeError("integer overflow computing byte length".to_string())
        })?;

    // 9. Map device.
    let (domain, _device_id) = if dl.device.device_type == K_DL_CPU {
        (MemoryDomain::Host, 0i32)
    } else {
        (
            MemoryDomain::Device {
                id: dl.device.device_id,
            },
            dl.device.device_id,
        )
    };

    // 10. Apply byte_offset.
    let byte_offset = usize::try_from(dl.byte_offset)
        .map_err(|_| ImageError::DlpackShapeError("byte_offset overflows usize".to_string()))?;

    // 11. Build storage with kornia-image's ForeignAllocator (no-op dealloc).
    // SAFETY: data pointer is valid for len_bytes (caller contract); keepalive keeps
    // the source alive; domain correctly reflects the DLDevice.
    let data_ptr = unsafe { (dl.data as *const u8).add(byte_offset) as *const T };
    let storage = unsafe {
        TensorStorage::from_borrowed(data_ptr, len_bytes, ForeignAllocator, domain, keepalive)
    };

    // 12. Build Tensor<T, 3, ForeignAllocator>.
    // Compute C-contiguous strides for HWC layout: [W*C, C, 1].
    let strides: [usize; 3] = [shape[1] * shape[2], shape[2], 1];
    let tensor = kornia_tensor::Tensor {
        storage,
        shape,
        strides,
    };

    // 13. Wrap into Image<T, C, ForeignAllocator>.
    // Safe because: ndim==3, shape[2]==C (validated above), ForeignAllocator is correct.
    Ok(Image(tensor))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use dlpack_rs::ffi::{DLManagedTensor, DLTensor};
    use dlpack_rs::safe;
    use kornia_tensor::CpuAllocator;

    use crate::{allocator::CpuAllocator as ImageCpuAllocator, image::ImageSize};

    // ── Test 1a: image_to_dlpack with ForeignAllocator drop-counter ─────────────
    //
    // Build an Image<u8,3,ForeignAllocator> with a drop-counter keepalive and verify
    // the deleter fires the counter exactly once.

    #[test]
    fn test_image_to_dlpack_foreign_deleter_fires_once() {
        let counter = Arc::new(AtomicUsize::new(0));

        struct DropGuard {
            counter: Arc<AtomicUsize>,
            _data: Vec<u8>,
        }
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Build a 2×3 RGB u8 image with values 0..18 via a borrowed foreign buffer.
        let data: Vec<u8> = (0u8..18).collect();
        let data_ptr = data.as_ptr();
        let shape_arr: Vec<i64> = vec![2i64, 3, 3];
        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(DropGuard {
            counter: counter.clone(),
            _data: data,
        });

        let dl_tensor = DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: safe::dtype_u8(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        // Import as Image<u8,3,ForeignAllocator> (keepalive holds the DropGuard).
        let img = unsafe { image_from_dlpack::<u8, 3>(&mut managed as *mut _, keepalive) }.unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 0, "not dropped yet");

        // Export to DLPack — moves img into the DLPack keepalive.
        let mt = image_to_dlpack(img);
        assert!(!mt.is_null(), "mt must be non-null");

        unsafe {
            let dl = &(*mt).dl_tensor;
            // Verify ndim==3, shape==[2,3,3]
            assert_eq!(dl.ndim, 3);
            let s = std::slice::from_raw_parts(dl.shape, 3);
            assert_eq!(s, &[2i64, 3, 3]);
            // dtype code must be K_DL_UINT (1) and bits==8
            use dlpack_rs::ffi::K_DL_UINT;
            assert_eq!(dl.dtype.code, K_DL_UINT as u8);
            assert_eq!(dl.dtype.bits, 8);
        }

        // Invoke deleter — must fire the keepalive drop exactly once.
        unsafe {
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
        }

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "keepalive must drop exactly once after deleter"
        );
    }

    // ── Test 1b: image_to_dlpack with CpuAllocator ──────────────────────────────
    //
    // Build a real Image<u8,3,CpuAllocator> from known data, export via image_to_dlpack,
    // verify shape/ndim/dtype, fire the deleter (no panic expected).

    #[test]
    fn test_image_to_dlpack_cpu_allocator() {
        let data: Vec<u8> = (0u8..18).collect();
        let img = Image::<u8, 3, ImageCpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 3,
            },
            data,
            ImageCpuAllocator,
        )
        .unwrap();

        let mt = image_to_dlpack(img);
        assert!(!mt.is_null(), "mt must be non-null");

        unsafe {
            let dl = &(*mt).dl_tensor;
            assert_eq!(dl.ndim, 3, "ndim must be 3");
            let s = std::slice::from_raw_parts(dl.shape, 3);
            assert_eq!(s, &[2i64, 3, 3], "shape must be [H=2, W=3, C=3]");
            use dlpack_rs::ffi::K_DL_UINT;
            assert_eq!(
                dl.dtype.code, K_DL_UINT as u8,
                "dtype code must be K_DL_UINT"
            );
            assert_eq!(dl.dtype.bits, 8, "dtype bits must be 8");

            // Fire the deleter — must not panic.
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
        }
    }

    // ── Test 2: image_from_dlpack::<f32,3> with drop-counter guard ──────────────

    #[test]
    fn test_image_from_dlpack_f32_3ch_with_drop_counter() {
        let counter = Arc::new(AtomicUsize::new(0));

        // Build a 2×3 F32 RGB buffer (18 elements).
        let data: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let shape_arr: Vec<i64> = vec![2i64, 3, 3];

        struct Guard {
            counter: Arc<AtomicUsize>,
            _data: Vec<f32>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        let data_ptr = data.as_ptr();
        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(Guard {
            counter: counter.clone(),
            _data: data,
        });

        let dl_tensor = DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: safe::dtype_f32(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let img =
            unsafe { image_from_dlpack::<f32, 3>(&mut managed as *mut _, keepalive) }.unwrap();

        // Verify shape [2,3,3], domain==Host, device_id==0
        assert_eq!(img.0.shape, [2, 3, 3]);
        assert_eq!(img.0.storage.domain(), MemoryDomain::Host);
        assert_eq!(img.0.storage.device_id(), 0);

        // Drop the image; the keepalive guard should fire.
        drop(img);
        assert_eq!(counter.load(Ordering::SeqCst), 1, "guard must drop once");
    }

    // ── Test 3: dtype-mismatch returns DlpackShapeError ─────────────────────────

    #[test]
    fn test_image_from_dlpack_dtype_mismatch() {
        // dtype_u8 but requesting f32
        let data: Vec<u8> = vec![0u8; 2 * 3 * 3];
        let shape_arr: Vec<i64> = vec![2i64, 3, 3];

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: safe::dtype_u8(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(data);
        let result = unsafe { image_from_dlpack::<f32, 3>(&mut managed as *mut _, keepalive) };

        assert!(
            matches!(result, Err(ImageError::DlpackShapeError(_))),
            "dtype mismatch must return DlpackShapeError"
        );
    }

    // ── Test 4: channel-mismatch returns InvalidChannelShape ────────────────────

    #[test]
    fn test_image_from_dlpack_channel_mismatch() {
        // shape [2,3,4] (RGBA = 4 channels) but requesting C=3
        let data: Vec<u8> = vec![0u8; 2 * 3 * 4];
        let shape_arr: Vec<i64> = vec![2i64, 3, 4];

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: safe::dtype_u8(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(data);
        let result = unsafe { image_from_dlpack::<u8, 3>(&mut managed as *mut _, keepalive) };

        assert!(
            matches!(result, Err(ImageError::InvalidChannelShape(4, 3))),
            "channel mismatch must return InvalidChannelShape(4, 3)"
        );
    }

    // ── Test 5: synthetic CUDA DLManagedTensor imports with Device domain ────────

    #[test]
    fn test_image_from_dlpack_cuda_device_domain() {
        use dlpack_rs::ffi::{DLDevice, K_DL_CUDA};

        // shape [2,4,1], f32, CUDA device 1
        let data: Vec<f32> = vec![0.0f32; 2 * 4 * 1];
        let shape_arr: Vec<i64> = vec![2i64, 4, 1];

        let device = DLDevice {
            device_type: K_DL_CUDA,
            device_id: 1,
        };

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device,
            ndim: 3,
            dtype: safe::dtype_f32(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(data);
        let img =
            unsafe { image_from_dlpack::<f32, 1>(&mut managed as *mut _, keepalive) }.unwrap();

        // Verify storage domain == Device and device_id == 1.
        assert_eq!(
            img.0.storage.domain(),
            MemoryDomain::Device { id: 1 },
            "must be Device domain"
        );
        assert_eq!(img.0.storage.device_id(), 1, "device_id must be 1");
        // Do NOT dereference/slice the image (device pointer).
    }

    // ── Test 6: ndim mismatch returns DlpackShapeError ──────────────────────────

    #[test]
    fn test_image_from_dlpack_ndim_mismatch() {
        // 1D tensor ndim=1
        let data: Vec<u8> = vec![0u8; 6];
        let shape_arr: Vec<i64> = vec![6i64];

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 1,
            dtype: safe::dtype_u8(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(data);
        let result = unsafe { image_from_dlpack::<u8, 3>(&mut managed as *mut _, keepalive) };

        assert!(
            matches!(result, Err(ImageError::DlpackShapeError(_))),
            "ndim mismatch must return DlpackShapeError"
        );
    }

    // ── Test 7: explicit C-contiguous strides are accepted ───────────────────────
    //
    // PyTorch and other producers may emit non-null but C-contiguous strides.
    // For a 2×3 RGB tensor, correct HWC strides are [W*C, C, 1] = [9, 3, 1].
    // image_from_dlpack must accept these without error.

    #[test]
    fn test_image_from_dlpack_explicit_c_contiguous_strides() {
        let data: Vec<u8> = (0u8..18).collect();
        let shape_arr: Vec<i64> = vec![2i64, 3, 3];
        // Explicit C-contiguous strides for HWC 2×3×3: [W*C, C, 1] = [9, 3, 1].
        let strides_arr: Vec<i64> = vec![9i64, 3, 1];

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: safe::dtype_u8(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: strides_arr.as_ptr() as *mut i64,
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(data);
        let result = unsafe { image_from_dlpack::<u8, 3>(&mut managed as *mut _, keepalive) };

        assert!(
            result.is_ok(),
            "explicit C-contiguous strides [9,3,1] must be accepted, got: {:?}",
            result.err()
        );
        let img = result.unwrap();
        assert_eq!(img.0.shape, [2, 3, 3], "shape must be [2,3,3]");
    }

    // ── Test 8: non-C-contiguous strides are rejected ───────────────────────────

    #[test]
    fn test_image_from_dlpack_non_contiguous_strides_rejected() {
        let data: Vec<u8> = vec![0u8; 18];
        let shape_arr: Vec<i64> = vec![2i64, 3, 3];
        // Wrong strides (e.g. Fortran-order or padded row stride).
        let strides_arr: Vec<i64> = vec![1i64, 2, 6];

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: safe::dtype_u8(),
            shape: shape_arr.as_ptr() as *mut i64,
            strides: strides_arr.as_ptr() as *mut i64,
            byte_offset: 0,
        };
        let mut managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(data);
        let result = unsafe { image_from_dlpack::<u8, 3>(&mut managed as *mut _, keepalive) };

        assert!(
            matches!(result, Err(ImageError::DlpackShapeError(_))),
            "non-C-contiguous strides must return DlpackShapeError"
        );
    }
}
