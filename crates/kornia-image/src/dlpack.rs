//! DLPack interoperability for [`DynImageBuf`].
//!
//! Provides [`dynimage_to_dlpack`] (export) and [`dynimage_from_dlpack_raw`] (import).
//! Both CPU and CUDA device images are accepted at the container level; host-slicing
//! of device images already errors in [`DynImageBuf::as_image`], which is the correct
//! safety barrier.

use std::{any::Any, sync::Arc};

use dlpack_rs::{
    ffi::{DLDevice, DLManagedTensor},
    safe::{self, TensorInfo},
};
use kornia_tensor::storage::MemoryDomain;

use crate::{
    color_space::ColorSpace,
    dyn_image_buf::DynImageBuf,
    error::ImageError,
    image::PixelFormat,
};

// ── pixel-format <-> DLDataType helpers ──────────────────────────────────────

/// Maps a [`PixelFormat`] to the corresponding DLPack data type descriptor.
fn pixel_format_to_dl_dtype(fmt: PixelFormat) -> dlpack_rs::ffi::DLDataType {
    match fmt {
        PixelFormat::U8 => safe::dtype_u8(),
        PixelFormat::U16 => safe::dtype_u16(),
        PixelFormat::F32 => safe::dtype_f32(),
    }
}

/// Maps a DLPack data type descriptor back to a [`PixelFormat`].
///
/// Returns `None` if the dtype is not one of the three supported formats.
fn dl_dtype_to_pixel_format(dtype: dlpack_rs::ffi::DLDataType) -> Option<PixelFormat> {
    use dlpack_rs::ffi::{K_DL_FLOAT, K_DL_UINT};
    match (dtype.code, dtype.bits) {
        (c, 8) if c == K_DL_UINT => Some(PixelFormat::U8),
        (c, 16) if c == K_DL_UINT => Some(PixelFormat::U16),
        (c, 32) if c == K_DL_FLOAT => Some(PixelFormat::F32),
        _ => None,
    }
}

// ── device helpers ────────────────────────────────────────────────────────────

/// Maps a [`MemoryDomain`] + device id to a [`DLDevice`].
fn memory_domain_to_dl_device(domain: MemoryDomain, device_id: i32) -> DLDevice {
    match domain {
        MemoryDomain::Host => safe::cpu_device(),
        MemoryDomain::Device => safe::cuda_device(device_id),
    }
}

/// Maps a [`DLDevice`] to `(MemoryDomain, device_id)`.
fn dl_device_to_memory_domain(device: DLDevice) -> (MemoryDomain, i32) {
    use dlpack_rs::ffi::K_DL_CPU;
    if device.device_type == K_DL_CPU {
        (MemoryDomain::Host, 0)
    } else {
        // kDLCUDA and any other device type -> Device domain.
        (MemoryDomain::Device, device.device_id)
    }
}

// ── dynimage_to_dlpack ────────────────────────────────────────────────────────

/// Exports a [`DynImageBuf`] to a heap-allocated `DLManagedTensor`.
///
/// The image itself is the keep-alive: the consumer's deleter will drop it, freeing
/// the buffer.  The export is zero-copy.
///
/// Shape is encoded as `[H, W, C]` (HWC layout).  Strides are `[W*C, C, 1]` (in
/// units of the element type, expressed as items, not bytes — DLPack strides are
/// always in element units).
///
/// # Returns
///
/// A raw pointer to a `DLManagedTensor` that the consumer owns.  The consumer
/// MUST call `managed.deleter(managed)` when done.
pub fn dynimage_to_dlpack(img: DynImageBuf) -> *mut DLManagedTensor {
    let dtype = pixel_format_to_dl_dtype(img.dtype());
    let device = memory_domain_to_dl_device(img.domain(), img.device_id());
    let data_ptr = img.data_ptr() as *mut std::ffi::c_void;

    let [h, w, c] = img.shape();
    let shape: Vec<i64> = vec![h as i64, w as i64, c as i64];
    // HWC strides: stride[0]=W*C, stride[1]=C, stride[2]=1 (in element units).
    let strides: Vec<i64> = vec![(w * c) as i64, c as i64, 1i64];

    let info = TensorInfo::strided(data_ptr, device, dtype, shape, strides);
    // SAFETY: `img` (moved into the keep-alive via `safe::pack`) owns the buffer
    // and outlives the DLManagedTensor.  The deleter provided by `safe::pack` drops
    // the keep-alive, which drops `img` and its backing storage.
    safe::pack(img, info)
}

// ── dynimage_from_dlpack_raw ──────────────────────────────────────────────────

/// Imports a [`DynImageBuf`] from a raw `DLManagedTensor` pointer without copying data.
///
/// Accepts both CPU (`kDLCPU`) and CUDA (`kDLCUDA`) tensors at the container level.
/// CUDA images will have `MemoryDomain::Device` — calling [`DynImageBuf::as_image`] on
/// them returns [`ImageError::UnsupportedDevice`], which is the correct behaviour (use
/// device-transfer APIs instead).
///
/// `keepalive` keeps the source object alive for the lifetime of the returned image.
/// The caller must NOT call the DLManagedTensor's own deleter after passing it here;
/// that responsibility is transferred to `keepalive`.
///
/// # Shape and color space
///
/// The tensor must be 3-dimensional with layout `[H, W, C]`.  The color space is
/// inferred from channel count: `1`→Gray, `3`→Rgb, `4`→Rgba.  Other channel counts
/// return [`ImageError::UnsupportedChannelCount`].
///
/// # Safety
///
/// - `mt` must be non-null and point to a valid, initialised `DLManagedTensor`.
/// - The `dl_tensor.data` pointer must be valid for at least `H * W * C * dtype.element_size()`
///   bytes for the full lifetime of `keepalive`.
/// - All dimensions in `dl_tensor.shape` must be positive.
///
/// # Errors
///
/// - [`ImageError::DlpackShapeError`] if `ndim != 3` or any dimension is non-positive.
/// - [`ImageError::DlpackShapeError`] if the dtype is not `U8`, `U16`, or `F32`.
/// - [`ImageError::UnsupportedChannelCount`] if channel count is not 1, 3, or 4.
pub unsafe fn dynimage_from_dlpack_raw(
    mt: *mut DLManagedTensor,
    keepalive: Arc<dyn Any + Send + Sync>,
) -> Result<DynImageBuf, ImageError> {
    // SAFETY: caller guarantees mt is non-null and valid.
    let dl = unsafe { &(*mt).dl_tensor };

    // Validate ndim must be 3.
    if dl.ndim != 3 {
        return Err(ImageError::DlpackShapeError(format!(
            "expected ndim=3, got {}",
            dl.ndim
        )));
    }

    // Read shape [H, W, C].
    // SAFETY: dl.shape is valid for dl.ndim (== 3) elements (caller contract + DLPack spec).
    let shape_slice = unsafe { std::slice::from_raw_parts(dl.shape, 3) };
    for (i, &dim) in shape_slice.iter().enumerate() {
        if dim <= 0 {
            return Err(ImageError::DlpackShapeError(format!(
                "dimension[{}] must be positive, got {}",
                i, dim
            )));
        }
    }
    let h = shape_slice[0] as usize;
    let w = shape_slice[1] as usize;
    let c = shape_slice[2] as usize;

    // Map dtype — returns DlpackShapeError for unrecognized DLDataType codes.
    let dtype = dl_dtype_to_pixel_format(dl.dtype).ok_or_else(|| {
        ImageError::DlpackShapeError(format!(
            "unsupported DLDataType code={} bits={}",
            dl.dtype.code, dl.dtype.bits
        ))
    })?;

    // Map device.
    let (domain, device_id) = dl_device_to_memory_domain(dl.device);

    // Infer color space from channel count.
    let color_space = match c {
        1 => ColorSpace::Gray,
        3 => ColorSpace::Rgb,
        4 => ColorSpace::Rgba,
        _ => return Err(ImageError::UnsupportedChannelCount(c)),
    };

    // Compute data pointer with byte offset applied.
    // SAFETY: dl.data is valid (caller contract); byte_offset is within the buffer.
    debug_assert!(
        dl.byte_offset <= usize::MAX as u64,
        "byte_offset overflows usize on this platform"
    );
    let data_ptr = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *mut u8 };

    // Build DynImageBuf as a zero-copy borrow.
    // SAFETY: data_ptr is valid for H*W*C*dtype.element_size() bytes (caller contract);
    // keepalive keeps the source alive; domain+device_id correctly reflect the DLDevice.
    unsafe {
        DynImageBuf::from_borrowed(
            data_ptr,
            dtype,
            [h, w, c],
            color_space,
            domain,
            device_id,
            true,   // readonly — consumer model
            keepalive,
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    // ── test 1: dynimage_to_dlpack — deleter fires exactly once ──────────────

    #[test]
    fn test_dynimage_to_dlpack_deleter_fires_once() {
        let counter = Arc::new(AtomicUsize::new(0));

        // Build a small 2×3 RGB U8 image; the keepalive Guard holds a counter that
        // we verify fires exactly once when the DLPack deleter runs.
        let inner_counter = counter.clone();

        // We build the image by creating a DynImageBuf, and insert the drop counter
        // as an Arc inside another Arc so the DynImageBuf owns it via from_borrowed.
        let mut raw_bytes = vec![0u8; 2 * 3 * 3];
        let ptr = raw_bytes.as_mut_ptr();

        struct Guard {
            counter: Arc<AtomicUsize>,
            _bytes: Vec<u8>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        let guard: Arc<dyn Any + Send + Sync> = Arc::new(Guard {
            counter: inner_counter,
            _bytes: raw_bytes.clone(),
        });

        let img = unsafe {
            DynImageBuf::from_borrowed(
                ptr,
                PixelFormat::U8,
                [2, 3, 3],
                ColorSpace::Rgb,
                MemoryDomain::Host,
                0,
                false,
                guard,
            )
        }
        .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 0, "not dropped yet");

        let mt = dynimage_to_dlpack(img);
        assert!(!mt.is_null());

        // Verify shape is [2, 3, 3] and HWC strides are [W*C, C, 1] = [9, 3, 1].
        unsafe {
            let dl = &(*mt).dl_tensor;
            assert_eq!(dl.ndim, 3);
            let s = std::slice::from_raw_parts(dl.shape, 3);
            assert_eq!(s, &[2i64, 3, 3]);
            // Shape [2,3,3] → HWC strides [W*C, C, 1] = [9, 3, 1]
            let strides = std::slice::from_raw_parts((*mt).dl_tensor.strides, 3);
            assert_eq!(strides, &[9i64, 3, 1], "HWC strides must be [W*C, C, 1]");
        }

        // Invoke deleter — must drop the DynImageBuf (and hence its keepalive).
        unsafe {
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
        }

        // Drop counter must have fired exactly once.
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "keepalive must drop exactly once after deleter"
        );
    }

    // ── test 2: dynimage_from_dlpack_raw — CPU path ──────────────────────────

    #[test]
    fn test_dynimage_from_dlpack_raw_cpu_path() {
        // Build a synthetic 4×6 RGB U8 DLManagedTensor over a heap buffer.
        let h: i64 = 4;
        let w: i64 = 6;
        let c: i64 = 3;
        let n = (h * w * c) as usize;
        let data: Vec<u8> = (0u8..).take(n).collect();
        let shape_arr: Vec<i64> = vec![h, w, c];

        let dl_tensor = dlpack_rs::ffi::DLTensor {
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

        let img = unsafe {
            dynimage_from_dlpack_raw(&mut managed as *mut _, keepalive)
        }
        .unwrap();

        // Verify domain.
        assert_eq!(img.domain(), MemoryDomain::Host, "must be Host domain");
        assert_eq!(img.device_id(), 0);

        // Verify shape and metadata.
        assert_eq!(img.shape(), [4, 6, 3]);
        assert_eq!(img.dtype(), PixelFormat::U8);
        assert_eq!(img.color_space(), ColorSpace::Rgb);
        assert!(img.readonly(), "imported images are readonly");

        // Verify data is accessible (host path).
        let slice = unsafe { std::slice::from_raw_parts(img.data_ptr(), img.nbytes()) };
        assert_eq!(slice[0], 0);
        assert_eq!(slice[1], 1);
    }

    // ── test 3: dynimage_from_dlpack_raw — synthetic CUDA path ───────────────

    #[test]
    fn test_dynimage_from_dlpack_raw_cuda_path() {
        use dlpack_rs::ffi::{DLDevice, K_DL_CUDA};

        // Simulate a kDLCUDA device_id=1 tensor — no real CUDA hardware needed.
        // We use a real allocation as the "device" pointer; we won't dereference it.
        let data: Vec<f32> = vec![0.0f32; 8];
        let data_ptr = data.as_ptr();
        let shape_arr: Vec<i64> = vec![2i64, 4, 1];

        let device = DLDevice {
            device_type: K_DL_CUDA,
            device_id: 1,
        };

        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
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

        let img = unsafe {
            dynimage_from_dlpack_raw(&mut managed as *mut _, keepalive)
        }
        .unwrap();

        // Verify domain == Device and device_id == 1.
        assert_eq!(img.domain(), MemoryDomain::Device, "must be Device domain");
        assert_eq!(img.device_id(), 1, "device_id must be 1");

        // Verify shape and dtype.
        assert_eq!(img.shape(), [2, 4, 1]);
        assert_eq!(img.dtype(), PixelFormat::F32);
        assert_eq!(img.color_space(), ColorSpace::Gray);

        // as_image must return UnsupportedDevice for device buffers.
        let result = unsafe { img.as_image::<f32, 1>() };
        assert!(
            matches!(result, Err(ImageError::UnsupportedDevice)),
            "as_image on CUDA buffer must return UnsupportedDevice"
        );
    }

    // ── test 4: ndim != 3 returns DlpackShapeError ───────────────────────────

    #[test]
    fn test_dynimage_from_dlpack_raw_wrong_ndim() {
        let data: Vec<u8> = vec![0u8; 4];
        let shape_arr: Vec<i64> = vec![4i64];

        let dl_tensor = dlpack_rs::ffi::DLTensor {
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

        let result = unsafe { dynimage_from_dlpack_raw(&mut managed as *mut _, keepalive) };
        assert!(
            matches!(result, Err(ImageError::DlpackShapeError(_))),
            "ndim != 3 must return DlpackShapeError"
        );
    }

    // ── test 5: unsupported channel count returns UnsupportedChannelCount ─────

    #[test]
    fn test_dynimage_from_dlpack_raw_unsupported_channels() {
        // 2-channel image — not Gray, RGB, or RGBA.
        let data: Vec<u8> = vec![0u8; 4 * 4 * 2];
        let shape_arr: Vec<i64> = vec![4i64, 4, 2];

        let dl_tensor = dlpack_rs::ffi::DLTensor {
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

        let result = unsafe { dynimage_from_dlpack_raw(&mut managed as *mut _, keepalive) };
        assert!(
            matches!(result, Err(ImageError::UnsupportedChannelCount(2))),
            "2-channel must return UnsupportedChannelCount(2)"
        );
    }

    // ── test 6: unsupported dtype (K_DL_INT / 8-bit signed) returns DlpackShapeError ──

    #[test]
    fn test_dynimage_from_dlpack_raw_unsupported_dtype() {
        use dlpack_rs::ffi::{DLDataType, K_DL_INT};

        // K_DL_INT 8-bit is not U8/U16/F32 — dl_dtype_to_pixel_format returns None.
        let int8_dtype = DLDataType {
            code: K_DL_INT,
            bits: 8,
            lanes: 1,
        };

        let data: Vec<i8> = vec![0i8; 4 * 4 * 3];
        let shape_arr: Vec<i64> = vec![4i64, 4, 3];

        let dl_tensor = dlpack_rs::ffi::DLTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            device: safe::cpu_device(),
            ndim: 3,
            dtype: int8_dtype,
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

        let result = unsafe { dynimage_from_dlpack_raw(&mut managed as *mut _, keepalive) };
        assert!(
            matches!(result, Err(ImageError::DlpackShapeError(_))),
            "unsupported dtype (K_DL_INT 8-bit) must return DlpackShapeError"
        );
    }
}
