//! Numpy-agnostic storage backing for the Python Image.
use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

use dlpack_rs::ffi::{DLDataType, K_DL_CPU, K_DL_FLOAT, K_DL_UINT};
use dlpack_rs::safe::{dtype_f32, dtype_u16, dtype_u8};
use kornia_image::allocator::host_alloc;
use kornia_image::{Image, ImageError, ImageSize};
use pyo3::prelude::*;

const ALIGN: usize = 64;

/// Element type of an Image buffer (v1 scope).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    U8,
    U16,
    F32,
}
impl Dtype {
    pub fn itemsize(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::U16 => 2,
            Dtype::F32 => 4,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Dtype::U8 => "uint8",
            Dtype::U16 => "uint16",
            Dtype::F32 => "float32",
        }
    }
    #[allow(dead_code)]
    pub fn from_numpy_str(s: &str) -> PyResult<Dtype> {
        match s {
            "uint8" | "u8" | "|u1" | "B" => Ok(Dtype::U8),
            "uint16" | "u16" | "<u2" | "=u2" | "H" => Ok(Dtype::U16),
            "float32" | "f32" | "<f4" | "=f4" | "f" => Ok(Dtype::F32),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported dtype {other:?}; expected uint8, uint16, or float32"
            ))),
        }
    }

    /// Convert this `Dtype` to the corresponding `DLDataType`.
    pub fn to_dldatatype(self) -> DLDataType {
        match self {
            Dtype::U8 => dtype_u8(),
            Dtype::U16 => dtype_u16(),
            Dtype::F32 => dtype_f32(),
        }
    }

    /// Convert a `DLDataType` to `Dtype`, or return a `ValueError`.
    pub fn from_dldatatype(dt: DLDataType) -> PyResult<Dtype> {
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
}

/// A 64-byte-aligned owned heap buffer (SIMD/DMA friendly).
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    #[allow(dead_code)]
    len: usize,
    layout: Layout,
}
// SAFETY: AlignedBytes uniquely owns a heap allocation of plain bytes.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}
impl AlignedBytes {
    pub fn zeroed(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), ALIGN).expect("layout");
        // SAFETY: layout has non-zero size (len.max(1)).
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len, layout }
    }
    /// Allocate `len` bytes **without zeroing**. This is how numpy/OpenCV allocate
    /// output buffers — pre-zeroing a buffer you're about to fully overwrite is
    /// pure waste (an extra full-buffer write).
    ///
    /// # Safety contract (caller-enforced, not in the type)
    /// The caller MUST fully initialize all `len` bytes before any read of this
    /// buffer (e.g. a full-overwrite op like crop, or `copy_nonoverlapping`).
    pub fn uninit(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), ALIGN).expect("layout");
        // SAFETY: layout has non-zero size (len.max(1)); the returned bytes are
        // uninitialized and must be fully written before being read.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len, layout }
    }
    pub fn from_slice(src: &[u8]) -> Self {
        // `uninit` is sound here: copy_nonoverlapping below writes every byte.
        let b = Self::uninit(src.len());
        // SAFETY: b.ptr owns len==src.len() bytes; regions don't overlap.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), b.ptr.as_ptr(), src.len()) };
        b
    }
    #[allow(dead_code)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr owns len bytes, allocated and valid for reads.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: ptr/layout came from alloc/alloc_zeroed with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Keeps a borrowed buffer's source alive for the Image's lifetime.
pub enum BorrowGuard {
    /// numpy ndarray (ptr = base) or PEP-3118 owner (ptr = Py_buffer.buf, view stored to release on drop).
    PyObject {
        obj: Py<PyAny>,
        buffer: Option<Box<pyo3::ffi::Py_buffer>>,
    },
}
impl Drop for BorrowGuard {
    fn drop(&mut self) {
        if let BorrowGuard::PyObject {
            buffer: Some(view), ..
        } = self
        {
            // SAFETY: `view` was filled by PyObject_GetBuffer in from_buffer; release exactly once.
            unsafe { pyo3::ffi::PyBuffer_Release(view.as_mut()) };
        }
        // Py<PyAny> drops itself.
    }
}

/// Image data backing: owned aligned bytes, or a zero-copy borrow with keep-alive.
///
/// # INVARIANT
///
/// No concurrent write to the backing buffer while a writable numpy view is live.
/// Enforced by GIL — all mutations go through `&mut self` methods called from Python.
pub enum Backing {
    Owned(AlignedBytes),
    Borrowed {
        ptr: NonNull<u8>,
        keep: BorrowGuard,
        readonly: bool,
        /// DLPack device of the borrowed buffer as `(device_type, device_id)`.
        /// Host buffers (numpy / PEP-3118) use `(K_DL_CPU, 0)`; a DLPack import
        /// carries the source tensor's real device (e.g. `(K_DL_CUDA, id)`).
        device: (i32, i32),
    },
    /// A device-resident image (CUDA). Owns its typed device buffer (and the
    /// `Arc<CudaStream>` carried inside it), so it can download, run kernels and
    /// export DLPack zero-copy — unlike a raw `Borrowed` device pointer. Shared
    /// via `Arc` so `.cuda()`/DLPack export can clone a keep-alive cheaply.
    #[cfg(feature = "cuda")]
    Device {
        img: std::sync::Arc<crate::device::DeviceImage>,
        readonly: bool,
    },
}
// SAFETY: Owned is Send+Sync; Borrowed holds Send keep-alives and a raw ptr with exclusive logical ownership.
// Sync is sound because all mutation of the pointed-to memory happens under the Python GIL.
unsafe impl Send for Backing {}
unsafe impl Sync for Backing {}
impl Backing {
    pub fn data_ptr(&self) -> *mut u8 {
        match self {
            Backing::Owned(b) => b.ptr.as_ptr(),
            Backing::Borrowed { ptr, .. } => ptr.as_ptr(),
            // Device pointer (CUdeviceptr). Only read by `__dlpack__` / the
            // `data_ptr` getter; never dereferenced on the host.
            #[cfg(feature = "cuda")]
            Backing::Device { img, .. } => img.as_ptr(),
        }
    }
    pub fn readonly(&self) -> bool {
        match self {
            Backing::Owned(_) => false,
            Backing::Borrowed { readonly, .. } => *readonly,
            #[cfg(feature = "cuda")]
            Backing::Device { readonly, .. } => *readonly,
        }
    }

    /// DLPack device of this backing as `(device_type, device_id)`.
    ///
    /// Owned buffers are always host CPU. Borrowed buffers carry the device
    /// captured at construction (host for numpy / PEP-3118 borrows, or the
    /// source tensor's device for a zero-copy DLPack import).
    pub fn device(&self) -> (i32, i32) {
        match self {
            Backing::Owned(_) => (K_DL_CPU as i32, 0),
            Backing::Borrowed { device, .. } => *device,
            #[cfg(feature = "cuda")]
            Backing::Device { img, .. } => (
                dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32,
                img.device_id(),
            ),
        }
    }

    /// `true` if the backing lives in host (CPU) memory and is safe to
    /// dereference from Rust.
    pub fn is_host(&self) -> bool {
        self.device().0 == K_DL_CPU as i32
    }

    /// Guard for host-only operations: returns an error if the backing is on a
    /// non-CPU device, where any host dereference would be undefined behaviour.
    ///
    /// Call this at the start of every method that reads or writes the buffer on
    /// the host (numpy export, pixel compute, encode/save, buffer protocol).
    pub fn ensure_host(&self) -> pyo3::PyResult<()> {
        if self.is_host() {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "operation requires a host (CPU) image; this image is on device \
                 (device_type={}); move it to the host first by calling .cpu() on \
                 this image before this operation",
                self.device().0
            )))
        }
    }
}

/// Build a typed compute borrow from a backing + shape. Validates channel count.
///
/// # Safety
///
/// Caller guarantees `b`'s buffer holds at least H*W*C elements of T and
/// stays alive for the returned Image's lifetime (the Image borrows it).
///
/// # Panics
///
/// Panics if `b` is not host-accessible. This wraps the pointer in a
/// `MemoryDomain::Host` Image (whose `as_slice` would then *not* panic), so a device
/// backing reaching here would be a silent host-deref of device memory. The assert
/// turns that latent UB into a panic; callers must gate with [`Backing::ensure_host`].
pub unsafe fn borrow_image<T: Clone, const C: usize>(
    b: &Backing,
    shape: [usize; 3],
) -> Result<Image<T, C>, ImageError> {
    assert!(
        b.is_host(),
        "borrow_image on a non-host backing (device_type={}); gate with ensure_host() first",
        b.device().0
    );
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != C {
        return Err(ImageError::InvalidChannelShape(c, C));
    }
    Image::from_raw_parts(
        ImageSize {
            width: w,
            height: h,
        },
        b.data_ptr() as *const T,
        h * w * c * std::mem::size_of::<T>(),
        host_alloc(),
    )
}

/// Compute the total byte length for an image with dimensions `(h, w, c)` and
/// element type `dtype`, using checked arithmetic to detect overflow.
///
/// Returns `PyOverflowError` if the product would exceed `usize::MAX`.
pub fn byte_len(h: usize, w: usize, c: usize, dtype: Dtype) -> pyo3::PyResult<usize> {
    h.checked_mul(w)
        .and_then(|x| x.checked_mul(c))
        .and_then(|x| x.checked_mul(dtype.itemsize()))
        .ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err("image dimensions overflow usize")
        })
}

/// Allocate a zeroed owned output buffer for an op of channel count C.
///
/// Uses checked arithmetic via [`byte_len`]; returns `PyOverflowError` if
/// dimensions would overflow `usize`.
pub fn alloc_output_owned<const C: usize>(
    dtype: Dtype,
    size: ImageSize,
) -> pyo3::PyResult<(AlignedBytes, ImageSize)> {
    let len = byte_len(size.height, size.width, C, dtype)?;
    Ok((AlignedBytes::zeroed(len), size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_bytes_is_64b_aligned_and_zeroed() {
        let b = AlignedBytes::zeroed(100);
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
        assert_eq!(b.len(), 100);
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }

    #[test]
    fn from_slice_copies() {
        let src = [1u8, 2, 3, 4, 5];
        let b = AlignedBytes::from_slice(&src);
        assert_eq!(b.as_slice(), &src);
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
    }

    #[test]
    fn dtype_roundtrip() {
        assert_eq!(Dtype::from_numpy_str("uint8").unwrap(), Dtype::U8);
        assert_eq!(Dtype::F32.itemsize(), 4);
        assert!(Dtype::from_numpy_str("int64").is_err());
    }

    #[test]
    fn alloc_output_owned_sizes_correctly() {
        let (b, sz) = alloc_output_owned::<3>(
            Dtype::F32,
            ImageSize {
                width: 4,
                height: 5,
            },
        )
        .unwrap();
        assert_eq!(b.len(), 4 * 5 * 3 * 4);
        assert_eq!((sz.width, sz.height), (4, 5));
    }

    #[test]
    fn byte_len_overflow_raises() {
        // usize::MAX * 4 (F32 itemsize) overflows usize.
        let huge = usize::MAX;
        assert!(byte_len(huge, 2, 3, Dtype::F32).is_err());
        // Normal case should succeed.
        assert_eq!(byte_len(5, 4, 3, Dtype::U8).unwrap(), 60);
    }
}
