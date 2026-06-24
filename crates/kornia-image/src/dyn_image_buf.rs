//! Runtime-typed image buffer with dynamic pixel format and color space.
//!
//! [`DynImageBuf`] is the central type for passing images across language or subsystem
//! boundaries (Python ↔ Rust, CPU ↔ CUDA, DLPack, etc.) where the pixel type is not
//! known at compile time.  It carries a raw byte buffer together with shape/dtype/color-
//! space metadata and can be zero-copy viewed as a typed [`Image`] when the domain and
//! dtype match.

use std::{alloc::Layout, any::Any, sync::Arc};

use kornia_tensor::{
    allocator::{AlignedCpuAllocator, TensorAllocator},
    storage::TensorStorage,
    ForeignAllocator as KtForeignAllocator, MemoryDomain, Tensor,
};

use crate::{
    allocator::ForeignAllocator,
    color_space::ColorSpace,
    error::ImageError,
    image::{Image, ImageSize, PixelFormat},
};

/// Internal backing store: either an owned 64-byte-aligned CPU allocation or a
/// zero-copy borrow of a foreign pointer (CUDA / numpy / DLPack).
enum Backing {
    /// Owned buffer allocated with 64-byte alignment by [`AlignedCpuAllocator`].
    Owned(TensorStorage<u8, AlignedCpuAllocator>),
    /// Borrowed foreign buffer; an [`Arc`] keep-alive is embedded inside the storage.
    Foreign(TensorStorage<u8, KtForeignAllocator>),
}

impl Backing {
    /// Returns a const pointer to the start of the buffer.
    fn as_ptr(&self) -> *const u8 {
        match self {
            Backing::Owned(s) => s.as_ptr(),
            Backing::Foreign(s) => s.as_ptr(),
        }
    }

    /// Returns a mutable pointer to the start of the buffer.
    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Backing::Owned(s) => s.as_mut_ptr(),
            Backing::Foreign(s) => s.as_mut_ptr(),
        }
    }

    /// Returns the [`MemoryDomain`] (Host or Device) of the buffer.
    fn domain(&self) -> MemoryDomain {
        match self {
            Backing::Owned(s) => s.domain(),
            Backing::Foreign(s) => s.domain(),
        }
    }

    /// Returns the CUDA device id (0 for host).
    fn device_id(&self) -> i32 {
        match self {
            Backing::Owned(s) => s.device_id(),
            Backing::Foreign(s) => s.device_id(),
        }
    }
}

/// A runtime-typed image buffer.
///
/// `DynImageBuf` stores raw bytes together with shape `[H, W, C]`, a [`PixelFormat`]
/// tag, and a [`ColorSpace`] tag.  The buffer may live on the host (CPU) or on a device
/// (GPU / CUDA).
///
/// # Typed views
///
/// Use [`DynImageBuf::as_image`] to obtain a zero-copy [`Image<T, C, ForeignAllocator>`]
/// view when the domain is [`MemoryDomain::Host`] and the dtype matches.
///
/// # Thread-safety
///
/// `DynImageBuf` is `Send`: the internal pointer is uniquely owned and the
/// [`Arc`] keep-alive is `Send + Sync`.
///
/// `DynImageBuf` is **not** automatically `Sync`: [`data_ptr_mut`](Self::data_ptr_mut)
/// hands out a `*mut u8` so callers must synchronise external access themselves.
pub struct DynImageBuf {
    backing: Backing,
    dtype: PixelFormat,
    shape: [usize; 3],
    color_space: ColorSpace,
    readonly: bool,
}

// SAFETY: The raw pointer inside Backing is owned exclusively by this DynImageBuf and
// the Arc keep-alive is Send+Sync, so DynImageBuf can safely be sent across threads.
unsafe impl Send for DynImageBuf {}

impl DynImageBuf {
    // ──────────────────────────────────── constructors ────────────────────────────────

    /// Allocates a zeroed, 64-byte-aligned host buffer for an image.
    ///
    /// # Arguments
    ///
    /// * `shape` – `[height, width, channels]`.
    /// * `dtype` – Element pixel format.
    /// * `color_space` – Color space tag.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] if the allocator fails (out of memory).
    pub fn new_owned(
        shape: [usize; 3],
        dtype: PixelFormat,
        color_space: ColorSpace,
    ) -> Result<Self, ImageError> {
        let nbytes = shape[0] * shape[1] * shape[2] * dtype.element_size();
        // Use nbytes.max(1) so that Layout::from_size_align never gets size 0.
        let layout = Layout::from_size_align(nbytes.max(1), 64)
            .map_err(|_| ImageError::ImageDataNotInitialized)?;
        // Note: reuses ImageDataNotInitialized for allocation failures (no dedicated OOM variant)
        let ptr = AlignedCpuAllocator
            .alloc(layout)
            .map_err(|_| ImageError::ImageDataNotInitialized)?;
        // SAFETY: ptr is valid, non-null, zeroed by AlignedCpuAllocator::alloc, and
        // `layout` is exactly what was used to allocate it.
        let storage =
            unsafe { TensorStorage::from_raw_host(ptr, nbytes, layout, AlignedCpuAllocator) };
        Ok(Self {
            backing: Backing::Owned(storage),
            dtype,
            shape,
            color_space,
            readonly: false,
        })
    }

    /// Copies `bytes` into a new 64-byte-aligned host buffer.
    ///
    /// # Arguments
    ///
    /// * `shape` – `[height, width, channels]`.
    /// * `dtype` – Element pixel format.
    /// * `color_space` – Color space tag.
    /// * `bytes` – Source slice; length must equal `H * W * C * dtype.element_size()`.
    ///
    /// # Errors
    ///
    /// Returns `ImageError::InvalidChannelShape(got, expected)` if `bytes.len()` does not
    /// equal `H * W * C * dtype.element_size()`. Note: semantically this is a size mismatch,
    /// but the variant is reused for this purpose (no dedicated "buffer size mismatch" variant).
    pub fn from_bytes(
        shape: [usize; 3],
        dtype: PixelFormat,
        color_space: ColorSpace,
        bytes: &[u8],
    ) -> Result<Self, ImageError> {
        let nbytes = shape[0] * shape[1] * shape[2] * dtype.element_size();
        if bytes.len() != nbytes {
            return Err(ImageError::InvalidChannelShape(bytes.len(), nbytes));
        }
        let layout = Layout::from_size_align(nbytes.max(1), 64)
            .map_err(|_| ImageError::ImageDataNotInitialized)?;
        // Note: reuses ImageDataNotInitialized for allocation failures (no dedicated OOM variant)
        let ptr = AlignedCpuAllocator
            .alloc(layout)
            .map_err(|_| ImageError::ImageDataNotInitialized)?;
        // SAFETY: ptr is valid for `nbytes` bytes; bytes.len() == nbytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, nbytes);
        }
        // SAFETY: ptr is valid, non-null; we filled it via copy above.
        let storage =
            unsafe { TensorStorage::from_raw_host(ptr, nbytes, layout, AlignedCpuAllocator) };
        Ok(Self {
            backing: Backing::Owned(storage),
            dtype,
            shape,
            color_space,
            readonly: false,
        })
    }

    /// Wraps a foreign pointer as a zero-copy borrow.
    ///
    /// # Arguments
    ///
    /// * `ptr` – Raw pointer to the buffer start.
    /// * `dtype` – Element pixel format.
    /// * `shape` – `[height, width, channels]`.
    /// * `color_space` – Color space tag.
    /// * `domain` – [`MemoryDomain::Host`] or [`MemoryDomain::Device`].
    /// * `device_id` – CUDA device id (0 for host).
    /// * `readonly` – If `true`, [`data_ptr_mut`](Self::data_ptr_mut) will panic.
    /// * `keepalive` – An [`Arc`] whose lifetime guarantees `ptr` remains valid.
    ///
    /// # Safety
    ///
    /// The memory region `[ptr, ptr + H*W*C*dtype.element_size())` must be valid and
    /// correctly described by `domain`/`device_id` for the entire lifetime of `keepalive`.
    ///
    /// # Errors
    ///
    /// Currently always returns `Ok`; the `Result` return is for forward compatibility.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_borrowed(
        ptr: *mut u8,
        dtype: PixelFormat,
        shape: [usize; 3],
        color_space: ColorSpace,
        domain: MemoryDomain,
        device_id: i32,
        readonly: bool,
        keepalive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, ImageError> {
        let nbytes = shape[0] * shape[1] * shape[2] * dtype.element_size();
        // SAFETY: caller guarantees ptr is valid for nbytes bytes.
        let storage = TensorStorage::<u8, KtForeignAllocator>::from_borrowed(
            ptr as *const u8,
            nbytes,
            KtForeignAllocator,
            domain,
            device_id,
            keepalive,
        );
        Ok(Self {
            backing: Backing::Foreign(storage),
            dtype,
            shape,
            color_space,
            readonly,
        })
    }

    // ──────────────────────────────────── accessors ───────────────────────────────────

    /// Returns the pixel format (dtype) of the buffer.
    pub fn dtype(&self) -> PixelFormat {
        self.dtype
    }

    /// Returns the number of channels (`shape[2]`).
    pub fn channels(&self) -> usize {
        self.shape[2]
    }

    /// Returns the image size (`width = shape[1]`, `height = shape[0]`).
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.shape[1],
            height: self.shape[0],
        }
    }

    /// Returns the shape `[height, width, channels]`.
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Returns the color space tag.
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Sets the color space tag.
    pub fn set_color_space(&mut self, cs: ColorSpace) {
        self.color_space = cs;
    }

    /// Returns a const pointer to the buffer start.
    ///
    /// For device buffers, dereferencing this pointer on the host is unsound.
    pub fn data_ptr(&self) -> *const u8 {
        self.backing.as_ptr()
    }

    /// Returns a mutable pointer to the buffer start.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was created with `readonly = true`.
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        assert!(
            !self.readonly,
            "DynImageBuf: data_ptr_mut called on a readonly buffer"
        );
        self.backing.as_mut_ptr()
    }

    /// Returns `true` if the buffer is read-only.
    pub fn readonly(&self) -> bool {
        self.readonly
    }

    /// Returns the memory domain (Host or Device).
    pub fn domain(&self) -> MemoryDomain {
        self.backing.domain()
    }

    /// Returns the CUDA device id (0 for host).
    pub fn device_id(&self) -> i32 {
        self.backing.device_id()
    }

    /// Returns the total size of the buffer in bytes (`H * W * C * dtype.element_size()`).
    pub fn nbytes(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2] * self.dtype.element_size()
    }

    // ────────────────────────────────── typed view ─────────────────────────────────

    /// Returns a zero-copy typed view of this buffer as [`Image<T, C, ForeignAllocator>`].
    ///
    /// # Type parameters
    ///
    /// * `T` – Element type (`u8`, `u16`, or `f32`).  Must match `self.dtype()`.
    /// * `C` – Channel count (const).  Must equal `self.channels()`.
    ///
    /// # Errors
    ///
    /// - [`ImageError::UnsupportedDevice`] if the buffer lives on a device.
    /// - [`ImageError::DtypeMismatch`] if `T` does not correspond to `self.dtype()`.
    /// - [`ImageError::InvalidChannelShape`] if `C != self.channels()`.
    ///
    /// # Safety
    ///
    /// The returned [`Image`] borrows the same memory as `self`.  The caller must ensure
    /// that `self` outlives the returned [`Image`] and that no mutable alias to the buffer
    /// exists while the [`Image`] is live.
    pub unsafe fn as_image<T: num_traits::NumCast + Copy + 'static, const C: usize>(
        &self,
    ) -> Result<Image<T, C, ForeignAllocator>, ImageError> {
        // 1. Domain check — slice access on device memory is unsound.
        if self.backing.domain() != MemoryDomain::Host {
            return Err(ImageError::UnsupportedDevice);
        }

        // 2. Dtype check via TypeId.
        // Unknown T: can't determine caller's format; both fields set to buffer's dtype as a best-effort signal
        let caller_fmt = dtype_for::<T>().ok_or(ImageError::DtypeMismatch {
            expected: self.dtype,
            got: self.dtype,
        })?;
        if caller_fmt != self.dtype {
            return Err(ImageError::DtypeMismatch {
                expected: caller_fmt,
                got: self.dtype,
            });
        }

        // 3. Channel count check.
        if self.shape[2] != C {
            return Err(ImageError::InvalidChannelShape(self.shape[2], C));
        }

        let [h, w, _c] = self.shape;
        let numel = h * w * C; // number of T-elements
        // TensorStorage::len is stored in bytes; Tensor::from_raw_parts passes len directly
        // to TensorStorage::from_raw_parts, so we must pass the byte count.
        let len_bytes = numel * std::mem::size_of::<T>();

        // 4. Build Tensor3<T, ForeignAllocator> zero-copy.
        //    Tensor::from_raw_parts sets owns_memory=false so ForeignAllocator::dealloc
        //    (a no-op) is called on drop — the memory is still owned by self.backing.
        //
        // SAFETY:
        //   - ptr is valid for `numel` elements of T on host (domain==Host checked above).
        //   - T size matches because element_size check (dtype) is validated above.
        //   - Caller guarantees self outlives the returned Image.
        let tensor: kornia_tensor::Tensor3<T, ForeignAllocator> = Tensor::from_raw_parts(
            [h, w, C],
            self.backing.as_ptr() as *const T,
            len_bytes,
            ForeignAllocator,
        )?;

        Ok(Image(tensor))
    }
}

/// Maps the Rust type `T` to the corresponding [`PixelFormat`] variant.
///
/// Returns `None` if `T` is not a recognised image element type.
fn dtype_for<T: 'static>() -> Option<PixelFormat> {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<u8>() {
        Some(PixelFormat::U8)
    } else if TypeId::of::<T>() == TypeId::of::<u16>() {
        Some(PixelFormat::U16)
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(PixelFormat::F32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageSize;

    // ── PixelFormat::element_size ─────────────────────────────────────────────────

    #[test]
    fn pixel_format_element_size() {
        assert_eq!(PixelFormat::U8.element_size(), 1);
        assert_eq!(PixelFormat::U16.element_size(), 2);
        assert_eq!(PixelFormat::F32.element_size(), 4);
    }

    // ── new_owned ─────────────────────────────────────────────────────────────────

    #[test]
    fn new_owned_creates_zeroed_buffer() {
        let buf = DynImageBuf::new_owned([4, 6, 3], PixelFormat::U8, ColorSpace::Rgb).unwrap();
        assert_eq!(buf.shape(), [4, 6, 3]);
        assert_eq!(buf.dtype(), PixelFormat::U8);
        assert_eq!(buf.channels(), 3);
        assert_eq!(buf.size(), ImageSize { height: 4, width: 6 });
        assert_eq!(buf.nbytes(), 4 * 6 * 3);
        assert_eq!(buf.domain(), MemoryDomain::Host);
        assert_eq!(buf.device_id(), 0);
        assert!(!buf.readonly());
        assert_eq!(buf.color_space(), ColorSpace::Rgb);
        // Buffer is zero-initialised.
        let slice = unsafe { std::slice::from_raw_parts(buf.data_ptr(), buf.nbytes()) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn new_owned_64_byte_aligned() {
        let buf = DynImageBuf::new_owned([8, 8, 1], PixelFormat::U8, ColorSpace::Gray).unwrap();
        assert_eq!(buf.data_ptr() as usize % 64, 0);
    }

    #[test]
    fn new_owned_f32_nbytes() {
        let buf =
            DynImageBuf::new_owned([2, 3, 1], PixelFormat::F32, ColorSpace::Gray).unwrap();
        assert_eq!(buf.nbytes(), 2 * 3 * 1 * 4);
    }

    // ── from_bytes ────────────────────────────────────────────────────────────────

    #[test]
    fn from_bytes_copies_data() {
        let data: Vec<u8> = (0u8..12).collect();
        let buf =
            DynImageBuf::from_bytes([2, 2, 3], PixelFormat::U8, ColorSpace::Rgb, &data).unwrap();
        assert_eq!(buf.nbytes(), 12);
        let slice = unsafe { std::slice::from_raw_parts(buf.data_ptr(), buf.nbytes()) };
        assert_eq!(slice, data.as_slice());
    }

    #[test]
    fn from_bytes_length_mismatch_errors() {
        let bad = vec![0u8; 5];
        let result =
            DynImageBuf::from_bytes([2, 2, 3], PixelFormat::U8, ColorSpace::Rgb, &bad);
        assert!(matches!(result, Err(ImageError::InvalidChannelShape(_, _))));
    }

    // ── from_borrowed ─────────────────────────────────────────────────────────────

    #[test]
    fn from_borrowed_wraps_without_copy() {
        let data = vec![1u8, 2, 3, 4, 5, 6];
        let ptr = data.as_ptr() as *mut u8;
        let ka: Arc<dyn Any + Send + Sync> = Arc::new(data.clone());
        let buf = unsafe {
            DynImageBuf::from_borrowed(
                ptr,
                PixelFormat::U8,
                [1, 2, 3],
                ColorSpace::Rgb,
                MemoryDomain::Host,
                0,
                false,
                ka,
            )
        }
        .unwrap();
        assert_eq!(buf.nbytes(), 6);
        assert_eq!(buf.domain(), MemoryDomain::Host);
        assert!(!buf.readonly());
        // Pointer is identical to source.
        assert_eq!(buf.data_ptr(), ptr as *const u8);
    }

    // ── readonly panic ────────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "readonly")]
    fn data_ptr_mut_panics_when_readonly() {
        let mut data = vec![0u8; 6];
        let ptr = data.as_mut_ptr();
        let ka: Arc<dyn Any + Send + Sync> = Arc::new(data.clone());
        let mut buf = unsafe {
            DynImageBuf::from_borrowed(
                ptr,
                PixelFormat::U8,
                [1, 2, 3],
                ColorSpace::Rgb,
                MemoryDomain::Host,
                0,
                true,
                ka,
            )
        }
        .unwrap();
        let _ = buf.data_ptr_mut();
    }

    // ── set_color_space ───────────────────────────────────────────────────────────

    #[test]
    fn set_color_space() {
        let mut buf =
            DynImageBuf::new_owned([1, 1, 3], PixelFormat::U8, ColorSpace::Rgb).unwrap();
        buf.set_color_space(ColorSpace::Bgr);
        assert_eq!(buf.color_space(), ColorSpace::Bgr);
    }

    // ── as_image ──────────────────────────────────────────────────────────────────

    #[test]
    fn as_image_u8_3ch_succeeds() {
        let data: Vec<u8> = (0u8..12).collect();
        let buf =
            DynImageBuf::from_bytes([2, 2, 3], PixelFormat::U8, ColorSpace::Rgb, &data).unwrap();
        let img = unsafe { buf.as_image::<u8, 3>() }.unwrap();
        assert_eq!(img.size(), ImageSize { height: 2, width: 2 });
        assert_eq!(img.num_channels(), 3);
        // Zero-copy: data is identical.
        assert_eq!(img.as_slice(), data.as_slice());
    }

    #[test]
    fn as_image_f32_1ch_succeeds() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = f32_slice_as_u8(&data);
        let buf =
            DynImageBuf::from_bytes([2, 2, 1], PixelFormat::F32, ColorSpace::Gray, bytes).unwrap();
        let img = unsafe { buf.as_image::<f32, 1>() }.unwrap();
        assert_eq!(img.as_slice(), data.as_slice());
    }

    #[test]
    fn as_image_dtype_mismatch_errors() {
        let buf =
            DynImageBuf::new_owned([2, 2, 3], PixelFormat::U8, ColorSpace::Rgb).unwrap();
        let result = unsafe { buf.as_image::<f32, 3>() };
        assert!(matches!(result, Err(ImageError::DtypeMismatch { .. })));
    }

    #[test]
    fn as_image_channel_mismatch_errors() {
        let buf =
            DynImageBuf::new_owned([2, 2, 3], PixelFormat::U8, ColorSpace::Rgb).unwrap();
        let result = unsafe { buf.as_image::<u8, 1>() };
        assert!(matches!(result, Err(ImageError::InvalidChannelShape(_, _))));
    }

    #[test]
    fn as_image_device_domain_errors() {
        let mut data = vec![0u8; 4];
        let ptr = data.as_mut_ptr();
        let ka: Arc<dyn Any + Send + Sync> = Arc::new(data.clone());
        let buf = unsafe {
            DynImageBuf::from_borrowed(
                ptr,
                PixelFormat::U8,
                [1, 2, 2],
                ColorSpace::Gray,
                MemoryDomain::Device,
                0,
                false,
                ka,
            )
        }
        .unwrap();
        let result = unsafe { buf.as_image::<u8, 2>() };
        assert!(matches!(result, Err(ImageError::UnsupportedDevice)));
    }

    // ── helper ────────────────────────────────────────────────────────────────────

    /// Reinterpret a `&[f32]` as `&[u8]` (native endian).
    fn f32_slice_as_u8(v: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                v.as_ptr() as *const u8,
                v.len() * std::mem::size_of::<f32>(),
            )
        }
    }
}
