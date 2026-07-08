//! CUDA device-memory integration for [`Image`].
//!
//! This module is enabled by the `cudarc` feature flag. It provides
//! [`Image::to_cuda`], which uploads a host-backed image to a new
//! device-backed [`kornia_tensor::Tensor`] via a host-to-device copy.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use kornia_image::color_spaces::Rgb8;
//! use cudarc::driver::CudaContext;
//! use std::sync::Arc;
//!
//! let rgb = Rgb8::from_size_val(
//!     kornia_image::ImageSize { width: 2, height: 2 },
//!     0u8,
//! ).unwrap();
//!
//! let ctx = CudaContext::new(0).unwrap();
//! let stream = ctx.default_stream();
//! // Typed device-resident image (stays Rgb8):
//! let dev = rgb.to_cuda(&stream).unwrap();
//! let _back = dev.to_host(&stream).unwrap();
//! # }
//! ```

use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::error::ImageError;
use crate::image::{Image, ImageSize};
use kornia_tensor::{CudaError, Tensor};

impl<T, const C: usize> Image<T, C>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    /// Upload this image's buffer to a new device tensor (H2D) on `stream`.
    ///
    /// The entire contiguous `H × W × C` byte buffer is copied from host to
    /// device in a single `clone_htod` call. The returned tensor has shape
    /// `[H, W, C]` and is backed by a [`CudaAllocator`].
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA failure.
    pub fn to_cuda(&self, stream: &Arc<CudaStream>) -> Result<Tensor<T, 3>, CudaError> {
        self.0.to_cuda(stream)
    }

    /// Upload this image to a new **device-resident `Image`** (H2D copy).
    ///
    /// Unlike [`to_cuda`](Self::to_cuda), the result keeps the `Image<T, C>`
    /// type, so device images flow through the same typed APIs as host images
    /// (e.g. residency-aware color conversion). The backing storage is a typed
    /// `CudaResource<T>`, so `as_cudaslice::<T>()` / `cuda_stream()` work on it.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on CUDA failure.
    pub fn to_cuda_image(&self, stream: &Arc<CudaStream>) -> Result<Image<T, C>, ImageError> {
        let dev = self
            .0
            .to_cuda(stream)
            .map_err(|e| ImageError::Cuda(e.to_string()))?;
        Image::try_from(dev)
    }

    /// Allocate a zero-initialised **device-resident `Image`** of `size` on `stream`.
    ///
    /// The backing storage is a typed `CudaResource<T>` (allocated via
    /// [`kornia_tensor::zeros_cuda`]), so `as_cudaslice::<T>()` and
    /// `cuda_stream()` work on the result. Device images intended for
    /// kernel dispatch must be created through this or
    /// [`to_cuda_image`](Self::to_cuda_image) — the raw `CudaAllocator` path
    /// stores untyped `CudaResource<u8>` and will not downcast for `T ≠ u8`.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on CUDA allocation failure.
    pub fn zeros_cuda(
        size: ImageSize,
        stream: &Arc<CudaStream>,
    ) -> Result<Image<T, C>, ImageError> {
        let dev = kornia_tensor::zeros_cuda::<T, 3>([size.height, size.width, C], stream)
            .map_err(|e| ImageError::Cuda(e.to_string()))?;
        Image::try_from(dev)
    }

    /// Allocate an **uninitialized** device-resident image — [`Self::zeros_cuda`]
    /// without the zeroing memset. The fast path for a destination a kernel will
    /// fully overwrite (e.g. a color conversion that writes every output pixel).
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on CUDA allocation failure.
    ///
    /// # Safety
    ///
    /// The image's device memory is uninitialized; the caller MUST fully write
    /// every pixel before any read. See [`kornia_tensor::uninit_cuda`].
    pub unsafe fn uninit_cuda(
        size: ImageSize,
        stream: &Arc<CudaStream>,
    ) -> Result<Image<T, C>, ImageError> {
        // SAFETY: forwarded to the caller — the returned image must be fully
        // written before it is read.
        let dev = unsafe { kornia_tensor::uninit_cuda::<T, 3>([size.height, size.width, C], stream) }
            .map_err(|e| ImageError::Cuda(e.to_string()))?;
        Image::try_from(dev)
    }

    /// Allocate a zero-initialised host `Image` in **page-locked (pinned)**
    /// memory — an ordinary host image for every CPU path, but H2D/D2H copies
    /// against it are direct DMA. Allocate once and reuse (`cuMemHostAlloc`
    /// page-locks and is too expensive for a frame loop).
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on allocation failure.
    pub fn zeros_pinned(
        size: ImageSize,
        ctx: &Arc<cudarc::driver::CudaContext>,
    ) -> Result<Image<T, C>, ImageError> {
        let t = kornia_tensor::zeros_pinned::<T, 3>([size.height, size.width, C], ctx)
            .map_err(|e| ImageError::Cuda(e.to_string()))?;
        Image::try_from(t)
    }

    /// Copy this device-resident image back to a new, owned host image using
    /// the image's **own carried stream** (no stream parameter to get wrong).
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on CUDA failure or if the image is not
    /// device-backed.
    pub fn to_host_owned(&self) -> Result<Image<T, C>, ImageError> {
        let host = self
            .0
            .to_host_owned()
            .map_err(|e| ImageError::Cuda(e.to_string()))?;
        Image::try_from(host)
    }

    /// D2H-copy this device-resident image directly into a caller-provided host
    /// slice (its own carried stream, synchronized before returning) — avoiding
    /// the extra allocation + copy that [`Self::to_host_image`] performs when the
    /// caller already owns the destination buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on CUDA failure, a `dst`-length mismatch, or
    /// if the image is not device-backed.
    pub fn to_host_into(&self, dst: &mut [T]) -> Result<(), ImageError> {
        self.0
            .to_host_into(dst)
            .map_err(|e| ImageError::Cuda(e.to_string()))
    }

    /// Copy this device-resident image back to a new host-backed `Image` (D2H).
    ///
    /// Synchronizes `stream` before returning so the host data is valid.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Cuda`] on CUDA failure or if the image is not
    /// device-backed by a typed `CudaResource<T>`.
    pub fn to_host_image(&self, stream: &Arc<CudaStream>) -> Result<Image<T, C>, ImageError> {
        let host = self
            .0
            .to_host(stream)
            .map_err(|e| ImageError::Cuda(e.to_string()))?;
        Image::try_from(host)
    }
}
