//! CUDA device-memory integration for [`Image`].
//!
//! This module is enabled by the `cudarc` feature flag. It provides
//! [`Image::to_cuda`], which uploads a host-backed image to a new
//! device-backed [`kornia_tensor::Tensor`] via a host-to-device copy.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "cudarc")]
//! # {
//! use kornia_image::color_spaces::Rgb8;
//! use kornia_tensor::CpuAllocator;
//! use cudarc::driver::CudaContext;
//! use std::sync::Arc;
//!
//! let rgb = Rgb8::from_size_val(
//!     kornia_image::ImageSize { width: 2, height: 2 },
//!     [0u8; 3],
//!     CpuAllocator,
//! ).unwrap();
//!
//! let ctx = CudaContext::new(0).unwrap();
//! let stream = ctx.default_stream();
//! // Works via Deref: Rgb8 → Image<u8,3,_> → to_cuda
//! let _dev = rgb.to_cuda(&stream).unwrap();
//! # }
//! ```

use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::allocator::ImageAllocator;
use crate::image::Image;
use kornia_tensor::{CudaAllocator, CudaError, Tensor};

impl<T, const C: usize, A: ImageAllocator> Image<T, C, A>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
    A: 'static,
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
    pub fn to_cuda(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Tensor<T, 3, CudaAllocator>, CudaError> {
        self.0.to_cuda(stream)
    }
}
