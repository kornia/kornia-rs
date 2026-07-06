#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Memory model
//!
//! `Image<T, C>` is a thin wrapper over a `kornia_tensor::Tensor3<T>` and inherits its
//! **runtime** memory model: there is no allocator/device type parameter. The backing
//! [`TensorStorage`](kornia_tensor::storage::TensorStorage) records a memory domain
//! (`Host` / `Device` / `Unified`) plus a runtime allocator handle; `as_slice` panics on a
//! non-host-accessible (device) image. See the `kornia_tensor` crate docs for the full model.
//!
//! # Unified constructor API
//!
//! The common host case needs no allocator argument; `_in` variants take an explicit
//! [`AllocHandle`](kornia_tensor::AllocHandle):
//!
//! ```rust
//! use kornia_image::{Image, ImageSize};
//!
//! let size = ImageSize { width: 4, height: 3 };
//! let img = Image::<u8, 3>::from_size_val(size, 0)?;            // host default
//! let img2 = Image::<u8, 3>::new(size, vec![0u8; 4 * 3 * 3])?;  // host default
//! # Ok::<(), kornia_image::ImageError>(())
//! ```
//!
//! `new`/`new_in`, `from_size_val`/`_in`, and `from_size_slice`/`_in` follow the same
//! host-default + `_in` pattern as `kornia_tensor`.

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

/// module containing ops implementations.
pub mod ops;

/// Typed color-space wrappers and runtime color-space vocabulary.
pub mod color_spaces;

/// Allocator re-exports for backward compatibility.
///
/// `kornia_image::allocator::host_alloc()` and `kornia_image::allocator::CpuAllocator`
/// resolve to the same items from `kornia_tensor`; prefer importing from `kornia_tensor`
/// directly in new code.
pub mod allocator;

pub use crate::color_spaces::{ColorSpace, DynImage};
pub use crate::error::ImageError;
pub use crate::image::{Image, ImageLayout, ImageSize, InterpolationMode, PixelFormat};
pub use kornia_tensor::{host_alloc, AllocHandle};

/// Arrow integration for converting images to Arrow format
#[cfg(feature = "arrow")]
pub mod arrow;

/// DLPack interoperability for typed [`Image`].
#[cfg(feature = "dlpack")]
pub mod dlpack;

/// CUDA device-memory integration: [`Image::to_cuda`].
#[cfg(feature = "cuda")]
pub mod cuda;
