#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

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
#[cfg(feature = "cudarc")]
pub mod cuda;
