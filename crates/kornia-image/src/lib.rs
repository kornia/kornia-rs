#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// allocator module containing the memory management utilities.
pub mod allocator;

/// image representation for computer vision purposes.
pub mod image;

/// Error types for the image module.
pub mod error;

/// module containing ops implementations.
pub mod ops;

/// Typed color space wrappers for compile-time type safety.
pub mod color_spaces;

pub use crate::error::ImageError;
pub use crate::image::{Image, ImageSize};
pub use crate::allocator::{CpuAllocator, ImageAllocator};
#[cfg(feature = "cuda")]
pub use crate::allocator::CudaAllocator;

// Convenience type aliases for CPU images
/// RGB8 image on CPU
pub type Image3u8 = Image<u8, 3, CpuAllocator>;
/// Grayscale image on CPU
pub type Image1u8 = Image<u8, 1, CpuAllocator>;
/// RGB32F image on CPU
pub type Image3f32 = Image<f32, 3, CpuAllocator>;

// Convenience type aliases for CUDA images
#[cfg(feature = "cuda")]
/// RGB8 image on CUDA
pub type CudaImage3u8 = Image<u8, 3, CudaAllocator>;
#[cfg(feature = "cuda")]
/// Grayscale image on CUDA
pub type CudaImage1u8 = Image<u8, 1, CudaAllocator>;
#[cfg(feature = "cuda")]
/// RGB32F image on CUDA
pub type CudaImage3f32 = Image<f32, 3, CudaAllocator>;

/// Arrow integration for converting images to Arrow format
#[cfg(feature = "arrow")]
pub mod arrow;
