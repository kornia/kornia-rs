#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Image
//!
//! High-performance image data structures with flexible memory management for computer vision applications.
//!
//! ## Key Features
//!
//! - **Custom Allocators**: Flexible memory management (CPU, GPU, etc.)
//! - **Type-Safe Color Spaces**: Compile-time validation of image channel configurations
//! - **Zero-Copy Operations**: Efficient image views and transformations
//! - **Arrow Integration**: Seamless interoperability with Apache Arrow (optional)
//!
//! ## Example
//!
//! ```rust
//! use kornia_image::{Image, ImageSize, CpuAllocator};
//!
//! // Create a 640x480 RGB image
//! let size = ImageSize {
//!     width: 640,
//!     height: 480,
//! };
//!
//! let image = Image::<u8, 3, _>::from_size_val(size, 0, CpuAllocator)?;
//! println!("Created {}x{} image with {} channels",
//!     image.width(), image.height(), image.channels());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Working with Different Color Spaces
//!
//! ```rust
//! use kornia_image::{Image, ImageSize, CpuAllocator};
//! use kornia_image::color_spaces::{Rgb, Gray};
//!
//! // Create a typed grayscale image
//! let gray_img: Image<u8, 1, CpuAllocator, Gray> =
//!     Image::from_size_val([480, 640].into(), 128, CpuAllocator)?;
//!
//! // Create a typed RGB image
//! let rgb_img: Image<u8, 3, CpuAllocator, Rgb> =
//!     Image::from_size_val([480, 640].into(), 0, CpuAllocator)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Allocator module containing memory management utilities.
///
/// Provides the [`TensorAllocator`](allocator::ImageAllocator) trait and implementations
/// for different memory backends.
pub mod allocator;

/// Image representation optimized for computer vision tasks.
///
/// Contains the core [`Image`](image::Image) struct and related types.
pub mod image;

/// Error types for image operations.
///
/// Defines [`ImageError`](error::ImageError) and related error handling.
pub mod error;

/// Operations on image data structures.
///
/// Provides utility functions for common image manipulations.
pub mod ops;

/// Typed color space wrappers for compile-time type safety.
///
/// Marker types for RGB, grayscale, and other color spaces.
pub mod color_spaces;

pub use crate::error::ImageError;
pub use crate::image::{Image, ImageSize};
pub use crate::allocator::{CpuAllocator, ImageAllocator};

/// Arrow integration for converting images to Arrow format.
///
/// Enables zero-copy data exchange with the Apache Arrow ecosystem.
#[cfg(feature = "arrow")]
pub mod arrow;
