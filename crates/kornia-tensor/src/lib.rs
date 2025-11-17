#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Overview
//!
//! `kornia-tensor` is a lightweight, high-performance tensor library designed specifically for
//! computer vision applications. It provides a flexible memory management system with custom
//! allocators and supports multi-dimensional arrays with efficient memory layouts.
//!
//! # Architecture
//!
//! The crate is organized into several key components:
//!
//! - **Tensor**: The main data structure representing multi-dimensional arrays with shape and stride information
//! - **TensorStorage**: Low-level memory buffer management with custom allocator support
//! - **TensorView**: Non-owning views into tensor data for efficient memory sharing
//! - **TensorAllocator**: Trait-based memory allocation system for different memory backends
//!
//! # Key Features
//!
//! - **Zero-copy operations**: Views and reshaping operations avoid data copying when possible
//! - **Custom allocators**: Support for different memory backends (CPU, GPU, etc.)
//! - **Type-safe dimensions**: Compile-time dimensional checking using const generics
//! - **Standard memory layouts**: Automatic stride calculation for contiguous memory
//! - **Thread-safe**: All components are Send + Sync when using thread-safe allocators
//!
//! # Quick Start
//!
//! Creating and manipulating tensors:
//!
//! ```rust
//! use kornia_tensor::{Tensor, CpuAllocator};
//!
//! // Create a 2x3 tensor from a vector
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::<f32, 2, _>::from_shape_vec([2, 3], data, CpuAllocator).unwrap();
//!
//! // Access elements
//! assert_eq!(tensor.get([0, 0]), Some(&1.0));
//! assert_eq!(tensor.get([1, 2]), Some(&6.0));
//!
//! // Reshape to a different shape
//! let reshaped = tensor.reshape([3, 2]).unwrap();
//! assert_eq!(reshaped.shape, [3, 2]);
//! ```
//!
//! Using tensor operations:
//!
//! ```rust
//! use kornia_tensor::{Tensor, CpuAllocator};
//!
//! // Create tensors with specific values
//! let zeros = Tensor::<f32, 2, _>::zeros([3, 3], CpuAllocator);
//! let ones = Tensor::<f32, 2, _>::from_shape_val([3, 3], 1.0, CpuAllocator);
//!
//! // Generate data with a function
//! let identity = Tensor::<f32, 2, _>::from_shape_fn([3, 3], CpuAllocator, |[i, j]| {
//!     if i == j { 1.0 } else { 0.0 }
//! });
//!
//! // Apply element-wise operations
//! let result = ones.map(|x| x * 2.0);
//! ```
//!
//! Working with views:
//!
//! ```rust
//! use kornia_tensor::{Tensor, CpuAllocator};
//!
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let tensor = Tensor::<i32, 1, _>::from_shape_vec([6], data, CpuAllocator).unwrap();
//!
//! // Create a reshaped view without copying data
//! let view = tensor.reshape([2, 3]).unwrap();
//!
//! // Permute dimensions
//! let transposed = tensor.reshape([2, 3]).unwrap().as_contiguous();
//! ```
//!
//! # Type Aliases
//!
//! The crate provides convenient type aliases for common tensor dimensions:
//!
//! - [`Tensor1`]: One-dimensional tensor (vector)
//! - [`Tensor2`]: Two-dimensional tensor (matrix)
//! - [`Tensor3`]: Three-dimensional tensor
//! - [`Tensor4`]: Four-dimensional tensor
//! - [`CpuTensor2`]: Two-dimensional CPU tensor (most common)

/// Allocator module containing memory management utilities.
///
/// This module provides the [`TensorAllocator`] trait and implementations for different
/// memory backends. The default [`CpuAllocator`] uses the system allocator for CPU memory.
pub mod allocator;

/// backend module containing device operation abstractions.
pub mod backend;

/// Bincode module for binary serialization and deserialization.
///
/// This module provides efficient binary serialization support for tensors when the
/// `bincode` feature is enabled.
#[cfg(feature = "bincode")]
pub mod bincode;

/// Device module containing device abstraction.
pub mod device;

/// device_marker module containing zero-cost device type markers.
pub mod device_marker;

/// Serde module for JSON/other format serialization and deserialization.
///
/// This module provides flexible serialization support for tensors when the
/// `serde` feature is enabled.
#[cfg(feature = "serde")]
pub mod serde;

/// Storage module containing low-level memory buffer implementations.
///
/// This module provides [`storage::TensorStorage`] which manages the actual memory buffer
/// for tensor data with custom allocator support.
pub mod storage;

/// Tensor module containing the main tensor implementation and error types.
///
/// This module provides the core [`tensor::Tensor`] struct and related functionality.
pub mod tensor;

/// View module containing non-owning tensor view implementations.
///
/// This module provides [`view::TensorView`] for creating efficient, zero-copy views
/// into existing tensor data.
pub mod view;

pub use crate::allocator::{CpuAllocator, TensorAllocator};
#[cfg(feature = "cuda")]
pub use crate::allocator::CudaAllocator;
pub use crate::backend::{Backend, CpuBackend};
#[cfg(feature = "cuda")]
pub use crate::backend::CudaBackend;
pub use crate::device::Device;
pub use crate::device_marker::{Cpu, DeviceMarker};
#[cfg(feature = "cuda")]
pub use crate::device_marker::Cuda;
pub(crate) use crate::tensor::get_strides_from_shape;
pub use crate::tensor::{Tensor, TensorError};

/// Type alias for a 1-dimensional tensor.
pub type Tensor1<T, A> = Tensor<T, 1, A>;

/// Type alias for a 2-dimensional tensor.
pub type Tensor2<T, A> = Tensor<T, 2, A>;

/// Type alias for a 3-dimensional tensor.
pub type Tensor3<T, A> = Tensor<T, 3, A>;

/// Type alias for a 4-dimensional tensor.
pub type Tensor4<T, A> = Tensor<T, 4, A>;

/// Type alias for a 2-dimensional tensor with CPU allocator.
pub type CpuTensor2<T> = Tensor2<T, CpuAllocator>;
