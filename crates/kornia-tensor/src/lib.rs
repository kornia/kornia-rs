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
//! - **TensorStorage**: Low-level memory buffer management holding a runtime allocator handle
//! - **TensorView**: Non-owning views into tensor data for efficient memory sharing
//! - **TensorAllocator**: Trait-based memory allocation system for different memory backends
//!
//! # Memory Model
//!
//! A tensor's location is described at **runtime**, not in its type. `Tensor<T, N>` has no
//! allocator/device type parameter; instead [`TensorStorage`](storage::TensorStorage) carries:
//!
//! - an `owner: Box<dyn `[`MemoryResource`](resource::MemoryResource)`>` — the single source of
//!   truth that frees the buffer on `Drop` and reports its [`MemoryDomain`](resource::MemoryDomain):
//!   - [`Host`](resource::MemoryDomain::Host) — kornia- or foreign-owned CPU memory,
//!   - [`Device { id }`](resource::MemoryDomain::Device) — CUDA device memory,
//!   - [`Unified { id }`](resource::MemoryDomain::Unified) — CUDA managed/unified memory;
//! - an [`AllocHandle`] (`Arc<dyn `[`TensorAllocator`]`>`) — a cheap, reference-counted runtime
//!   handle propagated to derived tensors (`cast`, `map`, `as_contiguous`, …). It is an **advisory
//!   tag**: host buffers are currently allocated through the global allocator (via `Vec`), so the
//!   handle is not yet consulted to allocate host memory; it is never touched on the hot path;
//! - a cached `NonNull<T>` pointer read directly by element access — so domain/handle add **zero**
//!   overhead to `as_ptr`/`as_slice`/indexing.
//!
//! **Host-accessibility gate.** Element access is checked at runtime, not by the type system:
//! [`as_slice`](storage::TensorStorage::as_slice)/`as_mut_slice` (and indexing) **panic** on a
//! non-host-accessible (`Device`) tensor. Move data with `to_host` / `to_cuda` first. A single
//! concrete `Tensor<T, N>` can therefore live in a `Vec` alongside tensors on other domains.
//!
//! # Unified Constructor API
//!
//! The common host case needs **no** allocator argument (mirrors `Vec::new` / `Vec::new_in`):
//!
//! ```rust
//! use kornia_tensor::{Tensor, host_alloc};
//!
//! // Host (default) — no allocator argument:
//! let a = Tensor::<f32, 2>::from_shape_vec([2, 3], vec![0.0; 6]).unwrap();
//! let z = Tensor::<f32, 2>::zeros([2, 3]);
//!
//! // Explicit allocator handle via the `_in` variants (custom/arena/device allocators):
//! let b = Tensor::<f32, 2>::from_shape_vec_in([2, 3], vec![0.0; 6], host_alloc()).unwrap();
//! ```
//!
//! Every host-default constructor has an `_in` counterpart taking an explicit [`AllocHandle`]
//! (which it threads onto the storage as the advisory tag described above — the buffer itself is
//! still allocated through the global allocator):
//! `from_shape_vec`/`_in`, `from_shape_slice`/`_in`, `from_shape_val`/`_in`, `from_shape_fn`/`_in`,
//! `zeros`/`_in`. Low-level / non-host constructors always take an explicit handle or resource:
//! [`from_vec`](storage::TensorStorage::from_vec), `from_raw_host`, `from_borrowed`, `from_raw_parts`.
//! Device tensors are created through the stream-based CUDA API rather than a handle —
//! `to_cuda`, `zeros_cuda`, `from_cudaslice` (feature `cudarc`) — which build the device
//! allocator handle internally from the supplied `CudaStream`.
//!
//! # Key Features
//!
//! - **Zero-copy operations**: Views and reshaping operations avoid data copying when possible
//! - **Runtime device model**: one concrete `Tensor<T, N>` spans Host / Device / Unified memory
//! - **Ergonomic host default**: host constructors need no allocator; `_in` variants for custom ones
//! - **Type-safe dimensions**: Compile-time dimensional checking using const generics
//! - **Standard memory layouts**: Automatic stride calculation for contiguous memory
//! - **Thread-safe**: All components are Send + Sync (allocator handle is `Arc<dyn TensorAllocator>`)
//!
//! # Quick Start
//!
//! Creating and manipulating tensors:
//!
//! ```rust
//! use kornia_tensor::Tensor;
//!
//! // Create a 2x3 tensor from a vector
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::<f32, 2>::from_shape_vec([2, 3], data).unwrap();
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
//! use kornia_tensor::Tensor;
//!
//! // Create tensors with specific values
//! let zeros = Tensor::<f32, 2>::zeros([3, 3]);
//! let ones = Tensor::<f32, 2>::from_shape_val([3, 3], 1.0);
//!
//! // Generate data with a function
//! let identity = Tensor::<f32, 2>::from_shape_fn([3, 3], |[i, j]| {
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
//! use kornia_tensor::Tensor;
//!
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let tensor = Tensor::<i32, 1>::from_shape_vec([6], data).unwrap();
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
//! - [`CpuTensor2`]: Two-dimensional tensor backed by CPU memory

/// Allocator module containing memory management utilities.
///
/// This module provides the [`TensorAllocator`] trait and implementations for different
/// memory backends. The default [`CpuAllocator`] uses the system allocator for CPU memory.
pub mod allocator;

/// DLPack interop — convert Tensor to/from DLManagedTensor (CPU and CUDA).
///
/// Enabled by the `dlpack` feature. Zero-copy for owned CPU tensors (the tensor
/// itself becomes the keepalive object passed to `dlpack_rs::safe::pack`).
#[cfg(feature = "dlpack")]
pub mod dlpack;

/// Bincode module for binary serialization and deserialization.
///
/// This module provides efficient binary serialization support for tensors when the
/// `bincode` feature is enabled.
#[cfg(feature = "bincode")]
pub mod bincode;

/// Serde module for JSON/other format serialization and deserialization.
///
/// This module provides flexible serialization support for tensors when the
/// `serde` feature is enabled.
#[cfg(feature = "serde")]
pub mod serde;

/// Resource module providing ownership handles for tensor backing memory.
///
/// This module defines the [`resource::MemoryResource`] trait, the three-state
/// [`resource::MemoryDomain`] enum, [`resource::HostResource`] (kornia-owned host memory),
/// and [`resource::ForeignResource`] (externally owned memory: numpy, gstreamer, dlpack, …).
pub mod resource;

/// CUDA device-memory integration via `cudarc 0.19`.
///
/// Enabled by the `cudarc` feature (default OFF). Provides [`cuda::CudaResource`],
/// [`cuda::CudaAllocator`], [`cuda::CudaKernel`], [`cuda::CudaLaunchBuilder`],
/// [`cuda::zeros_cuda`], and `Tensor` methods `to_cuda`/`to_host`/`from_cudaslice`/
/// `as_cudaslice`/`into_cudaslice` for first-class device tensor support.
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use crate::cuda::{
    pinned_alloc, zeros_cuda, zeros_pinned, CudaAllocator, CudaError, CudaKernel,
    CudaLaunchBuilder, PinnedAllocator, PinnedResource,
};

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

pub use crate::allocator::{host_alloc, AllocHandle, CpuAllocator, TensorAllocator};
pub use crate::resource::{ForeignResource, HostResource, MemoryDomain, MemoryResource};
// Keep backward-compatible re-export: `use kornia_tensor::storage::MemoryDomain` still resolves
// because storage.rs now re-exports from resource.
pub(crate) use crate::tensor::get_strides_from_shape;
pub use crate::tensor::{Tensor, TensorError};

// Note: type aliases no longer carry an allocator parameter — the allocator is runtime.

/// Type alias for a 1-dimensional tensor.
pub type Tensor1<T> = Tensor<T, 1>;

/// Type alias for a 2-dimensional tensor.
pub type Tensor2<T> = Tensor<T, 2>;

/// Type alias for a 3-dimensional tensor.
pub type Tensor3<T> = Tensor<T, 3>;

/// Type alias for a 4-dimensional tensor.
pub type Tensor4<T> = Tensor<T, 4>;

/// Type alias for a 2-dimensional tensor backed by CPU memory.
pub type CpuTensor2<T> = Tensor2<T>;
