//! GPU backend abstraction for `kornia-tensor`.
//!
//! This module defines the [`Backend`] trait and the [`GpuAllocator`] wrapper that bridges
//! any `Backend` impl into the existing [`TensorAllocator`](crate::allocator::TensorAllocator)
//! interface, letting `Tensor<T, N, GpuAllocator<B>>` work without changes to the core type.
//!
//! # Swap-in story
//!
//! Each concrete backend lives behind its own feature flag:
//! - `gpu-cubecl`      → [`cubecl`] module  (stable Rust, multi-platform JIT)
//! - `gpu-cuda-oxide`  → [`cuda_oxide`] module  (reserved; see `proto/cuda-oxide` branch)
//!
//! To add a new backend: implement [`Backend`] for your type, add a `gpu-<name>` feature in
//! `Cargo.toml`, and add a `#[cfg(feature = "gpu-<name>")] pub mod <name>;` entry here.

use std::alloc::Layout;

use crate::allocator::{TensorAllocator, TensorAllocatorError};

#[cfg(feature = "gpu-cubecl")]
pub mod cubecl;

#[cfg(feature = "gpu-cuda-oxide")]
pub mod cuda_oxide;

/// Raw device-memory provider.
///
/// The pointer returned by [`alloc`](Backend::alloc) is a **device address** —
/// callers must not dereference it on the host.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Backend-native error type returned from allocation failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate `size` bytes of device memory. Returns a device pointer.
    fn alloc(&self, size: usize) -> Result<*mut u8, Self::Error>;

    /// Free a device pointer previously returned by [`alloc`](Backend::alloc).
    ///
    /// # Safety
    /// `ptr` must have been returned by this backend's `alloc`, and `size` must match.
    unsafe fn dealloc(&self, ptr: *mut u8, size: usize);
}

/// [`TensorAllocator`] wrapper around any [`Backend`].
#[derive(Clone)]
pub struct GpuAllocator<B: Backend> {
    backend: B,
}

impl<B: Backend> GpuAllocator<B> {
    /// Wrap a backend in a `GpuAllocator`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: Backend> TensorAllocator for GpuAllocator<B> {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        self.backend
            .alloc(layout.size())
            .map_err(|_e| TensorAllocatorError::NullPointer)
    }

    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            unsafe { self.backend.dealloc(ptr, layout.size()) }
        }
    }
}
