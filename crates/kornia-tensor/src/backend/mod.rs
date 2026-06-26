//! GPU backend abstraction for `kornia-tensor`.
//!
//! This module defines the [`Backend`] trait and the [`GpuAllocator`] wrapper that bridges
//! any `Backend` impl into the existing [`TensorAllocator`](crate::allocator::TensorAllocator)
//! interface, letting `Tensor<T, N, GpuAllocator<B>>` work without changes to the core type.
//!
//! # Swap-in story
//!
//! The concrete backend lives behind the `gpu-cubecl` feature flag → [`cubecl`] module (stub).

use std::alloc::Layout;

use crate::allocator::{TensorAllocator, TensorAllocatorError};

#[cfg(feature = "gpu-cubecl")]
pub mod cubecl;

/// Raw device-memory provider.
///
/// The pointer returned by [`alloc`](Backend::alloc) is a **device address** —
/// callers must not dereference it on the host.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Backend-native error type returned from allocation failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate device memory for the given `layout`.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, Self::Error>;

    /// Free a device pointer previously returned by [`alloc`](Backend::alloc).
    ///
    /// # Safety
    /// `ptr` must have been returned by this backend's `alloc` with the same `layout`.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
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

// The trait method is intentionally safe; the unsafe block delegates to the Backend impl.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl<B: Backend> TensorAllocator for GpuAllocator<B> {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = self
            .backend
            .alloc(layout)
            .map_err(|_e| TensorAllocatorError::NullPointer)?;
        if ptr.is_null() {
            return Err(TensorAllocatorError::NullPointer);
        }
        Ok(ptr)
    }

    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            unsafe { self.backend.dealloc(ptr, layout) }
        }
    }
}
