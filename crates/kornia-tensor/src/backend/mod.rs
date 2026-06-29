//! GPU backend abstraction for `kornia-tensor`.
//!
//! This module defines the [`Backend`] trait and the [`GpuAllocator`] wrapper that bridges
//! any `Backend` impl into the existing [`TensorAllocator`](crate::allocator::TensorAllocator)
//! interface. `GpuAllocator<B>` implements `TensorAllocator` and can be wrapped in an
//! [`AllocHandle`](crate::allocator::AllocHandle) (`Arc<dyn AllocDyn>`) for use with `Tensor<T, N>`.
//!
//! # Swap-in story
//!
//! The concrete backend lives behind the `gpu-cubecl` feature flag → [`cubecl`] module (stub).

use std::alloc::Layout;
use std::any::Any;

use crate::allocator::{TensorAllocator, TensorAllocatorError};
use crate::resource::{MemoryDomain, MemoryResource};

#[cfg(feature = "gpu-cubecl")]
pub mod cubecl;

/// Raw device-memory provider.
///
/// The pointer returned by [`alloc`](Backend::alloc) is a **device address** —
/// callers must not dereference it on the host.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Backend-native error type returned from allocation failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate `layout.size()` bytes of device memory.
    ///
    /// Returns the device pointer (not host-dereferenceable) and the CUDA device id
    /// so that the caller can populate a [`GpuResource`].
    fn alloc(&self, layout: Layout) -> Result<(*mut u8, i32), Self::Error>;

    /// Free a device pointer previously returned by [`alloc`](Backend::alloc).
    ///
    /// # Safety
    /// `ptr` must have been returned by this backend's `alloc` with the same `layout`.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
}

// ── GpuResource ───────────────────────────────────────────────────────────────

/// An owning device-memory handle returned by [`GpuAllocator::allocate`].
///
/// Holds the raw device pointer plus enough state to call the backend's free routine
/// on drop.  The pointer is a device address — it must NOT be dereferenced on the host.
///
/// # Single-device / single-context assumption
///
/// `GpuResource` is `Send`, so it may be dropped on a thread other than the one that
/// allocated it.  The `cudarc` free path (`free_sync`) requires the allocating CUDA
/// context to be current on the calling thread.  In a single-GPU, single-context
/// program this is always satisfied.  Multi-GPU or multi-context programs must ensure
/// the allocating context is made current before the resource is dropped.
pub struct GpuResource<B: Backend> {
    /// Raw device pointer (not host-dereferenceable).
    ptr: *mut u8,
    /// Byte length of the allocation.
    len_bytes: usize,
    /// CUDA (or other accelerator) device id.
    device_id: i32,
    /// Layout used during allocation (needed to call `Backend::dealloc`).
    layout: Layout,
    /// Backend clone retained so `drop` can call `dealloc`.
    backend: B,
}

// SAFETY: the device pointer is never dereferenced on the host; `B` is Send + Sync.
unsafe impl<B: Backend> Send for GpuResource<B> {}
unsafe impl<B: Backend> Sync for GpuResource<B> {}

impl<B: Backend> MemoryResource for GpuResource<B> {
    /// Device pointer — NOT safe to dereference on the host.
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Byte length of the device allocation.
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Returns [`MemoryDomain::Device`] with the device id for this allocation.
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Device { id: self.device_id }
    }

    /// Downcast hook for type-erased access.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Mutable downcast hook for type-erased access.
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl<B: Backend> Drop for GpuResource<B> {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: `ptr` was returned by this backend's `alloc` with `self.layout`.
            unsafe { self.backend.dealloc(self.ptr, self.layout) }
        }
    }
}

// ── GpuAllocator ─────────────────────────────────────────────────────────────

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
    /// Allocate device memory and return an owning [`GpuResource`] handle.
    fn allocate(&self, layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        let (ptr, device_id) = self
            .backend
            .alloc(layout)
            .map_err(|e| TensorAllocatorError::AllocationFailed(e.to_string()))?;
        if layout.size() != 0 && ptr.is_null() {
            return Err(TensorAllocatorError::NullPointer);
        }
        Ok(Box::new(GpuResource {
            ptr,
            len_bytes: layout.size(),
            device_id,
            layout,
            backend: self.backend.clone(),
        }))
    }
}
