use std::alloc::Layout;

use thiserror::Error;

use crate::resource::{HostResource, MemoryResource};

/// Error type for tensor memory allocation operations.
///
/// This enum represents all possible errors that can occur during tensor memory
/// allocation and deallocation.
#[derive(Debug, Error, PartialEq)]
pub enum TensorAllocatorError {
    /// Invalid memory layout for tensor allocation.
    ///
    /// This error occurs when attempting to create a memory layout with invalid
    /// parameters (e.g., size too large, alignment requirements not met).
    #[error("Invalid tensor layout {0}")]
    LayoutError(core::alloc::LayoutError),

    /// Allocation returned a null pointer.
    ///
    /// This typically indicates an out-of-memory condition or other allocation failure.
    #[error("Null pointer")]
    NullPointer,

    /// A foreign allocator was asked to allocate memory, which it never does.
    ///
    /// [`ForeignAllocator`] exists only as a type tag for externally-owned buffers;
    /// calling `allocate` on it is always an error.
    #[error(
        "Cannot allocate with a foreign allocator — use from_borrowed or a wrapping constructor"
    )]
    CannotAllocateForeign,

    /// Backend allocation failed with an error message.
    ///
    /// This error carries the backend-specific error description, such as a CUDA
    /// driver error message from CubeCL.
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
}

/// Trait for custom tensor memory allocators.
///
/// `TensorAllocator` enables supporting different memory backends (CPU, GPU, shared memory, etc.)
/// by abstracting the allocation interface. Implementors return an owning [`MemoryResource`]
/// handle that frees the backing buffer correctly on [`Drop`].
///
/// # Thread Safety
///
/// Implementations must be `Clone + Send + Sync` as tensors can be shared across threads.
///
/// # Examples
///
/// Using the default CPU allocator:
///
/// ```rust
/// use std::alloc::Layout;
/// use kornia_tensor::{CpuAllocator, TensorAllocator};
///
/// let allocator = CpuAllocator;
/// let layout = Layout::from_size_align(64, 8).unwrap();
/// let resource = allocator.allocate(layout).unwrap();
/// assert_eq!(resource.len_bytes(), 64);
/// assert!(resource.domain().is_host_accessible());
/// ```
pub trait TensorAllocator: Clone + Send + Sync {
    /// Allocates memory for a tensor with the given layout and returns an owning handle.
    ///
    /// # Arguments
    ///
    /// * `layout` - The memory layout specifying size and alignment requirements.
    ///
    /// # Returns
    ///
    /// A boxed [`MemoryResource`] that owns the allocation and frees it on drop.
    ///
    /// # Errors
    ///
    /// - [`TensorAllocatorError::NullPointer`] if the allocator returns a null pointer.
    /// - [`TensorAllocatorError::CannotAllocateForeign`] if called on [`ForeignAllocator`].
    fn allocate(&self, layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError>;
}

/// CPU memory allocator using the system allocator.
///
/// `CpuAllocator` is the default allocator for tensors, providing standard zeroed heap
/// allocation using Rust's global allocator. Suitable for general-purpose CPU tensor ops.
///
/// # Examples
///
/// ```rust
/// use std::alloc::Layout;
/// use kornia_tensor::{CpuAllocator, TensorAllocator};
///
/// let layout = Layout::from_size_align(1024, 8).unwrap();
/// let resource = CpuAllocator.allocate(layout).unwrap();
/// assert_eq!(resource.len_bytes(), 1024);
/// ```
#[derive(Clone)]
pub struct CpuAllocator;

/// Provides a default instance of [`CpuAllocator`].
impl Default for CpuAllocator {
    fn default() -> Self {
        Self
    }
}

// SAFETY: CpuAllocator is a zero-size unit struct with no interior mutability.
unsafe impl Send for CpuAllocator {}
unsafe impl Sync for CpuAllocator {}

impl TensorAllocator for CpuAllocator {
    /// Allocates a zeroed host buffer via [`HostResource::from_layout`].
    fn allocate(&self, layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        Ok(Box::new(HostResource::from_layout(layout)?))
    }
}

/// A no-op allocator for foreign (externally managed) memory.
///
/// Used as a type tag when wrapping memory that is owned by an external system
/// (e.g. numpy arrays, GStreamer buffers, or other external allocations). Calling [`allocate`](TensorAllocator::allocate)
/// on `ForeignAllocator` always returns [`TensorAllocatorError::CannotAllocateForeign`];
/// use `from_borrowed` or a wrapping constructor instead.
#[derive(Clone)]
pub struct ForeignAllocator;

// SAFETY: ForeignAllocator is a zero-size unit struct with no interior mutability.
unsafe impl Send for ForeignAllocator {}
unsafe impl Sync for ForeignAllocator {}

impl TensorAllocator for ForeignAllocator {
    /// Always returns [`TensorAllocatorError::CannotAllocateForeign`].
    ///
    /// Foreign allocators never allocate — they are pure type tags for externally
    /// owned buffers.
    fn allocate(&self, _layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        Err(TensorAllocatorError::CannotAllocateForeign)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    /// Verify that `CpuAllocator::allocate` returns a zeroed, host-accessible buffer.
    #[test]
    fn cpu_allocate_zeroed_and_aligned() {
        let l = Layout::from_size_align(64, 1).unwrap();
        let r = CpuAllocator.allocate(l).unwrap();
        assert_eq!(r.len_bytes(), 64);
        assert!(r.domain().is_host_accessible());
        // Must be zeroed.
        unsafe { assert!((0..64).all(|i| *r.as_ptr().add(i) == 0)) };
    }

    /// `ForeignAllocator::allocate` must always return `CannotAllocateForeign`.
    #[test]
    fn foreign_allocate_returns_error() {
        let l = Layout::from_size_align(64, 1).unwrap();
        match ForeignAllocator.allocate(l) {
            Err(TensorAllocatorError::CannotAllocateForeign) => {}
            other => panic!("expected CannotAllocateForeign, got {:?}", other.err()),
        }
    }

    /// Allocator types must be `Clone`.
    #[test]
    fn allocators_are_clone() {
        let _a = CpuAllocator.clone();
        let _b = ForeignAllocator.clone();
    }
}
