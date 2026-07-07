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

    /// An allocator was asked to allocate memory but does not support it.
    ///
    /// Returned by allocators that wrap externally-owned buffers (e.g. Arrow, GStreamer);
    /// use `from_borrowed` or a wrapping constructor instead.
    #[error("Cannot allocate with this allocator — use from_borrowed or a wrapping constructor")]
    CannotAllocateForeign,

    /// Backend allocation failed with an error message.
    ///
    /// This error carries the backend-specific error description, such as a CUDA
    /// driver error message from CubeCL.
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    /// A CUDA allocation or driver call failed.
    ///
    /// Only produced when the `cudarc` feature is enabled.
    #[cfg(feature = "cuda")]
    #[error("CUDA allocator error: {0}")]
    CudaError(String),
}

/// Trait for custom tensor memory allocators.
///
/// `TensorAllocator` enables supporting different memory backends (CPU, GPU, shared memory, etc.)
/// by abstracting the allocation interface. Implementors return an owning [`MemoryResource`]
/// handle that frees the backing buffer correctly on [`Drop`].
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` as tensors can be shared across threads. The
/// allocator is held behind an [`AllocHandle`] (`Arc<dyn TensorAllocator>`), so it is
/// shared by reference-count, not cloned by value.
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
pub trait TensorAllocator: Send + Sync {
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
    /// - [`TensorAllocatorError::CannotAllocateForeign`] if the allocator wraps foreign memory.
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

// ── Runtime allocator handle ──────────────────────────────────────────────────

use std::sync::{Arc, LazyLock};

/// A cheaply-cloneable runtime allocator reference.
///
/// `TensorAllocator` is object-safe (one non-generic method, no `Clone` supertrait),
/// so the handle is a plain `Arc<dyn TensorAllocator>` — no separate object-safe shim.
pub type AllocHandle = Arc<dyn TensorAllocator>;

/// Process-global host allocator handle, initialised once on first use.
///
/// `CpuAllocator` is a stateless ZST, so a single shared `Arc` is safe to hand out to
/// every host tensor; cloning it is one atomic increment with no per-tensor heap box.
static HOST_ALLOC: LazyLock<AllocHandle> = LazyLock::new(|| Arc::new(CpuAllocator));

/// Returns the process-global host allocator handle.
pub fn host_alloc() -> AllocHandle {
    HOST_ALLOC.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    /// Verify that `CpuAllocator::allocate` returns a zeroed, host-accessible buffer.
    #[test]
    fn cpu_allocate_zeroed_and_aligned() {
        let l = Layout::from_size_align(64, 1).unwrap();
        let r = TensorAllocator::allocate(&CpuAllocator, l).unwrap();
        assert_eq!(r.len_bytes(), 64);
        assert!(r.domain().is_host_accessible());
        // Must be zeroed.
        unsafe { assert!((0..64).all(|i| *r.as_ptr().add(i) == 0)) };
    }
}
