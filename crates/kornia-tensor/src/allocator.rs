use std::alloc;
use std::alloc::Layout;

use thiserror::Error;

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
}

/// Trait for custom tensor memory allocators.
///
/// `TensorAllocator` enables supporting different memory backends (CPU, GPU, shared memory, etc.)
/// by abstracting the allocation and deallocation interface. Implementors can provide custom
/// memory management strategies while maintaining compatibility with the tensor library.
///
/// # Thread Safety
///
/// Implementations must be thread-safe (`Send` + `Sync`) as tensors can be shared across threads.
/// The allocator is typically wrapped in `Arc` or similar when shared between tensors.
///
/// # Lifecycle
///
/// - [`alloc`](Self::alloc): Called when creating new tensor storage
/// - [`dealloc`](Self::dealloc): Called when tensor storage is dropped
///
/// # Examples
///
/// Using the default CPU allocator:
///
/// ```rust
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let allocator = CpuAllocator;
/// let tensor = Tensor::<f32, 2, _>::zeros([100, 100], allocator);
/// ```
///
/// Implementing a custom allocator:
///
/// ```rust
/// use std::alloc::Layout;
/// use kornia_tensor::{TensorAllocator, allocator::TensorAllocatorError};
///
/// #[derive(Clone)]
/// struct AlignedAllocator {
///     alignment: usize,
/// }
///
/// impl TensorAllocator for AlignedAllocator {
///     fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
///         // Custom allocation with guaranteed alignment
///         let aligned_layout = Layout::from_size_align(layout.size(), self.alignment)
///             .map_err(TensorAllocatorError::LayoutError)?;
///         
///         let ptr = unsafe { std::alloc::alloc(aligned_layout) };
///         if ptr.is_null() {
///             return Err(TensorAllocatorError::NullPointer);
///         }
///         Ok(ptr)
///     }
///
///     fn dealloc(&self, ptr: *mut u8, layout: Layout) {
///         if !ptr.is_null() {
///             let aligned_layout = Layout::from_size_align(layout.size(), self.alignment)
///                 .unwrap();
///             unsafe { std::alloc::dealloc(ptr, aligned_layout) }
///         }
///     }
/// }
/// ```
pub trait TensorAllocator: Clone {
    /// Allocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The memory layout specifying size and alignment requirements
    ///
    /// # Returns
    ///
    /// A pointer to the allocated memory on success, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns [`TensorAllocatorError::NullPointer`] if allocation fails.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;

    /// Deallocates memory previously allocated by this allocator.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the memory to deallocate
    /// * `layout` - The layout used when allocating this memory
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by [`alloc`](Self::alloc) with the same layout.
    fn dealloc(&self, ptr: *mut u8, layout: Layout);
}

/// CPU memory allocator using the system allocator.
///
/// `CpuAllocator` is the default allocator for tensors, providing standard heap allocation
/// using Rust's global allocator. It's suitable for general-purpose CPU-based tensor operations.
///
/// # Examples
///
/// ```rust
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let tensor = Tensor::<f32, 2, _>::zeros([10, 10], CpuAllocator);
/// ```
#[derive(Clone)]
pub struct CpuAllocator;

/// Provides a default instance of [`CpuAllocator`].
impl Default for CpuAllocator {
    fn default() -> Self {
        Self
    }
}

/// Implements [`TensorAllocator`] using the Rust global allocator.
impl TensorAllocator for CpuAllocator {
    /// Allocates memory using the system allocator.
    ///
    /// This uses Rust's global allocator (typically the system's malloc/free) to allocate
    /// memory with the specified layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The memory layout specifying size and alignment
    ///
    /// # Returns
    ///
    /// A pointer to the allocated memory on success.
    ///
    /// # Errors
    ///
    /// Returns [`TensorAllocatorError::NullPointer`] if the allocation fails (typically
    /// due to insufficient memory).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::alloc::Layout;
    /// use kornia_tensor::{TensorAllocator, CpuAllocator};
    ///
    /// let allocator = CpuAllocator;
    /// let layout = Layout::from_size_align(1024, 8).unwrap();
    /// let ptr = allocator.alloc(layout).unwrap();
    /// allocator.dealloc(ptr, layout);
    /// ```
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }
        Ok(ptr)
    }

    /// Deallocates memory using the system allocator.
    ///
    /// This safely deallocates memory previously allocated by [`alloc`](Self::alloc).
    /// If the pointer is null, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to deallocate (can be null)
    /// * `layout` - The layout used when allocating this memory
    ///
    /// # Safety
    ///
    /// If `ptr` is non-null, it must have been allocated with this allocator using
    /// the same layout. The memory must not be accessed after deallocation.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            unsafe { alloc::dealloc(ptr, layout) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allocator() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let layout = Layout::from_size_align(1024, 64).unwrap();
        let ptr = allocator.alloc(layout)?;
        allocator.dealloc(ptr, layout);
        Ok(())
    }
}
