use std::alloc::{GlobalAlloc, Layout, System};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorAllocatorError {
    #[error("Invalid tensor layout {0}")]
    LayoutError(core::alloc::LayoutError),

    #[error("Null pointer")]
    NullPointer,
}

/// A trait for allocating and deallocating memory for tensors.
///
/// # Safety
///
/// The tensor allocator must be thread-safe.
///
/// # Methods
///
/// * `alloc` - Allocates memory for a tensor with the given layout.
/// * `dealloc` - Deallocates memory for a tensor with the given layout.
pub trait TensorAllocator: Clone {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;
    fn dealloc(&self, ptr: *mut u8, layout: Layout);
}

#[derive(Clone)]
/// A tensor allocator that uses the system allocator.
pub struct CpuAllocator;

/// Implement the `Default` trait for the `CpuAllocator` struct.
impl Default for CpuAllocator {
    fn default() -> Self {
        Self
    }
}

/// Implement the `TensorAllocator` trait for the `CpuAllocator` struct.
impl TensorAllocator for CpuAllocator {
    /// Allocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The layout of the tensor.
    ///
    /// # Returns
    ///
    /// A non-null pointer to the allocated memory if successful, otherwise an error.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { System.alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }
        Ok(ptr)
    }

    /// Deallocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `ptr` - A non-null pointer to the allocated memory.
    /// * `layout` - The layout of the tensor.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the layout must be correct.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
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
