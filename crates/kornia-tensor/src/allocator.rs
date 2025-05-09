use crate::ParentDeallocator;
use std::alloc;
use std::alloc::Layout;
use std::sync::Arc;
use thiserror::Error;

/// An error type for tensor allocator operations.
#[derive(Debug, Error, PartialEq)]
pub enum TensorAllocatorError {
    /// An error occurred during memory allocation.
    #[error("Invalid tensor layout {0}")]
    LayoutError(core::alloc::LayoutError),

    /// An error occurred during memory allocation.
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
    /// Allocates memory for a tensor with the given layout.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;

    /// Deallocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `ptr` - A non-null pointer to the allocated memory.
    /// * `layout` - The layout of the tensor.
    fn dealloc(&self, ptr: *mut u8, layout: Layout);
}

#[derive(Clone)]
/// A tensor allocator that uses the system allocator.
pub struct CpuAllocator {
    parent: Option<Arc<dyn ParentDeallocator>>,
}

/// Implement the `Default` trait for the `CpuAllocator` struct.
impl Default for CpuAllocator {
    fn default() -> Self {
        Self { parent: None }
    }
}

impl CpuAllocator {
    /// Creates a new `CpuAllocator` with a parent relation. The parent relation
    /// allows you to store a slice that is a reference to parent. For this, the parent
    /// needs to live long enough. When using this function the `CpuAllocator` will not
    /// drop the child _(reference)_, but instead use [ParentDeallocator::dealloc].
    pub fn with_parent_relation(parent: Arc<dyn ParentDeallocator>) -> Self {
        Self {
            parent: Some(parent),
        }
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
        let ptr = unsafe { alloc::alloc(layout) };
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
    /// * `parent` - An optional parent deallocator.
    /// * `layout` - The layout of the tensor.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the layout must be correct.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(parent) = self.parent.as_ref() {
            parent.dealloc();
        } else if !ptr.is_null() {
            unsafe {
                alloc::dealloc(ptr, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allocator() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator::default();
        let layout = Layout::from_size_align(1024, 64).unwrap();
        let ptr = allocator.alloc(layout)?;
        allocator.dealloc(ptr, layout);
        Ok(())
    }
}
