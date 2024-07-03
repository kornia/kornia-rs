use super::allocator::{TensorAllocator, TensorAllocatorError};
use arrow_buffer::{ArrowNativeType, Buffer};
use std::marker::PhantomData;
use std::sync::Arc;
use std::{alloc::Layout, ptr::NonNull};

/// represents a contiguous memory region that can be shared with other buffers and across thread boundaries.
///
/// NOTE: https://docs.rs/arrow/latest/arrow/buffer/struct.Buffer.html
///
/// # Safety
///
/// The tensor storage must be properly aligned and have the correct size.
///
/// # Fields
///
/// * `data` - The buffer containing the tensor storage.
/// * `alloc` - The allocator used to allocate the tensor storage.
/// * `marker` - The marker type for the tensor storage.
pub struct TensorStorage<T: ArrowNativeType, A: TensorAllocator> {
    pub data: Buffer,
    alloc: A,
    marker: PhantomData<T>,
}

/// Implement the `TensorStorage` struct.
impl<T, A: TensorAllocator> TensorStorage<T, A>
where
    T: ArrowNativeType + std::panic::RefUnwindSafe,
{
    /// Creates a new tensor storage with the given length and allocator.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the tensor storage.
    /// * `alloc` - The allocator used to allocate the tensor storage.
    ///
    /// # Returns
    ///
    /// A new tensor storage if successful, otherwise an error.
    pub fn new(len: usize, alloc: A) -> Result<Self, TensorAllocatorError> {
        // allocate memory for tensor storage
        let ptr =
            alloc.alloc(Layout::array::<T>(len).map_err(TensorAllocatorError::LayoutError)?)?;

        // create the buffer
        let buffer = unsafe {
            Buffer::from_custom_allocation(
                NonNull::new_unchecked(ptr),
                len * std::mem::size_of::<T>(),
                Arc::new(Vec::<T>::with_capacity(len)),
            )
        };

        Ok(Self {
            data: buffer,
            alloc,
            marker: PhantomData,
        })
    }

    /// Creates a new tensor storage from a vector with the given allocator without copying the data.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector to copy to the tensor storage.
    /// * `alloc` - The allocator used to allocate the tensor storage.
    ///
    /// # Safety
    ///
    /// The vector must have the correct length and alignment.
    pub fn from_vec(vec: Vec<T>, alloc: A) -> Result<Self, TensorAllocatorError> {
        // create immutable buffer from vec
        let buffer = unsafe {
            Buffer::from_custom_allocation(
                NonNull::new_unchecked(vec.as_ptr() as *mut u8),
                vec.len() * std::mem::size_of::<T>(),
                Arc::new(vec),
            )
        };

        // create tensor storage
        let storage = Self {
            data: buffer,
            alloc,
            marker: PhantomData,
        };

        Ok(storage)
    }

    /// Returns the allocator used to allocate the tensor storage.
    pub fn alloc(&self) -> &A {
        &self.alloc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::allocator::CpuAllocator;

    #[test]
    fn test_tensor_storage() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u8, _>::new(1024, allocator)?;
        assert_eq!(storage.data.len(), 1024);
        Ok(())
    }

    #[test]
    fn test_tensor_storage_ptr() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u8, _>::new(1024, allocator)?;
        let ptr = storage.data.as_ptr();
        assert!(!ptr.is_null());
        Ok(())
    }

    #[test]
    fn test_tensor_storage_from_vec() -> Result<(), TensorAllocatorError> {
        type CpuStorage = TensorStorage<u8, CpuAllocator>;
        let allocator = CpuAllocator;
        let vec = vec![0, 1, 2, 3, 4, 5];
        let storage = CpuStorage::from_vec(vec, allocator)?;
        assert_eq!(storage.data.len(), 6);
        Ok(())
    }
}
