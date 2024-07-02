use super::allocator::{TensorAllocator, TensorAllocatorError};
use core::ptr;
use std::{alloc::Layout, ptr::NonNull};

/// A contiguous block of memory for a tensor.
///
/// # Safety
///
/// The tensor storage must be properly aligned and have the correct size.
///
/// # Fields
///
/// * `ptr` - A pointer to the first element of the tensor storage.
/// * `len` - The number of elements in the tensor storage.
/// * `alloc` - The allocator used to allocate and deallocate the tensor storage.
pub struct TensorStorage<T, A: TensorAllocator> {
    ptr: NonNull<T>,
    len: usize,
    alloc: A,
}

/// Implement the `TensorStorage` struct.
impl<T, A: TensorAllocator> TensorStorage<T, A> {
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
        let ptr = alloc.alloc(
            Layout::array::<T>(len).map_err(|err| TensorAllocatorError::LayoutError(err))?,
        )?;
        Ok(Self {
            len,
            ptr: ptr.cast(),
            alloc,
        })
    }

    /// Creates a new tensor storage from a vector with the given allocator.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector to copy to the tensor storage.
    /// * `alloc` - The allocator used to allocate the tensor storage.
    ///
    /// # Returns
    ///
    /// A new tensor storage if successful, otherwise an error.
    ///
    /// # Safety
    ///
    /// The vector must have the correct length and alignment.
    pub fn from_vec(mut vec: Vec<T>, alloc: A) -> Result<Self, TensorAllocatorError> {
        let len = vec.len();
        let ptr = NonNull::new(vec.as_mut_ptr()).ok_or(TensorAllocatorError::InvalidPointer)?;
        std::mem::forget(vec); // Prevent the vector from being deallocated.

        let storage = Self {
            len,
            ptr: ptr.cast(),
            alloc,
        };

        Ok(storage)
    }

    /// Creates a new tensor storage from a value with the given length and allocator.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the tensor storage.
    /// * `val` - The value to copy to the tensor storage.
    /// * `alloc` - The allocator used to allocate the tensor storage.
    ///
    /// # Returns
    ///
    /// A new tensor storage if successful, otherwise an error.
    pub fn from_val(len: usize, val: T, alloc: A) -> Result<Self, TensorAllocatorError>
    where
        T: Clone,
    {
        let storage = Self::new(len, alloc)?;
        for i in 0..len {
            unsafe {
                storage.ptr.as_ptr().add(i).write(val.clone());
            }
        }
        Ok(storage)
    }

    /// Returns the number of elements in the tensor storage.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a pointer to the first element of the tensor storage.
    pub fn ptr(&self) -> ptr::NonNull<T> {
        self.ptr
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
        assert_eq!(storage.len(), 1024);
        Ok(())
    }

    #[test]
    fn test_tensor_storage_ptr() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u8, _>::new(1024, allocator)?;
        let ptr = storage.ptr();
        assert!(ptr.as_ptr() as usize != 0);
        Ok(())
    }

    #[test]
    fn test_tensor_storage_from_vec() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let vec = vec![0, 1, 2, 3, 4, 5];
        let storage = TensorStorage::from_vec(vec, allocator)?;
        assert_eq!(storage.len(), 6);
        Ok(())
    }

    #[test]
    fn test_tensor_storage_from_val() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::from_val(1024, 42, allocator)?;
        assert_eq!(storage.len(), 1024);
        Ok(())
    }
}
