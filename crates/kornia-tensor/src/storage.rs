use std::{alloc::Layout, sync::Arc};

use crate::allocator::TensorAllocator;

/// Definition of the buffer for a tensor.
pub struct TensorStorage<T, A: TensorAllocator> {
    /// The shared pointer to the tensor memory.
    pub(crate) data: Arc<Vec<T>>,
    /// The layout of the tensor memory.
    pub(crate) layout: Layout,
    /// The allocator used to allocate/deallocate the tensor memory.
    pub(crate) alloc: A,
}

impl<T, A: TensorAllocator> TensorStorage<T, A> {
    /// Returns the pointer to the tensor memory.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns the pointer to the tensor memory.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        // This cast is safe because we ensure exclusive access through &mut self
        // We only use this for internal mutation within tensor operations
        Arc::as_ptr(&self.data) as *mut T
    }

    /// Returns the data pointer as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the data pointer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] 
    where 
        T: Clone
    {
        // If there are other references to the data, clone it first
        if Arc::strong_count(&self.data) > 1 {
            let cloned_data = self.data.to_vec();
            self.data = Arc::new(cloned_data);
        }
        
        // Now we can safely get a mutable reference to the unique data
        // This is safe because we know we're the only owner of the Arc
        unsafe {
            let vec_ptr = Arc::as_ptr(&self.data) as *mut Vec<T>;
            &mut (*vec_ptr)[..]
        }
    }

    /// Returns the number of bytes contained in this `TensorStorage`.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    /// Returns true if the `TensorStorage` has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the layout of the tensor buffer.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the allocator of the tensor buffer.
    #[inline]
    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    /// Creates a new tensor buffer from a vector.
    pub fn from_vec(value: Vec<T>, alloc: A) -> Self {
        let layout = unsafe { Layout::array::<T>(value.capacity()).unwrap_unchecked() };
        Self {
            data: Arc::new(value),
            layout,
            alloc,
        }
    }

    /// Creates a new tensor buffer from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the length must be valid.
    pub unsafe fn from_raw_parts(data: *const T, len: usize, alloc: A) -> Self 
    where 
        T: Clone
    {
        let slice = std::slice::from_raw_parts(data, len);
        let vec = slice.to_vec();
        let layout = Layout::from_size_align_unchecked(len * std::mem::size_of::<T>(), std::mem::align_of::<T>());
        Self {
            data: Arc::new(vec),
            layout,
            alloc,
        }
    }

    /// Converts the `TensorStorage` into a `Vec<T>`.
    ///
    /// If this is the only reference to the data, it will be returned directly.
    /// Otherwise, a clone of the data will be created.
    pub fn into_vec(self) -> Vec<T> 
    where 
        T: Clone
    {
        // If there are other references to the data, clone it and return the clone
        if Arc::strong_count(&self.data) > 1 {
            return self.data.to_vec();
        }
        
        // Try to unwrap the Arc, falling back to cloning if there are other references
        match Arc::try_unwrap(self.data.clone()) {
            Ok(vec) => vec,
            Err(arc) => arc.to_vec(),
        }
    }
}

// Safety:
// TensorStorage is thread safe if the allocator is thread safe.
unsafe impl<T, A: TensorAllocator> Send for TensorStorage<T, A> {}
unsafe impl<T, A: TensorAllocator> Sync for TensorStorage<T, A> {}

impl<T, A: TensorAllocator> Drop for TensorStorage<T, A> {
    fn drop(&mut self) {
        // Arc handles the memory deallocation now
        // We don't need to call the allocator anymore
    }
}

/// Clone implementation for TensorStorage that efficiently shares the underlying data.
impl<T, A> Clone for TensorStorage<T, A>
where
    A: TensorAllocator + 'static,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(), // This is just cloning the Arc pointer, not the data
            layout: self.layout,
            alloc: self.alloc.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TensorStorage;
    use crate::allocator::CpuAllocator;
    use crate::allocator::TensorAllocatorError;

    #[test]
    fn test_tensor_buffer_create_raw() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let vec = vec![0u8; size];
        let vec_ptr = vec.as_ptr();
        
        // Create buffer from vec
        let buffer = TensorStorage::from_vec(vec, allocator);
        
        assert_eq!(buffer.as_ptr(), vec_ptr);
        assert!(!vec_ptr.is_null());
        assert_eq!(buffer.len(), size * std::mem::size_of::<u8>());
        assert!(!buffer.is_empty());

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_ptr() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let vec = vec![0u8; size];
        
        // Create buffer from vec
        let buffer = TensorStorage::from_vec(vec, allocator);
        
        // check alignment
        let ptr_raw = buffer.as_ptr() as usize;
        let alignment = std::mem::align_of::<u8>();
        assert_eq!(ptr_raw % alignment, 0);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_create_f32() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let vec = vec![0.0f32; size];
        let vec_ptr = vec.as_ptr();
        
        // Create buffer from vec
        let buffer = TensorStorage::from_vec(vec, allocator);
        
        assert_eq!(buffer.as_ptr(), vec_ptr);
        assert_eq!(buffer.len(), size * std::mem::size_of::<f32>());

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_lifecycle() -> Result<(), TensorAllocatorError> {
        // Create a simple buffer
        let vec = vec![0u8; 1024];
        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        
        // Clone the buffer (this should just clone the Arc, not the data)
        let buffer2 = buffer.clone();
        
        // Verify both buffers point to the same memory
        assert_eq!(buffer.as_ptr(), buffer2.as_ptr());
        
        // Convert buffer to vec (should clone since buffer2 still has a reference)
        let vec1 = buffer.into_vec();
        assert_eq!(vec1.len(), 1024);
        
        // Convert buffer2 to vec (should be able to reclaim without cloning)
        let vec2 = buffer2.into_vec();
        assert_eq!(vec2.len(), 1024);
        
        Ok(())
    }

    #[test]
    fn test_tensor_buffer_from_vec() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let vec_ptr = vec.as_ptr();
        let vec_len = vec.len();

        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        // check NO copy 
        let buffer_ptr = buffer.as_ptr();
        assert!(std::ptr::eq(buffer_ptr, vec_ptr));

        // check alignment
        let buffer_ptr = buffer.as_ptr() as usize;
        let alignment = std::mem::align_of::<i32>();
        assert_eq!(buffer_ptr % alignment, 0);

        // check accessors
        let data = buffer.as_slice();
        assert_eq!(data.len(), vec_len);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 3);
        assert_eq!(data[3], 4);
        assert_eq!(data[4], 5);

        assert_eq!(data.first(), Some(&1));
        assert_eq!(data.get(1), Some(&2));
        assert_eq!(data.get(2), Some(&3));
        assert_eq!(data.get(3), Some(&4));
        assert_eq!(data.get(4), Some(&5));
        assert_eq!(data.get(5), None);

        unsafe {
            assert_eq!(data.get_unchecked(0), &1);
            assert_eq!(data.get_unchecked(1), &2);
            assert_eq!(data.get_unchecked(2), &3);
            assert_eq!(data.get_unchecked(3), &4);
            assert_eq!(data.get_unchecked(4), &5);
        }

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_into_vec() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        // convert back to vec
        let result_vec = buffer.into_vec();

        // The pointer may be different now since Arc::try_unwrap is used
        // and we can't guarantee the exact same memory location
        assert_eq!(result_vec.len(), 5);
        assert_eq!(result_vec[0], 1);
        assert_eq!(result_vec[1], 2);
        assert_eq!(result_vec[2], 3);
        assert_eq!(result_vec[3], 4);
        assert_eq!(result_vec[4], 5);

        Ok(())
    }

    #[test]
    fn test_tensor_mutability() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        
        // Modify the data through as_mut_slice
        {
            let slice = buffer.as_mut_slice();
            slice[0] = 10;
        }
        
        assert_eq!(buffer.as_slice()[0], 10);
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_buffer_clone() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        
        // Clone the buffer - should share the same data
        let buffer_clone = buffer.clone();
        
        // Check that they point to the same memory
        assert_eq!(buffer.as_ptr(), buffer_clone.as_ptr());
        
        // Check that cloned buffer has the correct data
        let data = buffer_clone.as_slice();
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 3);
        assert_eq!(data[3], 4);
        assert_eq!(data[4], 5);
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_buffer_clone_modify() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        
        // Clone the buffer - should share the same data
        let mut buffer_clone = buffer.clone();
        
        // Modify the cloned buffer - should copy-on-write
        {
            let slice = buffer_clone.as_mut_slice();
            slice[0] = 10;
        }
        
        // Original buffer should be unchanged
        assert_eq!(buffer.as_slice()[0], 1);
        
        // Cloned buffer should have the new value
        assert_eq!(buffer_clone.as_slice()[0], 10);
        
        // Pointers should now be different
        assert_ne!(buffer.as_ptr(), buffer_clone.as_ptr());
        
        Ok(())
    }
}
