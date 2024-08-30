use super::allocator::{TensorAllocator, TensorAllocatorError};
use arrow_buffer::{Buffer, ScalarBuffer};
use std::sync::Arc;
use std::{alloc::Layout, ptr::NonNull};

/// A trait to define the types that can be used in a tensor.
pub trait SafeTensorType: arrow_buffer::ArrowNativeType + std::panic::RefUnwindSafe {}

/// Implement the `SafeTensorType` trait for the supported types.
impl SafeTensorType for u8 {}
impl SafeTensorType for u16 {}
impl SafeTensorType for u32 {}
impl SafeTensorType for u64 {}
impl SafeTensorType for i8 {}
impl SafeTensorType for i16 {}
impl SafeTensorType for i32 {}
impl SafeTensorType for i64 {}
impl SafeTensorType for f32 {}
impl SafeTensorType for f64 {}

/// represents a contiguous memory region that can be shared with other buffers and across thread boundaries.
///
/// NOTE: https://docs.rs/arrow-buffer/latest/arrow_buffer/buffer/struct.ScalarBuffer.html
///
/// # Safety
///
/// The tensor storage must be properly aligned and have the correct size.
///
/// # Fields
///
/// * `data` - The buffer containing the tensor storage.
/// * `alloc` - The allocator used to allocate the tensor storage.
pub struct TensorStorage<T, A: TensorAllocator>
where
    T: SafeTensorType,
{
    /// The buffer containing the tensor storage.
    data: ScalarBuffer<T>,
    alloc: A,
}

/// Implement the `TensorStorage` struct.
impl<T, A: TensorAllocator> TensorStorage<T, A>
where
    T: SafeTensorType,
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
            data: buffer.into(),
            alloc,
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
    pub fn from_vec(vec: Vec<T>, alloc: A) -> Self {
        // create immutable buffer from vec
        let buffer = unsafe {
            Buffer::from_custom_allocation(
                NonNull::new_unchecked(vec.as_ptr() as *mut u8),
                vec.len() * std::mem::size_of::<T>(),
                Arc::new(vec),
            )
        };

        // create tensor storage
        Self {
            data: buffer.into(),
            alloc,
        }
    }

    /// Returns the allocator used to allocate the tensor storage.
    #[inline]
    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    /// Returns the length of the tensor storage.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the tensor storage is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the data pointer
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    // TODO: remove this method once we don't need it anymore because of
    // ndarray::ArrayViewMut in kornia-imgproc
    /// Returns the data pointer as a mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }

    /// Return the data pointer as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len()) }
    }

    /// Return the data pointer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr() as *mut T, self.len()) }
    }

    /// Returns the data reference from the tensor storage checking the bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Returns the data reference from the tensor storage without checking the bounds.
    pub fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.data.get_unchecked(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::CpuAllocator;

    #[test]
    fn test_tensor_storage() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u8, _>::new(1024, allocator)?;
        let ptr = storage.as_ptr();
        assert_eq!(storage.len(), 1024);
        assert!(!storage.is_empty());
        assert!(!ptr.is_null());
        Ok(())
    }

    #[test]
    fn test_tensor_storage_ptr() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u64, _>::new(1024, allocator)?;

        // check alignment
        let ptr = storage.data.as_ptr() as usize;
        let alignment = std::mem::align_of::<u64>();
        assert_eq!(ptr % alignment, 0);
        Ok(())
    }

    #[test]
    fn test_tensor_storage_from_vec() -> Result<(), TensorAllocatorError> {
        type CpuStorage = TensorStorage<u8, CpuAllocator>;
        let allocator = CpuAllocator;

        let vec = vec![0, 1, 2, 3, 4, 5];
        let vec_ptr = vec.as_ptr();

        let storage = CpuStorage::from_vec(vec, allocator);
        assert_eq!(storage.len(), 6);

        // check NO copy
        let storage_data_ptr = storage.as_ptr();
        assert!(std::ptr::eq(storage_data_ptr, vec_ptr));

        // check alignment
        let storage_data_ptr = storage_data_ptr as usize;
        let alignment = std::mem::align_of::<u8>();
        assert_eq!(storage_data_ptr % alignment, 0);

        // check accessors
        let data = storage.as_slice();
        assert_eq!(data.len(), 6);
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 1);
        assert_eq!(data[2], 2);
        assert_eq!(data[3], 3);
        assert_eq!(data[4], 4);
        assert_eq!(data[5], 5);

        assert_eq!(storage.get(0), Some(&0));
        assert_eq!(storage.get(1), Some(&1));
        assert_eq!(storage.get(2), Some(&2));
        assert_eq!(storage.get(3), Some(&3));
        assert_eq!(storage.get(4), Some(&4));
        assert_eq!(storage.get(5), Some(&5));
        assert_eq!(storage.get(6), None);

        assert_eq!(storage.get_unchecked(0), &0);
        assert_eq!(storage.get_unchecked(1), &1);
        assert_eq!(storage.get_unchecked(2), &2);
        assert_eq!(storage.get_unchecked(3), &3);
        assert_eq!(storage.get_unchecked(4), &4);
        assert_eq!(storage.get_unchecked(5), &5);
        // TODO: fix this test
        // assert!(std::panic::catch_unwind(|| storage.get_unchecked(6)).is_err());

        Ok(())
    }
}
