use crate::allocator::{TensorAllocator, TensorAllocatorError};
use crate::buffer::TensorStorage;
use std::{alloc::Layout, ptr::NonNull};

/// Represents a contiguous memory region that can be shared with other buffers and across thread boundaries.
///
/// This struct provides methods to create, access, and manage tensor storage using a custom allocator.
///
/// # Safety
///
/// The tensor storage must be properly aligned and have the correct size.
pub struct TensorStorage<T, A: TensorAllocator> {
    /// The buffer containing the tensor storage.
    buffer: TensorStorage<T, A>,
}

impl<T, A> TensorStorage<T, A>
where
    A: TensorAllocator + 'static,
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
    pub fn new(size: usize, alloc: A) -> Result<Self, TensorAllocatorError> {
        // allocate memory for tensor storage
        let layout = Layout::array::<T>(size).map_err(TensorAllocatorError::LayoutError)?;
        let ptr = NonNull::new(alloc.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;

        // lenght of the buffer in bytes
        let len = size * std::mem::size_of::<T>();

        // create the buffer
        let buffer = TensorStorage {
            ptr: ptr.cast(),
            len,
            layout,
            alloc,
        };

        Ok(Self { buffer })
    }

    /// Creates a new tensor storage from a vector with the given allocator without copying the data.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector to use for the tensor storage.
    ///
    /// # Safety
    ///
    /// The vector must have the correct length and alignment.
    pub fn from_vec(vec: Vec<T>, alloc: A) -> Self {
        let buffer = TensorStorage::from_vec(vec, alloc);
        Self { buffer }
    }

    /// Converts the tensor storage into a `Vec<T>`.
    ///
    /// NOTE: useful for safe zero copies.
    ///
    /// This method attempts to convert the internal buffer of the tensor storage into a `Vec<T>`.
    /// If the conversion fails (e.g., due to reference counting issues), it constructs a new `Vec<T>`
    /// by copying the data from the raw pointer.
    ///
    /// # Safety
    ///
    /// This method is safe to call, but it may involve unsafe operations internally when
    /// constructing a new Vec from raw parts if the initial conversion fails.
    ///
    /// # Performance
    ///
    /// In the best case, this operation is O(1) when the internal buffer can be directly converted.
    /// In the worst case, it's O(n) where n is the number of elements, as it may need to copy all data.
    pub fn into_vec(self) -> Vec<T> {
        self.buffer.into_vec()
    }

    /// Returns the capacity of the tensor storage.
    //#[inline]
    //pub fn capacity(&self) -> usize {
    //    self.buffer.len()
    //}

    /// Returns the length of the tensor storage.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns whether the tensor storage is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the data pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buffer.ptr.as_ptr()
    }

    /// Returns the data pointer as a mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buffer.ptr.as_ptr()
    }

    /// Returns the data pointer as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.capacity()) }
    }

    /// Returns the data pointer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity()) }
    }

    /// Returns a reference to the data at the specified index, if it is within bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    /// Returns a reference to the data at the specified index without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    pub fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.as_slice().get_unchecked(index) }
    }

    /// Returns the allocator.
    ///
    /// # Safety
    ///
    /// The allocator must be valid.
    #[inline]
    pub fn alloc(&self) -> &A {
        &self.buffer.alloc
    }

    /// Converts the tensor storage into a `Vec<T>`.
    // TODO: this should be into_vec() to avoid extra copy.
    pub fn to_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }
}

/// A new `TensorStorage` instance with cloned data if successful, otherwise an error.
impl<T, A> Clone for TensorStorage<T, A>
where
    T: Clone,
    A: TensorAllocator + 'static,
{
    fn clone(&self) -> Self {
        // TODO: use zero copy
        let mut new_storage =
            Self::new(self.len(), self.alloc().clone()).expect("Failed to clone TensorStorage");

        for (d, s) in new_storage.as_mut_slice().iter_mut().zip(self.as_slice()) {
            *d = s.clone();
        }
        new_storage
    }
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//    use crate::allocator::CpuAllocator;
//    use std::alloc::Layout;
//    use std::cell::RefCell;
//    use std::rc::Rc;
//
//    #[test]
//    fn test_tensor_storage() -> Result<(), TensorAllocatorError> {
//        let allocator = CpuAllocator;
//        let storage = TensorStorage::<u8, _>::new(1024, allocator)?;
//        let ptr = storage.as_ptr();
//        assert_eq!(storage.len(), 1024);
//        assert!(!storage.is_empty());
//        assert!(!ptr.is_null());
//        Ok(())
//    }
//
//    #[test]
//    fn test_tensor_storage_ptr() -> Result<(), TensorAllocatorError> {
//        let allocator = CpuAllocator;
//        let storage = TensorStorage::<u64, _>::new(1024, allocator)?;
//
//        // check alignment
//        let ptr = storage.as_ptr() as usize;
//        let alignment = std::mem::align_of::<u64>();
//        assert_eq!(ptr % alignment, 0);
//        Ok(())
//    }
//
//    #[test]
//    fn test_tensor_storage_from_vec() -> Result<(), TensorAllocatorError> {
//        type CpuStorage = TensorStorage<u8, CpuAllocator>;
//
//        let vec = vec![0, 1, 2, 3, 4, 5];
//        let vec_ptr = vec.as_ptr();
//
//        let storage = CpuStorage::from_vec(vec, CpuAllocator);
//        assert_eq!(storage.len(), 6);
//
//        // check NO copy
//        let storage_data_ptr = storage.as_ptr();
//        assert!(std::ptr::eq(storage_data_ptr, vec_ptr));
//
//        // check alignment
//        let storage_data_ptr = storage_data_ptr as usize;
//        let alignment = std::mem::align_of::<u8>();
//        assert_eq!(storage_data_ptr % alignment, 0);
//
//        // check accessors
//        let data = storage.as_slice();
//        assert_eq!(data.len(), 6);
//        assert_eq!(data[0], 0);
//        assert_eq!(data[1], 1);
//        assert_eq!(data[2], 2);
//        assert_eq!(data[3], 3);
//        assert_eq!(data[4], 4);
//        assert_eq!(data[5], 5);
//
//        assert_eq!(storage.get(0), Some(&0));
//        assert_eq!(storage.get(1), Some(&1));
//        assert_eq!(storage.get(2), Some(&2));
//        assert_eq!(storage.get(3), Some(&3));
//        assert_eq!(storage.get(4), Some(&4));
//        assert_eq!(storage.get(5), Some(&5));
//        assert_eq!(storage.get(6), None);
//
//        assert_eq!(storage.get_unchecked(0), &0);
//        assert_eq!(storage.get_unchecked(1), &1);
//        assert_eq!(storage.get_unchecked(2), &2);
//        assert_eq!(storage.get_unchecked(3), &3);
//        assert_eq!(storage.get_unchecked(4), &4);
//        assert_eq!(storage.get_unchecked(5), &5);
//        // TODO: fix this test
//        // assert!(std::panic::catch_unwind(|| storage.get_unchecked(6)).is_err());
//
//        Ok(())
//    }
//
//    #[test]
//    fn test_tensor_storage_into_vec() {
//        let original_vec = vec![1, 2, 3, 4, 5];
//        let original_vec_ptr = original_vec.as_ptr();
//        let original_vec_capacity = original_vec.capacity();
//
//        let storage = TensorStorage::from_vec(original_vec, CpuAllocator);
//
//        // Convert the storage back to a vector
//        let result_vec = storage.into_vec();
//
//        // check NO copy
//        assert_eq!(result_vec.capacity(), original_vec_capacity);
//        assert!(std::ptr::eq(result_vec.as_ptr(), original_vec_ptr));
//    }
//
//    #[test]
//    fn test_tensor_storage_allocator() -> Result<(), TensorAllocatorError> {
//        // A test TensorAllocator that keeps a count of the bytes that are allocated but not yet
//        // deallocated via the allocator.
//        #[derive(Clone)]
//        struct TestAllocator {
//            bytes_allocated: Rc<RefCell<i32>>,
//        }
//        impl TensorAllocator for TestAllocator {
//            fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
//                *self.bytes_allocated.borrow_mut() += layout.size() as i32;
//                CpuAllocator.alloc(layout)
//            }
//            fn dealloc(&self, ptr: *mut u8, layout: Layout) {
//                *self.bytes_allocated.borrow_mut() -= layout.size() as i32;
//                CpuAllocator.dealloc(ptr, layout)
//            }
//        }
//
//        let allocator = TestAllocator {
//            bytes_allocated: Rc::new(RefCell::new(0)),
//        };
//        let len = 1024;
//
//        // TensorStorage::new()
//        // Deallocation should happen when `storage` goes out of scope.
//        {
//            let _storage = TensorStorage::<u8, _>::new(len, allocator.clone())?;
//            assert_eq!(*allocator.bytes_allocated.borrow(), len as i32);
//        }
//        assert_eq!(*allocator.bytes_allocated.borrow(), 0);
//
//        // TensorStorage::new() -> TensorStorage::into_vec()
//        // TensorStorage::into_vec() consumes the storage and creates a copy (in this case).
//        // This should cause deallocation of the original memory.
//        {
//            let storage = TensorStorage::<u8, _>::new(len, allocator.clone())?;
//            assert_eq!(*allocator.bytes_allocated.borrow(), len as i32);
//
//            //let _vec = storage.into_vec();
//            let _vec = storage.to_vec();
//            assert_eq!(*allocator.bytes_allocated.borrow(), 0);
//        }
//        assert_eq!(*allocator.bytes_allocated.borrow(), 0);
//
//        // TensorStorage::from_vec()  -> TensorStorage::into_vec()
//        // TensorStorage::from_vec() currently does not use the custom allocator, so the
//        // bytes_allocated value should not change.
//        {
//            let original_vec = Vec::<u8>::with_capacity(len);
//            let original_vec_ptr = original_vec.as_ptr();
//            let original_vec_capacity = original_vec.capacity();
//
//            let storage = TensorStorage::from_vec(original_vec, allocator.clone());
//            assert_eq!(*allocator.bytes_allocated.borrow(), 0);
//
//            let result_vec = storage.into_vec();
//            assert_eq!(*allocator.bytes_allocated.borrow(), 0);
//
//            assert_eq!(result_vec.capacity(), original_vec_capacity);
//            assert!(std::ptr::eq(result_vec.as_ptr(), original_vec_ptr));
//        }
//        assert_eq!(*allocator.bytes_allocated.borrow(), 0);
//
//        Ok(())
//    }
//
//    #[test]
//    fn test_tensor_storage_as_slice() -> Result<(), TensorAllocatorError> {
//        let data = vec![1, 2, 3, 4, 5];
//        let storage = TensorStorage::from_vec(data, CpuAllocator);
//        let slice = storage.as_slice();
//        assert_eq!(slice.len(), 5);
//        assert_eq!(slice, &[1, 2, 3, 4, 5]);
//        Ok(())
//    }
//}
