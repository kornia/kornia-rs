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

/// Represents the owner of custom Arrow Buffer memory allocations.
///
/// This struct is used to facilitate the automatic deallocation of the memory it owns,
/// using the `Drop` trait.
pub struct TensorCustomAllocationOwner<A: TensorAllocator> {
    /// The allocator used to allocate the tensor storage.
    alloc: Arc<A>,
    /// The layout used for the allocation.
    layout: Layout,
    /// The pointer to the allocated memory
    ptr: NonNull<u8>,
}

// SAFETY: TensorCustomAllocationOwner is never modifed from multiple threads.
impl<A: TensorAllocator> std::panic::RefUnwindSafe for TensorCustomAllocationOwner<A> {}
unsafe impl<A: TensorAllocator> Sync for TensorCustomAllocationOwner<A> {}
unsafe impl<A: TensorAllocator> Send for TensorCustomAllocationOwner<A> {}

impl<A: TensorAllocator> Drop for TensorCustomAllocationOwner<A> {
    fn drop(&mut self) {
        self.alloc.dealloc(self.ptr.as_ptr(), self.layout);
    }
}

/// Represents a contiguous memory region that can be shared with other buffers and across thread boundaries.
///
/// This struct provides methods to create, access, and manage tensor storage using a custom allocator.
///
/// # Safety
///
/// The tensor storage must be properly aligned and have the correct size.
pub struct TensorStorage<T, A: TensorAllocator>
where
    T: SafeTensorType,
{
    /// The buffer containing the tensor storage.
    data: ScalarBuffer<T>,
    /// The allocator used to allocate the tensor storage.
    alloc: Arc<A>,
}

impl<T, A> TensorStorage<T, A>
where
    T: SafeTensorType + Clone,
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
    pub fn new(len: usize, alloc: A) -> Result<Self, TensorAllocatorError> {
        // allocate memory for tensor storage
        let layout = Layout::array::<T>(len).map_err(TensorAllocatorError::LayoutError)?;
        let ptr = NonNull::new(alloc.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;
        let alloc = Arc::new(alloc);
        let owner = TensorCustomAllocationOwner {
            alloc: alloc.clone(),
            layout,
            ptr,
        };

        // create the buffer
        let buffer = unsafe {
            // SAFETY: `ptr` is non-null and properly aligned, and `len` is the correct size.
            Buffer::from_custom_allocation(ptr, len * std::mem::size_of::<T>(), Arc::new(owner))
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
    /// * `vec` - The vector to use for the tensor storage.
    /// * `alloc` - The allocator used to allocate the tensor storage.
    ///
    /// # Safety
    ///
    /// The vector must have the correct length and alignment.
    pub fn from_vec(vec: Vec<T>, alloc: A) -> Self {
        // NOTE: this is a temporary solution until we have a custom allocator for the buffer
        // create immutable buffer from vec
        // let _buffer = unsafe {
        //     // SAFETY: `vec` is properly aligned and has the correct length.
        //     Buffer::from_custom_allocation(
        //         NonNull::new_unchecked(vec.as_ptr() as *mut u8),
        //         vec.len() * std::mem::size_of::<T>(),
        //         Arc::new(vec),
        //     )
        // };

        // create immutable buffer from vec
        // NOTE: this is a temporary solution until we have a custom allocator for the buffer
        let buffer = Buffer::from_vec(vec);

        // create tensor storage
        Self {
            data: buffer.into(),
            alloc: Arc::new(alloc),
        }
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
        match self.data.into_inner().into_vec() {
            Ok(vec) => vec,
            Err(buf) => unsafe {
                std::slice::from_raw_parts(
                    buf.as_ptr() as *const T,
                    buf.len() / std::mem::size_of::<T>(),
                )
                .to_vec()
            },
        }
    }

    /// Creates a new `TensorStorage` from a slice of data.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice containing the data to be stored.
    /// * `alloc` - The allocator to use for creating the storage.
    ///
    /// # Returns
    ///
    /// A new `TensorStorage` instance containing a copy of the input data.
    ///
    /// # Errors
    ///
    /// Returns a `TensorAllocatorError` if the allocation fails.
    pub fn from_slice(data: &[T], alloc: A) -> Self {
        let buffer = Buffer::from_slice_ref(data);
        Self {
            data: buffer.into(),
            alloc: Arc::new(alloc),
        }
    }

    /// Creates a new tensor storage from an existing raw pointer with the given allocator.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The existing raw pointer to the tensor data.
    /// * `len` - The number of elements in the tensor storage.
    /// * `alloc` - A reference to the allocator used to allocate the tensor storage.
    ///
    /// # Safety
    ///
    /// The pointer must be properly aligned and have the correct length.
    pub unsafe fn from_ptr(ptr: *mut T, len: usize, alloc: &A) -> Self {
        // create the buffer
        let buffer = Buffer::from_custom_allocation(
            NonNull::new_unchecked(ptr as *mut u8),
            len * std::mem::size_of::<T>(),
            Arc::new(()),
        );

        // create tensor storage
        Self {
            data: buffer.into(),
            alloc: Arc::new(alloc.clone()),
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

    /// Returns the data pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns the data pointer as a mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }

    /// Returns the data pointer as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len()) }
    }

    /// Returns the data pointer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr() as *mut T, self.len()) }
    }

    /// Returns a reference to the data at the specified index, if it is within bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Returns a reference to the data at the specified index without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    pub fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.data.get_unchecked(index) }
    }
}

/// A new `TensorStorage` instance with cloned data if successful, otherwise an error.
impl<T, A> Clone for TensorStorage<T, A>
where
    T: SafeTensorType + Clone,
    A: TensorAllocator + Clone + 'static,
{
    fn clone(&self) -> Self {
        let mut new_storage = Self::new(self.len(), (*self.alloc).clone())
            .expect("Failed to allocate memory for cloned TensorStorage");
        new_storage.as_mut_slice().clone_from_slice(self.as_slice());
        new_storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::CpuAllocator;
    use std::alloc::Layout;
    use std::cell::RefCell;
    use std::rc::Rc;

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

    #[test]
    fn test_tensor_storage_from_cpu_ptr() -> Result<(), TensorAllocatorError> {
        let len = 1024;
        let layout = Layout::array::<u8>(len).unwrap();
        let allocator = CpuAllocator;

        // Allocate CPU memory
        let ptr = allocator.alloc(layout)?;

        // Wrap the existing CPU pointer in a `TensorStorage`
        let storage = unsafe { TensorStorage::from_ptr(ptr, len, &allocator) };

        // Use the `TensorStorage` as needed
        assert_eq!(storage.len(), len);
        assert!(!storage.is_empty());

        // Deallocate CPU memory
        allocator.dealloc(ptr, layout);

        Ok(())
    }

    #[test]
    fn test_tensor_storage_into_vec() {
        let allocator = CpuAllocator;
        let original_vec = vec![1, 2, 3, 4, 5];
        let original_vec_ptr = original_vec.as_ptr();
        let original_vec_capacity = original_vec.capacity();

        let storage = TensorStorage::<i32, _>::from_vec(original_vec, allocator);

        // Convert the storage back to a vector
        let result_vec = storage.into_vec();

        // check NO copy
        assert_eq!(result_vec.capacity(), original_vec_capacity);
        assert!(std::ptr::eq(result_vec.as_ptr(), original_vec_ptr));
    }

    #[test]
    fn test_tensor_storage_allocator() -> Result<(), TensorAllocatorError> {
        // A test TensorAllocator that keeps a count of the bytes that are allocated but not yet
        // deallocated via the allocator.
        #[derive(Clone)]
        struct TestAllocator {
            bytes_allocated: Rc<RefCell<i32>>,
        }
        impl TensorAllocator for TestAllocator {
            fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
                *self.bytes_allocated.borrow_mut() += layout.size() as i32;
                CpuAllocator.alloc(layout)
            }
            fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                *self.bytes_allocated.borrow_mut() -= layout.size() as i32;
                CpuAllocator.dealloc(ptr, layout)
            }
        }

        let allocator = TestAllocator {
            bytes_allocated: Rc::new(RefCell::new(0)),
        };
        let len = 1024;

        // TensorStorage::new()
        // Deallocation should happen when `storage` goes out of scope.
        {
            let _storage = TensorStorage::<u8, _>::new(len, allocator.clone())?;
            assert_eq!(*allocator.bytes_allocated.borrow(), len as i32);
        }
        assert_eq!(*allocator.bytes_allocated.borrow(), 0);

        // TensorStorage::new() -> TensorStorage::into_vec()
        // TensorStorage::into_vec() consumes the storage and creates a copy (in this case).
        // This should cause deallocation of the original memory.
        {
            let storage = TensorStorage::<u8, _>::new(len, allocator.clone())?;
            assert_eq!(*allocator.bytes_allocated.borrow(), len as i32);

            let _vec = storage.into_vec();
            assert_eq!(*allocator.bytes_allocated.borrow(), 0);
        }
        assert_eq!(*allocator.bytes_allocated.borrow(), 0);

        // TensorStorage::from_vec()  -> TensorStorage::into_vec()
        // TensorStorage::from_vec() currently does not use the custom allocator, so the
        // bytes_allocated value should not change.
        {
            let original_vec = Vec::<u8>::with_capacity(len);
            let original_vec_ptr = original_vec.as_ptr();
            let original_vec_capacity = original_vec.capacity();

            let storage = TensorStorage::<u8, _>::from_vec(original_vec, allocator.clone());
            assert_eq!(*allocator.bytes_allocated.borrow(), 0);

            let result_vec = storage.into_vec();
            assert_eq!(*allocator.bytes_allocated.borrow(), 0);

            assert_eq!(result_vec.capacity(), original_vec_capacity);
            assert!(std::ptr::eq(result_vec.as_ptr(), original_vec_ptr));
        }
        assert_eq!(*allocator.bytes_allocated.borrow(), 0);

        // TensorStorage::from_ptr()
        // TensorStorage::from_ptr() does not take ownership of buffer. So the memory should not be
        // deallocated when the TensorStorage goes out of scope.
        // In this case, the memory will be deallocated when the vector goes out of scope.
        {
            let mut original_vec = Vec::<u8>::with_capacity(len);
            let original_ptr = original_vec.as_ptr();
            {
                let storage = unsafe {
                    TensorStorage::<u8, _>::from_ptr(original_vec.as_mut_ptr(), len, &allocator)
                };
                assert_eq!(*allocator.bytes_allocated.borrow(), 0);

                assert_eq!(storage.as_ptr(), original_ptr);
            }
            assert_eq!(*allocator.bytes_allocated.borrow(), 0);
        }

        Ok(())
    }
}
