use std::{alloc::Layout, ptr::NonNull};

use crate::allocator::TensorAllocator;

/// Low-level memory buffer for tensor data.
///
/// `TensorStorage` manages a contiguous block of memory that holds the actual data for a tensor.
/// It uses a custom allocator system to support different memory backends (CPU, GPU, etc.).
///
/// # Memory Management
///
/// The storage owns its memory and automatically deallocates it when dropped. The memory is
/// allocated using the provided allocator, which must implement the [`TensorAllocator`] trait.
///
/// # Thread Safety
///
/// `TensorStorage` is `Send` and `Sync` when the allocator is thread-safe, allowing tensors
/// to be safely shared across threads.
///
/// # Examples
///
/// Creating storage from a vector:
///
/// ```rust
/// use kornia_tensor::{storage::TensorStorage, CpuAllocator};
///
/// let data = vec![1, 2, 3, 4, 5];
/// let storage = TensorStorage::from_vec(data, CpuAllocator);
///
/// assert_eq!(storage.as_slice(), &[1, 2, 3, 4, 5]);
/// assert!(!storage.is_empty());
/// ```
///
/// Converting back to a vector:
///
/// ```rust
/// use kornia_tensor::{storage::TensorStorage, CpuAllocator};
///
/// let data = vec![1.0, 2.0, 3.0];
/// let storage = TensorStorage::from_vec(data, CpuAllocator);
/// let recovered = storage.into_vec();
///
/// assert_eq!(recovered, vec![1.0, 2.0, 3.0]);
/// ```
pub struct TensorStorage<T, A: TensorAllocator> {
    /// The pointer to the tensor memory which must be non null.
    pub(crate) ptr: NonNull<T>,
    /// The length of the tensor memory in bytes.
    pub(crate) len: usize,
    /// The layout of the tensor memory.
    pub(crate) layout: Layout,
    /// The allocator used to allocate/deallocate the tensor memory.
    pub(crate) alloc: A,
}

impl<T, A: TensorAllocator> TensorStorage<T, A> {
    /// Returns a raw pointer to the storage's memory.
    ///
    /// # Returns
    ///
    /// A const pointer to the first element of the storage.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the storage's memory.
    ///
    /// # Returns
    ///
    /// A mutable pointer to the first element of the storage.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the storage data as a slice.
    ///
    /// This provides safe, immutable access to the storage's underlying data.
    ///
    /// # Returns
    ///
    /// A slice containing all elements in the storage.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len / std::mem::size_of::<T>()) }
    }

    /// Returns the storage data as a mutable slice.
    ///
    /// This provides safe, mutable access to the storage's underlying data.
    ///
    /// # Returns
    ///
    /// A mutable slice containing all elements in the storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len / std::mem::size_of::<T>())
        }
    }

    /// Returns the number of bytes contained in this storage.
    ///
    /// Note: This returns the size in bytes, not the number of elements.
    /// To get the number of elements, divide by `std::mem::size_of::<T>()`.
    ///
    /// # Returns
    ///
    /// The total size in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the storage has a length of 0.
    ///
    /// # Returns
    ///
    /// `true` if the storage is empty, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the memory layout of the storage.
    ///
    /// The layout describes the size and alignment requirements of the allocated memory.
    ///
    /// # Returns
    ///
    /// The memory layout used for this storage.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns a reference to the allocator used by this storage.
    ///
    /// # Returns
    ///
    /// A reference to the allocator.
    #[inline]
    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    /// Creates a new tensor storage from a vector.
    ///
    /// This takes ownership of the vector and wraps it in a `TensorStorage` without copying
    /// the data. The memory is transferred to the storage's management.
    ///
    /// # Arguments
    ///
    /// * `value` - The vector to convert into storage
    /// * `alloc` - The allocator to associate with this storage
    ///
    /// # Returns
    ///
    /// A new `TensorStorage` instance wrapping the vector's memory.
    ///
    /// # Note
    ///
    /// Currently, the provided allocator is stored but not used for the initial allocation,
    /// as the vector was already allocated. The allocator will be used when the storage is dropped.
    pub fn from_vec(value: Vec<T>, alloc: A) -> Self {
        //let buf = arrow_buffer::Buffer::from_vec(value);
        // Safety
        // Vec::as_ptr guaranteed to not be null
        let ptr = unsafe { NonNull::new_unchecked(value.as_ptr() as _) };
        let len = value.len() * std::mem::size_of::<T>();
        // Safety
        // Vec guaranteed to have a valid layout matching that of `Layout::array`
        // This is based on `RawVec::current_memory`
        let layout = unsafe { Layout::array::<T>(value.capacity()).unwrap_unchecked() };
        std::mem::forget(value);

        Self {
            ptr,
            len,
            layout,
            alloc,
        }
    }

    /// Creates a new tensor storage from raw parts.
    ///
    /// # Arguments
    ///
    /// * `data` - A pointer to the memory buffer
    /// * `len` - The length of the buffer in number of elements (not bytes)
    /// * `alloc` - The allocator to use for deallocation
    ///
    /// # Returns
    ///
    /// A new `TensorStorage` instance managing the provided memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The pointer is non-null and properly aligned
    /// - The memory region is valid for `len` elements of type `T`
    /// - The memory was allocated in a way compatible with the provided allocator
    /// - No other code will free this memory (ownership is transferred)
    pub unsafe fn from_raw_parts(data: *const T, len: usize, alloc: A) -> Self {
        let ptr = NonNull::new_unchecked(data as _);
        let layout = Layout::from_size_align_unchecked(len, std::mem::size_of::<T>());
        Self {
            ptr,
            len,
            layout,
            alloc,
        }
    }

    /// Consumes the storage and returns the underlying data as a vector.
    ///
    /// This transfers ownership of the memory from the storage to a `Vec<T>` without copying.
    /// The storage is consumed in the process.
    ///
    /// # Returns
    ///
    /// A vector containing all the elements from the storage.
    ///
    /// # Note
    ///
    /// The returned vector will have the same capacity as the storage's allocated memory.
    pub fn into_vec(self) -> Vec<T> {
        // TODO: check if the buffer is a cpu buffer or comes from a custom allocator
        let _layout = &self.layout;

        let vec_capacity = self.layout.size() / std::mem::size_of::<T>();
        //match Layout::array::<T>(vec_capacity) {
        //    Ok(expected) if layout == &expected => {}
        //    e => return Err(TensorAllocatorError::LayoutError(e.unwrap_err())),
        //}

        let length = self.len;
        let ptr = self.ptr;
        let vec_len = length / std::mem::size_of::<T>();

        // Safety
        std::mem::forget(self);
        unsafe { Vec::from_raw_parts(ptr.as_ptr(), vec_len, vec_capacity) }
    }
}

// Safety:
// TensorStorage is thread-safe if the allocator is thread-safe.
// The storage owns its data exclusively, and the allocator trait requires thread-safety.
unsafe impl<T, A: TensorAllocator> Send for TensorStorage<T, A> {}
unsafe impl<T, A: TensorAllocator> Sync for TensorStorage<T, A> {}

impl<T, A: TensorAllocator> Drop for TensorStorage<T, A> {
    /// Automatically deallocates the storage's memory when dropped.
    ///
    /// This uses the storage's allocator to properly free the memory.
    fn drop(&mut self) {
        // Only deallocate if there is actual heap memory to free
        if self.layout.size() > 0 {
            self.alloc
                .dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Clones the storage by creating a new storage with copied data.
///
/// This performs a deep copy of the storage data using the cloned allocator.
impl<T, A> Clone for TensorStorage<T, A>
where
    T: Clone,
    A: TensorAllocator,
{
    /// Creates a new storage with a copy of this storage's data.
    ///
    /// # Returns
    ///
    /// A new `TensorStorage` instance with cloned data.
    fn clone(&self) -> Self {
        Self::from_vec(self.as_slice().to_vec(), self.alloc.clone())
    }
}

#[cfg(test)]
mod tests {

    use super::TensorStorage;
    use crate::allocator::{CpuAllocator, TensorAllocatorError};
    use crate::TensorAllocator;
    use std::alloc::Layout;
    use std::cell::RefCell;
    use std::ptr::NonNull;
    use std::rc::Rc;

    #[test]
    fn test_tensor_buffer_create_raw() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let layout = Layout::array::<u8>(size).map_err(TensorAllocatorError::LayoutError)?;
        let ptr =
            NonNull::new(allocator.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;
        let ptr_raw = ptr.as_ptr();

        let buffer = TensorStorage {
            alloc: allocator,
            len: size * std::mem::size_of::<u8>(),
            layout,
            ptr,
        };

        assert_eq!(buffer.ptr.as_ptr(), ptr_raw);
        assert!(!ptr_raw.is_null());
        assert_eq!(buffer.layout, layout);
        assert_eq!(buffer.len(), size);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.len(), size * std::mem::size_of::<u8>());

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_ptr() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let layout = Layout::array::<u8>(size).map_err(TensorAllocatorError::LayoutError)?;
        let ptr =
            NonNull::new(allocator.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;

        // check alignment
        let ptr_raw = ptr.as_ptr() as usize;
        let alignment = std::mem::align_of::<u8>();
        assert_eq!(ptr_raw % alignment, 0);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_create_f32() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let layout = Layout::array::<f32>(size).map_err(TensorAllocatorError::LayoutError)?;
        let ptr =
            NonNull::new(allocator.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;

        let buffer = TensorStorage {
            alloc: allocator,
            len: size,
            layout,
            ptr: ptr.cast::<f32>(),
        };

        assert_eq!(buffer.as_ptr(), ptr.as_ptr() as *const f32);
        assert_eq!(buffer.layout, layout);
        assert_eq!(buffer.len(), size);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_lifecycle() -> Result<(), TensorAllocatorError> {
        /// A simple allocator that counts the number of bytes allocated and deallocated.
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
        assert_eq!(*allocator.bytes_allocated.borrow(), 0);

        let size = 1024;

        // TensorStorage::from_vec() -> TensorStorage::into_vec()
        // TensorStorage::from_vec() currently does not use the custom allocator, so the
        // bytes_allocated value should not change.
        {
            let vec = Vec::<u8>::with_capacity(size);
            let vec_ptr = vec.as_ptr();
            let vec_capacity = vec.capacity();

            let buffer = TensorStorage::from_vec(vec, allocator.clone());
            assert_eq!(*allocator.bytes_allocated.borrow(), 0);

            let result_vec = buffer.into_vec();
            assert_eq!(*allocator.bytes_allocated.borrow(), 0);

            assert_eq!(result_vec.capacity(), vec_capacity);
            assert!(std::ptr::eq(result_vec.as_ptr(), vec_ptr));
        }
        assert_eq!(*allocator.bytes_allocated.borrow(), 0);

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
    fn test_tensor_buffer_from_empty_vec() -> Result<(), TensorAllocatorError> {
        // This test checks if an empty vector can be handled correct
        // There was in issue in the Drop function which tried to dealloc an empty vector
        // which does not allocate memory, which lead to a segfault when Drop was called.
        let vec: Vec<i32> = vec![];
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

        // At the end it is dropped and test should not crash with seg fault
        Ok(())
    }

    #[test]
    fn test_tensor_buffer_into_vec() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let vec_ptr = vec.as_ptr();
        let vec_cap = vec.capacity();

        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        // convert back to vec
        let result_vec = buffer.into_vec();

        // check NO copy
        assert_eq!(result_vec.capacity(), vec_cap);
        assert!(std::ptr::eq(result_vec.as_ptr(), vec_ptr));

        Ok(())
    }

    #[test]
    fn test_tensor_mutability() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        let ptr_mut = buffer.as_mut_ptr();
        unsafe {
            *ptr_mut.add(0) = 10;
        }
        assert_eq!(buffer.into_vec(), vec![10, 2, 3, 4, 5]);
        Ok(())
    }
}
