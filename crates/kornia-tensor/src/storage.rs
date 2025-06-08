use std::{alloc::Layout, ptr::NonNull};

use crate::allocator::TensorAllocator;

/// Definition of the buffer for a tensor.
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
    /// Returns the pointer to the tensor memory.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the pointer to the tensor memory.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the data pointer as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len / std::mem::size_of::<T>()) }
    }

    /// Returns the data pointer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len / std::mem::size_of::<T>())
        }
    }

    /// Returns the number of bytes contained in this `TensorStorage`.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `TensorStorage` has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
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

    // TODO: use the allocator somehow
    /// Creates a new tensor buffer from a vector.
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

    /// Creates a new tensor buffer from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the length must be valid.
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

    /// Converts the `TensorStorage` into a `Vec<T>`.
    ///
    /// Returns `Err(self)` if the buffer does not have the same layout as the destination Vec.
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
// TensorStorage is thread safe if the allocator is thread safe.
unsafe impl<T, A: TensorAllocator> Send for TensorStorage<T, A> {}
unsafe impl<T, A: TensorAllocator> Sync for TensorStorage<T, A> {}

impl<T, A: TensorAllocator> Drop for TensorStorage<T, A> {
    fn drop(&mut self) {
        self.alloc
            .dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
    }
}
/// A new `TensorStorage` instance with cloned data if successful, otherwise an error.
impl<T, A> Clone for TensorStorage<T, A>
where
    T: Clone,
    A: TensorAllocator + 'static,
{
    fn clone(&self) -> Self {
        let mut new_vec = Vec::<T>::with_capacity(self.len());

        for i in self.as_slice() {
            new_vec.push(i.clone());
        }

        Self::from_vec(new_vec, self.alloc.clone())
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
