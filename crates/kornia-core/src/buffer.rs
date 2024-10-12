use std::{alloc::Layout, ptr::NonNull};

use crate::allocator::TensorAllocator;

/// Definition of the buffer for a tensor.
pub struct TensorBuffer<T, A: TensorAllocator> {
    /// The pointer to the tensor memory which must be non null.
    pub(crate) ptr: NonNull<T>,
    /// The length of the tensor memory in bytes.
    pub(crate) len: usize,
    /// The layout of the tensor memory.
    pub(crate) layout: Layout,
    /// The allocator used to allocate/deallocate the tensor memory.
    pub(crate) alloc: A,
}

// Safety:
// TensorBuffer is thread safe if the allocator is thread safe.
unsafe impl<T, A: TensorAllocator> Send for TensorBuffer<T, A> {}
unsafe impl<T, A: TensorAllocator> Sync for TensorBuffer<T, A> {}

impl<T, A: TensorAllocator> Drop for TensorBuffer<T, A> {
    fn drop(&mut self) {
        self.alloc
            .dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
    }
}

impl<T, A: TensorAllocator> Clone for TensorBuffer<T, A> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            layout: self.layout,
            alloc: self.alloc.clone(),
        }
    }
}

// TODO: pass the allocator to constructor
impl<T, A: TensorAllocator> From<Vec<T>> for TensorBuffer<T, A>
where
    A: Default,
{
    /// Creates a new tensor buffer from a vector.
    fn from(value: Vec<T>) -> Self {
        // Safety
        // Vec::as_ptr guaranteed to not be null
        let ptr = unsafe { NonNull::new_unchecked(value.as_ptr() as *mut T) };
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
            alloc: A::default(),
        }
    }
}

impl<T, A: TensorAllocator> TensorBuffer<T, A> {
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

    /// Returns the maximum number of elements that can be stored in this `TensorBuffer`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.len / std::mem::size_of::<T>()
    }

    /// Returns the number of bytes contained in this `TensorBuffer`.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `TensorBuffer` has a length of 0.
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
        // Safety
        // Vec::as_ptr guaranteed to not be null
        let ptr = unsafe { NonNull::new_unchecked(value.as_ptr() as *mut T) };
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

    /// Converts the `TensorBuffer` into a `Vec<T>`.
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

#[cfg(test)]
mod tests {

    use super::TensorBuffer;
    use crate::allocator::{CpuAllocator, TensorAllocatorError};
    use crate::TensorAllocator;
    use std::alloc::Layout;
    use std::cell::RefCell;
    use std::ptr::NonNull;
    use std::rc::Rc;

    #[test]
    fn test_tensor_buffer_create() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let layout = Layout::array::<u8>(size).map_err(TensorAllocatorError::LayoutError)?;
        let ptr =
            NonNull::new(allocator.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;

        let buffer = TensorBuffer {
            alloc: allocator,
            len: size * std::mem::size_of::<u8>(),
            layout,
            ptr,
        };

        assert_eq!(buffer.ptr.as_ptr(), ptr.as_ptr());
        assert_eq!(buffer.layout, layout);
        assert_eq!(buffer.capacity(), size);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_create_f32() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let allocator = CpuAllocator;
        let layout = Layout::array::<f32>(size).map_err(TensorAllocatorError::LayoutError)?;
        let ptr =
            NonNull::new(allocator.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;

        let buffer = TensorBuffer {
            alloc: allocator,
            len: size * std::mem::size_of::<f32>(),
            layout,
            ptr: ptr.cast::<f32>(),
        };

        assert_eq!(buffer.as_ptr(), ptr.as_ptr() as *const f32);
        assert_eq!(buffer.layout, layout);
        assert_eq!(buffer.capacity(), size);

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

        // Create a new buffer by allocating memory.
        // Deallocation should happen when `buffer` goes out of scope.
        {
            let layout = Layout::array::<u8>(size).map_err(TensorAllocatorError::LayoutError)?;
            let ptr =
                NonNull::new(allocator.alloc(layout)?).ok_or(TensorAllocatorError::NullPointer)?;

            let _buffer = TensorBuffer {
                alloc: allocator.clone(),
                len: size * std::mem::size_of::<u8>(),
                layout,
                ptr,
            };

            assert_eq!(*allocator.bytes_allocated.borrow(), size as i32);
        }
        assert_eq!(*allocator.bytes_allocated.borrow(), 0);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_from_vec() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let vec_ptr = vec.as_ptr();
        let vec_len = vec.len();
        let vec_cap = vec.capacity();
        let buffer = TensorBuffer::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        assert!(!buffer.is_empty());
        assert_eq!(buffer.as_ptr(), vec_ptr);
        assert_eq!(buffer.len(), vec_len * std::mem::size_of::<i32>());
        assert_eq!(buffer.layout, Layout::array::<i32>(vec_cap).unwrap());

        let vec2 = buffer.into_vec();
        assert_eq!(vec2.as_slice(), &[1, 2, 3, 4, 5]);
        assert_eq!(vec2.capacity(), vec_cap);
        assert!(std::ptr::eq(vec2.as_ptr(), vec_ptr));

        Ok(())
    }

    #[test]
    fn test_tensor_mutability() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut buffer = TensorBuffer::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        let ptr_mut = buffer.as_mut_ptr();
        unsafe {
            *ptr_mut.add(0) = 10;
        }
        assert_eq!(buffer.into_vec(), vec![10, 2, 3, 4, 5]);
        Ok(())
    }
}
