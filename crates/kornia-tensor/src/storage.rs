use std::{alloc::Layout, ptr::NonNull};

use crate::allocator::TensorAllocator;

// MemoryDomain is now defined in `resource` and re-exported from there.
// This re-export keeps all `storage::MemoryDomain` use-sites working unchanged.
pub use crate::resource::MemoryDomain;

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
    /// Whether this storage owns its memory (false for foreign/numpy-backed buffers).
    pub(crate) owns_memory: bool,
    /// Where the backing memory lives (host or device).
    pub(crate) domain: MemoryDomain,
    /// The CUDA device id (0 for host or single-GPU).
    pub(crate) device_id: i32,
    /// Optional keep-alive guard; when Some, the Arc is dropped with this storage,
    /// ensuring the source object (e.g. a DLPack producer or numpy array) outlives
    /// this borrowed view.  The field is never read — it is held purely for its
    /// Drop side-effect (releasing the Arc decrements the refcount and may free
    /// the underlying buffer).
    #[allow(dead_code)]
    pub(crate) keepalive: Option<std::sync::Arc<dyn core::any::Any + Send + Sync>>,
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

    /// Returns the memory domain for this storage.
    #[inline]
    pub fn domain(&self) -> MemoryDomain {
        self.domain
    }

    /// Returns the device id for this storage (0 for host; CUDA device id for Device domain).
    #[inline]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Returns the storage data as a slice.
    ///
    /// # Panics
    ///
    /// Panics if the storage lives on the device.
    /// Use explicit host-device transfer APIs to access device data.
    pub fn as_slice(&self) -> &[T] {
        assert_eq!(
            self.domain,
            MemoryDomain::Host,
            "as_slice called on device storage — use to_host() first"
        );
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len / std::mem::size_of::<T>()) }
    }

    /// Returns the storage data as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the storage lives on the device.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert_eq!(
            self.domain,
            MemoryDomain::Host,
            "as_mut_slice called on device storage — use to_host() first"
        );
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
            owns_memory: true,
            domain: MemoryDomain::Host,
            device_id: 0,
            keepalive: None,
        }
    }

    /// Creates a new tensor storage from raw parts.
    ///
    /// # Arguments
    ///
    /// * `data` - A pointer to the memory buffer
    /// * `len` - The length of the buffer in bytes (not number of elements)
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
    /// - The memory region is valid for `len` bytes (i.e. `len / size_of::<T>()` elements)
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
            owns_memory: false,
            domain: MemoryDomain::Host,
            device_id: 0,
            keepalive: None,
        }
    }

    /// Creates a new tensor storage that owns a host (CPU) allocation produced by `alloc`.
    ///
    /// Unlike [`from_vec`](Self::from_vec) (which wraps a `Vec` and inherits its alignment),
    /// this constructor allows callers to allocate with a custom layout (e.g. 64-byte alignment)
    /// via `alloc` and then hand the ownership to `TensorStorage`.
    ///
    /// # Safety
    ///
    /// - `data` must be a valid, non-null host pointer for at least `len_bytes` bytes.
    /// - `layout` must exactly match the layout passed to `alloc.alloc` that produced `data`.
    /// - Ownership is transferred; the allocator's `dealloc(data, layout)` will be called on drop.
    pub unsafe fn from_raw_host(
        data: *mut T,
        len_bytes: usize,
        layout: Layout,
        alloc: A,
    ) -> Self {
        let ptr = NonNull::new_unchecked(data);
        Self {
            ptr,
            len: len_bytes,
            layout,
            alloc,
            owns_memory: true,
            domain: MemoryDomain::Host,
            device_id: 0,
            keepalive: None,
        }
    }

    /// Creates a new tensor storage from a raw device pointer returned by a GPU allocator.
    ///
    /// The resulting storage has [`MemoryDomain::Device`]; calling [`as_slice`](Self::as_slice)
    /// on it will panic. Use explicit transfer APIs to bring data to the host.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid device pointer for at least `len_bytes` bytes.
    /// - `layout` must match the allocation that produced `ptr`.
    /// - Ownership is transferred; the allocator's `dealloc` will be called on drop.
    pub unsafe fn from_raw_device(
        data: *mut T,
        len_bytes: usize,
        layout: Layout,
        alloc: A,
    ) -> Self {
        let ptr = NonNull::new_unchecked(data);
        Self {
            ptr,
            len: len_bytes,
            layout,
            alloc,
            owns_memory: true,
            domain: MemoryDomain::Device { id: 0 },
            device_id: 0,
            keepalive: None,
        }
    }

    /// Creates a borrowed storage view that keeps `keepalive` alive until this storage is dropped.
    ///
    /// The storage does NOT own the memory — `Drop` will NOT call `alloc.dealloc`.
    /// Instead it holds `keepalive` in an `Arc` so the source object lives at least as long
    /// as this storage.
    ///
    /// # Safety
    ///
    /// - `data` must point to a valid, non-null allocation of at least `len_bytes` bytes valid for `T`.
    /// - The memory must remain valid for the full lifetime of this storage (guaranteed by `keepalive`).
    /// - `domain` and `device_id` must correctly describe where `data` lives:
    ///   - `(Host, 0)` for CPU memory,
    ///   - `(Device, id)` for CUDA device `id`.
    /// - For `Device` domain: do NOT call `as_slice`/`as_mut_slice` (they will panic).
    pub unsafe fn from_borrowed(
        data: *const T,
        len_bytes: usize,
        alloc: A,
        domain: MemoryDomain,
        device_id: i32,
        keepalive: std::sync::Arc<dyn core::any::Any + Send + Sync>,
    ) -> Self {
        // SAFETY: caller guarantees data is non-null and valid for len_bytes
        let ptr = NonNull::new_unchecked(data as *mut T);
        let layout = Layout::from_size_align_unchecked(len_bytes, std::mem::align_of::<T>());
        Self {
            ptr,
            len: len_bytes,
            layout,
            alloc,
            owns_memory: false,
            domain,
            device_id,
            keepalive: Some(keepalive),
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
        assert!(
            self.owns_memory,
            "cannot convert foreign-memory-backed storage into Vec"
        );

        let vec_capacity = self.layout.size() / std::mem::size_of::<T>();
        let vec_len = self.len / std::mem::size_of::<T>();
        let ptr = self.ptr;

        // Safety
        std::mem::forget(self);
        unsafe { Vec::from_raw_parts(ptr.as_ptr(), vec_len, vec_capacity) }
    }
}

// Safety:
// TensorStorage is thread-safe if the allocator is thread-safe.
// The storage owns its data exclusively, and the allocator trait requires thread-safety.
// The `keepalive` field is `Option<Arc<dyn Any+Send+Sync>>`, which is itself Send+Sync,
// so adding it does not weaken the overall thread-safety guarantees.
unsafe impl<T, A: TensorAllocator> Send for TensorStorage<T, A> {}
unsafe impl<T, A: TensorAllocator> Sync for TensorStorage<T, A> {}

impl<T, A: TensorAllocator> Drop for TensorStorage<T, A> {
    /// Automatically deallocates the storage's memory when dropped.
    ///
    /// This uses the storage's allocator to properly free the memory.
    fn drop(&mut self) {
        // Only deallocate if there is actual heap memory to free and we own it
        if self.owns_memory && self.layout.size() > 0 {
            self.alloc
                .dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Clones the storage by creating a new storage with copied data.
///
/// This performs a deep copy of the storage data using the cloned allocator.
///
/// # Panics
///
/// Panics if the storage lives on the device. Device-to-device copy is not yet
/// implemented; use an explicit transfer API when it becomes available.
impl<T, A> Clone for TensorStorage<T, A>
where
    T: Clone,
    A: TensorAllocator,
{
    fn clone(&self) -> Self {
        assert_eq!(
            self.domain,
            MemoryDomain::Host,
            "clone called on device storage — device-to-device copy is not yet implemented"
        );
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
            owns_memory: true,
            domain: super::MemoryDomain::Host,
            device_id: 0,
            keepalive: None,
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
            owns_memory: true,
            domain: super::MemoryDomain::Host,
            device_id: 0,
            keepalive: None,
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

    fn make_device_storage() -> TensorStorage<u8, CpuAllocator> {
        // Construct a Device-domain storage without actually touching GPU hardware.
        // The pointer value is irrelevant; we only test the domain guard.
        let layout = Layout::array::<u8>(1).unwrap();
        let ptr = CpuAllocator.alloc(layout).unwrap();
        unsafe { TensorStorage::from_raw_device(ptr, 1, layout, CpuAllocator) }
    }

    #[test]
    #[should_panic(expected = "as_slice called on device storage")]
    fn test_as_slice_panics_on_device() {
        let storage = make_device_storage();
        let _ = storage.as_slice();
    }

    #[test]
    #[should_panic(expected = "as_mut_slice called on device storage")]
    fn test_as_mut_slice_panics_on_device() {
        let mut storage = make_device_storage();
        let _ = storage.as_mut_slice();
    }

    #[test]
    #[should_panic(expected = "clone called on device storage")]
    fn test_clone_panics_on_device() {
        let storage = make_device_storage();
        let _ = storage.clone();
    }
}
