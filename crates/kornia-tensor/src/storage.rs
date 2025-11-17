//! Arc-based storage management for zero-copy views and efficient sharing.
//!
//! This module provides an improved storage implementation using `Arc` for
//! reference counting, enabling cheap clones and zero-copy tensor views.

use std::{alloc::Layout, ptr::NonNull, sync::Arc};

use crate::{device_marker::DeviceMarker, TensorAllocator};

/// Inner storage implementation that holds the actual memory.
///
/// This is wrapped in an `Arc` to enable reference counting and
/// zero-copy views with different offsets.
struct StorageImpl<T, D: DeviceMarker> {
    /// The pointer to the tensor memory which must be non-null.
    ptr: NonNull<T>,
    /// The total length of allocated memory in bytes.
    len: usize,
    /// The memory layout used for allocation.
    layout: Layout,
    /// Marker for device type (zero-sized).
    _device: std::marker::PhantomData<D>,
}

impl<T, D: DeviceMarker> std::fmt::Debug for StorageImpl<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageImpl")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("layout", &self.layout)
            .finish()
    }
}

/// Arc-based tensor storage enabling zero-copy views and efficient sharing.
///
/// This storage type uses `Arc` internally, making clones cheap (just
/// incrementing a reference count) and enabling multiple tensors to share
/// the same underlying memory with different views.
///
/// # Thread Safety
///
/// `TensorStorage` is `Send + Sync` when `T: Send + Sync`, allowing tensors
/// to be safely shared across threads. The Arc ensures proper synchronization
/// of the reference count.
///
/// # Memory Management
///
/// Memory is allocated using the device-specific allocator and automatically
/// freed when the last reference is dropped.
pub struct TensorStorage<T, D: DeviceMarker> {
    /// Reference-counted inner storage.
    inner: Arc<StorageImpl<T, D>>,
    /// Offset into the storage for views (in number of elements).
    ///
    /// This allows creating zero-copy views that point to a subset of the
    /// underlying storage.
    offset: usize,
    /// Number of elements accessible from this view.
    ///
    /// May be less than `inner.len` for views.
    view_len: usize,
}

impl<T, D: DeviceMarker> TensorStorage<T, D> {
    /// Returns the pointer to the tensor memory, accounting for offset.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        // SAFETY: offset is always within bounds (validated at construction)
        unsafe { self.inner.ptr.as_ptr().add(self.offset) }
    }

    /// Returns the mutable pointer to the tensor memory, accounting for offset.
    ///
    /// # Safety
    ///
    /// This returns a mutable pointer even though we only have `&self`.
    /// The caller must ensure exclusive access when dereferencing.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        // SAFETY: offset is always within bounds (validated at construction)
        unsafe { self.inner.ptr.as_ptr().add(self.offset) }
    }

    /// Returns the storage data as a slice.
    ///
    /// This provides safe, immutable access to the storage's underlying data.
    ///
    /// # Returns
    ///
    /// A slice containing all elements in the storage.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not on CPU.
    pub fn as_slice(&self) -> &[T] {
        assert!(
            self.is_cpu(),
            "Cannot access GPU tensor as slice. Use to_cpu() to transfer data first."
        );
        let elem_count = self.view_len / std::mem::size_of::<T>();
        // SAFETY: ptr is valid for view_len bytes, properly aligned, and on CPU (checked above)
        // offset was validated at construction
        unsafe { std::slice::from_raw_parts(self.as_ptr(), elem_count) }
    }

    /// Returns the storage data as a mutable slice.
    ///
    /// This provides safe, mutable access to the storage's underlying data.
    ///
    /// # Returns
    ///
    /// A mutable slice containing all elements in the storage.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not on CPU or if there are other references to this storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            self.is_cpu(),
            "Cannot access GPU tensor as mutable slice. Use to_cpu() to transfer data first."
        );
        assert!(
            Arc::strong_count(&self.inner) == 1,
            "Cannot get mutable slice when storage is shared. Clone the tensor first."
        );
        let elem_count = self.view_len / std::mem::size_of::<T>();
        // SAFETY: ptr is valid for view_len bytes, properly aligned, on CPU, and exclusively owned
        // offset was validated at construction
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), elem_count) }
    }

    /// Returns the device where the tensor data is allocated.
    #[inline]
    pub fn device(&self) -> crate::device::Device {
        D::device_info()
    }

    /// Returns true if the tensor is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        D::device_info().is_cpu()
    }

    /// Returns true if the tensor is on GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        D::device_info().is_gpu()
    }

    /// Returns the number of bytes accessible from this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.view_len
    }

    /// Returns true if this view has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.view_len == 0
    }

    /// Returns the memory layout of the underlying storage.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.inner.layout
    }

    /// Returns true if this storage is uniquely owned (no other Arc references).
    ///
    /// This is useful for determining if mutation is safe without cloning.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Returns the current offset into the underlying storage (in elements).
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Creates a new tensor buffer from a vector.
    ///
    /// # Errors
    ///
    /// Returns an error if memory allocation fails.
    pub fn from_vec(value: Vec<T>) -> Result<Self, crate::TensorError> {
        let alloc = D::allocator()?;
        
        let buf_len = value.len() * std::mem::size_of::<T>();
        let buf_layout = Layout::from_size_align(buf_len, std::mem::align_of::<T>())
            .map_err(|e| crate::TensorError::StorageError(
                crate::allocator::TensorAllocatorError::LayoutError(e)
            ))?;
        let raw_ptr = alloc.alloc(buf_layout)
            .map_err(crate::TensorError::StorageError)?;
        let buf_ptr = raw_ptr as *mut T;
        
        let buf_ptr = NonNull::new(buf_ptr)
            .ok_or(crate::TensorError::StorageError(
                crate::allocator::TensorAllocatorError::NullPointer
            ))?;

        // SAFETY: buf_ptr is valid (just allocated), value.as_ptr() is valid, regions don't overlap
        unsafe {
            std::ptr::copy_nonoverlapping(value.as_ptr(), buf_ptr.as_ptr(), value.len());
        }
        
        std::mem::forget(value);

        Ok(Self {
            inner: Arc::new(StorageImpl {
                ptr: buf_ptr,
                len: buf_len,
                layout: buf_layout,
                _device: std::marker::PhantomData,
            }),
            offset: 0,
            view_len: buf_len,
        })
    }

    /// Creates a new tensor buffer from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is valid for reads of `len` bytes
    /// - `ptr` is properly aligned for type `T`
    /// - The memory at `ptr` was allocated by an allocator compatible with device `D`
    /// - The memory will not be accessed after this storage is dropped
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Result<Self, crate::TensorError> {
        // SAFETY: Caller guarantees len and alignment are valid
        let layout = Layout::from_size_align_unchecked(len, std::mem::align_of::<T>());
        let ptr = NonNull::new(ptr as *mut T)
            .ok_or(crate::TensorError::StorageError(
                crate::allocator::TensorAllocatorError::NullPointer
            ))?;
        Ok(Self {
            inner: Arc::new(StorageImpl {
                ptr,
                len,
                layout,
                _device: std::marker::PhantomData,
            }),
            offset: 0,
            view_len: len,
        })
    }

    /// Consumes the storage and returns the data as a vector.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not on CPU or if the storage is shared.
    pub fn into_vec(self) -> Vec<T> {
        assert!(self.is_cpu(), "Cannot convert GPU tensor to Vec. Use to_cpu() first.");
        assert!(
            self.is_unique() && self.offset == 0,
            "Cannot convert shared or offset storage to Vec. Clone first."
        );
        
        // Try to unwrap the Arc - this should succeed since we checked is_unique()
        let inner = Arc::try_unwrap(self.inner)
            .expect("Storage should be unique after is_unique() check");
        
        let vec_capacity = inner.layout.size() / std::mem::size_of::<T>();
        let vec_len = inner.len / std::mem::size_of::<T>();
        let ptr = inner.ptr;

        // SAFETY: Prevent double-free by forgetting inner
        std::mem::forget(inner);
        
        // SAFETY: ptr, vec_len, and vec_capacity are valid, and we've prevented double-free
        unsafe { Vec::from_raw_parts(ptr.as_ptr(), vec_len, vec_capacity) }
    }

    /// Creates a new view into this storage with the specified offset and length.
    ///
    /// This is a zero-copy operation that creates a new `TensorStorage` pointing
    /// to a subset of the underlying memory.
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset in elements (not bytes)
    /// * `len` - Length in elements (not bytes)
    ///
    /// # Errors
    ///
    /// Returns an error if the offset + len exceeds the storage bounds.
    pub fn view(&self, offset: usize, len: usize) -> Result<Self, crate::TensorError> {
        let byte_offset = offset * std::mem::size_of::<T>();
        let byte_len = len * std::mem::size_of::<T>();
        
        let end_pos = self.offset + byte_offset + byte_len;
        if end_pos > self.inner.len {
            return Err(crate::TensorError::index_out_of_bounds(end_pos, self.inner.len));
        }

        Ok(Self {
            inner: Arc::clone(&self.inner),
            offset: self.offset + offset,
            view_len: byte_len,
        })
    }
}

// SAFETY: TensorStorage can be sent between threads because:
// - Arc<StorageImpl> is Send when T: Send
// - D: DeviceMarker implies Send + Sync
// - offset and view_len are primitive types (Send + Sync)
unsafe impl<T: Send, D: DeviceMarker> Send for TensorStorage<T, D> {}

// SAFETY: TensorStorage can be shared between threads because:
// - Arc provides synchronized access to the inner storage
// - Access to data requires &self (immutable) or &mut self (exclusive via as_mut_slice)
// - D: DeviceMarker implies Send + Sync
// - T: Sync is required by the impl bound
unsafe impl<T: Sync, D: DeviceMarker> Sync for TensorStorage<T, D> {}

impl<T, D: DeviceMarker> Drop for StorageImpl<T, D> {
    fn drop(&mut self) {
        // SAFETY: ptr and layout were created together during allocation
        // This is the final drop of StorageImpl, so no other Arc references exist
        if let Ok(alloc) = D::allocator() {
            alloc.dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

impl<T, D: DeviceMarker> Clone for TensorStorage<T, D> {
    /// Creates a cheap clone by incrementing the Arc reference count.
    ///
    /// This is a O(1) operation that doesn't copy the underlying data.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            offset: self.offset,
            view_len: self.view_len,
        }
    }
}

impl<T, D: DeviceMarker> std::fmt::Debug for TensorStorage<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorStorage")
            .field("ptr", &self.inner.ptr)
            .field("len", &self.inner.len)
            .field("offset", &self.offset)
            .field("view_len", &self.view_len)
            .field("device", &self.device())
            .field("is_unique", &self.is_unique())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_marker::Cpu;
    use crate::TensorError;

    #[test]
    fn test_tensor_buffer_create_f32() -> Result<(), TensorError> {
        let data = vec![0.0_f32; 10];
        let buffer = TensorStorage::<f32, Cpu>::from_vec(data)?;
        assert_eq!(buffer.len(), 10 * std::mem::size_of::<f32>());
        assert!(!buffer.is_empty());
        assert!(buffer.is_cpu());
        assert!(!buffer.is_gpu());
        Ok(())
    }

    #[test]
    fn test_tensor_buffer_from_vec() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let buffer = TensorStorage::<i32, Cpu>::from_vec(data)?;
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn test_tensor_buffer_into_vec() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let buffer = TensorStorage::<i32, Cpu>::from_vec(data)?;
        let vec = buffer.into_vec();
        assert_eq!(vec, vec![1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn test_tensor_buffer_lifecycle() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4];
        let buffer = TensorStorage::<i32, Cpu>::from_vec(data)?;
        assert_eq!(buffer.len(), 4 * std::mem::size_of::<i32>());
        drop(buffer); // Explicitly drop
        Ok(())
    }

    #[test]
    fn test_tensor_buffer_ptr() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4];
        let buffer = TensorStorage::<i32, Cpu>::from_vec(data)?;
        let ptr = buffer.as_ptr();
        assert!(!ptr.is_null());
        Ok(())
    }

    #[test]
    fn test_tensor_mutability() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4];
        let mut buffer = TensorStorage::<i32, Cpu>::from_vec(data)?;
        {
            let slice = buffer.as_mut_slice();
            slice[0] = 10;
        }
        assert_eq!(buffer.as_slice()[0], 10);
        Ok(())
    }

    #[test]
    fn test_arc_storage_create() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let storage = TensorStorage::<i32, Cpu>::from_vec(data)?;
        assert_eq!(storage.as_slice(), &[1, 2, 3, 4, 5]);
        assert!(storage.is_unique());
        Ok(())
    }

    #[test]
    fn test_arc_storage_cheap_clone() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let storage1 = TensorStorage::<i32, Cpu>::from_vec(data)?;
        
        // Clone should be cheap (just Arc increment)
        let storage2 = storage1.clone();
        
        assert_eq!(storage1.as_slice(), &[1, 2, 3, 4, 5]);
        assert_eq!(storage2.as_slice(), &[1, 2, 3, 4, 5]);
        assert!(!storage1.is_unique());
        assert!(!storage2.is_unique());
        
        Ok(())
    }

    #[test]
    fn test_arc_storage_view() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let storage = TensorStorage::<i32, Cpu>::from_vec(data)?;
        
        // Create a view of elements [1, 2, 3] (indices 1-3)
        let view = storage.view(1, 3)?;
        assert_eq!(view.as_slice(), &[2, 3, 4]);
        assert_eq!(view.offset(), 1);
        
        Ok(())
    }

    #[test]
    fn test_arc_storage_shared_mutation_panics() {
        let data = vec![1, 2, 3, 4, 5];
        let mut storage1 = TensorStorage::<i32, Cpu>::from_vec(data).unwrap();
        let _storage2 = storage1.clone();
        
        // Should panic because storage is shared
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = storage1.as_mut_slice();
        }));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_arc_storage_unique_mutation() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let mut storage = TensorStorage::<i32, Cpu>::from_vec(data)?;
        
        assert!(storage.is_unique());
        
        // Should work because storage is unique
        {
            let slice = storage.as_mut_slice();
            slice[0] = 10;
        }
        
        assert_eq!(storage.as_slice()[0], 10);
        Ok(())
    }
}
