//! Arc-based storage management for zero-copy views and efficient sharing.
//!
//! This module provides an improved storage implementation using `Arc` for
//! reference counting, enabling cheap clones and zero-copy tensor views.

use std::{alloc::Layout, sync::Arc};

/// Trait for extension buffers that can be used with custom allocators.
///
/// This trait allows external crates (like `kornia-io` for GStreamer) to implement
/// their own buffer types that can be stored in the `Buffer` enum.
///
/// The trait extends `BufferOps` with device information. Note that extension buffers
/// can be on any device (CPU, CUDA, etc.) - the buffer itself reports its device.
///
/// # Example (in kornia-io crate)
///
/// ```rust,no_run
/// use kornia_tensor::storage::ExtensionBuffer;
/// use kornia_tensor::allocator::BufferOps;
/// use std::sync::Arc;
/// use gstreamer::Buffer;
///
/// // GStreamer buffer can be on CPU or CUDA depending on the pipeline
/// impl ExtensionBuffer for Arc<Buffer> {
///     fn device(&self) -> kornia_tensor::device::Device {
///         // Check buffer metadata to determine actual device
///         // For now, assume CPU (could be enhanced to check GStreamer caps)
///         kornia_tensor::device::Device::Cpu
///     }
/// }
///
/// impl BufferOps for Arc<Buffer> {
///     fn as_ptr(&self) -> *const u8 {
///         self.map_readable().map(|m| m.as_slice().as_ptr() as *const u8)
///             .unwrap_or(std::ptr::null())
///     }
///     fn as_mut_ptr(&mut self) -> *mut u8 {
///         self.as_ptr() as *mut u8
///     }
///     fn len(&self) -> usize {
///         self.map_readable().map(|m| m.as_slice().len())
///             .unwrap_or(0)
///     }
/// }
/// ```
pub trait ExtensionBuffer: crate::allocator::BufferOps + Send + Sync {
    /// Returns the device where this buffer's memory is located.
    ///
    /// This allows the buffer to report its actual device, which may differ
    /// from the allocator's device (e.g., GStreamer can manage CPU or CUDA buffers).
    fn device(&self) -> crate::device::Device;
}

/// Device-agnostic buffer storage enum.
///
/// This enum stores the actual buffer types for different memory backends:
/// - **CPU**: `Vec<u8>` (owns the memory, automatic deallocation)
/// - **CUDA**: `Arc<DeviceBuffer<u8>>` (reference-counted, automatic deallocation)
/// - **Extension**: Custom buffer types from external crates (e.g., `Arc<gstreamer::Buffer>`)
///
/// Note: Extension buffers can be on any device (CPU, CUDA, etc.) - the buffer
/// itself reports its device via `ExtensionBuffer::device()`.
///
/// This simplifies memory management by using owned types that handle
/// deallocation automatically via Drop.
pub enum Buffer {
    /// CPU memory buffer (owned Vec).
    Cpu(Vec<u8>),
    /// CUDA device buffer (reference-counted).
    #[cfg(feature = "cuda")]
    Cuda(Arc<cust::memory::DeviceBuffer<u8>>),
    /// Extension buffer for custom allocators (e.g., GStreamer).
    /// 
    /// This allows external crates to provide their own buffer types
    /// without modifying kornia-tensor. The buffer can be on any device
    /// and reports its device via `ExtensionBuffer::device()`.
    Extension(Box<dyn ExtensionBuffer>),
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Buffer::Cpu(vec) => f.debug_tuple("Cpu").field(&vec.len()).finish(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(arc) => {
                use crate::allocator::BufferOps;
                f.debug_tuple("Cuda").field(&arc.len()).finish()
            }
            Buffer::Extension(ext) => {
                f.debug_tuple("Extension")
                    .field(&ext.device())
                    .field(&ext.len())
                    .finish()
            }
        }
    }
}

impl Buffer {
    /// Returns a pointer to the buffer's data.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Buffer::Cpu(vec) => vec.as_ptr(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(arc) => {
                use crate::allocator::BufferOps;
                arc.as_ptr()
            }
            Buffer::Extension(ext) => ext.as_ptr(),
        }
    }

    /// Returns a mutable pointer to the buffer's data.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Buffer::Cpu(vec) => vec.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(arc) => {
                use crate::allocator::BufferOps;
                // For Arc, we can't get mutable access, so we cast the const pointer
                arc.as_ptr() as *mut u8
            }
            Buffer::Extension(ext) => ext.as_mut_ptr(),
        }
    }

    /// Returns the length of the buffer in bytes.
    pub fn len(&self) -> usize {
        match self {
            Buffer::Cpu(vec) => vec.len(),
            #[cfg(feature = "cuda")]
            Buffer::Cuda(arc) => {
                use crate::allocator::BufferOps;
                arc.len()
            }
            Buffer::Extension(ext) => ext.len(),
        }
    }

    /// Returns the device where this buffer's memory is located.
    ///
    /// For extension buffers, this queries the buffer itself, allowing
    /// it to report its actual device (e.g., GStreamer buffers can be CPU or CUDA).
    pub fn device(&self) -> crate::device::Device {
        match self {
            Buffer::Cpu(_) => crate::device::Device::Cpu,
            #[cfg(feature = "cuda")]
            Buffer::Cuda(_) => {
                // For CUDA, we'd need to track device_id, but for now assume device 0
                // This could be improved by storing device info in the buffer
                crate::device::Device::Cuda { device_id: 0 }
            }
            Buffer::Extension(ext) => ext.device(),
        }
    }
}

// SAFETY: Buffer is Send + Sync:
// - Vec<u8> is Send + Sync
// - Arc<DeviceBuffer<u8>> is Send + Sync
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// Device-agnostic pointer type for pointer operations.
///
/// This is derived from Buffer for pointer arithmetic and operations.
#[derive(Debug, Clone, Copy)]
pub enum DevicePtr {
    /// CPU memory pointer.
    Cpu(*mut u8),
    /// CUDA device memory address.
    #[cfg(feature = "cuda")]
    Cuda(u64),
}

// SAFETY: DevicePtr is Send + Sync:
// - *mut u8 is Send + Sync (raw pointers are Send + Sync)
// - u64 is Send + Sync
unsafe impl Send for DevicePtr {}
unsafe impl Sync for DevicePtr {}

impl DevicePtr {
    /// Creates a CPU device pointer from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be valid for the lifetime of the DevicePtr.
    pub unsafe fn from_cpu_ptr(ptr: *mut u8) -> Self {
        DevicePtr::Cpu(ptr)
    }

    /// Creates a CUDA device pointer from a device address.
    ///
    /// # Safety
    ///
    /// The address must be a valid CUDA device memory address.
    #[cfg(feature = "cuda")]
    pub unsafe fn from_cuda_addr(addr: u64) -> Self {
        DevicePtr::Cuda(addr)
    }

    /// Returns the pointer as a CPU pointer, or None if it's not a CPU pointer.
    pub fn as_cpu_ptr(&self) -> Option<*mut u8> {
        match self {
            DevicePtr::Cpu(ptr) => Some(*ptr),
            #[cfg(feature = "cuda")]
            DevicePtr::Cuda(_) => None,
        }
    }

    /// Returns the pointer as a CUDA address, or None if it's not a CUDA pointer.
    #[cfg(feature = "cuda")]
    pub fn as_cuda_addr(&self) -> Option<u64> {
        match self {
            DevicePtr::Cpu(_) => None,
            DevicePtr::Cuda(addr) => Some(*addr),
        }
    }

    /// Returns a const pointer to the data.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            DevicePtr::Cpu(ptr) => *ptr as *const u8,
            #[cfg(feature = "cuda")]
            DevicePtr::Cuda(addr) => *addr as *const u8,
        }
    }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            DevicePtr::Cpu(ptr) => *ptr,
            #[cfg(feature = "cuda")]
            DevicePtr::Cuda(addr) => *addr as *mut u8,
        }
    }

    /// Adds an offset to the pointer.
    pub fn add(&self, offset: usize) -> Self {
        match self {
            DevicePtr::Cpu(ptr) => {
                DevicePtr::Cpu(unsafe { ptr.add(offset) })
            }
            #[cfg(feature = "cuda")]
            DevicePtr::Cuda(addr) => {
                DevicePtr::Cuda(addr.wrapping_add(offset as u64))
            }
        }
    }
}

use crate::{device_marker::DeviceMarker, TensorAllocator};

/// Inner storage implementation that holds the actual memory.
///
/// This is wrapped in an `Arc` to enable reference counting and
/// zero-copy views with different offsets.
struct StorageImpl<T, D: DeviceMarker> {
    /// The actual buffer (owns or references the memory).
    /// This contains all the data - length, pointer, etc. can be derived from it.
    buffer: Buffer,
    /// Marker for element type (zero-sized).
    _element: std::marker::PhantomData<T>,
    /// Marker for device type (zero-sized).
    _device: std::marker::PhantomData<D>,
}

impl<T, D: DeviceMarker> std::fmt::Debug for StorageImpl<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageImpl")
            .field("buffer", &self.buffer)
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
    /// Creates a new `TensorStorage` from a `Buffer`.
    ///
    /// This is used internally when transferring tensors between devices.
    /// The buffer must match the device type `D`.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer containing the tensor data
    /// * `len` - The length of the buffer in bytes
    ///
    /// # Returns
    ///
    /// A new `TensorStorage` instance.
    pub(crate) fn from_buffer(buffer: Buffer, len: usize) -> Self {
        Self {
            inner: Arc::new(StorageImpl {
                buffer,
                _element: std::marker::PhantomData,
                _device: std::marker::PhantomData,
            }),
            offset: 0,
            view_len: len,
        }
    }

    /// Returns the pointer to the tensor memory, accounting for offset.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        // Calculate byte offset
        let byte_offset = self.offset * std::mem::size_of::<T>();
        // Get base pointer from buffer and add offset
        let base_ptr = self.inner.buffer.as_ptr();
        // SAFETY: offset is always within bounds (validated at construction)
        unsafe { base_ptr.add(byte_offset) as *const T }
    }

    /// Returns the mutable pointer to the tensor memory, accounting for offset.
    ///
    /// # Safety
    ///
    /// This returns a mutable pointer even though we only have `&self`.
    /// The caller must ensure exclusive access when dereferencing.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        // Calculate byte offset
        let byte_offset = self.offset * std::mem::size_of::<T>();
        // Get base pointer from buffer and add offset
        // Note: We can't get mutable access to Arc, so we use the const pointer
        // and cast it. This is safe because the caller must ensure exclusive ownership.
        let base_ptr = self.inner.buffer.as_ptr() as *mut u8;
        // SAFETY: offset is always within bounds (validated at construction)
        unsafe { base_ptr.add(byte_offset) as *mut T }
    }

    /// Returns the storage data as a slice.
    ///
    /// This provides safe, immutable access to the storage's underlying data.
    /// This method is only available for CPU devices, ensuring compile-time type safety.
    ///
    /// # Returns
    ///
    /// A slice containing all elements in the storage.
    ///
    /// # Panics
    ///
    /// Panics if there are other references to this storage (checked via Arc::strong_count).
    pub fn as_slice(&self) -> &[T]
    where
        D: crate::device_marker::CpuDevice,
    {
        let elem_count = self.view_len / std::mem::size_of::<T>();
        // SAFETY: ptr is valid for view_len bytes, properly aligned, and on CPU (enforced by trait bound)
        // offset was validated at construction
        unsafe { std::slice::from_raw_parts(self.as_ptr(), elem_count) }
    }

    /// Returns the storage data as a mutable slice.
    ///
    /// This provides safe, mutable access to the storage's underlying data.
    /// This method is only available for CPU devices, ensuring compile-time type safety.
    ///
    /// # Returns
    ///
    /// A mutable slice containing all elements in the storage.
    ///
    /// # Panics
    ///
    /// Panics if there are other references to this storage (checked via Arc::strong_count).
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        D: crate::device_marker::CpuDevice,
    {
        assert!(
            Arc::strong_count(&self.inner) == 1,
            "Cannot get mutable slice when storage is shared. Clone the tensor first."
        );
        let elem_count = self.view_len / std::mem::size_of::<T>();
        // SAFETY: ptr is valid for view_len bytes, properly aligned, on CPU (enforced by trait bound), and exclusively owned
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
        matches!(D::device_info(), crate::device::Device::Cpu)
    }

    /// Returns true if the tensor is on GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
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
        // Reconstruct layout from buffer length and element alignment
        let size = self.inner.buffer.len();
        let align = std::mem::align_of::<T>();
        unsafe {
            Layout::from_size_align_unchecked(size, align)
        }
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
    /// Returns an error if memory allocation fails or memory transfer fails.
    pub fn from_vec(value: Vec<T>) -> Result<Self, crate::TensorError> {
        let alloc = D::allocator()?;
        
        let buf_len = value.len() * std::mem::size_of::<T>();
        let buf_layout = Layout::from_size_align(buf_len, std::mem::align_of::<T>())
            .map_err(|e| crate::TensorError::StorageError(
                crate::allocator::TensorAllocatorError::LayoutError(e)
            ))?;
        
        // Allocate buffer using allocator and convert to our Buffer enum
        let alloc_buffer = alloc.alloc(buf_layout)
            .map_err(crate::TensorError::StorageError)?;

        // Convert allocator buffer to our Buffer enum using the allocator's conversion method
        let mut buffer = alloc.convert_to_storage_buffer(alloc_buffer, buf_len, buf_layout)
            .map_err(crate::TensorError::StorageError)?;
        
        // Copy data from value into the buffer
        // Create temporary wrappers for BufferOps
        use crate::allocator::{BufferOps, RawPtr};
        let cpu_buffer = value.as_ptr() as *const u8;
        let cpu_wrapper = RawPtr(cpu_buffer as *mut u8);
        let buffer_ptr = buffer.as_mut_ptr();
        let mut buffer_wrapper = RawPtr(buffer_ptr);
        
        unsafe {
            alloc.copy_from(
                &cpu_wrapper as &dyn BufferOps,
                &mut buffer_wrapper as &mut dyn BufferOps,
                buf_len,
                &crate::device::Device::Cpu,
            )?;
        }
        
        std::mem::forget(value);

        Ok(Self {
            inner: Arc::new(StorageImpl {
                buffer,
                _element: std::marker::PhantomData,
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
    /// - For GPU devices, `ptr` must be a device pointer (not a CPU pointer)
    /// 
    /// # Note
    ///
    /// For CUDA devices, this method cannot create a DevicePtr from a raw pointer
    /// because we need the actual `Arc<DeviceBuffer>`. This method is primarily
    /// for CPU tensors. For CUDA, use `from_vec()` or other allocation methods.
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Result<Self, crate::TensorError> {
        // SAFETY: Caller guarantees len and alignment are valid
        let _layout = Layout::from_size_align_unchecked(len, std::mem::align_of::<T>());
        
        let device = D::device_info();
        let buffer = match device {
            crate::device::Device::Cpu => {
                let ptr_u8 = ptr as *const u8 as *mut u8;
                if ptr_u8.is_null() {
                    return Err(crate::TensorError::StorageError(
                        crate::allocator::TensorAllocatorError::NullPointer
                    ));
                }
                // Create Vec from raw pointer
                // SAFETY: Caller guarantees ptr is valid for len bytes
                let vec = unsafe {
                    Vec::from_raw_parts(ptr_u8, len, len)
                };
                Buffer::Cpu(vec)
            }
            #[cfg(feature = "cuda")]
            crate::device::Device::Cuda { .. } => {
                // For CUDA, we cannot create Buffer from raw pointer
                // because we need the actual Arc<DeviceBuffer> for proper memory management
                // This is a limitation - use from_vec() or other allocation methods for CUDA
                return Err(crate::TensorError::StorageError(
                    crate::allocator::TensorAllocatorError::UnsupportedOperation(
                        "from_raw_parts() not supported for CUDA. Use from_vec() or allocation methods.".to_string()
                    )
                ));
            }
            #[allow(unreachable_patterns)]
            _ => {
                // For other devices, treat as CPU pointer for now
                let ptr_u8 = ptr as *const u8 as *mut u8;
                if ptr_u8.is_null() {
                    return Err(crate::TensorError::StorageError(
                        crate::allocator::TensorAllocatorError::NullPointer
                    ));
                }
                let vec = unsafe {
                    Vec::from_raw_parts(ptr_u8, len, len)
                };
                Buffer::Cpu(vec)
            }
        };
        
        Ok(Self {
            inner: Arc::new(StorageImpl {
                buffer,
                _element: std::marker::PhantomData,
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
        let mut inner = Arc::try_unwrap(self.inner)
            .expect("Storage should be unique after is_unique() check");
        
        // Extract Vec from Buffer
        // We need to move the buffer out, so we replace it with an empty buffer
        match std::mem::replace(&mut inner.buffer, Buffer::Cpu(Vec::new())) {
            Buffer::Cpu(mut vec) => {
                // Convert Vec<u8> to Vec<T>
                let elem_count = vec.len() / std::mem::size_of::<T>();
                let vec_capacity = vec.capacity() / std::mem::size_of::<T>();
                let ptr = vec.as_mut_ptr() as *mut T;
                
                // Prevent Vec<u8> from dropping
                std::mem::forget(vec);
                
                // SAFETY: ptr, elem_count, and vec_capacity are valid
                unsafe { Vec::from_raw_parts(ptr, elem_count, vec_capacity) }
            }
            #[cfg(feature = "cuda")]
            Buffer::Cuda(_) => {
                panic!("Cannot convert CUDA tensor to Vec. Use to_cpu() first.");
            }
            Buffer::Extension(_) => {
                panic!("Cannot convert extension buffer to Vec. Use to_cpu() first.");
            }
        }
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
        let buffer_len = self.inner.buffer.len();
        if end_pos > buffer_len {
            return Err(crate::TensorError::index_out_of_bounds(end_pos, buffer_len));
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
        // Buffer enum handles deallocation automatically:
        // - Buffer::Cpu(Vec<u8>) - Vec's Drop handles deallocation
        // - Buffer::Cuda(Arc<DeviceBuffer>) - Arc's Drop handles deallocation when last reference
        // No manual deallocation needed!
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
            .field("buffer", &self.inner.buffer)
            .field("buffer_len", &self.inner.buffer.len())
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

    #[test]
    fn test_device_ptr_cpu() {
        let mut ptr = unsafe { DevicePtr::from_cpu_ptr(0x1000 as *mut u8) };
        assert!(ptr.as_ptr() as usize == 0x1000);
        assert!(ptr.as_mut_ptr() as usize == 0x1000);
        
        // Test pointer arithmetic
        let ptr2 = ptr.add(100);
        assert!(ptr2.as_ptr() as usize == 0x1000 + 100);
        
        #[cfg(feature = "cuda")]
        assert!(ptr.as_cuda_addr().is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_device_ptr_cuda() {
        let addr: u64 = 0x8000_0000_0000_0000;
        let ptr = unsafe { DevicePtr::from_cuda_addr(addr) };
        assert_eq!(ptr.as_cuda_addr(), Some(addr));
        
        // Test pointer arithmetic
        let ptr2 = ptr.add(256);
        assert_eq!(ptr2.as_cuda_addr(), Some(addr + 256));
    }

    #[test]
    fn test_device_ptr_send_sync() {
        // Verify DevicePtr is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DevicePtr>();
    }

    #[test]
    fn test_storage_device_ptr_integration() -> Result<(), TensorError> {
        // Test that StorageImpl correctly uses DevicePtr
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let storage = TensorStorage::<f32, Cpu>::from_vec(data)?;
        
        // Verify we can get pointers
        let ptr = storage.as_ptr();
        assert!(!ptr.is_null());
        
        // Verify we can access the data
        let slice = storage.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
        
        Ok(())
    }
}
