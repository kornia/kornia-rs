use std::alloc;
use std::alloc::Layout;

use thiserror::Error;

use crate::device::Device;

/// An error type for tensor allocator operations.
///
/// This enum provides detailed error information for memory allocation,
/// deallocation, and transfer operations across different devices.
#[derive(Debug, Error, PartialEq)]
pub enum TensorAllocatorError {
    /// Memory layout is invalid for the requested allocation.
    ///
    /// This typically occurs when the size or alignment requirements
    /// exceed system limits or are internally inconsistent.
    ///
    /// # Possible Causes
    /// - Requested size exceeds `isize::MAX`
    /// - Alignment is not a power of two
    /// - Size is not a multiple of alignment
    #[error("Invalid memory layout: {0}. Check that size fits in isize::MAX and alignment is a power of 2.")]
    LayoutError(core::alloc::LayoutError),

    /// Memory allocation returned a null pointer.
    ///
    /// This indicates that the system was unable to allocate the requested memory,
    /// typically due to insufficient available memory.
    ///
    /// # Possible Causes
    /// - Out of memory (OOM)
    /// - Requested allocation size too large
    /// - Memory fragmentation
    ///
    /// # Recommended Actions
    /// - Reduce tensor size
    /// - Free unused tensors
    /// - Check system memory availability
    #[error("Memory allocation failed: received null pointer. System may be out of memory.")]
>>>>>>> c8c28ff (implement cuda backend)
    NullPointer,

    /// Attempted memory transfer between incompatible devices.
    ///
    /// This occurs when trying to copy data between devices that don't support
    /// direct memory transfers without going through the host.
    ///
    /// # Example
    /// Copying directly between two different CUDA devices without peer-to-peer support.
    #[error("Device mismatch: cannot transfer from {0} to {1}. Consider using host as intermediate.")]
    DeviceMismatch(String, String),

    /// Memory transfer operation failed.
    ///
    /// This can occur during host-to-device, device-to-host, or device-to-device
    /// memory transfers.
    ///
    /// # Possible Causes
    /// - Invalid memory addresses
    /// - Insufficient device memory
    /// - Device communication error
    #[error("Memory transfer failed: {0}")]
    MemoryTransferError(String),

    /// CUDA-specific error occurred.
    ///
    /// This wraps errors from CUDA driver or runtime API calls.
    ///
    /// # Common CUDA Errors
    /// - Out of memory
    /// - Invalid device
    /// - Invalid context
    /// - Launch failure
    ///
    /// # Debugging Tips
    /// - Check device memory availability with `nvidia-smi`
    /// - Verify device ID is valid
    /// - Ensure CUDA drivers are up to date
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Operation not supported on this device or configuration.
    ///
    /// This error indicates that the requested operation is not available
    /// for the current device or tensor configuration.
    ///
    /// # Examples
    /// - Attempting to use features not supported by hardware
    /// - Invalid operation for the current device type
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

impl TensorAllocatorError {
    /// Returns true if this error is recoverable by freeing memory.
    ///
    /// # Returns
    ///
    /// `true` if the error might be resolved by freeing memory and retrying.
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            TensorAllocatorError::NullPointer => true,
            #[cfg(feature = "cuda")]
            TensorAllocatorError::CudaError(s) if s.contains("out of memory") => true,
            _ => false,
        }
    }

    /// Returns true if this error indicates a programming error rather than a runtime issue.
    ///
    /// # Returns
    ///
    /// `true` if the error is likely due to incorrect API usage.
    pub fn is_programming_error(&self) -> bool {
        matches!(
            self,
            TensorAllocatorError::LayoutError(_)
                | TensorAllocatorError::DeviceMismatch(_, _)
                | TensorAllocatorError::UnsupportedOperation(_)
        )
    }

    /// Returns a user-friendly suggestion for resolving the error.
    pub fn suggestion(&self) -> &str {
        match self {
            TensorAllocatorError::LayoutError(_) => {
                "Verify tensor dimensions don't exceed system limits (size < isize::MAX)"
            }
            TensorAllocatorError::NullPointer => {
                "Free unused tensors or reduce tensor size. Check available memory with system tools."
            }
            TensorAllocatorError::DeviceMismatch(_, _) => {
                "Use intermediate host memory for transfers between incompatible devices"
            }
            TensorAllocatorError::MemoryTransferError(_) => {
                "Check that source and destination pointers are valid and devices are accessible"
            }
            #[cfg(feature = "cuda")]
            TensorAllocatorError::CudaError(e) if e.contains("out of memory") => {
                "Free GPU memory or reduce batch size. Check GPU memory with nvidia-smi"
            }
            #[cfg(feature = "cuda")]
            TensorAllocatorError::CudaError(_) => {
                "Check CUDA device status and ensure drivers are up to date"
            }
            TensorAllocatorError::UnsupportedOperation(_) => {
                "Check device capabilities and API documentation for supported operations"
            }
        }
    }
}

/// Trait for custom tensor memory allocators.
///
/// `TensorAllocator` enables supporting different memory backends (CPU, GPU, shared memory, etc.)
/// by abstracting the allocation and deallocation interface. Implementors can provide custom
/// memory management strategies while maintaining compatibility with the tensor library.
///
/// # Thread Safety
///
/// Implementations must be thread-safe (`Send` + `Sync`) as tensors can be shared across threads.
/// The allocator is typically wrapped in `Arc` or similar when shared between tensors.
///
/// # Lifecycle
///
/// - [`alloc`](Self::alloc): Called when creating new tensor storage
/// - [`dealloc`](Self::dealloc): Called when tensor storage is dropped
/// - [`device`](Self::device): Returns the device associated with this allocator
/// - [`copy_from`](Self::copy_from): Copies data from a source device to this device
///
/// # Examples
///
/// Using the default CPU allocator:
///
/// ```rust
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let allocator = CpuAllocator;
/// let tensor = Tensor::<f32, 2, _>::zeros([100, 100], allocator);
/// ```
///
/// Implementing a custom allocator:
///
/// ```rust
/// use std::alloc::Layout;
/// use kornia_tensor::{TensorAllocator, allocator::TensorAllocatorError, device::Device};
///
/// #[derive(Clone)]
/// struct AlignedAllocator {
///     alignment: usize,
/// }
///
/// impl TensorAllocator for AlignedAllocator {
///     fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
///         // Custom allocation with guaranteed alignment
///         let aligned_layout = Layout::from_size_align(layout.size(), self.alignment)
///             .map_err(TensorAllocatorError::LayoutError)?;
///         
///         let ptr = unsafe { std::alloc::alloc(aligned_layout) };
///         if ptr.is_null() {
///             return Err(TensorAllocatorError::NullPointer);
///         }
///         Ok(ptr)
///     }
///
///     fn dealloc(&self, ptr: *mut u8, layout: Layout) {
///         if !ptr.is_null() {
///             let aligned_layout = Layout::from_size_align(layout.size(), self.alignment)
///                 .unwrap();
///             unsafe { std::alloc::dealloc(ptr, aligned_layout) }
///         }
///     }
///
///     fn device(&self) -> Device {
///         Device::Cpu
///     }
///
///     fn copy_from(
///         &self,
///         src_ptr: *const u8,
///         dst_ptr: *mut u8,
///         len: usize,
///         src_device: &Device,
///     ) -> Result<(), TensorAllocatorError> {
///         if matches!(src_device, Device::Cpu) {
///             unsafe {
///                 std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
///             }
///             Ok(())
///         } else {
///             Err(TensorAllocatorError::MemoryTransferError(
///                 "Unsupported device".to_string()
///             ))
///         }
///     }
/// }
/// ```
pub trait TensorAllocator: Clone {
    /// Allocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The memory layout specifying size and alignment requirements
    ///
    /// # Returns
    ///
    /// A pointer to the allocated memory on success, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns [`TensorAllocatorError::NullPointer`] if allocation fails.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;

    /// Deallocates memory previously allocated by this allocator.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the memory to deallocate
    /// * `layout` - The layout used when allocating this memory
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by [`alloc`](Self::alloc) with the same layout.
    fn dealloc(&self, ptr: *mut u8, layout: Layout);

    /// Returns the device associated with this allocator.
    fn device(&self) -> Device;

    /// Copies data from source pointer to destination pointer.
    ///
    /// # Safety
    ///
    /// This function dereferences raw pointers. Callers must ensure:
    /// - `src_ptr` and `dst_ptr` are valid for reads/writes of `len` bytes
    /// - The memory regions must not overlap
    /// - Both pointers must be properly aligned
    ///
    /// # Arguments
    ///
    /// * `src_ptr` - Source pointer
    /// * `dst_ptr` - Destination pointer
    /// * `len` - Number of bytes to copy
    /// * `src_device` - Source device
    unsafe fn copy_from(
        &self,
        src_ptr: *const u8,
        dst_ptr: *mut u8,
        len: usize,
        src_device: &Device,
    ) -> Result<(), TensorAllocatorError>;
}

/// CPU memory allocator using the system allocator.
///
/// `CpuAllocator` is the default allocator for tensors, providing standard heap allocation
/// using Rust's global allocator. It's suitable for general-purpose CPU-based tensor operations.
///
/// # Examples
///
/// ```rust
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let tensor = Tensor::<f32, 2, _>::zeros([10, 10], CpuAllocator);
/// ```
#[derive(Clone)]
pub struct CpuAllocator;

/// Provides a default instance of [`CpuAllocator`].
impl Default for CpuAllocator {
    fn default() -> Self {
        Self
    }
}

/// Implements [`TensorAllocator`] using the Rust global allocator.
impl TensorAllocator for CpuAllocator {
    /// Allocates memory using the system allocator.
    ///
    /// This uses Rust's global allocator (typically the system's malloc/free) to allocate
    /// memory with the specified layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The memory layout specifying size and alignment
    ///
    /// # Returns
    ///
    /// A pointer to the allocated memory on success.
    ///
    /// # Errors
    ///
    /// Returns [`TensorAllocatorError::NullPointer`] if the allocation fails (typically
    /// due to insufficient memory).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::alloc::Layout;
    /// use kornia_tensor::{TensorAllocator, CpuAllocator};
    ///
    /// let allocator = CpuAllocator;
    /// let layout = Layout::from_size_align(1024, 8).unwrap();
    /// let ptr = allocator.alloc(layout).unwrap();
    /// allocator.dealloc(ptr, layout);
    /// ```
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }
        Ok(ptr)
    }

    /// Deallocates memory using the system allocator.
    ///
    /// This safely deallocates memory previously allocated by [`alloc`](Self::alloc).
    /// If the pointer is null, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to deallocate (can be null)
    /// * `layout` - The layout used when allocating this memory
    ///
    /// # Safety
    ///
    /// If `ptr` is non-null, it must have been allocated with this allocator using
    /// the same layout. The memory must not be accessed after deallocation.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            unsafe { alloc::dealloc(ptr, layout) }
        }
    }

    /// Returns the device associated with this allocator.
    fn device(&self) -> Device {
        Device::Cpu
    }

    /// Copies data from source pointer to destination pointer.
    unsafe fn copy_from(
        &self,
        src_ptr: *const u8,
        dst_ptr: *mut u8,
        len: usize,
        src_device: &Device,
    ) -> Result<(), TensorAllocatorError> {
        // For CPU, we support copy from CPU and any GPU device (download)
        #[cfg(feature = "cuda")]
        if let Device::Cuda { .. } = src_device {
            // CUDA to CPU copy using raw CUDA API
            use cust::sys::cuMemcpyDtoH_v2;
            
            let result = unsafe {
                cuMemcpyDtoH_v2(
                    dst_ptr as *mut std::ffi::c_void,
                    src_ptr as u64,
                    len,
                )
            };
            
            if result != cust::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(TensorAllocatorError::CudaError(format!(
                    "DtoH copy failed with error code: {:?}",
                    result
                )));
            }
            
            return Ok(());
        }

        match src_device {
            Device::Cpu => {
                // CPU to CPU copy using memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
                }
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => Err(TensorAllocatorError::MemoryTransferError(format!(
                "Unsupported copy from {:?} to CPU",
                src_device
            ))),
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA allocator for GPU tensors.
#[derive(Clone)]
pub struct CudaAllocator {
    device_id: usize,
    _context: std::sync::Arc<cust::context::Context>,
}

#[cfg(feature = "cuda")]
impl CudaAllocator {
    /// Creates a new CUDA allocator for the specified device.
    pub fn new(device_id: usize) -> Result<Self, TensorAllocatorError> {
        use once_cell::sync::Lazy;
        use std::sync::Mutex;
        
        // Global CUDA context cache
        static CUDA_CONTEXTS: Lazy<Mutex<std::collections::HashMap<usize, std::sync::Arc<cust::context::Context>>>> = 
            Lazy::new(|| Mutex::new(std::collections::HashMap::new()));
        
        let mut contexts = CUDA_CONTEXTS.lock().unwrap();
        
        let context = if let Some(ctx) = contexts.get(&device_id) {
            ctx.clone()
        } else {
            // Initialize CUDA if not already done
            let _ = cust::init(cust::CudaFlags::empty());
            
            // Get device
            let device = cust::device::Device::get_device(device_id as u32)
                .map_err(|e| TensorAllocatorError::CudaError(format!("Failed to get device: {:?}", e)))?;
            
            // Create a new context for this device
            let ctx = cust::context::Context::new(device)
                .map_err(|e| TensorAllocatorError::CudaError(format!("Failed to create context: {:?}", e)))?;
            
            let ctx_arc = std::sync::Arc::new(ctx);
            contexts.insert(device_id, ctx_arc.clone());
            ctx_arc
        };
        
        Ok(Self {
            device_id,
            _context: context,
        })
    }

    /// Returns the device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Set this context as current on the calling thread
    fn set_current(&self) -> Result<(), TensorAllocatorError> {
        use cust::context::CurrentContext;
        CurrentContext::set_current(self._context.as_ref())
            .map_err(|e| TensorAllocatorError::CudaError(format!("Failed to set context: {:?}", e)))
    }
}

#[cfg(feature = "cuda")]
impl Default for CudaAllocator {
    fn default() -> Self {
        Self::new(0).expect("Failed to create default CUDA allocator")
    }
}

#[cfg(feature = "cuda")]
impl TensorAllocator for CudaAllocator {
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        use cust::memory::DeviceBuffer;
        
        // Set context as current before allocation
        self.set_current()?;
        
        // Allocate device memory
        let buffer = DeviceBuffer::<u8>::zeroed(layout.size())
            .map_err(|e| TensorAllocatorError::CudaError(format!("CUDA malloc failed: {:?}", e)))?;
        
        // Get raw pointer and leak the buffer (we'll manage memory manually)
        let ptr = buffer.as_device_ptr().as_raw() as *mut u8;
        std::mem::forget(buffer);
        
        Ok(ptr)
    }

    fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        if !ptr.is_null() {
            // Set context as current before deallocation
            if self.set_current().is_ok() {
                use cust::memory::DevicePointer;
                // Convert back to DevicePointer and drop it
                let _dev_ptr = DevicePointer::<u8>::from_raw(ptr as u64);
                // dev_ptr will be dropped automatically, freeing the memory
            }
        }
    }

    fn device(&self) -> Device {
        Device::Cuda {
            device_id: self.device_id,
        }
    }

    unsafe fn copy_from(
        &self,
        src_ptr: *const u8,
        dst_ptr: *mut u8,
        len: usize,
        src_device: &Device,
    ) -> Result<(), TensorAllocatorError> {
        // Set context as current before copy operations
        self.set_current()?;
        
        match src_device {
            Device::Cpu => {
                // CPU to CUDA copy (upload) using raw CUDA API
                use cust::sys::cuMemcpyHtoD_v2;
                
                let result = unsafe {
                    cuMemcpyHtoD_v2(
                        dst_ptr as u64,
                        src_ptr as *const std::ffi::c_void,
                        len,
                    )
                };
                
                if result != cust::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(TensorAllocatorError::CudaError(format!(
                        "HtoD copy failed with error code: {:?}",
                        result
                    )));
                }
                
                Ok(())
            }
            Device::Cuda { device_id } => {
                if *device_id != self.device_id {
                    return Err(TensorAllocatorError::DeviceMismatch(
                        format!("cuda:{}", device_id),
                        format!("cuda:{}", self.device_id),
                    ));
                }
                // CUDA to CUDA copy (same device) - use raw memcpy
                use cust::sys::cuMemcpy;
                
                let result = unsafe {
                    cuMemcpy(
                        dst_ptr as u64,
                        src_ptr as u64,
                        len,
                    )
                };
                
                if result != cust::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(TensorAllocatorError::CudaError(format!(
                        "DtoD copy failed with error code: {:?}",
                        result
                    )));
                }
                
                Ok(())
            }
            #[cfg(feature = "metal")]
            Device::Metal { .. } => Err(TensorAllocatorError::MemoryTransferError(
                "Metal to CUDA copy not supported".to_string(),
            )),
            #[cfg(feature = "vulkan")]
            Device::Vulkan { .. } => Err(TensorAllocatorError::MemoryTransferError(
                "Vulkan to CUDA copy not supported".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allocator() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let layout = Layout::from_size_align(1024, 64).unwrap();
        let ptr = allocator.alloc(layout)?;
        allocator.dealloc(ptr, layout);
        Ok(())
    }

    #[test]
    fn test_cpu_allocator_device() {
        let allocator = CpuAllocator;
        assert_eq!(allocator.device(), Device::Cpu);
        assert!(allocator.device().is_cpu());
        assert!(!allocator.device().is_gpu());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_allocator_device() {
        if let Ok(allocator) = CudaAllocator::new(0) {
            assert_eq!(allocator.device(), Device::cuda(0));
            assert!(!allocator.device().is_cpu());
            assert!(allocator.device().is_gpu());
        }
    }
}
