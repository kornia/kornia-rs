use std::alloc;
use std::alloc::Layout;

use thiserror::Error;

use crate::device::Device;

/// An error type for tensor allocator operations.
#[derive(Debug, Error, PartialEq)]
pub enum TensorAllocatorError {
    /// An error occurred during memory allocation.
    #[error("Invalid tensor layout {0}")]
    LayoutError(core::alloc::LayoutError),

    /// An error occurred during memory allocation.
    #[error("Null pointer")]
    NullPointer,

    /// Device mismatch error for memory transfer operations.
    #[error("Device mismatch: source device {0}, destination device {1}")]
    DeviceMismatch(String, String),

    /// Memory transfer error.
    #[error("Memory transfer failed: {0}")]
    MemoryTransferError(String),

    /// CUDA error
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),
}

/// A trait for allocating and deallocating memory for tensors.
///
/// # Safety
///
/// The tensor allocator must be thread-safe.
///
/// # Methods
///
/// * `alloc` - Allocates memory for a tensor with the given layout.
/// * `dealloc` - Deallocates memory for a tensor with the given layout.
/// * `device` - Returns the device associated with this allocator.
/// * `copy_from` - Copies data from a source device to this device.
pub trait TensorAllocator: Clone {
    /// Allocates memory for a tensor with the given layout.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;

    /// Deallocates memory for a tensor with the given layout.
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

#[derive(Clone)]
/// A tensor allocator that uses the system allocator.
pub struct CpuAllocator;

/// Implement the `Default` trait for the `CpuAllocator` struct.
impl Default for CpuAllocator {
    fn default() -> Self {
        Self
    }
}

/// Implement the `TensorAllocator` trait for the `CpuAllocator` struct.
impl TensorAllocator for CpuAllocator {
    /// Allocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The layout of the tensor.
    ///
    /// # Returns
    ///
    /// A non-null pointer to the allocated memory if successful, otherwise an error.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }
        Ok(ptr)
    }

    /// Deallocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `ptr` - A non-null pointer to the allocated memory.
    /// * `layout` - The layout of the tensor.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the layout must be correct.
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
