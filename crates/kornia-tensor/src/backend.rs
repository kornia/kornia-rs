//! Backend abstraction for device operations.
//!
//! This module provides a trait-based abstraction for device-specific operations,
//! making it easier to add new backends (Metal, Vulkan, etc.) in the future.

use std::alloc::Layout;

use crate::{
    device::Device,
    allocator::{TensorAllocator, TensorAllocatorError, CpuAllocator},
};

#[cfg(feature = "cuda")]
use crate::allocator::CudaAllocator;

/// Backend trait defining core device operations.
///
/// This trait abstracts device-specific operations like memory allocation,
/// copying, and synchronization. Each device (CPU, CUDA, Metal, etc.)
/// implements this trait to provide its specific behavior.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow safe usage across threads.
pub trait Backend: Send + Sync + 'static {
    /// Returns the device type for this backend.
    fn device(&self) -> Device;

    /// Allocates memory on the device.
    ///
    /// # Arguments
    ///
    /// * `layout` - The memory layout to allocate
    ///
    /// # Returns
    ///
    /// A pointer to the allocated memory.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;

    /// Deallocates memory on the device.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated by this backend
    /// - `ptr` is not used after deallocation
    /// - `layout` matches the original allocation
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);

    /// Copies data between devices.
    ///
    /// # Arguments
    ///
    /// * `src` - Source pointer
    /// * `dst` - Destination pointer
    /// * `len` - Number of bytes to copy
    /// * `src_device` - Source device type
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `src` is valid for reads of `len` bytes
    /// - `dst` is valid for writes of `len` bytes
    /// - Pointers are on correct devices
    unsafe fn copy(
        &self,
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_device: &Device,
    ) -> Result<(), TensorAllocatorError>;

    /// Synchronizes device operations.
    ///
    /// This ensures all pending operations on the device have completed.
    /// For CPU, this is a no-op since operations are synchronous.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails.
    fn synchronize(&self) -> Result<(), TensorAllocatorError> {
        // Default implementation: no-op
        Ok(())
    }
}

/// CPU backend implementation.
///
/// This backend performs all operations on the CPU using standard Rust
/// memory allocation and operations.
#[derive(Clone)]
pub struct CpuBackend {
    allocator: CpuAllocator,
}

impl CpuBackend {
    /// Creates a new CPU backend.
    pub fn new() -> Self {
        Self {
            allocator: CpuAllocator,
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        self.allocator.device()
    }

    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        self.allocator.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.allocator.dealloc(ptr, layout);
    }

    unsafe fn copy(
        &self,
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_device: &Device,
    ) -> Result<(), TensorAllocatorError> {
        self.allocator.copy_from(src, dst, len, src_device)
    }

    fn synchronize(&self) -> Result<(), TensorAllocatorError> {
        // CPU operations are synchronous
        Ok(())
    }
}

/// CUDA backend implementation.
///
/// This backend performs operations on NVIDIA GPUs using CUDA.
/// It wraps the existing `CudaAllocator` to provide the Backend interface.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CudaBackend {
    allocator: CudaAllocator,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Creates a new CUDA backend for the specified device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The CUDA device ID
    ///
    /// # Errors
    ///
    /// Returns an error if the device cannot be initialized.
    pub fn new(device_id: usize) -> Result<Self, TensorAllocatorError> {
        Ok(Self {
            allocator: CudaAllocator::new(device_id)?,
        })
    }

    /// Returns the CUDA device ID.
    pub fn device_id(&self) -> usize {
        self.allocator.device_id()
    }
}

#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    fn device(&self) -> Device {
        self.allocator.device()
    }

    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        self.allocator.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.allocator.dealloc(ptr, layout);
    }

    unsafe fn copy(
        &self,
        src: *const u8,
        dst: *mut u8,
        len: usize,
        src_device: &Device,
    ) -> Result<(), TensorAllocatorError> {
        self.allocator.copy_from(src, dst, len, src_device)
    }

    fn synchronize(&self) -> Result<(), TensorAllocatorError> {
        // CUDA operations are asynchronous, but our allocator handles this
        // No explicit synchronization needed for now
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_alloc_dealloc() {
        let backend = CpuBackend::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();

        let ptr = backend.alloc(layout).unwrap();
        assert!(!ptr.is_null());

        unsafe {
            backend.dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_cpu_backend_copy() {
        let backend = CpuBackend::new();
        let src = vec![1u8, 2, 3, 4, 5];
        let mut dst = vec![0u8; 5];

        unsafe {
            backend
                .copy(src.as_ptr(), dst.as_mut_ptr(), 5, &Device::Cpu)
                .unwrap();
        }

        assert_eq!(dst, src);
    }

    #[test]
    fn test_cpu_backend_synchronize() {
        let backend = CpuBackend::new();
        assert!(backend.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_backend_device() {
        let backend = CpuBackend::new();
        assert_eq!(backend.device(), Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_backend_create() {
        // Skip if no CUDA device available
        if let Ok(backend) = CudaBackend::new(0) {
            assert_eq!(backend.device_id(), 0);
            assert!(matches!(backend.device(), Device::Cuda { device_id: 0 }));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_backend_alloc_dealloc() {
        if let Ok(backend) = CudaBackend::new(0) {
            let layout = Layout::from_size_align(1024, 8).unwrap();

            let ptr = backend.alloc(layout).unwrap();
            assert!(!ptr.is_null());

            unsafe {
                backend.dealloc(ptr, layout);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_backend_copy_host_device() {
        if let Ok(backend) = CudaBackend::new(0) {
            let layout = Layout::from_size_align(5, 1).unwrap();
            let device_ptr = backend.alloc(layout).unwrap();

            let src = vec![1u8, 2, 3, 4, 5];
            let cpu_layout = Layout::from_size_align(5, 1).unwrap();
            let cpu_backend = CpuBackend::new();
            let host_ptr = cpu_backend.alloc(cpu_layout).unwrap();

            unsafe {
                // Copy data to host buffer
                std::ptr::copy_nonoverlapping(src.as_ptr(), host_ptr, 5);

                // Copy to device
                backend
                    .copy(host_ptr, device_ptr, 5, &Device::Cpu)
                    .unwrap();

                // Sync
                backend.synchronize().unwrap();

                // Copy back to host
                let mut dst = vec![0u8; 5];
                backend
                    .copy(device_ptr, dst.as_mut_ptr(), 5, &backend.device())
                    .unwrap();

                backend.dealloc(device_ptr, layout);
                cpu_backend.dealloc(host_ptr, cpu_layout);

                assert_eq!(dst, src);
            }
        }
    }

    #[test]
    fn test_backend_trait_object() {
        // Test that Backend can be used as a trait object
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(CpuBackend::new()),
        ];

        for backend in backends {
            assert!(backend.synchronize().is_ok());
        }
    }
}
