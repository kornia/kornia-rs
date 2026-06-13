//! CubeCL backend for `kornia-tensor` (enabled by feature `gpu-cubecl`).
//!
//! The pointer returned by [`CubeclBackend::alloc`] is a CUDA device address cast to
//! `*mut u8`. It must not be dereferenced on the host.
//!
//! For non-zero allocations, `cudarc::driver::result::malloc_sync` controls alignment
//! (typically 256 bytes per the CUDA spec) and the `layout.align()` value passed by the
//! caller is ignored. For zero-sized allocations a non-null dangling pointer aligned to
//! `layout.align()` is returned without calling into the driver.

// CUDA device pointers are 64-bit values; this backend is only correct on 64-bit targets.
#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "kornia-tensor's CubeclBackend requires a 64-bit target (CUDA device pointers are 64-bit)"
);

use std::alloc::Layout;

use cubecl_core::client::ComputeClient;
use cubecl_core::Runtime;

use super::{Backend, GpuAllocator};

/// Errors produced by [`CubeclBackend`].
#[derive(Debug, thiserror::Error)]
pub enum CubeclBackendError {
    /// The CUDA driver returned a null/zero device pointer.
    #[error("CUDA driver returned a null device pointer")]
    NullDevicePointer,
    /// The CUDA driver returned an error during allocation.
    #[error("CUDA driver allocation failed: {0}")]
    AllocationFailed(String),
}

/// CubeCL backend wrapping a [`ComputeClient`] for the chosen runtime.
///
/// Raw allocation uses `cudarc` directly (`malloc_sync` / `free_sync`) so that
/// the returned pointer is a plain CUDA device address. The `client` is retained
/// so kernel launches can reuse the same CUDA context.
pub struct CubeclBackend<R: Runtime> {
    /// CubeCL compute client — initialises the CUDA context and will be used
    /// for kernel launches in a future PR.
    client: ComputeClient<R>,
}

impl<R: Runtime> CubeclBackend<R> {
    /// Wrap an existing cubecl [`ComputeClient`].
    pub fn new(client: ComputeClient<R>) -> Self {
        Self { client }
    }
}

impl<R: Runtime> Clone for CubeclBackend<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }
}

impl<R: Runtime> Backend for CubeclBackend<R> {
    type Error = CubeclBackendError;

    fn alloc(&self, layout: Layout) -> Result<*mut u8, Self::Error> {
        if layout.size() == 0 {
            // layout.align() >= 1 by invariant; this pointer is never dereferenced.
            return Ok(std::ptr::without_provenance_mut(layout.align()));
        }
        // SAFETY: a CUDA context is active — CudaRuntime::client() initialised it.
        let device_ptr = unsafe { cudarc::driver::result::malloc_sync(layout.size()) }
            .map_err(|e| CubeclBackendError::AllocationFailed(e.to_string()))?;
        if device_ptr == 0 {
            return Err(CubeclBackendError::NullDevicePointer);
        }
        // CUDA device address → host-side opaque pointer; never dereferenced on host.
        Ok(std::ptr::without_provenance_mut(device_ptr as usize))
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // Ignore free errors (driver already shut down, or double-free caught at debug level).
        let _ = unsafe { cudarc::driver::result::free_sync(ptr as u64) };
    }
}

/// Create a [`GpuAllocator`] backed by the default CUDA device.
///
/// Hides the cubecl runtime types from callers; use this instead of constructing
/// [`CubeclBackend`] and [`GpuAllocator`] manually in tests and application code.
pub fn new_cuda_allocator() -> GpuAllocator<CubeclBackend<cubecl_cuda::CudaRuntime>> {
    use cubecl_core::Runtime;
    let device = <cubecl_cuda::CudaRuntime as Runtime>::Device::default();
    let client = cubecl_cuda::CudaRuntime::client(&device);
    GpuAllocator::new(CubeclBackend::new(client))
}
