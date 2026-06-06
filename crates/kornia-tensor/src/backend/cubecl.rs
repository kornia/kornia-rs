//! CubeCL backend for `kornia-tensor` (enabled by feature `gpu-cubecl`).
//!
//! The pointer returned by [`CubeclBackend::alloc`] is a CUDA device address cast to
//! `*mut u8`. It must not be dereferenced on the host.
//!
//! For non-zero allocations, cubecl-cuda controls alignment (typically 256 bytes) and
//! the `layout.align()` value passed by the caller is ignored. For zero-sized allocations
//! a non-null dangling pointer aligned to `layout.align()` is returned without calling
//! into cubecl.

// CUDA device pointers are 64-bit values; this backend is only correct on 64-bit targets.
#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "kornia-tensor's CubeclBackend requires a 64-bit target (CUDA device pointers are 64-bit)"
);

use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use super::{Backend, GpuAllocator};

/// Errors produced by [`CubeclBackend`].
#[derive(Debug, thiserror::Error)]
pub enum CubeclBackendError {
    /// The cubecl server returned a null/zero device pointer.
    #[error("cubecl backend returned a null device pointer")]
    NullDevicePointer,
    /// Failed to resolve the cubecl `Handle` to a device resource.
    #[error("cubecl backend failed to resolve handle resource: {0}")]
    ResourceLookup(String),
    /// The allocator side-table mutex was poisoned by a prior panic.
    #[error("cubecl backend side-table mutex is poisoned")]
    MutexPoisoned,
}

/// CubeCL backend wrapping a [`ComputeClient`] for the chosen runtime.
///
/// Allocations call `client.empty(size)` and keep the resulting [`Handle`] alive in a
/// side-table keyed by the device pointer. Dealloc removes the entry, dropping the
/// handle and letting cubecl's reference-counted memory manager free the allocation.
pub struct CubeclBackend<R: Runtime> {
    client: ComputeClient<R>,
    live: Arc<Mutex<HashMap<u64, Handle>>>,
}

impl<R: Runtime> CubeclBackend<R> {
    /// Wrap an existing cubecl [`ComputeClient`].
    pub fn new(client: ComputeClient<R>) -> Self {
        Self {
            client,
            live: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<R: Runtime> Clone for CubeclBackend<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            live: self.live.clone(),
        }
    }
}

/// CUDA-specific device-pointer extraction. The `Resource` type associated with
/// cubecl-cuda's server is `cubecl_cuda::compute::storage::gpu::GpuResource`, which
/// exposes `pub ptr: u64`. We bridge via a private trait so the backend stays
/// runtime-generic at the type level while only supporting cuda concretely today.
trait DevicePtr {
    fn device_ptr(&self) -> u64;
}

impl DevicePtr for cubecl_cuda::compute::storage::gpu::GpuResource {
    fn device_ptr(&self) -> u64 {
        self.ptr
    }
}

impl<R: Runtime> Backend for CubeclBackend<R>
where
    <<R as Runtime>::Server as cubecl::compute::ComputeServer>::Storage:
        cubecl::storage::ComputeStorage,
    <<<R as Runtime>::Server as cubecl::compute::ComputeServer>::Storage as cubecl::storage::ComputeStorage>::Resource: DevicePtr,
{
    type Error = CubeclBackendError;

    fn alloc(&self, layout: Layout) -> Result<*mut u8, Self::Error> {
        // ZST: return a non-null dangling pointer aligned to layout.align().
        // layout.align() is always >= 1, so NonNull::new succeeds.
        if layout.size() == 0 {
            let dangling = NonNull::new(layout.align() as *mut u8)
                .expect("layout.align() is always nonzero");
            return Ok(dangling.as_ptr());
        }
        let handle = self.client.empty(layout.size());
        let managed = self
            .client
            .get_resource(handle.clone())
            .map_err(|e| CubeclBackendError::ResourceLookup(format!("{e:?}")))?;
        let device_ptr = managed.resource().device_ptr();
        if device_ptr == 0 {
            return Err(CubeclBackendError::NullDevicePointer);
        }
        self.live
            .lock()
            .map_err(|_| CubeclBackendError::MutexPoisoned)?
            .insert(device_ptr, handle);
        Ok(device_ptr as *mut u8)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // If the mutex is poisoned a prior panic already corrupted state; skip removal
        // rather than propagating into an infallible dealloc path.
        if let Ok(mut guard) = self.live.lock() {
            guard.remove(&(ptr as u64));
        }
    }
}

/// Create a [`GpuAllocator`] backed by the default CUDA device.
///
/// Hides the cubecl runtime types from callers; use this instead of constructing
/// [`CubeclBackend`] and [`GpuAllocator`] manually in tests and application code.
pub fn new_cuda_allocator() -> GpuAllocator<CubeclBackend<cubecl_cuda::CudaRuntime>> {
    use cubecl::Runtime;
    let device = <cubecl_cuda::CudaRuntime as Runtime>::Device::default();
    let client = cubecl_cuda::CudaRuntime::client(&device);
    GpuAllocator::new(CubeclBackend::new(client))
}
