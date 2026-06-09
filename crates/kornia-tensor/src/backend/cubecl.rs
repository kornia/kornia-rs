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
use std::sync::{Arc, Mutex};

use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use cubecl::Runtime;

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
        if layout.size() == 0 {
            // layout.align() >= 1 by invariant; this pointer is never dereferenced.
            // without_provenance_mut avoids an integer-to-pointer cast under strict provenance.
            return Ok(std::ptr::without_provenance_mut(layout.align()));
        }
        let handle = self.client.empty(layout.size());
        let managed = self
            .client
            .get_resource(handle.clone())
            .map_err(|e| CubeclBackendError::ResourceLookup(e.to_string()))?;
        let device_ptr = managed.resource().device_ptr();
        if device_ptr == 0 {
            return Err(CubeclBackendError::NullDevicePointer);
        }
        // Recover from a poisoned mutex (same as dealloc) so a prior panic
        // in an unrelated thread does not permanently block future allocations.
        let mut guard = match self.live.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.insert(device_ptr, handle);
        // CUDA device address → host-side opaque pointer; never dereferenced on host.
        Ok(std::ptr::without_provenance_mut(device_ptr as usize))
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // Recover from a poisoned mutex rather than leaking the Handle.
        // A poisoned guard still holds valid data; we take it and remove the entry.
        let mut guard = match self.live.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let key = ptr as u64;
        let removed = guard.remove(&key);
        debug_assert!(
            removed.is_some(),
            "CubeclBackend::dealloc called with unknown pointer {key:#x} — possible double-free or wrong allocator"
        );
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
