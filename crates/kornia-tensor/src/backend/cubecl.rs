//! CubeCL backend for `kornia-tensor` (enabled by feature `gpu-cubecl`).
//!
//! The pointer returned by [`CubeclBackend::alloc`] is a CUDA device address cast to
//! `*mut u8`. It must not be dereferenced on the host. cubecl's `Layout::align` is
//! ignored — the cubecl-cuda backend uses its own alignment (typically 256 bytes).

use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use super::Backend;

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
    live: Arc<Mutex<HashMap<usize, Handle>>>,
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
        let handle = self.client.empty(layout.size());
        let managed = self
            .client
            .get_resource(handle.clone())
            .map_err(|e| CubeclBackendError::ResourceLookup(format!("{e:?}")))?;
        let device_ptr = managed.resource().device_ptr();
        if device_ptr == 0 {
            return Err(CubeclBackendError::NullDevicePointer);
        }
        let ptr = device_ptr as *mut u8;
        self.live
            .lock()
            .expect("CubeclBackend side-table poisoned")
            .insert(ptr as usize, handle);
        Ok(ptr)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        let _ = self
            .live
            .lock()
            .expect("CubeclBackend side-table poisoned")
            .remove(&(ptr as usize));
    }
}
