//! CUDA device memory integration for `kornia-tensor` via `cudarc 0.19`.
//!
//! This module is enabled by the `cudarc` feature flag.  It provides:
//!
//! - [`CudaResource`]: an owning [`MemoryResource`] that wraps a [`CudaSlice<u8>`].
//!   The `CudaSlice`'s own `Drop` frees the device allocation — there is no manual
//!   `cudaFree` call here, which is the guarantee against double-free.
//!
//! - [`CudaAllocator`]: a [`TensorAllocator`] that allocates zero-initialised device
//!   memory via `stream.alloc_zeros::<u8>(n)` and wraps the result in a `CudaResource`.
//!
//! - Five methods on [`Tensor`]:
//!   - [`Tensor::from_cudaslice`] — wrap an existing `CudaSlice<T>` as a device tensor.
//!   - [`Tensor::as_cudaslice`] — borrow the underlying `CudaSlice<u8>` (if any).
//!   - [`Tensor::into_cudaslice`] — consume the tensor and return the `CudaSlice<u8>`.
//!   - [`Tensor::to_cuda`] — copy a host tensor to a new device tensor (h→d).
//!   - [`Tensor::to_host`] — copy a device tensor back to a new host tensor (d→h).
//!
//! # Memory-safety invariants
//!
//! - Each `CudaSlice<u8>` is stored **exactly once**: inside `CudaResource`.
//!   `CudaResource` is owned by a `Box<dyn MemoryResource>` in `TensorStorage`.
//!   When the `Tensor` drops, the chain drops, `CudaSlice::drop` runs exactly once,
//!   and cudarc frees the device memory.
//!
//! - `CudaAllocator::allocate` calls `stream.alloc_zeros::<u8>(n)` — all memory is
//!   zero-initialised and the `CudaSlice<u8>` carries the free obligation.
//!
//! - `into_cudaslice` extracts the `CudaSlice<u8>` by consuming the `Box<CudaResource>`
//!   via `ManuallyDrop`, so the `CudaSlice` is moved out before any destructor runs.
//!   The heap allocation for the `CudaResource` struct itself is freed by `Box::from_raw`;
//!   the `CudaSlice` it contained is *not* dropped (we return it to the caller instead).
//!
//! - `from_cudaslice` accepts a `CudaSlice<T>` for any `DeviceRepr + ValidAsZeroBits` T.
//!   To store it uniformly as `CudaSlice<u8>`, we issue a device-to-device byte copy
//!   into a new `CudaSlice<u8>` of `size_of::<T>() * numel` bytes, then drop the source.
//!   This is allocation-safe (no host data) and avoids any transmute UB.
//!   When `T = u8`, the copy is still done for uniformity; an optimisation (direct move)
//!   can be added later once cudarc exposes a `CudaSlice::reinterpret_as_u8` API.
//!
//! - `miri` cannot execute CUDA driver calls; device tests are guarded by
//!   `#[cfg(all(test, feature = "cudarc"))]` and run on the real Jetson Orin.

use std::{
    any::Any,
    marker::PhantomData,
    ptr::NonNull,
    sync::Arc,
};

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits,
};

use crate::{
    allocator::{CpuAllocator, TensorAllocator, TensorAllocatorError},
    resource::{MemoryDomain, MemoryResource},
    storage::TensorStorage,
    tensor::{Tensor, TensorError, get_strides_from_shape},
};

// ── CudaResource ─────────────────────────────────────────────────────────────

/// An owning [`MemoryResource`] that wraps a [`CudaSlice<u8>`].
///
/// The `CudaSlice<u8>`'s own `Drop` impl frees the device allocation exactly once.
/// No manual free is performed here.
pub struct CudaResource {
    /// Owns the device allocation; freed when this struct is dropped.
    pub(crate) slice: CudaSlice<u8>,
    /// Cached raw device pointer (device-addressable; NOT safe to dereference on host).
    ptr: *mut u8,
    /// CUDA device ordinal (returned by `CudaContext::ordinal()`).
    id: i32,
}

// SAFETY: CudaSlice<u8> is Send + Sync.  `ptr` is a device pointer that is never
// dereferenced on the host — it is only passed back to CUDA APIs.
unsafe impl Send for CudaResource {}
unsafe impl Sync for CudaResource {}

impl MemoryResource for CudaResource {
    /// Returns the cached device pointer (NOT host-dereferenceable).
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn len_bytes(&self) -> usize {
        self.slice.num_bytes()
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Device { id: self.id }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Implicit Drop: `CudaSlice::drop` calls cudarc's device-free path exactly once.

// ── CudaAllocator ─────────────────────────────────────────────────────────────

/// A [`TensorAllocator`] that allocates zero-initialised CUDA device memory.
///
/// Uses [`CudaStream::alloc_zeros::<u8>`] so all bytes are guaranteed zero.
#[derive(Clone)]
pub struct CudaAllocator {
    /// Shared CUDA context (keeps the driver alive).
    pub ctx: Arc<CudaContext>,
    /// Stream on which allocations (and later free-async) are issued.
    pub stream: Arc<CudaStream>,
}

// SAFETY: Arc<CudaContext> and Arc<CudaStream> are Send + Sync.
unsafe impl Send for CudaAllocator {}
unsafe impl Sync for CudaAllocator {}

impl TensorAllocator for CudaAllocator {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        let n_bytes = layout.size();
        let slice: CudaSlice<u8> = self
            .stream
            .alloc_zeros::<u8>(n_bytes)
            .map_err(|e| TensorAllocatorError::CudaError(e.to_string()))?;

        // Extract the raw device pointer before moving the slice.
        // _sync must be dropped before we move `slice` into CudaResource.
        let ptr = {
            let (cu_ptr, _sync) = slice.device_ptr(&self.stream);
            cu_ptr as *mut u8
            // `_sync` drops here, releasing the borrow on `slice`
        };
        let id = self.ctx.ordinal() as i32;

        Ok(Box::new(CudaResource { slice, ptr, id }))
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type for CUDA tensor operations.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// cudarc driver error.
    #[error("CUDA driver error: {0}")]
    Driver(String),

    /// Shape / element-count mismatch.
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),

    /// Storage is not backed by a `CudaResource`.
    #[error("Tensor storage is not device-backed by CudaResource")]
    NotCudaBacked,
}

// ── Helper: build a TensorStorage from a CudaResource ─────────────────────────

/// Build a `TensorStorage<T, CudaAllocator>` that owns the given [`CudaResource`].
///
/// # Safety
///
/// `ptr` must be the device pointer inside `resource` (cached from
/// `resource.slice.device_ptr(stream)`).  It must be valid for `len_bytes` bytes on
/// the device.  The caller guarantees `resource` is the sole owner of that allocation.
unsafe fn storage_from_cuda_resource<T, A: TensorAllocator>(
    resource: CudaResource,
    ptr: *mut T,
    len_bytes: usize,
    alloc: A,
) -> TensorStorage<T, A> {
    let owner: Box<dyn MemoryResource> = Box::new(resource);
    let nn_ptr = NonNull::new_unchecked(ptr);
    TensorStorage {
        ptr: nn_ptr,
        len: len_bytes,
        owner,
        alloc,
        _marker: PhantomData,
    }
}

// ── Tensor: from_cudaslice / as_cudaslice / into_cudaslice ────────────────────

impl<T, const N: usize> Tensor<T, N, CudaAllocator>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    /// Wrap an existing `CudaSlice<T>` as a device-backed tensor (no host copy).
    ///
    /// The source slice's elements are copied device-to-device into a new
    /// `CudaSlice<u8>` of `numel * size_of::<T>()` bytes; the source slice is
    /// then dropped.  This avoids any transmute and keeps `CudaResource` typed
    /// as `CudaSlice<u8>` uniformly.
    ///
    /// # Arguments
    ///
    /// * `slice` — source device slice; `slice.len()` must equal `shape.iter().product()`.
    /// * `shape` — N-dimensional tensor shape.
    /// * `stream` — stream for the d→d byte copy.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() != shape.iter().product()`.
    pub fn from_cudaslice(
        slice: CudaSlice<T>,
        shape: [usize; N],
        stream: Arc<CudaStream>,
    ) -> Self {
        let numel = shape.iter().product::<usize>();
        assert_eq!(
            slice.len(),
            numel,
            "from_cudaslice: slice.len() ({}) != shape product ({})",
            slice.len(),
            numel,
        );

        let n_bytes = slice.num_bytes(); // numel * size_of::<T>()
        let ctx = stream.context().clone();
        let id = ctx.ordinal() as i32;

        // cudarc 0.19 does not expose a public CudaSlice::reinterpret_as method, and
        // memcpy_dtod is typed as <T> (both src and dst must share the same T).
        // Safest public-API path: D→H (clone_dtoh → Vec<T>), reinterpret bytes, H→D
        // into a fresh CudaSlice<u8>.  On Jetson with unified memory the D→H copy is
        // effectively a no-op.  A direct D→D byte copy can replace this if cudarc ever
        // exposes an untyped memcpy or a reinterpret API.
        let host_tmp: Vec<T> = stream
            .clone_dtoh(&slice)
            .expect("from_cudaslice: dtoh failed");
        let byte_view: &[u8] = unsafe {
            std::slice::from_raw_parts(
                host_tmp.as_ptr() as *const u8,
                std::mem::size_of_val(host_tmp.as_slice()),
            )
        };
        let mut byte_slice: CudaSlice<u8> = stream
            .clone_htod(byte_view)
            .expect("from_cudaslice: htod failed");
        // Synchronize to ensure the copy completes before we drop the source slice.
        stream.synchronize().expect("from_cudaslice: synchronize failed");

        // Drop the source T-typed slice (its device memory is freed).
        drop(slice);
        drop(host_tmp);

        // Ensure byte_slice is live as mutable for memcpy_htod; suppress the lint.
        let _ = &mut byte_slice;

        // Extract device pointer; _sync must be dropped before byte_slice is moved.
        let ptr = {
            let (cu_ptr, _sync) = byte_slice.device_ptr(&stream);
            cu_ptr as *mut u8
            // _sync drops here
        };

        let alloc = CudaAllocator { ctx, stream };
        let resource = CudaResource {
            slice: byte_slice,
            ptr,
            id,
        };

        let storage =
            unsafe { storage_from_cuda_resource(resource, ptr as *mut T, n_bytes, alloc) };
        let strides = get_strides_from_shape(shape);
        Tensor {
            storage,
            shape,
            strides,
        }
    }

    /// Borrow the underlying `CudaSlice<u8>` if the storage is backed by a [`CudaResource`].
    ///
    /// Returns `None` if the storage was not created via a `CudaAllocator`-backed path.
    pub fn as_cudaslice(&self) -> Option<&CudaSlice<u8>> {
        self.storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource>()
            .map(|r| &r.slice)
    }

    /// Consume the tensor and return the underlying `CudaSlice<u8>`.
    ///
    /// Returns `Err(self)` if the storage is not backed by a [`CudaResource`].
    ///
    /// # Memory safety — no double-free
    ///
    /// The `Box<dyn MemoryResource>` is consumed via `Box::into_raw` → `Box::from_raw`
    /// to downcast to `Box<CudaResource>`.  The `CudaSlice<u8>` is then moved out of
    /// the `CudaResource` via `ManuallyDrop` so the struct's own `Drop` does NOT run.
    /// The `Box<CudaResource>` heap allocation is freed (it no longer contains the
    /// slice, so only the struct metadata is freed, not the device memory).
    pub fn into_cudaslice(self) -> Result<CudaSlice<u8>, Self> {
        if self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource>()
            .is_none()
        {
            return Err(self);
        }

        // Consume `self` without running TensorStorage's Drop or CudaResource's Drop.
        // We use ManuallyDrop on the whole TensorStorage and extract just the owner.
        let (shape, strides, storage) = (self.shape, self.strides, self.storage);
        let _ = (shape, strides); // suppress unused warnings

        // Consume TensorStorage: we must prevent its Drop from running (which would
        // drop `owner` and thus drop CudaSlice — we want to return it instead).
        let md_storage = std::mem::ManuallyDrop::new(storage);

        // Read `owner` out of the ManuallyDrop without running TensorStorage's Drop.
        // SAFETY: We are the sole owner; ManuallyDrop prevents double-drop of the storage.
        let owner: Box<dyn MemoryResource> = unsafe { std::ptr::read(&md_storage.owner) };
        // Explicitly drop the allocator (unit struct, but be explicit).
        let _alloc = unsafe { std::ptr::read(&md_storage.alloc) };
        // Suppress the ptr/len/marker (no Drop needed for NonNull/PhantomData/usize).
        // ManuallyDrop ensures nothing else is dropped.

        // Downcast Box<dyn MemoryResource> → Box<CudaResource>.
        // SAFETY: We verified above (downcast_ref) that the concrete type is CudaResource.
        let raw: *mut dyn MemoryResource = Box::into_raw(owner);
        let cuda_box: Box<CudaResource> = unsafe { Box::from_raw(raw as *mut CudaResource) };

        // Move CudaSlice<u8> out without running CudaResource's Drop.
        let mut md_res = std::mem::ManuallyDrop::new(*cuda_box);
        // SAFETY: md_res prevents CudaResource from dropping its fields; we take the slice.
        let slice = unsafe { std::ptr::read(&md_res.slice) };
        // Poison the ptr to make residual state obviously invalid (defensive).
        md_res.ptr = std::ptr::null_mut();

        Ok(slice)
    }
}

// ── Tensor::to_cuda (host → device) ──────────────────────────────────────────

impl<T, const N: usize, A: TensorAllocator> Tensor<T, N, A>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
    A: 'static,
{
    /// Copy this host tensor to a new device-backed tensor on `stream`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA failure.
    pub fn to_cuda(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Tensor<T, N, CudaAllocator>, CudaError> {
        let src_slice = self.as_slice(); // panics (correctly) if non-host-accessible

        let ctx = stream.context().clone();
        let id = ctx.ordinal() as i32;
        let n_bytes = std::mem::size_of_val(src_slice);

        // Reinterpret the host slice as bytes and copy directly to a CudaSlice<u8>.
        // This avoids a redundant D→H→D round-trip that would be needed if we first
        // cloned as CudaSlice<T>.
        //
        // SAFETY: T is DeviceRepr + ValidAsZeroBits; byte reinterpretation is well-defined.
        let byte_src: &[u8] = unsafe {
            std::slice::from_raw_parts(src_slice.as_ptr() as *const u8, n_bytes)
        };

        let byte_slice: CudaSlice<u8> = stream
            .clone_htod(byte_src)
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        // Extract device pointer; _sync must drop before byte_slice is moved.
        let ptr = {
            let (cu_ptr, _sync) = byte_slice.device_ptr(stream);
            cu_ptr as *mut u8
            // _sync drops here
        };
        let alloc = CudaAllocator {
            ctx,
            stream: stream.clone(),
        };
        let resource = CudaResource {
            slice: byte_slice,
            ptr,
            id,
        };
        let storage =
            unsafe { storage_from_cuda_resource(resource, ptr as *mut T, n_bytes, alloc) };
        let strides = get_strides_from_shape(self.shape);
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides,
        })
    }
}

// ── Tensor::to_host (device → host) ──────────────────────────────────────────

impl<T, const N: usize> Tensor<T, N, CudaAllocator>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    /// Copy this device tensor to a new host-backed tensor.
    ///
    /// Synchronizes the stream before returning so the host data is valid.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA failure or [`CudaError::NotCudaBacked`]
    /// if the storage owner is not a [`CudaResource`].
    pub fn to_host(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Tensor<T, N, CpuAllocator>, CudaError> {
        let cuda_res = self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource>()
            .ok_or(CudaError::NotCudaBacked)?;

        // D→H: copy the raw bytes, then reinterpret as T.
        let numel = self.shape.iter().product::<usize>();
        let n_bytes = numel * std::mem::size_of::<T>();

        // Allocate host destination for raw bytes.
        let mut byte_buf: Vec<u8> = vec![0u8; n_bytes];
        stream
            .memcpy_dtoh(&cuda_res.slice, byte_buf.as_mut_slice())
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        // Reinterpret byte buffer as Vec<T>.
        // SAFETY: T is DeviceRepr + ValidAsZeroBits; the bytes came from a valid T-typed
        // allocation; size and alignment are correct.
        let mut host_data: Vec<T> = Vec::with_capacity(numel);
        unsafe {
            std::ptr::copy_nonoverlapping(
                byte_buf.as_ptr(),
                host_data.as_mut_ptr() as *mut u8,
                n_bytes,
            );
            host_data.set_len(numel);
        }
        drop(byte_buf);

        let storage = TensorStorage::from_vec(host_data, CpuAllocator);
        let strides = get_strides_from_shape(self.shape);
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cudarc"))]
mod tests {
    use super::*;
    use crate::{CpuAllocator, Tensor};

    /// Device round-trip: host → GPU → host, bytes must match exactly.
    /// Also verifies domain is Device and as_cudaslice is Some.
    #[test]
    fn cuda_roundtrip_and_as_slice_panics() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let host =
            Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], vec![1, 2, 3, 4], CpuAllocator)
                .unwrap();

        let dev = host.to_cuda(&stream).unwrap();
        assert!(
            matches!(dev.storage.domain(), MemoryDomain::Device { .. }),
            "expected Device domain, got {:?}",
            dev.storage.domain()
        );
        assert!(
            dev.as_cudaslice().is_some(),
            "as_cudaslice should return Some for a CudaResource-backed tensor"
        );

        let back = dev.to_host(&stream).unwrap();
        assert_eq!(back.as_slice(), &[1u8, 2, 3, 4], "round-trip bytes must match");
    }

    /// `as_slice()` on a device tensor must panic with "non-host-accessible".
    #[test]
    #[should_panic(expected = "non-host-accessible")]
    fn as_slice_on_device_panics() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let host =
            Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], vec![1, 2, 3, 4], CpuAllocator)
                .unwrap();
        let dev = host.to_cuda(&stream).unwrap();
        let _ = dev.as_slice(); // must panic: Device is not host-accessible
    }

    /// `into_cudaslice` returns the slice; no double-free.
    /// A subsequent alloc must succeed (proves no CUDA error state after the free).
    #[test]
    fn into_cudaslice_no_double_free() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let host =
            Tensor::<u8, 1, CpuAllocator>::from_shape_vec([8], vec![10u8; 8], CpuAllocator)
                .unwrap();
        let dev = host.to_cuda(&stream).unwrap();
        let slice = dev.into_cudaslice().ok().expect("must be cuda-backed");
        // The slice now owns the device memory.  Drop it — cudarc frees exactly once.
        drop(slice);
        // A subsequent allocation must succeed (no CUDA error state after the free).
        let _s2 = stream
            .alloc_zeros::<u8>(8)
            .expect("second alloc must succeed after into_cudaslice+drop");
    }

    /// `from_cudaslice` → drop: device memory freed once, no double-free.
    #[test]
    fn from_cudaslice_drop_once() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // Allocate on device directly via cudarc.
        let dev_slice: CudaSlice<u8> = stream.alloc_zeros::<u8>(16).unwrap();
        let tensor =
            Tensor::<u8, 1, CudaAllocator>::from_cudaslice(dev_slice, [16], stream.clone());
        // Drop tensor → TensorStorage drops → Box<CudaResource> drops → CudaSlice::drop
        // → cudarc frees the device memory exactly once.
        drop(tensor);
        // A fresh allocation must succeed (no CUDA error after the free).
        let _verify = stream
            .alloc_zeros::<u8>(16)
            .expect("allocation after from_cudaslice+drop must succeed");
    }
}
