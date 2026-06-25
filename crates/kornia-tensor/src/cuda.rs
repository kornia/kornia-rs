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
//! - `from_cudaslice` ZERO-COPY WRAPS the input `CudaSlice<T>`: it caches the device
//!   pointer (via `DevicePtr::device_ptr`) and moves the slice, unchanged, into a
//!   generic `CudaResource<T>`. The resulting tensor aliases the same device allocation
//!   — no host round-trip, no device copy. `CudaResource` is generic over `T` precisely
//!   so this requires no transmute or byte coercion. (`to_cuda`/`to_host` DO copy —
//!   they are host↔device transfers, which is correct.)
//!
//! - `miri` cannot execute CUDA driver calls; device tests are guarded by
//!   `#[cfg(all(test, feature = "cudarc"))]` and run on the real Jetson Orin.

use std::{any::Any, marker::PhantomData, ptr::NonNull, sync::Arc};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};

use crate::{
    allocator::{CpuAllocator, TensorAllocator, TensorAllocatorError},
    resource::{MemoryDomain, MemoryResource},
    storage::TensorStorage,
    tensor::{get_strides_from_shape, Tensor, TensorError},
};

// ── CudaResource ─────────────────────────────────────────────────────────────

/// An owning [`MemoryResource`] that wraps a [`CudaSlice<T>`].
///
/// The wrapped `CudaSlice<T>` is the **sole owner** of the device allocation and its
/// own `Drop` impl frees the device memory exactly once.  No manual free is performed
/// here — the `CudaResource` is purely a keepalive + type-erasable handle.
///
/// `CudaResource` is generic over the element type `T` so that
/// [`Tensor::from_cudaslice`] can **zero-copy wrap** (alias) an existing
/// `CudaSlice<T>` without any host round-trip or byte coercion.  The
/// [`CudaAllocator`] produces `CudaResource<u8>`.
pub struct CudaResource<T> {
    /// Owns the device allocation; freed when this struct is dropped.
    pub(crate) slice: CudaSlice<T>,
    /// Cached raw device pointer (device-addressable; NOT safe to dereference on host).
    ptr: *mut u8,
    /// CUDA device ordinal (returned by `CudaContext::ordinal()`).
    id: i32,
}

// SAFETY: CudaSlice<T> is Send + Sync.  `ptr` is a device pointer that is never
// dereferenced on the host — it is only passed back to CUDA APIs.
unsafe impl<T> Send for CudaResource<T> {}
unsafe impl<T> Sync for CudaResource<T> {}

// SAFETY: CudaResource<T> is unconditionally Send + Sync (see the unsafe impls above);
// `T: 'static` is required only so the value is `Any`-downcastable via `as_any`.
impl<T: 'static> MemoryResource for CudaResource<T> {
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

        Ok(Box::new(CudaResource::<u8> { slice, ptr, id }))
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

/// Build a `TensorStorage<T, A>` that owns the given [`CudaResource<R>`].
///
/// `R` is the element type of the wrapped `CudaSlice` (e.g. `u8` from the allocator,
/// or `T` from `from_cudaslice`); `T` is the tensor's element type.
///
/// # Safety
///
/// `ptr` must be the device pointer inside `resource` (cached from
/// `resource.slice.device_ptr(stream)`).  It must be valid for `len_bytes` bytes on
/// the device.  The caller guarantees `resource` is the sole owner of that allocation.
unsafe fn storage_from_cuda_resource<T, R, A>(
    resource: CudaResource<R>,
    ptr: *mut T,
    len_bytes: usize,
    alloc: A,
) -> TensorStorage<T, A>
where
    R: 'static,
    A: TensorAllocator,
{
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
    /// Zero-copy wrap an existing `CudaSlice<T>` as a device-backed tensor.
    ///
    /// The resulting tensor **aliases the same device allocation** as `slice` — no
    /// host round-trip and no device-to-device copy occur.  The `CudaSlice<T>` is
    /// moved (unchanged) into a [`CudaResource<T>`]; its own `Drop` remains the sole
    /// owner of the device memory and frees it exactly once when the tensor drops.
    ///
    /// The tensor's cached device pointer equals the input slice's device pointer.
    ///
    /// # Arguments
    ///
    /// * `slice` — source device slice; `slice.len()` must equal `shape.iter().product()`.
    /// * `shape` — N-dimensional tensor shape.
    /// * `stream` — stream owning `slice`'s context; retained in the `CudaAllocator`.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() != shape.iter().product()`.
    pub fn from_cudaslice(slice: CudaSlice<T>, shape: [usize; N], stream: Arc<CudaStream>) -> Self {
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

        // Cache the raw device pointer of the EXISTING allocation (no copy).
        // _sync must be dropped before we move `slice` into CudaResource.
        let ptr = {
            let (cu_ptr, _sync) = slice.device_ptr(&stream);
            cu_ptr as *mut u8
            // _sync drops here, releasing the borrow on `slice`
        };

        let alloc = CudaAllocator { ctx, stream };
        // Move the original CudaSlice<T> in, unchanged — this is the aliasing wrap.
        let resource = CudaResource::<T> { slice, ptr, id };

        let storage =
            unsafe { storage_from_cuda_resource(resource, ptr as *mut T, n_bytes, alloc) };
        let strides = get_strides_from_shape(shape);
        Tensor {
            storage,
            shape,
            strides,
        }
    }

    /// Borrow the underlying `CudaSlice<T>` if the storage is backed by a
    /// [`CudaResource<T>`] (i.e. was built via [`from_cudaslice`](Self::from_cudaslice)).
    ///
    /// Returns `None` if the storage is not a `CudaResource<T>` (e.g. it was allocated
    /// by [`CudaAllocator`] which produces `CudaResource<u8>`, or T differs).
    pub fn as_cudaslice(&self) -> Option<&CudaSlice<T>> {
        self.storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .map(|r| &r.slice)
    }

    /// Consume the tensor and return the underlying `CudaSlice<T>`.
    ///
    /// Returns `Err(self)` if the storage is not backed by a [`CudaResource<T>`].
    ///
    /// # Memory safety — no double-free
    ///
    /// The `Box<dyn MemoryResource>` is downcast to `Box<CudaResource<T>>` via
    /// `Box::into_raw` → `Box::from_raw`.  The `CudaSlice<T>` is then moved out via
    /// `ManuallyDrop` so the `CudaResource`'s own `Drop` does NOT run.  The
    /// `Box<CudaResource<T>>` heap allocation is freed (struct metadata only); the
    /// device memory is now owned by the returned `CudaSlice<T>`.
    pub fn into_cudaslice(self) -> Result<CudaSlice<T>, Self> {
        if self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .is_none()
        {
            return Err(self);
        }

        // Consume `self` without running TensorStorage's Drop or CudaResource's Drop.
        let storage = self.storage;
        let md_storage = std::mem::ManuallyDrop::new(storage);

        // Read `owner` out of the ManuallyDrop without running TensorStorage's Drop.
        // SAFETY: We are the sole owner; ManuallyDrop prevents double-drop of the storage.
        let owner: Box<dyn MemoryResource> = unsafe { std::ptr::read(&md_storage.owner) };
        // Drop the allocator field explicitly (Arc fields drop normally).
        let _alloc = unsafe { std::ptr::read(&md_storage.alloc) };
        // ptr/len/marker are Copy/ZST — nothing else to drop.

        // Downcast Box<dyn MemoryResource> → Box<CudaResource<T>>.
        // SAFETY: We verified above (downcast_ref) that the concrete type is CudaResource<T>.
        let raw: *mut dyn MemoryResource = Box::into_raw(owner);
        let cuda_box: Box<CudaResource<T>> = unsafe { Box::from_raw(raw as *mut CudaResource<T>) };

        // Move CudaSlice<T> out without running CudaResource's Drop.
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

        // Copy host slice → a new device CudaSlice<T> (this is a transfer, copy is correct).
        let dev_slice: CudaSlice<T> = stream
            .clone_htod(src_slice)
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        // Extract device pointer; _sync must drop before dev_slice is moved.
        let ptr = {
            let (cu_ptr, _sync) = dev_slice.device_ptr(stream);
            cu_ptr as *mut u8
            // _sync drops here
        };
        let alloc = CudaAllocator {
            ctx,
            stream: stream.clone(),
        };
        // Store as CudaResource<T> so as_cudaslice::<T>() also works on to_cuda results.
        let resource = CudaResource::<T> {
            slice: dev_slice,
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
    /// if the storage owner is not a [`CudaResource<T>`].
    pub fn to_host(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Tensor<T, N, CpuAllocator>, CudaError> {
        let cuda_res = self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .ok_or(CudaError::NotCudaBacked)?;

        // D→H typed copy into a Vec<T> (this is a transfer, copy is correct).
        let host_data: Vec<T> = stream
            .clone_dtoh(&cuda_res.slice)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(e.to_string()))?;

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
        assert_eq!(
            back.as_slice(),
            &[1u8, 2, 3, 4],
            "round-trip bytes must match"
        );
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

        let host = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([8], vec![10u8; 8], CpuAllocator)
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

    /// `from_cudaslice` must ZERO-COPY WRAP (alias) the existing device allocation:
    /// the resulting tensor's device pointer must equal the source slice's device pointer.
    #[test]
    fn from_cudaslice_zero_copy_alias() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // Allocate on device; record its device pointer.
        let dev_slice: CudaSlice<u8> = stream.alloc_zeros::<u8>(32).unwrap();
        let orig_ptr = {
            let (cu_ptr, _sync) = dev_slice.device_ptr(&stream);
            cu_ptr as usize
        };

        let tensor =
            Tensor::<u8, 1, CudaAllocator>::from_cudaslice(dev_slice, [32], stream.clone());

        // The tensor's cached device pointer must equal the original allocation's pointer
        // → same device memory → aliased, not copied.
        let tensor_ptr = tensor.as_ptr() as usize;
        assert_eq!(
            tensor_ptr, orig_ptr,
            "from_cudaslice must alias the original device allocation \
             (tensor ptr {tensor_ptr:#x} != orig ptr {orig_ptr:#x})"
        );

        // as_cudaslice must also report the same device pointer.
        let wrapped = tensor
            .as_cudaslice()
            .expect("must be CudaResource<u8>-backed");
        let wrapped_ptr = {
            let (cu_ptr, _sync) = wrapped.device_ptr(&stream);
            cu_ptr as usize
        };
        assert_eq!(
            wrapped_ptr, orig_ptr,
            "as_cudaslice must report the aliased pointer"
        );
    }
}
