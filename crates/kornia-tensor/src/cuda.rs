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
//!   `#[cfg(all(test, feature = "cuda"))]` and run on the real Jetson Orin.

use std::{
    any::Any,
    collections::HashMap,
    marker::PhantomData,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
    LaunchArgs, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::{
    allocator::{host_alloc, AllocHandle, TensorAllocator, TensorAllocatorError},
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
    /// Owns the device allocation; freed when this struct is dropped —
    /// unless `foreign` is set, in which case the slice is leaked (the memory
    /// belongs to an external producer) and the keepalive is released instead.
    /// `ManuallyDrop` so [`Drop`] can decide between the two.
    pub(crate) slice: std::mem::ManuallyDrop<CudaSlice<T>>,
    /// Cached raw device pointer (device-addressable; NOT safe to dereference on host).
    ptr: *mut u8,
    /// CUDA device ordinal (returned by `CudaContext::ordinal()`).
    id: i32,
    /// Stream this allocation was created on. Carried inside the resource so that
    /// device-dispatching code (e.g. residency-aware color conversion) can recover
    /// a stream from the tensor itself via [`Tensor::cuda_stream`] — no global or
    /// thread-local stream state.
    pub(crate) stream: Arc<CudaStream>,
    /// `Some` when the device memory belongs to an external producer (e.g. a
    /// zero-copy DLPack import): the boxed value keeps the producer alive and
    /// is dropped — releasing it — after the slice is leaked.
    pub(crate) foreign: Option<Box<dyn std::any::Any + Send>>,
}

impl<T> Drop for CudaResource<T> {
    fn drop(&mut self) {
        // SAFETY: Drop runs exactly once and `slice` is never used afterwards.
        let slice = unsafe { std::mem::ManuallyDrop::take(&mut self.slice) };
        if self.foreign.is_some() {
            // Foreign memory is not ours to free: leak the aliasing slice so
            // cuMemFree never runs. The `foreign` keepalive drops right after
            // this fn returns, releasing the producer's reference.
            slice.leak();
        }
    }
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

    fn as_any_mut(&mut self) -> &mut dyn Any {
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

        Ok(Box::new(CudaResource::<u8> {
            slice: std::mem::ManuallyDrop::new(slice),
            ptr,
            id,
            stream: self.stream.clone(),
            foreign: None,
        }))
    }
}

// ── PinnedResource / PinnedAllocator ─────────────────────────────────────────

/// An owning [`MemoryResource`] over page-locked (pinned) **host** memory
/// (`cuMemHostAlloc`, cacheable — no write-combining, so CPU reads stay fast).
///
/// Pinned memory is host-accessible like any heap allocation (domain is
/// [`MemoryDomain::Host`]; `as_slice` and every CPU path work unchanged), but
/// the CUDA driver recognises the registered range and services H2D/D2H
/// copies with direct DMA instead of staging through an internal bounce
/// buffer — on Jetson this roughly halves transfer cost. Freed with
/// `cuMemFreeHost` exactly once on drop.
pub struct PinnedResource {
    ptr: NonNull<u8>,
    len_bytes: usize,
    /// Keeps the driver context alive for the free call.
    _ctx: Arc<CudaContext>,
}

// SAFETY: pinned host memory is plain host memory; the pointer is uniquely
// owned by this resource.
unsafe impl Send for PinnedResource {}
unsafe impl Sync for PinnedResource {}

impl MemoryResource for PinnedResource {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Drop for PinnedResource {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from cuMemHostAlloc and is freed exactly once.
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr.as_ptr() as *mut _);
        }
    }
}

/// A [`TensorAllocator`] that allocates zero-initialised page-locked host
/// memory. Use via [`pinned_alloc`] with the `_in` constructors
/// (`Image::from_size_val_in`, `Tensor::from_shape_vec` etc. take a plain
/// handle) to make a host image whose uploads/downloads are DMA-fast.
#[derive(Clone)]
pub struct PinnedAllocator {
    /// Context that owns the pinned registration.
    pub ctx: Arc<CudaContext>,
}

impl TensorAllocator for PinnedAllocator {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        let n_bytes = layout.size().max(1);
        // SAFETY: flags = 0 → cacheable, portable pinned host memory.
        let ptr = unsafe { cudarc::driver::result::malloc_host(n_bytes, 0) }
            .map_err(|e| TensorAllocatorError::CudaError(e.to_string()))?;
        let ptr = NonNull::new(ptr as *mut u8)
            .ok_or_else(|| TensorAllocatorError::CudaError("null pinned alloc".into()))?;
        // Zero-initialise to match the other allocators' contract.
        // SAFETY: freshly allocated n_bytes at ptr.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, n_bytes) };
        Ok(Box::new(PinnedResource {
            ptr,
            len_bytes: layout.size(),
            _ctx: self.ctx.clone(),
        }))
    }
}

/// Convenience: an [`AllocHandle`] for page-locked host memory on `ctx`.
///
/// Note: the `Vec`-based `_in` constructors (`from_shape_vec_in` etc.) adopt
/// the vec's own heap memory and only *carry* the handle — they do not route
/// the allocation through it. To get storage that actually lives in pinned
/// memory, use [`zeros_pinned`] and fill via `as_slice_mut`.
pub fn pinned_alloc(ctx: &Arc<CudaContext>) -> AllocHandle {
    Arc::new(PinnedAllocator { ctx: ctx.clone() })
}

/// Allocate a zero-initialised **host** tensor in page-locked (pinned) memory.
///
/// The result is an ordinary host tensor (every CPU path works on it), but
/// H2D/D2H copies against it are direct DMA — pair with
/// [`Tensor::to_cuda`] / [`Tensor::to_host_in`] for the fast transfer path.
///
/// # Errors
///
/// Returns [`CudaError::Driver`] on allocation failure.
pub fn zeros_pinned<T, const N: usize>(
    shape: [usize; N],
    ctx: &Arc<CudaContext>,
) -> Result<Tensor<T, N>, CudaError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    let alloc = pinned_alloc(ctx);
    let numel: usize = shape.iter().product();
    let layout =
        std::alloc::Layout::array::<T>(numel).map_err(|e| CudaError::Driver(e.to_string()))?;
    let owner = alloc
        .allocate(layout)
        .map_err(|e| CudaError::Driver(e.to_string()))?;
    let ptr = owner.as_ptr() as *mut T;
    // SAFETY: allocator returned non-null storage of `layout.size()` bytes.
    let nn_ptr = unsafe { NonNull::new_unchecked(ptr) };
    let storage = TensorStorage {
        ptr: nn_ptr,
        len: layout.size(),
        owner,
        alloc,
        _marker: PhantomData,
    };
    let strides = get_strides_from_shape(shape);
    Ok(Tensor {
        storage,
        shape,
        strides,
    })
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

// ── CudaKernel ────────────────────────────────────────────────────────────────

/// A compiled CUDA kernel that wraps a [`CudaFunction`] (which internally keeps
/// its owning `CudaModule` alive).
///
/// Construct via [`CudaKernel::compile`], then launch via
/// [`CudaKernel::launch_builder`].
pub struct CudaKernel {
    /// The loaded kernel. `CudaFunction` internally holds an `Arc<CudaModule>`,
    /// so the module stays alive as long as this kernel exists — no separate
    /// keep-alive field is needed.
    func: CudaFunction,
}

impl CudaKernel {
    /// Compile a CUDA C source string and return a ready-to-launch kernel.
    ///
    /// # Arguments
    ///
    /// * `ctx`     — CUDA context; used to detect compute capability and load the module.
    /// * `src`     — CUDA C (`.cu`) source string.
    /// * `fn_name` — name of the `extern "C" __global__` function to load.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on nvrtc compile failure or module/function
    /// load failure.
    pub fn compile(ctx: &Arc<CudaContext>, src: &str, fn_name: &str) -> Result<Self, CudaError> {
        let module = Self::load_module(ctx, src)?;
        // `load_function` returns a `CudaFunction` that holds its own
        // `Arc<CudaModule>`, so `module` may drop at the end of this scope.
        let func = module
            .load_function(fn_name)
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        Ok(CudaKernel { func })
    }

    /// Compile `src` **once** and load several `extern "C" __global__` functions
    /// from it — for a source file that defines a whole kernel suite (NMS, top-K,
    /// matching, …). Returns one [`CudaKernel`] per name, in order. Avoids paying
    /// the nvrtc compile cost once per function.
    pub fn compile_many(
        ctx: &Arc<CudaContext>,
        src: &str,
        fn_names: &[&str],
    ) -> Result<Vec<Self>, CudaError> {
        let module = Self::load_module(ctx, src)?;
        fn_names
            .iter()
            .map(|name| {
                module
                    .load_function(name)
                    .map(|func| CudaKernel { func })
                    .map_err(|e| CudaError::Driver(e.to_string()))
            })
            .collect()
    }

    /// nvrtc-compile `src` and load the module on `ctx`'s device, with the target
    /// arch (`compute_XY`) auto-detected from the device's compute capability.
    fn load_module(ctx: &Arc<CudaContext>, src: &str) -> Result<Arc<CudaModule>, CudaError> {
        // Detect arch from the context's device.
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        // `CompileOptions.arch` is `Option<&'static str>`.
        // We cache the leaked `&'static str` per (major, minor) compute capability so
        // that at most one string is ever leaked per distinct arch value (typically a
        // handful in the lifetime of a process), rather than one per `compile` call.
        static ARCH_CACHE: OnceLock<Mutex<HashMap<(i32, i32), &'static str>>> = OnceLock::new();
        let arch_str: &'static str = {
            let mut map = ARCH_CACHE
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .expect("ARCH_CACHE mutex poisoned");
            if let Some(&s) = map.get(&(major, minor)) {
                s
            } else {
                let s: &'static str =
                    Box::leak(format!("compute_{}{}", major, minor).into_boxed_str());
                map.insert((major, minor), s);
                s
            }
        };

        let ptx = compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some(arch_str),
                ..Default::default()
            },
        )
        .map_err(|e| CudaError::Driver(format!("{e:?}")))?;

        ctx.load_module(ptx)
            .map_err(|e| CudaError::Driver(e.to_string()))
    }

    /// Create a [`CudaLaunchBuilder`] pre-bound to this kernel and the given stream.
    pub fn launch_builder<'a>(&'a self, stream: &'a Arc<CudaStream>) -> CudaLaunchBuilder<'a> {
        CudaLaunchBuilder {
            inner: stream.launch_builder(&self.func),
        }
    }

    /// Set the L1/shared-memory carveout for this kernel.
    ///
    /// Pass `0` to give L1 the maximum possible space (all shared memory budget
    /// allocated to L1 data cache).  This improves `__ldg` hit rates when the
    /// kernel does not use shared memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] if the driver call fails.
    pub fn prefer_l1_cache(&self) -> Result<(), CudaError> {
        use cudarc::driver::sys::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_L1;
        self.func
            .set_function_cache_config(CU_FUNC_CACHE_PREFER_L1)
            .map_err(|e| CudaError::Driver(e.to_string()))
    }
}

// ── CudaLaunchBuilder ─────────────────────────────────────────────────────────

/// Ergonomic wrapper around cudarc's [`LaunchArgs`] that accumulates kernel
/// arguments and provides a one-liner `launch_1d` helper.
///
/// Obtain via [`CudaKernel::launch_builder`].
///
/// # Example
///
/// ```ignore
/// kernel.launch_builder(&stream)
///     .arg(&input_slice)
///     .arg(&mut output_slice)
///     .arg(&n_i32)
///     .launch_1d(n as u32)?;
/// ```
pub struct CudaLaunchBuilder<'a> {
    inner: LaunchArgs<'a>,
}

impl<'a> CudaLaunchBuilder<'a> {
    /// Push a kernel argument.  `T` must implement [`PushKernelArg`] for
    /// [`LaunchArgs<'a>`], which is satisfied by `&T` (for `T: DeviceRepr`),
    /// `&CudaSlice<T>`, `&mut CudaSlice<T>`, and similar cudarc types.
    pub fn arg<T>(mut self, v: T) -> Self
    where
        LaunchArgs<'a>: PushKernelArg<T>,
    {
        self.inner.arg(v);
        self
    }

    /// Launch the kernel with a 1-D grid sized to cover `n` elements.
    ///
    /// Uses a fixed block size of 256 threads.
    ///
    /// # Safety (internal)
    ///
    /// The `unsafe { self.inner.launch(cfg) }` call is inherently unsafe because
    /// CUDA cannot verify that the accumulated arguments are valid.  This wrapper
    /// provides the ergonomic surface; callers must ensure argument types and
    /// counts match the kernel signature.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA launch failure.
    pub fn launch_1d(self, n: u32) -> Result<(), CudaError> {
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            block_dim: (BLOCK, 1, 1),
            grid_dim: (n.div_ceil(BLOCK), 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch_cfg(cfg)
    }

    /// Launch with an explicit [`LaunchConfig`] — for kernels that need a specific
    /// grid/block shape or shared memory (2-D image kernels, one-block-per-item
    /// cooperative reductions, fixed-block-size tiled kernels) that `launch_1d`'s
    /// flat 256-thread blocks cannot express.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA launch failure.
    pub fn launch_cfg(mut self, cfg: LaunchConfig) -> Result<(), CudaError> {
        // SAFETY: The caller is responsible for ensuring the kernel arguments match
        // the kernel's parameter list in type, count, and alignment.
        unsafe { self.inner.launch(cfg) }
            .map(|_| ()) // discard optional (CudaEvent, CudaEvent) timing pair
            .map_err(|e| CudaError::Driver(e.to_string()))
    }

    /// Launch the kernel with a caller-supplied [`LaunchConfig`].
    ///
    /// Use this when the kernel requires a 2-D or 3-D grid (e.g. image kernels
    /// where `blockIdx.x/y` index the output pixel column/row).
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA launch failure.
    pub fn launch_2d(
        mut self,
        _width: u32,
        _height: u32,
        cfg: LaunchConfig,
    ) -> Result<(), CudaError> {
        // SAFETY: The caller is responsible for ensuring the kernel arguments match
        // the kernel's parameter list in type, count, and alignment.
        unsafe { self.inner.launch(cfg) }
            .map(|_| ())
            .map_err(|e| CudaError::Driver(e.to_string()))
    }
}

// ── zeros_cuda ────────────────────────────────────────────────────────────────

/// Allocate a zero-initialised device tensor with shape `shape` on `stream`.
///
/// The storage is backed by a typed [`CudaResource<T>`], so the resulting tensor's
/// [`as_cudaslice`](Tensor::as_cudaslice) / [`as_cudaslice_mut`](Tensor::as_cudaslice_mut)
/// return `Some(&CudaSlice<T>)` — the tensor can be passed straight as a kernel
/// argument or written in place, with no `from_cudaslice` round-trip.
///
/// # Type parameters
///
/// * `T` — element type; must satisfy [`DeviceRepr`] + [`ValidAsZeroBits`].
/// * `N` — number of dimensions.
///
/// # Errors
///
/// Returns [`CudaError::Driver`] on CUDA allocation failure. A zero-element
/// shape is allowed and yields an empty (0-byte) device tensor.
pub fn zeros_cuda<T, const N: usize>(
    shape: [usize; N],
    stream: &Arc<CudaStream>,
) -> Result<Tensor<T, N>, CudaError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    let numel: usize = shape.iter().product();
    // Zero-initialised, **typed** device memory so the storage is backed by
    // `CudaResource<T>` (not `CudaResource<u8>`) — `as_cudaslice::<T>()` then works.
    let slice: CudaSlice<T> = stream
        .alloc_zeros::<T>(numel)
        .map_err(|e| CudaError::Driver(e.to_string()))?;
    Ok(wrap_device_slice(slice, shape, stream))
}

/// Allocate **uninitialized**, typed device memory backed by `CudaResource<T>` —
/// [`zeros_cuda`] without the zeroing memset.
///
/// This is the fast path for a destination a kernel will fully overwrite: the
/// per-allocation device-wide `memset` that [`zeros_cuda`] pays is pure waste
/// when every output element is written before it is ever read.
///
/// # Safety
///
/// The returned tensor's device memory is uninitialized. The caller MUST fully
/// write every element (e.g. launch a kernel that writes the whole output)
/// before any host or device code reads it. Reading before the overwrite yields
/// indeterminate values. Prefer [`zeros_cuda`] unless the full-overwrite is
/// guaranteed.
pub unsafe fn uninit_cuda<T, const N: usize>(
    shape: [usize; N],
    stream: &Arc<CudaStream>,
) -> Result<Tensor<T, N>, CudaError>
where
    T: DeviceRepr + 'static,
{
    let numel: usize = shape.iter().product();
    // SAFETY: the caller's contract (documented above) guarantees the memory is
    // fully written before it is read; this mirrors `zeros_cuda` minus the memset.
    let slice: CudaSlice<T> = unsafe { stream.alloc::<T>(numel) }
        .map_err(|e| CudaError::Driver(e.to_string()))?;
    Ok(wrap_device_slice(slice, shape, stream))
}

/// Wrap an already-allocated typed device `slice` into a `Tensor<T, N>` backed by
/// a `CudaResource<T>`. The single owner/pointer-caching path shared by
/// [`zeros_cuda`] and [`uninit_cuda`] — they differ only in how `slice` is
/// allocated (zeroed vs uninitialized).
fn wrap_device_slice<T, const N: usize>(
    slice: CudaSlice<T>,
    shape: [usize; N],
    stream: &Arc<CudaStream>,
) -> Tensor<T, N>
where
    T: DeviceRepr + 'static,
{
    let ctx = stream.context().clone();
    let numel: usize = shape.iter().product();
    let n_bytes = numel * std::mem::size_of::<T>();

    // Cache device pointer before moving `slice` into CudaResource.
    let id = ctx.ordinal() as i32;
    let ptr = {
        let (cu_ptr, _sync) = slice.device_ptr(stream);
        cu_ptr as *mut u8
        // _sync drops here
    };

    let resource = CudaResource::<T> {
        slice: std::mem::ManuallyDrop::new(slice),
        ptr,
        id,
        stream: stream.clone(),
        foreign: None,
    };
    let alloc: AllocHandle = Arc::new(CudaAllocator {
        ctx: ctx.clone(),
        stream: stream.clone(),
    });

    // SAFETY: `ptr` is the device pointer inside `resource`; `n_bytes` is the
    // allocation size; `resource` is the sole owner of that device memory.
    let storage = unsafe { storage_from_cuda_resource(resource, ptr as *mut T, n_bytes, alloc) };
    let strides = get_strides_from_shape(shape);
    Tensor {
        storage,
        shape,
        strides,
    }
}

// ── Helper: build a TensorStorage from a CudaResource ─────────────────────────

/// Build a `TensorStorage<T>` that owns the given [`CudaResource<R>`].
///
/// `R` is the element type of the wrapped `CudaSlice` (e.g. `u8` from the allocator,
/// or `T` from `from_cudaslice`); `T` is the tensor's element type.
///
/// # Safety
///
/// `ptr` must be the device pointer inside `resource` (cached from
/// `resource.slice.device_ptr(stream)`).  It must be valid for `len_bytes` bytes on
/// the device.  The caller guarantees `resource` is the sole owner of that allocation.
unsafe fn storage_from_cuda_resource<T, R>(
    resource: CudaResource<R>,
    ptr: *mut T,
    len_bytes: usize,
    alloc: AllocHandle,
) -> TensorStorage<T>
where
    R: 'static,
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

impl<T, const N: usize> Tensor<T, N>
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
        Self::from_cudaslice_impl(slice, shape, stream, None)
    }

    /// Wrap a **foreign** device allocation (e.g. a DLPack producer's memory
    /// aliased via `upgrade_device_ptr`) as a tensor, zero-copy.
    ///
    /// The tensor never frees the memory: on drop the aliasing `CudaSlice` is
    /// leaked and `keepalive` is dropped instead — the keepalive must hold
    /// whatever reference keeps the producer's allocation valid.
    /// [`into_cudaslice`](Self::into_cudaslice) refuses such tensors (the
    /// slice must not outlive the keepalive).
    pub fn from_foreign_cudaslice(
        slice: CudaSlice<T>,
        shape: [usize; N],
        stream: Arc<CudaStream>,
        keepalive: Box<dyn std::any::Any + Send>,
    ) -> Self {
        Self::from_cudaslice_impl(slice, shape, stream, Some(keepalive))
    }

    fn from_cudaslice_impl(
        slice: CudaSlice<T>,
        shape: [usize; N],
        stream: Arc<CudaStream>,
        foreign: Option<Box<dyn std::any::Any + Send>>,
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

        // Cache the raw device pointer of the EXISTING allocation (no copy).
        // _sync must be dropped before we move `slice` into CudaResource.
        let ptr = {
            let (cu_ptr, _sync) = slice.device_ptr(&stream);
            cu_ptr as *mut u8
            // _sync drops here, releasing the borrow on `slice`
        };

        let alloc: AllocHandle = Arc::new(CudaAllocator {
            ctx,
            stream: stream.clone(),
        });
        // Move the original CudaSlice<T> in, unchanged — this is the aliasing wrap.
        let resource = CudaResource::<T> {
            slice: std::mem::ManuallyDrop::new(slice),
            ptr,
            id,
            stream,
            foreign,
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
            .map(|r| &*r.slice)
    }

    /// Return the stream this device tensor's allocation was created on, if the
    /// storage is backed by a [`CudaResource<T>`].
    ///
    /// This is how residency-aware dispatch (e.g. color conversion on device
    /// images) recovers a stream without any global state: the stream travels
    /// inside the tensor. Returns `None` for host tensors or element-type
    /// mismatches (same rules as [`as_cudaslice`](Self::as_cudaslice)).
    pub fn cuda_stream(&self) -> Option<&Arc<CudaStream>> {
        self.storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .map(|r| &r.stream)
    }

    /// Mutably borrow the underlying `CudaSlice<T>` if the storage is backed by a
    /// [`CudaResource<T>`].
    ///
    /// This is the mutable sibling of [`as_cudaslice`](Self::as_cudaslice); it lets a
    /// device-owning output tensor be passed as a mutable kernel argument
    /// (e.g. `kernel.launch_builder(&stream).arg(out.as_cudaslice_mut().unwrap())`).
    ///
    /// Returns `None` if the storage is not a `CudaResource<T>` (e.g. host-backed, or T differs).
    pub fn as_cudaslice_mut(&mut self) -> Option<&mut CudaSlice<T>> {
        self.storage
            .owner
            .as_any_mut()
            .downcast_mut::<CudaResource<T>>()
            .map(|r| &mut *r.slice)
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
        match self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
        {
            // Foreign memory: handing out an owned CudaSlice would let it
            // outlive the keepalive (and its drop would cuMemFree memory we
            // do not own).
            Some(r) if r.foreign.is_some() => return Err(self),
            Some(_) => {}
            None => return Err(self),
        }

        // Consume `self` without running TensorStorage's Drop or CudaResource's Drop.
        let storage = self.storage;
        let md_storage = std::mem::ManuallyDrop::new(storage);

        // Read `owner` out of the ManuallyDrop without running TensorStorage's Drop.
        // SAFETY: We are the sole owner; ManuallyDrop prevents double-drop of the storage.
        let owner: Box<dyn MemoryResource> = unsafe { std::ptr::read(&md_storage.owner) };
        // Drop the allocator handle (Arc) explicitly.
        let _alloc = unsafe { std::ptr::read(&md_storage.alloc) };
        // ptr/len/marker are Copy/ZST — nothing else to drop.

        // Downcast Box<dyn MemoryResource> → Box<CudaResource<T>>.
        // SAFETY: We verified above (downcast_ref) that the concrete type is CudaResource<T>.
        let raw: *mut dyn MemoryResource = Box::into_raw(owner);
        let cuda_box: Box<CudaResource<T>> = unsafe { Box::from_raw(raw as *mut CudaResource<T>) };

        // Move CudaSlice<T> out without running CudaResource's Drop.
        let mut md_res = std::mem::ManuallyDrop::new(*cuda_box);
        // SAFETY: md_res prevents CudaResource from dropping its fields; we take the slice.
        let slice = std::mem::ManuallyDrop::into_inner(unsafe { std::ptr::read(&md_res.slice) });
        // Drop the stream Arc explicitly — ManuallyDrop would otherwise leak it.
        let _stream = unsafe { std::ptr::read(&md_res.stream) };
        // `foreign` is None (checked above) but must still be read out so the
        // Option itself is not leaked.
        let _foreign = unsafe { std::ptr::read(&md_res.foreign) };
        // Poison the ptr to make residual state obviously invalid (defensive).
        md_res.ptr = std::ptr::null_mut();

        Ok(slice)
    }
}

// ── Tensor::to_cuda (host → device) ──────────────────────────────────────────

impl<T, const N: usize> Tensor<T, N>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    /// Copy this host tensor to a new device-backed tensor on `stream`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA failure.
    pub fn to_cuda(&self, stream: &Arc<CudaStream>) -> Result<Tensor<T, N>, CudaError> {
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
        let alloc: AllocHandle = Arc::new(CudaAllocator {
            ctx,
            stream: stream.clone(),
        });
        // Store as CudaResource<T> so as_cudaslice::<T>() also works on to_cuda results.
        let resource = CudaResource::<T> {
            slice: std::mem::ManuallyDrop::new(dev_slice),
            ptr,
            id,
            stream: stream.clone(),
            foreign: None,
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

    /// Copy this device tensor to a new host-backed tensor.
    ///
    /// Synchronizes the stream before returning so the host data is valid.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA failure or [`CudaError::NotCudaBacked`]
    /// if the storage owner is not a [`CudaResource<T>`].
    pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<Tensor<T, N>, CudaError> {
        let cuda_res = self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .ok_or(CudaError::NotCudaBacked)?;

        // D→H typed copy into a Vec<T> (this is a transfer, copy is correct).
        let host_data: Vec<T> = stream
            .clone_dtoh(&*cuda_res.slice)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        let storage = TensorStorage::from_vec(host_data, host_alloc());
        let strides = get_strides_from_shape(self.shape);
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides,
        })
    }

    /// Copy this device tensor to a new host tensor using the tensor's **own
    /// carried stream** — the one it was allocated/uploaded on — removing the
    /// footgun of passing a different stream than the data's producer (a
    /// read-before-write hazard). Synchronizes before returning.
    ///
    /// # Errors
    ///
    /// [`CudaError::NotCudaBacked`] if the tensor is not device-backed by a
    /// typed [`CudaResource<T>`], or [`CudaError::Driver`] on CUDA failure.
    pub fn download(&self) -> Result<Tensor<T, N>, CudaError> {
        let stream = self.cuda_stream().ok_or(CudaError::NotCudaBacked)?.clone();
        self.to_host(&stream)
    }

    /// D2H-copy this device tensor directly into a caller-provided host slice,
    /// using the tensor's own carried stream and synchronizing before returning.
    ///
    /// Unlike [`Self::download`] / [`Self::to_host`] (which allocate a fresh
    /// `Vec<T>` destination), this targets memory the caller already owns — e.g.
    /// an aligned buffer that will back a numpy view — so no second host copy is
    /// needed to move the pixels into their final home.
    ///
    /// # Errors
    ///
    /// [`CudaError::NotCudaBacked`] if the tensor is not device-backed by a typed
    /// [`CudaResource<T>`], or [`CudaError::Driver`] on a `dst`-length mismatch or
    /// CUDA failure.
    pub fn download_into(&self, dst: &mut [T]) -> Result<(), CudaError> {
        let numel: usize = self.shape.iter().product();
        if dst.len() != numel {
            return Err(CudaError::Driver(format!(
                "download_into: dst len {} != tensor element count {numel}",
                dst.len()
            )));
        }
        let cuda_res = self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .ok_or(CudaError::NotCudaBacked)?;
        let stream = self.cuda_stream().ok_or(CudaError::NotCudaBacked)?;
        stream
            .memcpy_dtoh(&*cuda_res.slice, dst)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        Ok(())
    }

    /// Copy this device tensor to a new host tensor whose storage comes from
    /// `alloc` — e.g. [`pinned_alloc`] for a page-locked destination, which
    /// lets the driver DMA the D2H copy directly instead of staging it.
    ///
    /// Synchronizes the stream before returning so the host data is valid.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::Driver`] on CUDA/allocation failure or
    /// [`CudaError::NotCudaBacked`] if the storage owner is not a
    /// [`CudaResource<T>`].
    pub fn to_host_in(
        &self,
        stream: &Arc<CudaStream>,
        alloc: AllocHandle,
    ) -> Result<Tensor<T, N>, CudaError> {
        let cuda_res = self
            .storage
            .owner
            .as_any()
            .downcast_ref::<CudaResource<T>>()
            .ok_or(CudaError::NotCudaBacked)?;

        let numel: usize = self.shape.iter().product();
        let layout =
            std::alloc::Layout::array::<T>(numel).map_err(|e| CudaError::Driver(e.to_string()))?;
        let owner = alloc
            .allocate(layout)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        let ptr = owner.as_ptr() as *mut T;

        // SAFETY: `owner` holds at least `numel * size_of::<T>()` bytes at `ptr`.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, numel) };
        stream
            .memcpy_dtoh(&*cuda_res.slice, dst)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        // SAFETY: ptr is non-null (checked by the allocator) and owned by `owner`.
        let nn_ptr = unsafe { NonNull::new_unchecked(ptr) };
        let storage = TensorStorage {
            ptr: nn_ptr,
            len: layout.size(),
            owner,
            alloc,
            _marker: PhantomData,
        };
        let strides = get_strides_from_shape(self.shape);
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::Tensor;

    /// Device round-trip: host → GPU → host, bytes must match exactly.
    /// Also verifies domain is Device and as_cudaslice is Some.
    #[test]
    fn cuda_roundtrip_and_as_slice_panics() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let host = Tensor::<u8, 1>::from_shape_vec([4], vec![1, 2, 3, 4]).unwrap();

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

    /// `download_into` D2H-copies into a caller-provided slice (no intermediate
    /// Vec) and rejects a length mismatch.
    #[test]
    fn download_into_matches_and_checks_len() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let host = Tensor::<u8, 1>::from_shape_vec([4], vec![10, 20, 30, 40]).unwrap();
        let dev = host.to_cuda(&stream).unwrap();

        let mut dst = vec![0u8; 4];
        dev.download_into(&mut dst).unwrap();
        assert_eq!(dst, vec![10, 20, 30, 40], "download_into bytes must match");

        let mut wrong = vec![0u8; 3];
        assert!(
            dev.download_into(&mut wrong).is_err(),
            "a dst-length mismatch must be rejected"
        );
    }

    /// `uninit_cuda` allocates a typed device tensor that is correct once fully
    /// overwritten (the full-overwrite safety contract).
    #[test]
    fn uninit_cuda_full_overwrite_roundtrip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // SAFETY: the tensor is fully overwritten via memcpy_htod before any read.
        let mut dev = unsafe { uninit_cuda::<u8, 1>([4], &stream) }.unwrap();
        assert!(dev.as_cudaslice().is_some());

        let slice = dev.as_cudaslice_mut().unwrap();
        stream.memcpy_htod(&[7u8, 8, 9, 11], slice).unwrap();

        let back = dev.download().unwrap();
        assert_eq!(back.as_slice(), &[7u8, 8, 9, 11]);
    }

    /// `as_cudaslice_mut` returns `Some` for a device tensor (CudaResource<T>-backed) and
    /// `None` when the concrete element type does not match the stored `CudaResource`.
    /// Mutating through the returned `&mut CudaSlice` must be observable after `to_host`.
    #[test]
    fn as_cudaslice_mut_some_for_device_none_for_mismatch() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // Device tensor backed by CudaResource<u8> (via to_cuda).
        let host = Tensor::<u8, 1>::from_shape_vec([4], vec![1, 2, 3, 4]).unwrap();
        let mut dev = host.to_cuda(&stream).unwrap();

        // Some: same element type u8.
        assert!(
            dev.as_cudaslice_mut().is_some(),
            "as_cudaslice_mut should return Some for a CudaResource<u8>-backed device tensor"
        );

        // Mutate the whole slice on-device through the &mut CudaSlice, then verify D2H.
        {
            let slice = dev.as_cudaslice_mut().expect("must be device-backed");
            stream.memset_zeros(slice).unwrap();
        }
        let back = dev.to_host(&stream).unwrap();
        assert_eq!(
            back.as_slice(),
            &[0u8, 0, 0, 0],
            "mutation through as_cudaslice_mut must be observable after to_host"
        );

        // `zeros_cuda` stores a TYPED `CudaResource<i16>`, so an i16 device tensor's
        // owner downcasts to `CudaResource<i16>` — the mutable query is Some.
        let mut dev_i16: Tensor<i16, 1> = zeros_cuda([4], &stream).unwrap();
        assert!(
            dev_i16.as_cudaslice_mut().is_some(),
            "zeros_cuda stores CudaResource<i16>; querying as CudaSlice<i16> must be Some"
        );
    }

    /// `as_slice()` on a device tensor must panic with "non-host-accessible".
    #[test]
    #[should_panic(expected = "non-host-accessible")]
    fn as_slice_on_device_panics() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let host = Tensor::<u8, 1>::from_shape_vec([4], vec![1, 2, 3, 4]).unwrap();
        let dev = host.to_cuda(&stream).unwrap();
        let _ = dev.as_slice(); // must panic: Device is not host-accessible
    }

    /// `into_cudaslice` returns the slice; no double-free.
    /// A subsequent alloc must succeed (proves no CUDA error state after the free).
    #[test]
    fn into_cudaslice_no_double_free() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let host = Tensor::<u8, 1>::from_shape_vec([8], vec![10u8; 8]).unwrap();
        let dev = host.to_cuda(&stream).unwrap();
        let slice = dev.into_cudaslice().ok().expect("must be cuda-backed");
        // The slice now owns the device memory.  Drop it — cudarc frees exactly once.
        drop(slice);
        // A subsequent allocation must succeed (no CUDA error state after the free).
        let _s2 = stream
            .alloc_zeros::<u8>(8)
            .expect("second alloc must succeed after into_cudaslice+drop");
    }

    /// Foreign tensors never free the producer's memory: the owner keeps its
    /// allocation valid, `into_cudaslice` refuses, and the keepalive is
    /// released exactly once on drop.
    #[test]
    fn from_foreign_cudaslice_leaks_slice_releases_keepalive() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        struct Keepalive(StdArc<AtomicUsize>);
        impl Drop for Keepalive {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // "Producer" allocation that must survive the tensor.
        let producer: CudaSlice<u8> = stream.alloc_zeros::<u8>(32).unwrap();
        let (cu_ptr, _sync) = producer.device_ptr(&stream);
        drop(_sync);
        // Alias it the way a DLPack import does.
        let alias = unsafe { stream.upgrade_device_ptr::<u8>(cu_ptr, 32) };

        let released = StdArc::new(AtomicUsize::new(0));
        let tensor = Tensor::<u8, 1>::from_foreign_cudaslice(
            alias,
            [32],
            stream.clone(),
            Box::new(Keepalive(released.clone())),
        );
        assert!(tensor.as_cudaslice().is_some());

        // Foreign tensors must not hand out an owned slice.
        let Err(tensor) = tensor.into_cudaslice() else {
            panic!("foreign must refuse into_cudaslice");
        };

        drop(tensor);
        assert_eq!(
            released.load(Ordering::SeqCst),
            1,
            "keepalive released once"
        );

        // Producer memory must still be usable (no cuMemFree ran on it).
        let host: Vec<u8> = stream.clone_dtoh(&producer).unwrap();
        assert_eq!(host, vec![0u8; 32]);
    }

    /// `from_cudaslice` → drop: device memory freed once, no double-free.
    #[test]
    fn from_cudaslice_drop_once() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // Allocate on device directly via cudarc.
        let dev_slice: CudaSlice<u8> = stream.alloc_zeros::<u8>(16).unwrap();
        let tensor = Tensor::<u8, 1>::from_cudaslice(dev_slice, [16], stream.clone());
        // Drop tensor → TensorStorage drops → Box<CudaResource> drops → CudaSlice::drop
        // → cudarc frees the device memory exactly once.
        drop(tensor);
        // A fresh allocation must succeed (no CUDA error after the free).
        let _verify = stream
            .alloc_zeros::<u8>(16)
            .expect("allocation after from_cudaslice+drop must succeed");
    }

    /// Compile a trivial `copy_bytes` kernel, launch it via `CudaKernel` + `CudaLaunchBuilder`,
    /// and verify that the output slice matches the input.
    #[test]
    fn cuda_kernel_compile_and_launch() {
        const COPY_BYTES_SRC: &str = r#"
            extern "C" __global__ void copy_bytes(
                const unsigned char* __restrict__ src,
                unsigned char* __restrict__ dst,
                int n)
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) { dst[i] = src[i]; }
            }
        "#;

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let kernel = CudaKernel::compile(&ctx, COPY_BYTES_SRC, "copy_bytes")
            .expect("kernel compile must succeed");

        const N: usize = 16;
        let input_data: Vec<u8> = (0u8..N as u8).collect();
        let input: CudaSlice<u8> = stream.clone_htod(&input_data).unwrap();
        let mut output: CudaSlice<u8> = stream.alloc_zeros::<u8>(N).unwrap();

        let n_i32: i32 = N as i32;
        kernel
            .launch_builder(&stream)
            .arg(&input)
            .arg(&mut output)
            .arg(&n_i32)
            .launch_1d(N as u32)
            .expect("kernel launch must succeed");

        let result: Vec<u8> = stream.clone_dtoh(&output).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            result, input_data,
            "copy_bytes kernel output must match input"
        );
    }

    /// Pinned round-trip: pinned host tensor → device → pinned host via
    /// `to_host_in(pinned_alloc)`; bytes must match and CPU access must work.
    #[test]
    fn pinned_alloc_roundtrip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // Host tensor in pinned memory — behaves like any host tensor.
        let data: Vec<u8> = (0u8..64).collect();
        let mut host = zeros_pinned::<u8, 1>([64], &ctx).unwrap();
        host.as_slice_mut().copy_from_slice(&data);
        assert_eq!(
            host.as_slice(),
            &data[..],
            "pinned tensor is host-accessible"
        );
        assert!(
            matches!(host.storage.domain(), MemoryDomain::Host),
            "pinned memory is host-domain"
        );

        // Upload (driver DMA path), download into a pinned destination.
        let dev = host.to_cuda(&stream).unwrap();
        let back = dev.to_host_in(&stream, pinned_alloc(&ctx)).unwrap();
        assert_eq!(back.as_slice(), &data[..], "pinned round-trip must match");
    }

    /// `zeros_cuda` allocates a device tensor; `to_host` must return all-zero bytes.
    #[test]
    fn zeros_cuda_test() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let dev = zeros_cuda::<u8, 1>([8], &stream).expect("zeros_cuda must succeed");
        let host = dev.to_host(&stream).expect("to_host must succeed");
        assert_eq!(
            host.as_slice(),
            &[0u8; 8],
            "zeros_cuda result must be all zeros"
        );
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

        let tensor = Tensor::<u8, 1>::from_cudaslice(dev_slice, [32], stream.clone());

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
