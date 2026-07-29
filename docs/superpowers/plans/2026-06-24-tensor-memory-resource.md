# Tensor MemoryResource Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `kornia-tensor`'s four-field ownership model with one owning `MemoryResource` handle + a three-state `MemoryDomain`, so a tensor's buffer can be kornia-host, foreign-host (numpy/gstreamer/v4l/dlpack), or device (cudarc), uniformly and memory-safely, with natural cudarc integration and a backend-agnostic core.

**Architecture:** `TensorStorage` holds a cached hot-path `ptr` plus a single `owner: Box<dyn MemoryResource>` that frees correctly on `Drop` (collapsing today's `owns_memory`/`keepalive`/`dealloc`-by-ptr). `TensorAllocator::allocate` returns a `Box<dyn MemoryResource>`; `from_*`/`wrap_*` constructors build foreign resources. `MemoryDomain{Host, Device{id}, Unified{id}}` is the accessibility axis and gates `as_slice`. cudarc/gstreamer/v4l are each a feature-gated `MemoryResource` impl; core names none of them.

**Tech Stack:** Rust, `kornia-tensor`/`kornia-image`/`kornia-imgproc`/`kornia-io`/`kornia-py`, cudarc 0.19 (CUDA 12.6, Jetson Orin sm_87, nvrtc), gstreamer-rs, miri.

## Global Constraints

- Branch: `feat/tensor-memory-resource` off `feat/gray-neon-kernels` (#944). Do NOT branch off main.
- Backend-agnostic core: `kornia-tensor` core + `kornia-image`/`imgproc`/`py` name NO GPU/streaming runtime. cudarc/cubecl/gstreamer appear ONLY behind optional features. Default build pulls none — verify `cargo tree -p kornia-tensor -e features | grep -i cudarc` is empty.
- Behavior-preserving: public accessor signatures (`as_ptr`/`as_mut_ptr`/`as_slice`/`as_mut_slice`/`domain`/`device_id`/`len`/`is_empty`/`layout`/`alloc`/`from_vec`/`into_vec`) keep working. The full existing Rust test suite + kornia-py pytest is the gate (only the 3 known pre-existing failures allowed: 2 flaky cv2-perf benches + apriltag fixture skip).
- Memory safety paramount: a memory-safety review + miri gate Waves A, B, C. Unifying ownership must REDUCE UB surface.
- No markdown docs committed to a PR.
- Naming: `<output>_from_<input>`; new public Rust fns return named structs.
- cudarc/gstreamer tests run on this Jetson (CUDA 12.6 + libnvrtc present; gstreamer installed).

---

## File Structure

- `crates/kornia-tensor/src/resource.rs` **(new)** — `MemoryResource` trait, `HostResource`, `ForeignResource`. One responsibility: ownership/release handles.
- `crates/kornia-tensor/src/storage.rs` — `MemoryDomain` → 3-state; `TensorStorage` holds `owner: Box<dyn MemoryResource>`; constructors + `Drop` + `as_slice` gating.
- `crates/kornia-tensor/src/allocator.rs` — `TensorAllocator::allocate(layout) -> Box<dyn MemoryResource>`; migrate `CpuAllocator`/`AlignedCpuAllocator`/`ForeignAllocator`.
- `crates/kornia-tensor/src/cuda.rs` **(new, feature `cudarc`)** — `CudaResource`, `CudaAllocator`, `Tensor::{from_cudaslice,as_cudaslice,into_cudaslice,to_cuda,to_host}`.
- `crates/kornia-tensor/src/lib.rs` — module decls + re-exports.
- `crates/kornia-tensor/Cargo.toml` — `cudarc` feature.
- `crates/kornia-image/src/dlpack.rs`, `crates/kornia-image/src/arrow.rs`, `crates/kornia-image/src/image.rs` — migrate direct storage construction to new constructors (accessor-stable).
- `crates/kornia-imgproc/src/{flip.rs,warp/kernels.rs,gpu/color.rs}` — same.
- `kornia-py/src/pipeline.rs` — same.
- `crates/kornia-imgproc/examples/cuda_stream_imgproc.rs` **(new, feature `cuda-example`)** — device-owning Tensor demo.
- `crates/kornia-io/src/gstreamer/mod.rs`, `crates/kornia-io/src/gstreamer/capture.rs` — `GstResource` replaces `GstAllocator` hack.
- `crates/kornia-io/src/v4l/stream.rs` — `V4lResource` replaces the mmap-guard-as-allocator.

---

## Wave A — Core MemoryResource model (kornia-tensor), behavior-preserving

### Task 1: `MemoryResource` trait + `HostResource` + `ForeignResource` + 3-state `MemoryDomain`

**Files:**
- Create: `crates/kornia-tensor/src/resource.rs`
- Modify: `crates/kornia-tensor/src/storage.rs:9-15` (MemoryDomain), `crates/kornia-tensor/src/lib.rs` (module + re-exports)
- Test: in `resource.rs` `#[cfg(test)]`

**Interfaces:**
- Produces:
  - `pub enum MemoryDomain { Host, Device { id: i32 }, Unified { id: i32 } }` with `pub fn is_host_accessible(&self) -> bool` (Host|Unified), `pub fn is_device_accessible(&self) -> bool` (Device|Unified), `pub fn device_id(&self) -> i32` (0 for Host).
  - `pub trait MemoryResource: Send + Sync { fn as_ptr(&self) -> *mut u8; fn len_bytes(&self) -> usize; fn domain(&self) -> MemoryDomain; fn as_any(&self) -> &dyn core::any::Any; }`
  - `pub struct HostResource { ptr: NonNull<u8>, layout: Layout }` + `unsafe fn from_layout(layout) -> Result<Self, TensorAllocatorError>` (alloc_zeroed) and `unsafe fn from_raw(ptr: *mut u8, layout: Layout) -> Self` (owns it). `Drop` calls `std::alloc::dealloc(ptr, layout)`.
  - `pub struct ForeignResource { ptr: NonNull<u8>, len_bytes: usize, domain: MemoryDomain, _keep: Option<Arc<dyn Any + Send + Sync>> }` + `unsafe fn new(ptr, len_bytes, domain, keep) -> Self`. `Drop` is a no-op for the bytes (only `_keep` drops).

- [ ] **Step 1: Failing test — MemoryDomain accessibility**

In `crates/kornia-tensor/src/resource.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn domain_accessibility() {
        assert!(MemoryDomain::Host.is_host_accessible());
        assert!(!MemoryDomain::Host.is_device_accessible());
        assert!(!MemoryDomain::Device { id: 1 }.is_host_accessible());
        assert!(MemoryDomain::Device { id: 1 }.is_device_accessible());
        assert!(MemoryDomain::Unified { id: 0 }.is_host_accessible());
        assert!(MemoryDomain::Unified { id: 0 }.is_device_accessible());
        assert_eq!(MemoryDomain::Device { id: 3 }.device_id(), 3);
        assert_eq!(MemoryDomain::Host.device_id(), 0);
    }
}
```

- [ ] **Step 2: Run, verify fails to compile** — `cargo test -p kornia-tensor resource:: 2>&1 | tail` → FAIL (types not defined).

- [ ] **Step 3: Implement `resource.rs`**

```rust
//! Ownership handles for a tensor's backing memory (host or device).
use std::{alloc::Layout, any::Any, ptr::NonNull, sync::Arc};
use crate::allocator::TensorAllocatorError;

/// Where a tensor's buffer can be legally dereferenced (the accessibility axis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryDomain {
    /// Host (CPU) memory; slice access is safe.
    Host,
    /// Device memory; host slice access is unsound.
    Device { id: i32 },
    /// Host- AND device-accessible (pinned / managed / unified / NVMM-mapped).
    Unified { id: i32 },
}

impl MemoryDomain {
    /// True when the pointer may be dereferenced on the host (slice APIs).
    pub fn is_host_accessible(&self) -> bool {
        matches!(self, MemoryDomain::Host | MemoryDomain::Unified { .. })
    }
    /// True when the pointer may be passed to a device kernel.
    pub fn is_device_accessible(&self) -> bool {
        matches!(self, MemoryDomain::Device { .. } | MemoryDomain::Unified { .. })
    }
    /// CUDA device id (0 for host).
    pub fn device_id(&self) -> i32 {
        match self {
            MemoryDomain::Host => 0,
            MemoryDomain::Device { id } | MemoryDomain::Unified { id } => *id,
        }
    }
}

/// An owning handle to a tensor's backing memory. Frees correctly on `Drop`.
///
/// # Safety
/// Implementors must guarantee `as_ptr()` is valid for `len_bytes()` for the
/// lifetime of the resource, and that `Drop` releases the memory exactly once.
pub trait MemoryResource: Send + Sync {
    /// Base pointer (host- or device-addressable per `domain`).
    fn as_ptr(&self) -> *mut u8;
    /// Length of the backing buffer in bytes.
    fn len_bytes(&self) -> usize;
    /// Accessibility of the backing buffer.
    fn domain(&self) -> MemoryDomain;
    /// Downcast hook (e.g. recover a `&CudaSlice`).
    fn as_any(&self) -> &dyn Any;
}

/// Host memory owned by kornia (allocated here, freed here).
pub struct HostResource {
    ptr: NonNull<u8>,
    layout: Layout,
}
impl HostResource {
    /// Allocate a zeroed host buffer of `layout`.
    pub fn from_layout(layout: Layout) -> Result<Self, TensorAllocatorError> {
        if layout.size() == 0 {
            return Ok(Self { ptr: NonNull::dangling(), layout });
        }
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(TensorAllocatorError::NullPointer)?;
        Ok(Self { ptr, layout })
    }
    /// Adopt a host pointer previously allocated with `layout`'s global allocator.
    /// # Safety: `ptr` must come from the global allocator with this exact `layout`.
    pub unsafe fn from_raw(ptr: *mut u8, layout: Layout) -> Result<Self, TensorAllocatorError> {
        Ok(Self { ptr: NonNull::new(ptr).ok_or(TensorAllocatorError::NullPointer)?, layout })
    }
}
impl MemoryResource for HostResource {
    fn as_ptr(&self) -> *mut u8 { self.ptr.as_ptr() }
    fn len_bytes(&self) -> usize { self.layout.size() }
    fn domain(&self) -> MemoryDomain { MemoryDomain::Host }
    fn as_any(&self) -> &dyn Any { self }
}
impl Drop for HostResource {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
        }
    }
}
// SAFETY: the pointer is uniquely owned; no interior mutability shared across threads.
unsafe impl Send for HostResource {}
unsafe impl Sync for HostResource {}

/// Foreign memory kornia does NOT own: numpy/gstreamer/v4l/dlpack/cudarc-wrap.
/// `Drop` frees nothing itself — it only drops `_keep`, whose own `Drop` releases
/// the source (decref the PyObject, unmap the gst buffer, free the CudaSlice, ...).
pub struct ForeignResource {
    ptr: NonNull<u8>,
    len_bytes: usize,
    domain: MemoryDomain,
    _keep: Option<Arc<dyn Any + Send + Sync>>,
}
impl ForeignResource {
    /// # Safety: `ptr` valid for `len_bytes` as long as `keep` is alive; `keep`
    /// must own (and on its `Drop` release) the underlying allocation.
    pub unsafe fn new(
        ptr: *mut u8,
        len_bytes: usize,
        domain: MemoryDomain,
        keep: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Self, TensorAllocatorError> {
        Ok(Self { ptr: NonNull::new(ptr).ok_or(TensorAllocatorError::NullPointer)?, len_bytes, domain, _keep: keep })
    }
}
impl MemoryResource for ForeignResource {
    fn as_ptr(&self) -> *mut u8 { self.ptr.as_ptr() }
    fn len_bytes(&self) -> usize { self.len_bytes }
    fn domain(&self) -> MemoryDomain { self.domain }
    fn as_any(&self) -> &dyn Any { self }
}
unsafe impl Send for ForeignResource {}
unsafe impl Sync for ForeignResource {}
```
Add `TensorAllocatorError::NullPointer` if missing (check `allocator.rs`; add a variant `#[error("null pointer")] NullPointer`). Replace `MemoryDomain` in `storage.rs:9-15` with `pub use crate::resource::MemoryDomain;` (delete the old enum there). Add to `lib.rs`: `pub mod resource;` and `pub use resource::{MemoryResource, MemoryDomain, HostResource, ForeignResource};` (keep the existing `pub use crate::storage::MemoryDomain;` working — re-export from resource instead).

- [ ] **Step 4: Run tests** — `cargo test -p kornia-tensor resource:: 2>&1 | tail` → PASS. Then `cargo build -p kornia-tensor` (storage.rs still references the old 2-variant `MemoryDomain` in matches — fix those compile errors minimally by mapping `MemoryDomain::Device` → `MemoryDomain::Device { id: self.device_id }` at the existing construction sites; this is transitional and fully replaced in Task 3).

- [ ] **Step 5: Commit** — `git add crates/kornia-tensor/src/resource.rs crates/kornia-tensor/src/lib.rs crates/kornia-tensor/src/storage.rs crates/kornia-tensor/src/allocator.rs && git commit -m "feat(kornia-tensor): MemoryResource trait + Host/Foreign resources + 3-state MemoryDomain"`

---

### Task 2: `TensorAllocator::allocate -> Box<dyn MemoryResource>`

**Files:**
- Modify: `crates/kornia-tensor/src/allocator.rs` (trait + `CpuAllocator`, `AlignedCpuAllocator`, `ForeignAllocator`)
- Test: in `allocator.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `MemoryResource`, `HostResource` (Task 1).
- Produces: `trait TensorAllocator: Clone + Send + Sync { fn allocate(&self, layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError>; }`. `CpuAllocator`/`AlignedCpuAllocator::allocate` return `Box<HostResource>` (Aligned forces 64-byte align). `ForeignAllocator::allocate` returns `Err(TensorAllocatorError::CannotAllocateForeign)` (it never allocates; it only exists as a type tag for foreign tensors).

- [ ] **Step 1: Failing test**
```rust
#[test]
fn cpu_allocate_zeroed_and_aligned() {
    let l = Layout::from_size_align(64, 1).unwrap();
    let r = CpuAllocator.allocate(l).unwrap();
    assert_eq!(r.len_bytes(), 64);
    assert!(r.domain().is_host_accessible());
    unsafe { assert!((0..64).all(|i| *r.as_ptr().add(i) == 0)); }

    let la = Layout::from_size_align(100, 1).unwrap();
    let ra = AlignedCpuAllocator.allocate(la).unwrap();
    assert_eq!(ra.as_ptr() as usize % 64, 0);
}
```
- [ ] **Step 2: Run, verify fails** — `cargo test -p kornia-tensor allocator:: 2>&1 | tail` → FAIL.
- [ ] **Step 3: Implement** — replace the `alloc`/`dealloc` methods on the trait with `allocate` (keep the trait `Clone + Send + Sync`). `CpuAllocator::allocate` = `Ok(Box::new(HostResource::from_layout(layout)?))`. `AlignedCpuAllocator::allocate` = build `Layout::from_size_align(layout.size(), 64).expect("64 valid")` then `HostResource::from_layout`. `ForeignAllocator::allocate` = `Err(TensorAllocatorError::CannotAllocateForeign)` (add the error variant). Remove the now-unused `dealloc` impls. Keep the allocator unit structs + `Clone`.
- [ ] **Step 4: Run tests** — `cargo test -p kornia-tensor allocator:: 2>&1 | tail` → PASS. (`storage.rs` will not compile yet — that's Task 3.)
- [ ] **Step 5: Commit** — `git add crates/kornia-tensor/src/allocator.rs && git commit -m "feat(kornia-tensor): TensorAllocator::allocate returns a MemoryResource"`

---

### Task 3: Refactor `TensorStorage` onto a single `owner: Box<dyn MemoryResource>`

**Files:**
- Modify: `crates/kornia-tensor/src/storage.rs` (struct, all constructors, `Drop`, `as_slice`/`as_mut_slice`, `domain`/`device_id`, `into_vec`)
- Test: extend `storage.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `TensorAllocator::allocate`, `MemoryResource`, `HostResource`, `ForeignResource`, `MemoryDomain` (Tasks 1-2).
- Produces (signatures preserved unless noted):
  - struct `TensorStorage<T, A> { ptr: NonNull<T>, len: usize, owner: Box<dyn MemoryResource>, alloc: A, _marker: PhantomData<T> }` (removes `layout`, `owns_memory`, `keepalive`, `domain`, `device_id` fields — `domain`/`device_id`/`layout` now derive from `owner`).
  - `from_vec(Vec<T>, A) -> Self` (unchanged signature; leaks the Vec into a `HostResource::from_raw`).
  - `unsafe from_raw_parts(*const T, len, A) -> Self`, `unsafe from_raw_host(...)`, `unsafe from_raw_device(...)`, `unsafe from_borrowed(...)` — keep signatures; build the right resource. `from_raw_device(ptr,len,alloc,device_id)` → `ForeignResource::new(.., Device{id:device_id}, None)`. `from_borrowed(ptr,len,alloc,domain,device_id,keepalive)` → `ForeignResource::new(.., domain-with-id, keepalive)`.
  - `domain(&self) -> MemoryDomain` = `self.owner.domain()`; `device_id(&self)` = `self.owner.domain().device_id()`; `layout(&self)` rebuilt from `len`+`size_of::<T>()`+align_of (or `Layout::from_size_align(self.owner.len_bytes(), align_of::<T>())`).
  - `as_slice`/`as_mut_slice` panic unless `self.owner.domain().is_host_accessible()` (message: `"as_slice on non-host-accessible memory (domain={:?})"`).
  - `into_vec(self) -> Vec<T>` keeps the existing `assert!` that it is host-owned; assert `owner.as_any().is::<HostResource>()` (foreign/device → panic, as today).

- [ ] **Step 1: Failing tests** (add to storage tests)
```rust
#[test]
fn from_vec_roundtrip_and_host_domain() {
    let s = TensorStorage::from_vec(vec![1u8, 2, 3, 4], CpuAllocator);
    assert_eq!(s.as_slice(), &[1, 2, 3, 4]);
    assert!(matches!(s.domain(), MemoryDomain::Host));
    assert_eq!(s.into_vec(), vec![1, 2, 3, 4]);
}
#[test]
#[should_panic(expected = "non-host-accessible")]
fn device_storage_as_slice_panics() {
    // synthetic device storage via from_raw_device
    let buf = vec![0u8; 16];
    let s = unsafe {
        TensorStorage::<u8, CpuAllocator>::from_raw_device(buf.as_ptr(), 16, CpuAllocator, 0)
    };
    let _ = s.as_slice();   // must panic: Device is not host-accessible
    std::mem::forget(buf);
}
#[test]
fn borrowed_keepalive_drops_once() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
    let n = Arc::new(AtomicUsize::new(0));
    let buf = vec![7u8; 8];
    {
        let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(Guard(n.clone()));
        let s = unsafe {
            TensorStorage::<u8, ForeignAllocator>::from_borrowed(
                buf.as_ptr(), 8, ForeignAllocator, MemoryDomain::Host, 0, keep)
        };
        assert_eq!(s.as_slice(), &[7u8; 8]);
    }
    assert_eq!(n.load(Ordering::SeqCst), 1); // guard dropped exactly once
}
```
- [ ] **Step 2: Run, verify fails** — `cargo test -p kornia-tensor storage:: 2>&1 | tail` → FAIL/compile error.
- [ ] **Step 3: Implement** — rewrite the struct + constructors + Drop + accessors per Interfaces. `from_vec`: take the Vec's ptr/len/cap, `Layout::array::<T>(cap)`, `std::mem::forget(vec)`, build `HostResource::from_raw(ptr as *mut u8, layout)`; `ptr = NonNull::new(ptr)`. `Drop` for `TensorStorage` becomes empty (the `owner: Box<dyn MemoryResource>` field drops itself and frees) — remove the manual `dealloc`. Keep `unsafe impl Send/Sync` (now require `A: Send+Sync`; `owner` is `Send+Sync`). Update every internal `MemoryDomain::Device` (2-variant) reference to the 3-variant form.
- [ ] **Step 4: Run** — `cargo test -p kornia-tensor 2>&1 | tail -15` → all pass (incl. existing tests). Then `cargo test -p kornia-tensor --features dlpack 2>&1 | tail` (dlpack import builds ForeignResource via `from_borrowed` — should still pass; fix `tensor_from_dlpack_raw` if it set `domain`/`device_id` fields directly → it must call `from_borrowed`). `cargo clippy -p kornia-tensor --no-deps -- -D warnings`.
- [ ] **Step 5: Commit** — `git add crates/kornia-tensor/src/storage.rs && git commit -m "refactor(kornia-tensor): TensorStorage owns one MemoryResource; domain/device_id derive from it; as_slice gated on host-accessibility"`

---

### Task 4: Migrate downstream direct-construction call sites

**Files:**
- Modify: `crates/kornia-image/src/{dlpack.rs,arrow.rs,image.rs}`, `crates/kornia-imgproc/src/{flip.rs,warp/kernels.rs,gpu/color.rs}`, `kornia-py/src/pipeline.rs`
- Test: existing crate test suites + kornia-py pytest

**Interfaces:**
- Consumes: the preserved constructor signatures from Task 3.

- [ ] **Step 1: Build to surface breakage** — `cargo build -p kornia-image -p kornia-imgproc 2>&1 | tail -30`. Most call sites use `Image::from_*`/`Tensor::from_*` (stable) and accessors (stable) → minimal breakage. For any site that read removed fields (`storage.owns_memory`, `storage.layout`, `storage.keepalive`, direct `domain`/`device_id` field access), switch to the accessor methods (`.domain()`, `.device_id()`, `.layout()`).
- [ ] **Step 2: Fix each compile error** to use accessors/constructors; no behavior change.
- [ ] **Step 3: Run Rust suites** — `cargo test -p kornia-tensor -p kornia-image -p kornia-imgproc --lib 2>&1 | grep "test result"` → all pass. `cargo test -p kornia-image --features dlpack 2>&1 | grep "test result"`.
- [ ] **Step 4: Build + test kornia-py** — `cd kornia-py && pixi run -e py312 maturin develop --release --uv` then `pixi run -e py312 pytest tests/ -q 2>&1 | tail -5` → only the 3 known pre-existing failures.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "refactor: migrate downstream storage construction to MemoryResource accessors"`

---

## Wave B — cudarc first-class (feature `cudarc`)

### Task 5: `CudaResource` + `CudaAllocator` + Tensor cudarc methods

**Files:**
- Create: `crates/kornia-tensor/src/cuda.rs`
- Modify: `crates/kornia-tensor/Cargo.toml` (feature `cudarc = ["dep:cudarc"]`, `cudarc = { workspace = true, optional = true, features = ["cuda-12060","driver","nvrtc","std"] }` — verify exact 0.19 feature names against the installed crate), `crates/kornia-tensor/src/lib.rs` (`#[cfg(feature="cudarc")] pub mod cuda;`)
- Test: `cuda.rs` `#[cfg(all(test, feature="cudarc"))]` (runs on this Jetson)

**Interfaces:**
- Consumes: `MemoryResource`, `ForeignResource`, `MemoryDomain::Device`, storage constructors.
- Produces:
  - `pub struct CudaResource { slice: cudarc::driver::CudaSlice<u8>, ptr: *mut u8, id: i32 }` impl `MemoryResource` (domain `Device{id}`; `as_any` returns `&self` so callers downcast and read `.slice`). `Drop` is implicit (the `CudaSlice` frees).
  - `pub struct CudaAllocator { ctx: Arc<CudaContext>, stream: Arc<CudaStream> }` impl `TensorAllocator` (`allocate` = `stream.alloc_zeros::<u8>(layout.size())` → device ptr via `slice.device_ptr(&stream)` → `Box::new(CudaResource{..})`).
  - `impl<T: ValidTensorType + ..., const N: usize> Tensor<T, N, CudaAllocator>` (and a generic-A variant where sensible): `from_cudaslice(slice: CudaSlice<T>, shape: [usize;N], stream: &CudaStream) -> Self` (transmute-len to bytes, build CudaResource, storage via `from_borrowed`-like path with the resource); `as_cudaslice(&self) -> Option<&CudaSlice<u8>>` (downcast owner→CudaResource→`&.slice`); `into_cudaslice(self) -> Result<CudaSlice<u8>, Self>`; `to_cuda(&self, stream) -> Result<Tensor<T,N,CudaAllocator>, _>` (host→device memcpy_stod); `to_host(&self, stream) -> Result<Tensor<T,N,CpuAllocator>, _>` (device→host memcpy_dtov).

- [ ] **Step 1: Failing test (device round-trip)**
```rust
#[test]
fn cuda_roundtrip_and_as_slice_panics() {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let host = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], vec![1,2,3,4], CpuAllocator).unwrap();
    let dev = host.to_cuda(&stream).unwrap();
    assert!(matches!(dev.storage.domain(), MemoryDomain::Device { .. }));
    assert!(dev.as_cudaslice().is_some());
    let back = dev.to_host(&stream).unwrap();
    assert_eq!(back.as_slice(), &[1,2,3,4]);
}
```
- [ ] **Step 2: Run, fails** — `cargo test -p kornia-tensor --features cudarc cuda:: 2>&1 | tail`.
- [ ] **Step 3: Implement** `cuda.rs` per Interfaces. Read the installed cudarc 0.19 source under `~/.cargo/registry/src/*/cudarc-0.19*/` to pin exact APIs (`CudaContext::new`, `default_stream`/`new_stream`, `alloc_zeros`, `memcpy_stod`, `memcpy_dtov`, `slice.device_ptr(&stream)`).
- [ ] **Step 4: Run** — `cargo test -p kornia-tensor --features cudarc 2>&1 | tail`; `cargo build -p kornia-tensor` (default — no cudarc) and `cargo tree -p kornia-tensor -e features | grep -i cudarc` (only under the feature). Clippy both modes.
- [ ] **Step 5: Commit** — `git add crates/kornia-tensor/src/cuda.rs crates/kornia-tensor/Cargo.toml crates/kornia-tensor/src/lib.rs Cargo.lock && git commit -m "feat(kornia-tensor): cudarc feature — CudaResource/CudaAllocator + from_cudaslice/as_cudaslice/to_cuda/to_host"`

---

### Task 6: CUDA-stream imgproc example (device-owning Tensor)

**Files:**
- Create: `crates/kornia-imgproc/examples/cuda_stream_imgproc.rs`
- Modify: `crates/kornia-imgproc/Cargo.toml` (`cuda-example = ["dep:cudarc","kornia-tensor/cudarc"]`, optional cudarc dep, `[[example]] name="cuda_stream_imgproc" required-features=["cuda-example"]`)

**Interfaces:**
- Consumes: `Tensor::from_cudaslice`/`as_cudaslice`, kornia-io image read, cudarc nvrtc.

- [ ] **Step 1: Write the example** — load `Image<u8,3>` (with synthetic-gradient fallback); H2D into a `CudaSlice` on a stream; wrap it as a kornia `Tensor`/`Image` via `from_cudaslice` (so the Image OWNS the device memory); assert `image.as_slice()` would panic (demonstrate with a `std::panic::catch_unwind` printing "as_slice correctly refused host access"); nvrtc-compile an `rgb_to_gray` kernel (arch `compute_87`); launch on the stream reading the device-backed image's `data_ptr()`; D2H; print checksum; drop the Image → device memory freed (print confirmation).
- [ ] **Step 2: Build** — `cargo build -p kornia-imgproc --example cuda_stream_imgproc --features cuda-example 2>&1 | tail` → compiles.
- [ ] **Step 3: Run** — `cargo run -p kornia-imgproc --example cuda_stream_imgproc --features cuda-example 2>&1 | tail -15` → prints kernel checksum + "as_slice refused" + "device freed".
- [ ] **Step 4: Default unaffected** — `cargo build -p kornia-imgproc 2>&1 | tail -2`; clippy on the example with the feature.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "example(imgproc): CUDA-stream imgproc on a Tensor that owns a CudaSlice (feature cuda-example)"`

---

### Task 6b: cutile-rs backend parity (backend-agnosticism proof)

**Goal:** Prove the `MemoryResource` rework is genuinely backend-agnostic by running the SAME imgproc op (rgb→gray) through a SECOND device backend — **cutile-rs** (`cutile` crate, NVlabs, JIT through CUDA Tile IR) — wrapped as a kornia `Tensor`, and verifying its output **matches the raw-cudarc example byte-for-byte**.

**Files:**
- Create: `crates/kornia-imgproc/examples/cutile_imgproc.rs` (feature `cutile-example`).
- Maybe create: `crates/kornia-tensor/src/cutile.rs` (feature `cutile`) — a `CutileResource` (`MemoryResource` impl) + `Tensor::from_cutile`/`as_cutile` interop, ONLY if cutile exposes a device pointer / tensor handle we can wrap (determine from the crate source).
- Modify: `crates/kornia-imgproc/Cargo.toml` (`cutile-example = ["dep:cutile", "kornia-tensor/cutile"]`).

**FEASIBILITY GATE (do this FIRST, report honestly, do not fake):**
- cuTile supports **sm_80+** (Orin sm_87 OK) but **recommends CUDA 13.3**; this box has **CUDA 12.6**. STEP 0: `cargo add cutile@0.2` (in a scratch build), `cargo build --features cutile-example`, and a trivial `api::zeros` + `kernel` launch. If it does NOT build/run on CUDA 12.6 / Orin, STOP, document the exact blocker in the report, and DO NOT proceed (leave the cudarc example as the device demo; note cutile parity is blocked by CUDA version). The redesign's agnosticism is still demonstrated structurally (a `CutileResource` impl) even if the JIT can't run here — but only claim "verified" if it actually ran.

**If it builds/runs:**
- [ ] **Step 1:** Read the `cutile` crate source/docs (`~/.cargo/registry/src/*/cutile-0.2*/`) to find how its host-side `Tensor` exposes device memory (raw device ptr? a cudarc `CudaSlice`? a buffer handle?). Document the exact type/fn.
- [ ] **Step 2:** Implement the interop: wrap a cutile device buffer as a kornia `Tensor` via a `CutileResource` (`MemoryResource`: `as_ptr` = the device ptr, `domain = Device{id}`, Drop = drop the cutile handle), OR feed a kornia-`Tensor`-owned device buffer INTO a cutile kernel — whichever the cutile API supports. This is the "new Tensor rework handles cutile-rs" proof.
- [ ] **Step 3:** `cutile_imgproc.rs`: load the SAME input image as `cuda_stream_imgproc.rs`, run rgb→gray via a cutile kernel over a kornia `Tensor`, copy back to host.
- [ ] **Step 4:** Compare: run BOTH examples on the same input (or a shared test) and assert the cutile gray output == the cudarc gray output (max abs diff 0, or ≤1 LSB if rounding differs — document which). A small `tests/`/example that runs both and diffs, OR each example prints a checksum and a wrapper diffs them.
- [ ] **Step 5:** Verify default build unaffected (no `cutile` by default; `cargo tree | grep cutile`), clippy clean, then `git add -A && git commit -m "example(imgproc): cutile-rs rgb->gray over a kornia Tensor; verified byte-parity with the cudarc backend"`.

**Report** (`/home/nvidia/kornia-rs/.git/sdd/task-6b-report.md`): did cutile build/run on CUDA 12.6/Orin? the cutile memory API used; how the kornia Tensor wrapped it; byte-parity result vs cudarc (max diff); or the exact blocker if infeasible here.

---

## Wave C — gstreamer + v4l migration (first-class)

### Task 7: `GstResource` replaces the `GstAllocator` hack

**Files:**
- Modify: `crates/kornia-io/src/gstreamer/mod.rs` (delete `GstAllocator`; add `GstResource`), `crates/kornia-io/src/gstreamer/capture.rs:152-176` (build via foreign-wrap constructor)
- Test: `crates/kornia-io/tests` or existing gstreamer tests (gated on gstreamer feature)

**Interfaces:**
- Consumes: `ForeignResource`/`MemoryResource` or a new `Image::from_foreign(size, ptr, len, domain, keep)` helper (add a thin constructor to `kornia-image` that wraps `TensorStorage::from_borrowed`).
- Produces: `pub struct GstResource { _map: gstreamer::buffer::MappedBuffer<Readable> }` impl `MemoryResource` (domain `Host` for sysmem; `as_ptr` = map.as_ptr() as *mut; `Drop` via the `MappedBuffer`'s own Drop = unmap, and it holds the `Buffer` ref). NOTE: storing the `MappedBuffer` directly keeps both the map and the buffer alive — cleaner than `into_buffer()`.

- [ ] **Step 1: Test (frame buffer alias + drop)** — adapt the existing gstreamer capture test (or add one) asserting a captured `Image` reads the gst data and that dropping the Image releases the buffer (no leak across N frames). Use the existing test pipeline (`videotestsrc`).
- [ ] **Step 2: Run, fails** — build kornia-io with the gstreamer feature → fails (GstAllocator removed / GstResource missing).
- [ ] **Step 3: Implement** — `GstResource` as above; in `capture.rs`, replace `let alloc = GstAllocator(mapped_buffer.into_buffer()); Image::from_raw_parts(size, data_ptr, data_len, alloc)` with building a `GstResource` from the `mapped_buffer` and constructing the Image via the foreign-wrap constructor (the Image's allocator type becomes a plain tag, e.g. `ForeignAllocator`). Delete the `GstAllocator` struct + its `TensorAllocator` impl.
- [ ] **Step 4: Run** — kornia-io gstreamer tests pass; `cargo clippy -p kornia-io --features gstreamer --no-deps -- -D warnings`.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "refactor(kornia-io): GstResource (MemoryResource) replaces the GstAllocator keepalive hack"`

---

### Task 8: `V4lResource` replaces the mmap-guard allocator

**Files:**
- Modify: `crates/kornia-io/src/v4l/stream.rs:100-160` (the `MmapBuffer`/allocator + the Image construction)
- Test: existing v4l tests if a device is present, else a unit test of `V4lResource` Drop with a synthetic mmap.

**Interfaces:**
- Consumes: `MemoryResource`/foreign-wrap constructor (Task 7).
- Produces: `pub struct V4lResource { ptr: NonNull<u8>, len: usize }` (or wrap the existing `MmapBuffer`) impl `MemoryResource` (domain `Host`); `Drop` = `munmap` (move the existing unmap logic into `V4lResource::Drop`).

- [ ] **Step 1: Test** — unit test: build a `V4lResource` over an `mmap`'d anonymous region, read through the tensor, drop, assert no double-unmap (use a counter or just that drop doesn't fault). If a `/dev/video*` exists, run the existing capture test.
- [ ] **Step 2: Run, fails.**
- [ ] **Step 3: Implement** — move the mmap-unmap Drop into `V4lResource`; construct the Image via the foreign-wrap constructor; remove the allocator-as-keepalive path.
- [ ] **Step 4: Run** — `cargo test -p kornia-io --features v4l 2>&1 | tail`; clippy.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "refactor(kornia-io): V4lResource (MemoryResource) replaces the mmap-guard allocator"`

---

## Wave D — verification + memory safety

### Task 9: Whole-workspace verification, miri, feature-isolation, safety review

**Files:** none (verification) — fixes land in the relevant task's files if issues surface.

- [ ] **Step 1: Full Rust suite** — `cargo test -p kornia-tensor -p kornia-image -p kornia-imgproc --lib 2>&1 | grep "test result"`; `cargo test -p kornia-tensor -p kornia-image --features dlpack 2>&1 | grep "test result"`; `cargo test -p kornia-tensor --features cudarc 2>&1 | grep "test result"` — all pass.
- [ ] **Step 2: kornia-py** — maturin + `pixi run -e py312 pytest tests/ -q 2>&1 | tail -5` → only 3 known pre-existing failures.
- [ ] **Step 3: Feature isolation** — `cargo tree -p kornia-tensor -e features 2>&1 | grep -iE "cudarc|gstreamer" || echo "clean default"` (must be clean by default); `cargo build --workspace 2>&1 | tail -2`.
- [ ] **Step 4: miri** — `cargo +nightly miri test -p kornia-tensor resource:: storage:: 2>&1 | tail` (HostResource/ForeignResource Drop, from_vec/from_borrowed, as_slice gating). cudarc paths can't run under miri — exclude.
- [ ] **Step 5: Clippy workspace** — `cargo clippy --workspace --no-deps --all-targets -- -D warnings` (move untracked example files aside if needed).
- [ ] **Step 6: Memory-safety review** — dispatch a memory-safety reviewer over the whole-branch diff (`git diff feat/gray-neon-kernels...HEAD`): focus on the single-owner `Drop` (no double-free / no leak), `as_slice` gating correctness for `Unified`, `from_cudaslice` device-ptr validity + `CudaSlice` ownership, gst/v4l `Drop` exactly-once. Fix Critical/Important.
- [ ] **Step 7: Commit any fixes** — `git commit -m "fix: address memory-safety review for MemoryResource redesign"`

---

## Self-Review

**Spec coverage:** two-axis model (Task 1: MemoryResource + 3-state MemoryDomain) ✓; storage single-owner (Task 3) ✓; allocator→resource (Task 2) ✓; downstream migration (Task 4) ✓; cudarc first-class + example (Tasks 5-6) ✓; gstreamer/v4l first-class migration (Tasks 7-8) ✓; behavior-preserving + miri + feature-isolation + safety review (Task 9) ✓; backend-agnostic constraint enforced in Tasks 5/7/8 + verified Task 9 ✓; deferred (cubecl, NVMM→CUDA) noted, not tasked ✓.

**Placeholder scan:** constructor/trait/resource code shown in full for the high-risk core (Tasks 1-3); cudarc/gst/v4l give exact types + the precise call-sites to change + the existing line numbers. No "TBD"/"handle edge cases".

**Type consistency:** `MemoryResource`/`MemoryDomain{Host,Device{id},Unified{id}}`/`HostResource`/`ForeignResource`/`CudaResource`/`CudaAllocator`/`GstResource`/`V4lResource` and `allocate`/`from_cudaslice`/`as_cudaslice`/`into_cudaslice`/`to_cuda`/`to_host` used consistently across tasks.
