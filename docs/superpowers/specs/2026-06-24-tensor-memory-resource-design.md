# Tensor Memory-Resource Redesign — Design Spec

**Date:** 2026-06-24
**Status:** Design approved (pending spec review) — not yet planned/implemented
**Scope crate(s):** `kornia-tensor` (core), `kornia-image`/`kornia-imgproc`/`kornia-py` (accessor-only ripple), new feature-gated resource impls (`cudarc`, later `gstreamer`, `cubecl`)

## Goal

Redesign `kornia-tensor`'s memory model so a tensor's backing buffer can come from **any provenance** — kornia-owned host, foreign host (numpy, GStreamer, mmap, DLPack), owned device (cudarc, cubecl), or dual-accessible (CUDA pinned/unified, Jetson NVMM) — under **one uniform ownership abstraction**, with **natural cudarc integration**, while the core stays **backend-agnostic** and the existing **host path is behavior-preserving**.

## Motivating insight: two orthogonal axes

The current `MemoryDomain{Host, Device}` conflates two independent properties, and the storage tracks ownership with four loosely-coupled fields (`ptr`, `owns_memory`, `alloc`, `keepalive`) whose freeing rule ("if `owns_memory` call `alloc.dealloc`, else drop `keepalive`") is the source of recurring memory-safety findings.

Split into two axes:

1. **Ownership / release** — *who frees it, and how?* → one owning RAII handle, `MemoryResource`, whose `Drop` does the correct release (host `dealloc`, `cudaFree`, `Py_buffer` release, DLPack deleter, `gst_buffer_unmap`+`unref`, …).
2. **Accessibility** — *where can the pointer be legally dereferenced?* → `Host`, `Device{id}`, or `Unified{id}` (both). Gates `as_slice()` (host) and kernel use (device).

GStreamer is the validating case: a `GstBuffer` is **host** memory kornia must **not** `free` (release = unmap+unref), while an **NVMM** buffer is **device** memory, and CUDA pinned/unified is **both** — one `Host/Device` bit cannot express this.

## Architecture

### `MemoryResource` — the ownership abstraction

```rust
/// Owning handle to a tensor's backing memory. Frees correctly on Drop.
pub trait MemoryResource: Send + Sync {
    fn as_ptr(&self) -> *mut u8;       // host-addr or device-addr (see domain)
    fn len_bytes(&self) -> usize;
    fn domain(&self) -> MemoryDomain;  // accessibility
    fn as_any(&self) -> &dyn core::any::Any;   // downcast (e.g. back to &CudaSlice)
    // Drop is where each provenance frees itself.
}
```

### `MemoryDomain` — the accessibility axis (replaces the binary enum)

```rust
pub enum MemoryDomain {
    Host,                  // malloc, numpy, gst sysmem
    Device  { id: i32 },   // cudaMalloc, CudaSlice, NVMM not host-mapped
    Unified { id: i32 },   // pinned / managed / unified / NVMM host-mapped — BOTH
}
```

- `as_slice()` / `as_slice_mut()` → `Ok` for `Host | Unified`; **panic/Err** for `Device`.
- device-pointer-for-kernel → valid for `Device | Unified`.
- `device_id` folds into the enum (no separate field; meaningless for `Host`).

### Storage shrinks to one owner + a cached pointer

```rust
pub struct TensorStorage<T, A: TensorAllocator> {
    ptr: NonNull<T>,                 // cached hot-path pointer (= owner.as_ptr())
    len: usize,
    owner: Box<dyn MemoryResource>,  // single source of "free correctly"
    alloc: A,                        // retained for (re)allocation / clone
    marker: PhantomData<T>,
}
```

- `Drop` = drop `owner` (frees). The `owns_memory` flag, the `keepalive` field, and `dealloc`-by-ptr are **removed** — collapsed into `owner`.
- Hot-path accessors read the cached `ptr` directly (no vtable). `Box<dyn>` dispatch is cold-path only (alloc/drop/metadata) → zero perf cost.

### Allocator: produce a resource, not a pointer

```rust
pub trait TensorAllocator: Clone + Send + Sync {
    fn allocate(&self, layout: Layout) -> Result<Box<dyn MemoryResource>, AllocError>;
}
```

`allocate` produces **owned** resources. The `from_*`/`wrap_*` constructors produce **foreign** resources. Both yield `Box<dyn MemoryResource>`; the storage is indifferent.

### Resource impls (one per provenance)

| Resource | Provenance | Drop frees via | domain |
|---|---|---|---|
| `HostResource{ptr, layout}` | `CpuAllocator`/`AlignedCpuAllocator` | host `dealloc` | `Host` |
| `ForeignResource{ptr, len, domain, keep: Arc<dyn Any>}` | numpy / mmap / DLPack import / `from_cudaslice` | drops `keep` (frees nothing itself) | as imported |
| `CudaResource{slice: CudaSlice<u8>, ptr, id}` | cudarc owned (`CudaAllocator`) | the `CudaSlice` (`cudaFree`) | `Device{id}` |
| `GstResource{buffer, map_info}` *(later)* | GStreamer | `gst_buffer_unmap` + `unref` | `Host` (sysmem) / `Device`/`Unified` (NVMM) |
| `GpuResource{handle}` *(later)* | cubecl | the `Backend` | `Device{id}` |

### Natural cudarc integration (feature `cudarc`, default-off)

```rust
// inherent methods, gated — read naturally, no extension-trait import
impl<T, const N: usize, A> Tensor<T, N, A> {
    pub fn from_cudaslice(slice: CudaSlice<T>, shape: [usize; N], stream: &CudaStream) -> Self; // wrap (foreign/Cuda resource)
    pub fn as_cudaslice(&self) -> Option<&CudaSlice<T>>;        // as_any downcast
    pub fn into_cudaslice(self) -> Result<CudaSlice<T>, Self>;  // unwrap owner
    pub fn to_cuda(&self, stream: &CudaStream) -> Result<Self, _>; // host -> device (CudaAllocator)
    pub fn to_host(&self, stream: &CudaStream) -> Result<Self, _>; // device -> host
}
```

`CudaAllocator{ ctx, stream }::allocate` → `stream.alloc_zeros` → `CudaResource`, so owned device allocation flows through the *same* `TensorAllocator` trait as host allocation. (cudarc 0.19 needs a `&CudaStream` to read a `CudaSlice`'s device pointer — `from_cudaslice` takes one; natural since you always have a stream in cudarc-land.)

## Global Constraints

- **Backend-agnostic core:** `kornia-tensor` core (and `kornia-image`/`imgproc`/`py`) name **no** GPU/streaming runtime. cudarc, cubecl, gstreamer each appear **only** as an optional feature providing a `MemoryResource`/`TensorAllocator` impl. Default build pulls none (`cargo tree -e features` verifies).
- **Behavior-preserving host path:** existing public API (`Tensor`/`Image` accessors `data_ptr`/`as_slice`/`domain`/etc.) keeps signatures; the full existing test suite (Rust + kornia-py pytest) is the gate. Downstream ripple stays inside `kornia-tensor`.
- **Memory safety is paramount:** unifying ownership into `MemoryResource` must remove (not add) UB surface; a memory-safety review (+ miri on the resource Drop paths, foreign/borrow, cudarc wrap, dlpack import) gates each wave.
- **No markdown docs added to PR #944** (this spec lives under `docs/superpowers/specs/`, not committed to the PR).
- Naming: `<output>_from_<input>`; new public Rust fns return named structs; new Python bindings `#[pyclass]`.

## Scope

**In (first plan):**
1. Core: `MemoryResource` trait, 3-state `MemoryDomain`, storage refactor (`owner` replaces `owns_memory`/`keepalive`/`dealloc`), `TensorAllocator::allocate`, host resource impls (`HostResource` for `CpuAllocator`/`AlignedCpuAllocator`, `ForeignResource`), accessibility-gated `as_slice`, migrate all constructors (`from_vec`/`from_raw_parts`/`from_raw_host`/`from_raw_device`/`from_borrowed`).
2. DLPack import/export updated to the resource model (import → `ForeignResource`; export unchanged semantics).
3. `cudarc` feature: `CudaResource`, `CudaAllocator`, `from_cudaslice`/`as_cudaslice`/`into_cudaslice`/`to_cuda`/`to_host`.
4. The CUDA example: a `Tensor` that genuinely **owns** a `CudaSlice`, runs an nvrtc kernel on `data_ptr()`, `as_slice()` correctly refuses host access, drop frees device memory.

5. **GStreamer + v4l migration (first-class).** kornia-io already ships zero-copy GStreamer/v4l capture, but via a hack: `GstAllocator(gstreamer::Buffer)` — a fake `TensorAllocator` whose `dealloc` is a no-op, abused to hold the gst `Buffer` alive for the `Image`'s lifetime (`crates/kornia-io/src/gstreamer/mod.rs`); v4l uses an `MmapBuffer` guard whose `Drop` unmaps (`crates/kornia-io/src/v4l/stream.rs`). These are exactly `MemoryResource` impls. Migrate both: `GstResource{ buffer, map }` (Drop = unmap+unref; `Host` for sysmem, `Device`/`Unified` for NVMM) and `V4lResource` (Drop = munmap), replacing the allocator-as-keepalive hack. kornia-io's capture constructors switch from `Image::from_raw_parts(.., GstAllocator(buf))` to a foreign-wrap constructor carrying the resource.

**Designed-for but deferred (follow-up plans):**
- `cubecl` migration to `GpuResource`/the `Backend` path under the new model.
- NVMM/DMABuf true zero-copy-to-CUDA (`GstResource` with a CUDA-external-memory import) — the `Host` gst path migrates now; the NVMM→CUDA device path is a follow-up once the sysmem path is proven.

**Out:** crates.io publish; changing image/imgproc algorithms.

## Open decisions (for spec review)

1. **Branch/PR strategy.** This redesign *supersedes* #944's Wave-1 storage internals (`owns_memory`/`keepalive`/`device_id` fields). Options: (a) evolve it **inside #944** (replacing Wave 1's storage), or (b) a **follow-up branch off #944**. Recommendation: (b) — keep #944 reviewable; land the redesign as its own PR on top.
2. **GStreamer now or later?** Recommend **later** (needs gstreamer-rs deps + camera path); include it in the design (above) so `ForeignResource`/custom-`Drop` provably covers it, but defer the impl.

## Risks

- Storage internals touch every constructor + `Drop` — highest memory-safety risk; mitigated by behavior-preserving gate + miri + per-wave review.
- `Box<dyn MemoryResource>` must stay off the hot path (cached `ptr`); verify no accessor regresses.
- `into_cudaslice`/`into_vec` need `Arc`/owner single-ownership (`try_unwrap` + downcast); define clear `Err(self)` fallback when shared.
