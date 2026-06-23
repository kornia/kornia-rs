# Numpy-agnostic Image core + DLPack — design spec

**Date:** 2026-06-24
**Status:** Approved (design)
**Branch / PR:** `feat/gray-neon-kernels` (PR #944) — lands in the same PR.
**Depends on:** the high-level color-conversion API (same branch) and `kornia/dlpack-rs` `v0.2.0` (PR kornia/dlpack-rs#5, tag pushed).
**Tracking:** task #24 (subsumes #23, the ROS2 bgr8 → owned-tensor → TensorRT path).

## Goal

Make the Python `kornia_rs.Image` **own a Rust buffer** instead of being a numpy wrapper, with numpy reduced to one interop adapter (`from_numpy`/`to_numpy`). Add framework-neutral zero-copy export (`__dlpack__`, buffer protocol) for Torch/CuPy/TensorRT, and robotics-grade ingest (`from_buffer`/`from_bytes`/`from_dlpack`) — all with strict keep-alive so a borrow can never outlive its source. Optimized for the Jetson Orin robotics path: ingest a transient frame (e.g. a ROS2 `sensor_msgs/Image` bgr8) zero-copy, run the fused pipeline, return an **owned** tensor whose pointer is safe to hand to cudarc/TensorRT after the source is recycled.

## Background (current state)

`PyImageApi` today stores `ImageData = { U8|U16|F32 }(Py<PyArray3<…>>)` — storage is 100% numpy. The **compute is already numpy-agnostic**: every kernel borrows `Image<T,C,ForeignAllocator>` from the numpy pointer via shared helpers (`numpy_as_image*`, `alloc_output_pyarray*`, `wrap*`). The numpy coupling concentrates in: the `ImageData` type, those shared helpers, the buffer protocol (`__getbuffer__`/`__releasebuffer__`, which currently *delegate* to numpy), `__reduce__`/pickle, and a few direct-numpy methods (`copy`, `to_numpy`, `tobytes`, `__array__`, `convert`). `kornia_image` provides only compile-time-typed owned containers (`Image<T,C,CpuAllocator>`); there is no type-erased owned buffer.

## Approved decisions

1. **Full agnostic core** — the Image owns its buffer; numpy is an adapter.
2. **Type-erased backing** — `enum Backing { Owned(AlignedBytes) | Numpy(Py<PyArray3>) | Dlpack(PyTensor) }` plus `(dtype, shape, color_space, mode)` metadata; typed `Image<T,C,ForeignAllocator>` views built on demand for compute (the same mechanism `numpy_as_image` uses today).
3. **`from_numpy` borrows by default** (`copy=True` to own); every backing variant carries a keep-alive so borrows can't outlive their source.
4. **Op outputs become `Owned`** buffers — done by reworking the shared alloc/wrap helpers (contained change, not 57 method rewrites). This is what makes the core genuinely numpy-agnostic.
5. **Owned buffers are 64-byte aligned** (SIMD/DMA).
6. **DLPack via the revived `dlpack-rs v0.2.0`** (git-tag dep, `features=["pyo3"]`), not a hand-rolled or third-party crate.
7. **dtype scope v1 = `{U8,U16,F32}`** (matches today); other dtypes rejected with a clear error.

Everything is additive to the public Python surface (the 57 existing methods keep working); the storage layer underneath changes.

---

## Section 1 — `Backing` abstraction (core)

```rust
// kornia-py/src/backing.rs
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dtype { U8, U16, F32 }
impl Dtype { pub fn itemsize(self) -> usize; pub fn name(self) -> &'static str; }

/// 64-byte-aligned owned heap buffer.
pub struct AlignedBytes { ptr: NonNull<u8>, len: usize, layout: std::alloc::Layout }
impl AlignedBytes {
    pub fn zeroed(len: usize) -> Self;            // alloc_zeroed, 64-byte align
    pub fn from_slice(src: &[u8]) -> Self;        // alloc + copy
    pub fn as_ptr(&self) -> *const u8;
    pub fn as_mut_ptr(&mut self) -> *mut u8;
    pub fn as_slice(&self) -> &[u8];
    pub fn len(&self) -> usize;
}
// Drop frees via the stored Layout. Send + Sync (plain bytes).

pub enum Backing {
    Owned(AlignedBytes),
    Numpy(Py<PyAny>),          // a live numpy ndarray we borrow; real dtype tracked by the `dtype` field
    Dlpack(dlpack_rs::pyo3_glue::PyTensor),
}
impl Backing {
    /// Host byte address of element [0,0,0]. The single source of truth.
    pub fn data_ptr(&self, py: Python<'_>) -> *mut u8;
    pub fn nbytes(&self, py: Python<'_>) -> usize;
}
```

`PyImageApi` becomes:
```rust
#[pyclass(name = "Image", weakref, module = "kornia_rs.image")]
pub struct PyImageApi {
    backing: Backing,
    dtype: Dtype,
    shape: [usize; 3],            // (H, W, C)
    color_space: ColorSpace,
    mode: String,
    format: Option<&'static str>,
}
```

**Keep-alive invariant:** `Numpy` holds the `Py<PyArray3>` (refcount), `Dlpack` holds the `PyTensor` (which holds the capsule and calls the producer's deleter on drop), `Owned` owns its bytes. A borrowed view is always backed by a live source.

Note: `Backing::Numpy` stores the ndarray as an opaque `Py<PyAny>` (the buffer we borrow); its element type is recorded once in `self.dtype`, and `data_ptr` reads the array's base pointer. The public contract is "a live numpy array whose buffer we borrow."

## Section 2 — Compute integration (high-leverage)

Rework the shared helpers so all methods route through `Backing`:
- `borrow_image::<T, const C: usize>(&self, py) -> Image<T,C,ForeignAllocator>` — builds the typed borrow from `backing.data_ptr()` + `shape`, validating `Dtype`/channels. Replaces `numpy_as_image::<C>` at call sites (which currently assume numpy).
- `alloc_output_owned::<const C: usize>(py, size, dtype) -> (OwnedWriteGuard, /*for return*/ PyImageApiParts)` — allocates an `AlignedBytes`, returns a writable `Image<T,C,ForeignAllocator>` over it; the op writes, then we wrap the `Owned` backing into the returned `Image`. Replaces `alloc_output_pyarray*`.
- `wrap_owned(parts) -> PyImageApi` and the existing `wrap_u8_result`/`clone_handle`/`copy` reroute to produce/preserve `Backing` (carrying `color_space`/`mode`).

Effect: `resize/blur/rotate/crop/flip/color/adjust_*/resize_normalize_to_tensor` produce **Owned** outputs with no per-method rewrite (they call the shared helpers). `require_u8`/dtype gates unchanged.

## Section 3 — Ingest + egress adapters

**Ingest** (all set the keep-alive + metadata + default `color_space` by channels):
- `from_numpy(arr, *, copy=False)` → `Backing::Numpy` (borrow) or `Owned` (copy). The `Image(arr, …)` constructor and `frombuffer`/`fromarray` delegate here.
- `from_bytes(data, width, height, channels=3, dtype="uint8")` → `Owned` (copy). (Existing `frombytes` semantics, now producing `Owned`.)
- `from_buffer(obj)` → borrow a PEP-3118 buffer-protocol object zero-copy with a keep-alive to `obj` (robotics ingest of a memoryview/bytearray, e.g. a ROS2 message payload). Validates C-contiguous + supported dtype/shape.
- `from_dlpack(obj)` → `Backing::Dlpack` via `PyTensor::from_pyany`; validates CPU device, contiguous, ndim==3 (or 2→(H,W,1)), dtype ∈ {u8,u16,f32}, C ∈ {1,3,4}.

**Egress:**
- `to_numpy(*, copy=False)` → zero-copy ndarray **view** with numpy `base = self` (keeps the Image alive) via the buffer protocol; `copy=True` returns an independent array.
- `__dlpack__(self, *, stream=None)` → PyCapsule via `dlpack_rs::IntoDLPack` (CPU device; uses `into_capsule`/`into_capsule_versioned`); the keep-alive wrapper owns a clone/handle of the backing so the exported tensor stays valid until the consumer's deleter runs. `__dlpack_device__` → `(kDLCPU=1, 0)`. Reject non-CPU with `NotImplementedError` (CUDA is future).
- `data_ptr`, `tobytes`, `numpy()`, `dtype`, `nbytes`, `shape` preserved over `Backing`.

## Section 4 — Buffer protocol + pickle over `Backing`

- Reimplement `__getbuffer__`/`__releasebuffer__` to fill a `Py_buffer` directly from `backing.data_ptr()`/`shape`/`dtype` (format string per dtype; strides for (H,W,C); readonly=false for `Owned`, readonly per-source for borrows). This is the zero-copy substrate that powers `np.asarray(img)`, `memoryview(img)`, `torch.frombuffer`, and `to_numpy(copy=False)` uniformly for owned buffers.
- `__array__(dtype=None, copy=None)` → build a numpy view via the buffer protocol (or copy/astype per args).
- `__reduce__`/pickle → serialize `(bytes, dtype_str, shape, color_space, mode)`, reconstruct an `Owned` image (bytes-based form; consistent with the color-cvt pickle fix). `copy()` → `Owned` deep copy. `convert(mode)` rerouted through typed views / existing kernels.

## Section 5 — Error handling

- `from_dlpack`/`from_buffer`/`from_numpy` raise `ValueError` for: non-contiguous, unsupported dtype, unsupported channel count, non-CPU device (dlpack). `__dlpack__` raises `NotImplementedError` for any non-CPU request. Messages name the exact constraint and the remediation.
- Borrow keep-alive is structural (the `Py` handle / `PyTensor` is a field), so misuse can't compile into a dangling view.

## Section 6 — Testing

- **Backing unit tests:** `AlignedBytes` 64-byte alignment, zeroed/from_slice correctness, Drop frees (no leak under Miri where feasible).
- **Aliasing/keep-alive:** `from_numpy(copy=False)` shares memory (mutation visible both ways); the Image survives `del arr`; `from_dlpack` import survives the producer dropping its handle; `Owned` op output is independent of any source mutation/free.
- **DLPack round-trip:** `kornia Image → torch.from_dlpack → Tensor` and `torch.Tensor.__dlpack__() → Image.from_dlpack`, values equal, no leak (the live test deferred from dlpack-rs lands here).
- **Buffer protocol:** `np.asarray(img)` / `memoryview(img)` zero-copy for `Owned`; `to_numpy(copy=False)` view keeps `img` alive; `copy=True` independent.
- **Robotics path (folds in #23):** simulate a ROS2 bgr8 buffer (a `bytearray`); `from_buffer` zero-copy → `cvt_color(Bgr→Rgb)`/`resize_normalize_to_tensor` → **owned** CHW f32 tensor; assert the output is independent of (survives mutation/free of) the source buffer and exposes a stable `data_ptr`/`__dlpack__`.
- **Full regression:** the entire existing Python suite (resize/color/io/pickle/buffer/zero-copy/serialize) stays green — proves the 57-method surface is preserved.
- Rust: `cargo clippy --workspace --no-deps --all-targets --features ci -- -D warnings`, lib tests; build `kornia-py` via maturin.

## Section 7 — File layout

| File | Change |
|---|---|
| `kornia-py/src/backing.rs` *(new)* | `Dtype`, `AlignedBytes`, `Backing`, `borrow_image`, `alloc_output_owned`, wrap/keep-alive helpers |
| `kornia-py/src/dlpack.rs` *(new)* | `__dlpack__`/`__dlpack_device__` export glue + `from_dlpack` import (over `dlpack_rs`) |
| `kornia-py/src/image.rs` | `PyImageApi` fields → `Backing`; reroute constructors, accessors, buffer protocol, pickle, `to_numpy`/`copy`/`tobytes`/`__array__`/`convert`, and the shared alloc/wrap helpers |
| `kornia-py/src/lib.rs` | register `from_dlpack` (and any new free fn) |
| `kornia-py/Cargo.toml` | `dlpack-rs = { git = "https://github.com/kornia/dlpack-rs", tag = "v0.2.0", features = ["pyo3"] }` |
| `kornia-py/python/kornia_rs/image.pyi` | stubs: `from_numpy`/`to_numpy`/`from_bytes`/`from_buffer`/`from_dlpack`/`__dlpack__`/`__dlpack_device__` |
| `kornia-py/tests/test_dlpack.py`, `test_backing.py` *(new)* | the tests above |

## Out of scope (future)
- CUDA device support + stream sync for `__dlpack__`/`from_dlpack` (Jetson GPU path) — leave device/stream hooks.
- crates.io publish of dlpack-rs (currently git-tag dep).
- Broadening dtype set beyond {u8,u16,f32}.

## Verification
```
cargo clippy --workspace --no-deps --all-targets --features ci -- -D warnings   # (move untracked example files aside; see ledger note)
cargo test -p kornia-image -p kornia-imgproc --lib
cd kornia-py && pixi run -e py312 maturin develop --release
pixi run -e py312 pytest tests/   # full suite incl. test_dlpack.py, test_backing.py, robotics path
```
Acceptance: full existing suite green; DLPack round-trips with torch; robotics path yields an owned tensor independent of its source; zero-copy paths verified to share memory and keep sources alive.
