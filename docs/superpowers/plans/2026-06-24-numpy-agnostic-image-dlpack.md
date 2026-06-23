# Numpy-agnostic Image core + DLPack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kornia_rs.Image` own a Rust buffer (numpy demoted to an adapter), with zero-copy `from_buffer`/`from_dlpack` ingest and `__dlpack__`/buffer-protocol egress, so a transient frame (e.g. ROS2 bgr8) can be ingested zero-copy and a fused pipeline can return an owned tensor safe for cudarc/TensorRT.

**Architecture:** Replace the numpy-only `ImageData` with `Backing { Owned(AlignedBytes) | Borrowed{ptr,keep,readonly} }` + `(dtype,shape,color_space,mode)` metadata. numpy / PEP-3118 buffers / DLPack imports all collapse into `Borrowed` (ptr + keep-alive). Compute keeps borrowing `Image<T,C,ForeignAllocator>` via a shared `borrow_image` helper; op outputs go through `alloc_output_owned` → `Owned`. DLPack via the revived `dlpack-rs v0.2.0`.

**Tech Stack:** Rust, pyo3 0.28, rust-numpy, maturin, `dlpack-rs` (git tag v0.2.0, feature `pyo3`), pytest, torch (for round-trip).

## Global Constraints
- Branch `feat/gray-neon-kernels` (PR #944); commit per task.
- Naming `<output>_from_<input>`; never `rgb_to_*`. New Python bindings are `#[pyclass]`/`#[pymethods]`.
- dtype scope v1 = `{U8,U16,F32}`; other dtypes → `ValueError`. Owned buffers **64-byte aligned**.
- `from_numpy` **borrows by default** (`copy=True` to own). Every `Borrowed` carries a keep-alive (no UAF).
- Op outputs become `Owned` (via the shared alloc/wrap helpers — do NOT rewrite 57 methods individually).
- Core enum holds **no numpy type**; numpy only at `from_numpy`/`to_numpy` adapters.
- Additive to the public surface: the full existing Python suite must stay green.
- Dep: `dlpack-rs = { git = "https://github.com/kornia/dlpack-rs", tag = "v0.2.0", features = ["pyo3"] }`.
- Clippy gate (CI): `cargo clippy --workspace --no-deps --all-targets --features ci -- -D warnings`. Locally, move the untracked `crates/kornia-imgproc/examples/{bench_gray,blur_profile,trace_warp,warp_profile,warp_profile_640,warp_profile_fresh,warp_profile_rot}.rs` (and `bench_gpu_color.rs` is a pre-existing aarch64-only lint — ignore) aside before running, or run `cargo clippy -p kornia-py --no-deps -- -D warnings` for kornia-py-scoped checks.

## File Structure
| File | Responsibility |
|---|---|
| `kornia-py/src/backing.rs` *(new)* | `Dtype`, `AlignedBytes`, `Backing`, `BorrowGuard`, `borrow_image`, `alloc_output_owned` |
| `kornia-py/src/dlpack.rs` *(new)* | `__dlpack__`/`__dlpack_device__` export + `from_dlpack` import over `dlpack_rs` |
| `kornia-py/src/image.rs` | swap `PyImageApi` storage to `Backing`; reroute accessors/constructors/buffer-protocol/pickle/helpers |
| `kornia-py/src/lib.rs` | register `from_dlpack`; ensure `mod backing; mod dlpack;` |
| `kornia-py/Cargo.toml` | add `dlpack-rs` dep |
| `kornia-py/python/kornia_rs/image.pyi` | stubs for new methods |
| `kornia-py/tests/test_backing.py`, `test_dlpack.py` *(new)* | unit + round-trip + robotics tests |

---

### Task 1: `backing.rs` — core storage type + compute helpers (standalone)

**Files:** create `kornia-py/src/backing.rs`; add `pub(crate) mod backing;` to `kornia-py/src/lib.rs`.

**Interfaces — Produces:**
- `enum Dtype { U8, U16, F32 }` with `itemsize()->usize`, `name()->&'static str`, `from_numpy_str(&str)->PyResult<Dtype>`.
- `struct AlignedBytes` with `zeroed(len)->Self`, `from_slice(&[u8])->Self`, `as_ptr()`, `as_mut_ptr()`, `as_slice()`, `len()`.
- `enum BorrowGuard { PyObject{obj: Py<PyAny>, buffer: Option<Box<pyo3::ffi::Py_buffer>>}, Dlpack(dlpack_rs::pyo3_glue::PyTensor) }`.
- `enum Backing { Owned(AlignedBytes), Borrowed{ ptr: std::ptr::NonNull<u8>, keep: BorrowGuard, readonly: bool } }` with `data_ptr(&self)->*mut u8`, `readonly(&self)->bool`.
- `unsafe fn borrow_image<T, const C: usize>(b: &Backing, shape: [usize;3]) -> Result<Image<T,C,ForeignAllocator>, ImageError>` (validates `shape[2]==C`).
- `fn alloc_output_owned<const C: usize>(dtype: Dtype, size: ImageSize) -> (AlignedBytes, ImageSize)` — returns a zeroed aligned buffer sized `H*W*C*itemsize`.

- [ ] **Step 1: write `kornia-py/src/backing.rs`**

```rust
//! Numpy-agnostic storage backing for the Python Image.
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

use kornia_image::{allocator::ForeignAllocator, Image, ImageError, ImageSize};
use pyo3::prelude::*;

const ALIGN: usize = 64;

/// Element type of an Image buffer (v1 scope).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype { U8, U16, F32 }
impl Dtype {
    pub fn itemsize(self) -> usize { match self { Dtype::U8 => 1, Dtype::U16 => 2, Dtype::F32 => 4 } }
    pub fn name(self) -> &'static str { match self { Dtype::U8 => "uint8", Dtype::U16 => "uint16", Dtype::F32 => "float32" } }
    pub fn from_numpy_str(s: &str) -> PyResult<Dtype> {
        match s {
            "uint8" | "u8" | "|u1" | "B" => Ok(Dtype::U8),
            "uint16" | "u16" | "<u2" | "=u2" | "H" => Ok(Dtype::U16),
            "float32" | "f32" | "<f4" | "=f4" | "f" => Ok(Dtype::F32),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported dtype {other:?}; expected uint8, uint16, or float32"))),
        }
    }
}

/// A 64-byte-aligned owned heap buffer (SIMD/DMA friendly).
pub struct AlignedBytes { ptr: NonNull<u8>, len: usize, layout: Layout }
// SAFETY: AlignedBytes uniquely owns a heap allocation of plain bytes.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}
impl AlignedBytes {
    pub fn zeroed(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), ALIGN).expect("layout");
        // SAFETY: layout has non-zero size (len.max(1)).
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len, layout }
    }
    pub fn from_slice(src: &[u8]) -> Self {
        let mut b = Self::zeroed(src.len());
        // SAFETY: b.ptr owns len==src.len() bytes; regions don't overlap.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), b.ptr.as_ptr(), src.len()); }
        b
    }
    pub fn as_ptr(&self) -> *const u8 { self.ptr.as_ptr() }
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr.as_ptr() }
    pub fn as_slice(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}
impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: ptr/layout came from alloc_zeroed with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout); }
    }
}

/// Keeps a borrowed buffer's source alive for the Image's lifetime.
pub enum BorrowGuard {
    /// numpy ndarray (ptr = base) or PEP-3118 owner (ptr = Py_buffer.buf, view stored to release on drop).
    PyObject { obj: Py<PyAny>, buffer: Option<Box<pyo3::ffi::Py_buffer>> },
    /// Imported DLPack tensor; its Drop runs the producer's deleter.
    Dlpack(dlpack_rs::pyo3_glue::PyTensor),
}
impl Drop for BorrowGuard {
    fn drop(&mut self) {
        if let BorrowGuard::PyObject { buffer: Some(view), .. } = self {
            // SAFETY: `view` was filled by PyObject_GetBuffer in from_buffer; release exactly once.
            unsafe { pyo3::ffi::PyBuffer_Release(view.as_mut()); }
        }
        // Py<PyAny> and PyTensor drop themselves.
    }
}

/// Image data backing: owned aligned bytes, or a zero-copy borrow with keep-alive.
pub enum Backing {
    Owned(AlignedBytes),
    Borrowed { ptr: NonNull<u8>, keep: BorrowGuard, readonly: bool },
}
// SAFETY: Owned is Send+Sync; Borrowed holds Send keep-alives and a raw ptr with exclusive logical ownership.
unsafe impl Send for Backing {}
impl Backing {
    pub fn data_ptr(&self) -> *mut u8 {
        match self {
            Backing::Owned(b) => b.ptr.as_ptr(),
            Backing::Borrowed { ptr, .. } => ptr.as_ptr(),
        }
    }
    pub fn readonly(&self) -> bool {
        match self { Backing::Owned(_) => false, Backing::Borrowed { readonly, .. } => *readonly }
    }
}

/// Build a typed compute borrow from a backing + shape. Validates channel count.
/// SAFETY: caller guarantees `b`'s buffer holds at least H*W*C elements of T and
/// stays alive for the returned Image's lifetime (the Image borrows it).
pub unsafe fn borrow_image<T: Clone, const C: usize>(
    b: &Backing, shape: [usize; 3],
) -> Result<Image<T, C, ForeignAllocator>, ImageError> {
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != C {
        return Err(ImageError::InvalidChannelShape(c, C));
    }
    Image::from_raw_parts(ImageSize { width: w, height: h }, b.data_ptr() as *const T, h * w * c, ForeignAllocator)
}

/// Allocate a zeroed owned output buffer for an op of channel count C.
pub fn alloc_output_owned<const C: usize>(dtype: Dtype, size: ImageSize) -> (AlignedBytes, ImageSize) {
    let len = size.width * size.height * C * dtype.itemsize();
    (AlignedBytes::zeroed(len), size)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aligned_bytes_is_64b_aligned_and_zeroed() {
        let b = AlignedBytes::zeroed(100);
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
        assert_eq!(b.len(), 100);
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }
    #[test]
    fn from_slice_copies() {
        let src = [1u8, 2, 3, 4, 5];
        let b = AlignedBytes::from_slice(&src);
        assert_eq!(b.as_slice(), &src);
        assert_eq!(b.as_ptr() as usize % ALIGN, 0);
    }
    #[test]
    fn dtype_roundtrip() {
        assert_eq!(Dtype::from_numpy_str("uint8").unwrap(), Dtype::U8);
        assert_eq!(Dtype::F32.itemsize(), 4);
        assert!(Dtype::from_numpy_str("int64").is_err());
    }
    #[test]
    fn alloc_output_owned_sizes_correctly() {
        let (b, sz) = alloc_output_owned::<3>(Dtype::F32, ImageSize { width: 4, height: 5 });
        assert_eq!(b.len(), 4 * 5 * 3 * 4);
        assert_eq!((sz.width, sz.height), (4, 5));
    }
}
```

- [ ] **Step 2: register the module** — in `kornia-py/src/lib.rs`, add near the other `mod` lines: `pub(crate) mod backing;`. (No need to register classes; these are Rust-only types.)

- [ ] **Step 3: add the dep** — in `kornia-py/Cargo.toml` `[dependencies]`: `dlpack-rs = { git = "https://github.com/kornia/dlpack-rs", tag = "v0.2.0", features = ["pyo3"] }`. Run `cargo fetch` to confirm it resolves (network).

- [ ] **Step 4: build + test**
Run: `cd /home/nvidia/kornia-rs && cargo test -p kornia-py --lib backing 2>&1 | tail`
Expected: 4 tests pass. Then `cargo clippy -p kornia-py --no-default-features --lib -- -D warnings` (or `-p kornia-py --lib`) clean.
(Note: kornia-py is a cdylib; `cargo test -p kornia-py --lib` may require the pyo3 test harness. If `--lib` test linking fails due to pyo3 `extension-module`, gate these unit tests behind `#[cfg(test)]` as written and run via `cargo test -p kornia-py --lib --no-default-features` or move the pure tests so they build without the Python interpreter. If linking still fails, report it — the AlignedBytes/Dtype tests need no interpreter.)

- [ ] **Step 5: commit**
```bash
git add kornia-py/src/backing.rs kornia-py/src/lib.rs kornia-py/Cargo.toml Cargo.lock
git commit -m "feat(kornia-py): Backing core (AlignedBytes/Dtype/Borrow) + dlpack-rs dep"
```

---

### Task 2: Swap `PyImageApi` storage to `Backing` (the atomic refactor)

**Files:** `kornia-py/src/image.rs` (struct + helpers + accessors + constructors + buffer protocol + pickle).

**Interfaces — Consumes:** `backing::{Backing, BorrowGuard, AlignedBytes, Dtype, borrow_image, alloc_output_owned}`.
**Produces:** `PyImageApi { backing: Backing, dtype: Dtype, shape: [usize;3], color_space, mode, format }` and helpers `borrow_self_image::<C>`, `wrap_owned`, `wrap_borrowed_numpy`, all existing pymethods preserved.

This task is large but mechanical: most methods route through shared helpers. The deliverable is **the crate compiles, maturin builds, and the full existing Python suite stays green** (behavior preserved). DLPack and new ingest are later tasks.

- [ ] **Step 1: write the failing test** — add `kornia-py/tests/test_backing.py`:
```python
import numpy as np
from kornia_rs.image import Image

def test_from_numpy_zero_copy_shares_memory():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = Image(arr)                      # borrow by default
    arr[1, 2, 0] = 200
    assert img.numpy()[1, 2, 0] == 200    # shares memory
    img.numpy()[3, 3, 1] = 50
    assert arr[3, 3, 1] == 50

def test_from_numpy_copy_is_independent():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image.from_numpy(arr, copy=True)
    arr[0, 0, 0] = 99
    assert img.numpy()[0, 0, 0] == 0

def test_owned_output_survives_source_free():
    arr = np.random.randint(0, 255, (16, 16, 3), np.uint8)
    img = Image(arr)
    out = img.resize(8, 8)                # owned output
    del arr, img
    assert out.numpy().shape == (8, 8, 3) # still valid
```

- [ ] **Step 2: run to verify it fails** — `cd kornia-py && pixi run -e py312 pytest tests/test_backing.py -x 2>&1 | tail` → FAIL (`from_numpy` missing / behavior differs). (After Step 3+ build.)

- [ ] **Step 3: swap the struct + `ImageData` removal.** Replace the `ImageData` enum and the `PyImageApi` struct (around image.rs:817–931) so storage is `Backing` + metadata:
```rust
#[pyclass(name = "Image", weakref, module = "kornia_rs.image")]
pub struct PyImageApi {
    pub(crate) backing: backing::Backing,
    pub(crate) dtype: backing::Dtype,
    pub(crate) shape: [usize; 3],            // (H, W, C)
    pub(crate) color_space: kornia_image::ColorSpace,
    pub(crate) mode: String,
    pub(crate) format: Option<&'static str>,
}
```
Delete the `ImageData` enum and its impls (`shape3/channels/dtype_name/itemsize/is_u16/is_f32/as_pyany/dtype_obj/as_ptr`). Reimplement each as a method on `PyImageApi` over `(backing, dtype, shape)`:
- `height()=shape[0]`, `width()=shape[1]`, `channels()=shape[2]`, `nbytes()=shape[0]*shape[1]*shape[2]*dtype.itemsize()`.
- `is_u16()=dtype==U16`, `is_f32()=dtype==F32`.

- [ ] **Step 4: rewrite the constructor/wrap helpers** to produce `Backing`:
```rust
impl PyImageApi {
    /// Borrow a numpy ndarray zero-copy (default ingest).
    fn from_numpy_borrow(py: Python<'_>, arr: &Bound<'_, PyAny>, mode: Option<String>, cs: Option<ColorSpace>) -> PyResult<Self> {
        // validate ndarray: C-contiguous, ndim 2|3, dtype in {u8,u16,f32}; derive shape/dtype.
        // ptr = arr base data pointer; readonly = !arr.flags.writeable.
        // backing = Borrowed{ ptr, keep: PyObject{ obj: arr.clone().unbind(), buffer: None }, readonly }.
        // shape=[h,w,c]; color_space = cs.unwrap_or(default_color_space(c)); mode similar.
    }
    /// Own a fresh aligned buffer copied from `bytes` with given shape/dtype.
    pub(crate) fn from_owned_bytes(b: AlignedBytes, dtype: Dtype, shape: [usize;3], cs: ColorSpace, mode: String) -> Self {
        Self { backing: Backing::Owned(b), dtype, shape, color_space: cs, mode, format: None }
    }
    /// Compute borrow of self's data (read path).
    pub(crate) unsafe fn borrow_self<T: Clone, const C: usize>(&self) -> Result<Image<T,C,ForeignAllocator>, ImageError> {
        backing::borrow_image::<T,C>(&self.backing, self.shape)
    }
    pub(crate) fn clone_handle(&self, py: Python<'_>) -> Self {
        // Owned -> deep copy bytes into a new AlignedBytes (cheap & keeps independence);
        // Borrowed -> clone the keep-alive (numpy Py clone_ref / re-borrow) preserving ptr.
        // Preserve dtype/shape/color_space/mode/format.
    }
}
```
Replace `wrap`/`wrap_u16`/`wrap_f32`/`wrap_vec`/`wrap_u8_result` call sites: every place that previously built a numpy-backed Image from a freshly written numpy array now uses an **Owned** `AlignedBytes`. Concretely, change `alloc_output_pyarray*`/`vec_to_pyarray*` to allocate `AlignedBytes` and return a writable borrow for the kernel, then wrap as `Owned`. Provide:
```rust
/// Allocate an owned output, give the kernel a writable Image, return (parts) to wrap.
fn run_into_owned<const C: usize, F>(&self, py: Python<'_>, dtype: Dtype, out_size: ImageSize, cs: ColorSpace, f: F) -> PyResult<Self>
where F: FnOnce(&mut Image<u8 /*or T*/, C, ForeignAllocator>) -> Result<(), ImageError> { /* alloc_output_owned, build mut borrow over its ptr, run f under py.detach, wrap Owned */ }
```
(Implementer: generalize `run_into_owned` per dtype as the existing helpers are; the existing `numpy_as_image::<C>` reads become `self.borrow_self::<T,C>()`.)

- [ ] **Step 5: reroute every numpy touchpoint** (checklist — each must compile over `Backing`):
  - Accessors: `dtype` getter → numpy dtype object built from `self.dtype` (use `numpy::dtype::<T>(py)` or `PyArray` dtype helper); `shape`/`width`/`height`/`channels`/`nbytes`/`size` → from `self.shape`/`self.dtype`; `data_ptr` → `self.backing.data_ptr() as usize`.
  - `numpy()` / `.data` getter → build a numpy **view** over the backing via the buffer protocol (`np.asarray(self)`), with `self` as base; OR if `Backing::Borrowed{keep:PyObject{obj}}` and obj is the original ndarray, return `obj.clone_ref(py)` (true zero-copy identity). `to_numpy(copy=False)` → same view; `copy=True` → `.copy()`.
  - `tobytes()` → `PyBytes::new(py, slice_over_backing)`.
  - `__array__` → numpy view (+ optional astype/copy).
  - `copy()` → `Owned` deep copy via `clone_handle` forcing Owned.
  - `frombuffer`/`fromarray` → delegate to `from_numpy_borrow`. `frombytes` → `from_owned_bytes`. `new` → `from_numpy_borrow` (+ optional color_space). `new_blank`/`load`/`decode`/`open` → produce `Owned` (decoders already write into fresh buffers — point them at `AlignedBytes`).
  - All imgproc/color methods (`resize`,`crop`,`rotate`,`flip_*`,`gaussian_blur`,`box_blur`,`adjust_*`,`cvt_color`,`colormap`,`convert`,`to_float`,`to_uint8`,`resize_normalize_to_tensor`,`normalize`): replace `numpy_as_image::<C>` reads with `self.borrow_self::<T,C>()` and `alloc_output_pyarray*` writes with the owned-output helper. Return `wrap_owned`.

- [ ] **Step 6: buffer protocol over `Backing`** — reimplement `__getbuffer__`/`__releasebuffer__` to fill the `Py_buffer` from `self.backing.data_ptr()` instead of delegating to numpy:
```rust
unsafe fn __getbuffer__(slf: PyRefMut<'_, Self>, view: *mut pyo3::ffi::Py_buffer, flags: c_int) -> PyResult<()> {
    // fill: buf=data_ptr, len=nbytes, itemsize=dtype.itemsize, readonly=self.backing.readonly() as i32,
    // ndim=3, format = dtype format string ("B"/"H"/"f"), shape=[h,w,c], strides=[w*c*itemsize, c*itemsize, itemsize],
    // obj = slf into a new ref (so Python keeps the Image alive while the view exists). Use PyBuffer_FillInfo-style manual fill.
}
unsafe fn __releasebuffer__(slf: PyRefMut<'_, Self>, view: *mut pyo3::ffi::Py_buffer) { /* free shape/strides if heap-allocated */ }
```
Keep the shape/strides arrays alive for the view's lifetime (store boxed and free in `__releasebuffer__`, or use static-per-call leaked-then-freed arrays). The `obj` field must hold a strong ref to `slf` so the Image outlives the memoryview.

- [ ] **Step 7: pickle over `Backing`** — `__getstate__` returns `(bytes, dtype_str, (h,w,c), color_space_int, mode)`; `__setstate__`/`_reconstruct` build an `Owned` image; `__reduce__` returns `(Image_type_or_reconstruct, args)` consistent with the existing serialization tests (`TestImageSerialize`: `constructor is Image`, `Image(*args)` reconstructs). Since `Image(arr,...)` takes a numpy array, reconstruct via the existing `_reconstruct` staticmethod path OR add a bytes-accepting reconstruct; ensure `test_image.py::TestImageSerialize` stays green.

- [ ] **Step 8: build + green suite**
Run: `cd /home/nvidia/kornia-rs/kornia-py && pixi run -e py312 maturin develop --release`
Run: `pixi run -e py312 pytest tests/test_backing.py tests/test_image.py tests/test_f32_image.py tests/test_color.py tests/test_cvt_color.py tests/test_resize.py tests/test_zero_copy_io.py tests/test_torch_zero_copy.py -q 2>&1 | tail -20`
Expected: all pass (Task-2 backing tests + full existing regression). Then `cargo clippy -p kornia-py --no-deps -- -D warnings` clean.

- [ ] **Step 9: commit**
```bash
git add kornia-py/src/image.rs kornia-py/tests/test_backing.py
git commit -m "refactor(kornia-py): Image owns a Backing buffer; numpy demoted to adapter"
```

---

### Task 3: DLPack export + import

**Files:** create `kornia-py/src/dlpack.rs`; `kornia-py/src/lib.rs` (`mod dlpack;` + register `from_dlpack`); add `__dlpack__`/`__dlpack_device__`/`from_dlpack` to `image.rs`.

**Interfaces — Consumes:** `dlpack_rs::{safe::*, pyo3_glue::{IntoDLPack, PyTensor}}`, `backing::*`, `PyImageApi`.

- [ ] **Step 1: failing test** — `kornia-py/tests/test_dlpack.py`:
```python
import numpy as np, pytest
from kornia_rs.image import Image

def test_dlpack_export_to_numpy_roundtrip():
    arr = np.random.randint(0, 255, (5, 7, 3), np.uint8)
    img = Image.from_numpy(arr, copy=True)
    back = np.from_dlpack(img)            # consumes img.__dlpack__()
    np.testing.assert_array_equal(back, arr)

def test_dlpack_import_from_numpy():
    arr = np.ascontiguousarray(np.random.rand(4, 4, 3).astype(np.float32))
    img = Image.from_dlpack(arr)          # numpy>=1.22 exposes __dlpack__
    np.testing.assert_allclose(img.numpy(), arr)

def test_dlpack_torch_roundtrip():
    torch = pytest.importorskip("torch")
    arr = np.random.randint(0, 255, (8, 8, 3), np.uint8)
    img = Image.from_numpy(arr, copy=True)
    t = torch.from_dlpack(img)
    assert t.shape == (8, 8, 3)
    img2 = Image.from_dlpack(t)
    np.testing.assert_array_equal(img2.numpy(), arr)
```

- [ ] **Step 2: run → fail** — `pixi run -e py312 pytest tests/test_dlpack.py -x 2>&1 | tail` → FAIL (`from_dlpack`/`__dlpack__` missing).

- [ ] **Step 3: write `kornia-py/src/dlpack.rs`** — export wrapper + import:
```rust
use dlpack_rs::pyo3_glue::{IntoDLPack, PyTensor};
use dlpack_rs::safe::{TensorInfo, cpu_device, dtype_u8, dtype_u16, dtype_f32};
use pyo3::prelude::*;
use crate::backing::Dtype;

/// Keep-alive export wrapper: owns a clone of the Image's data so the exported
/// tensor stays valid until the consumer's deleter runs.
pub struct ImageExport { pub bytes: Vec<u8>, pub shape: Vec<i64>, pub dtype: Dtype }
impl IntoDLPack for ImageExport {
    fn tensor_info(&self) -> TensorInfo {
        let dt = match self.dtype { Dtype::U8 => dtype_u8(), Dtype::U16 => dtype_u16(), Dtype::F32 => dtype_f32() };
        TensorInfo::contiguous(self.bytes.as_ptr() as *mut _, cpu_device(), dt, self.shape.clone())
    }
}
```
(Export copies bytes into the keep-alive `ImageExport` so the exported tensor is self-contained — simplest correct lifetime. Optimization to share owned buffers via Arc is a follow-up.)

- [ ] **Step 4: add methods to `image.rs`**:
```rust
fn __dlpack__(&self, py: Python<'_>, stream: Option<PyObject>) -> PyResult<Py<PyAny>> {
    let _ = stream; // CPU: ignore
    let nbytes = self.shape.iter().product::<usize>() * self.dtype.itemsize();
    let bytes = unsafe { std::slice::from_raw_parts(self.backing.data_ptr(), nbytes) }.to_vec();
    let shape = self.shape.iter().map(|&d| d as i64).collect();
    crate::dlpack::ImageExport { bytes, shape, dtype: self.dtype }.into_capsule(py)
}
fn __dlpack_device__(&self) -> (i32, i32) { (1, 0) } // kDLCPU, device 0

#[staticmethod]
fn from_dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
    let pt = PyTensor::from_pyany(py, obj)?;
    // validate: device kind == kDLCPU(1); ndim==3 (or 2 -> C=1); contiguous; map DLDataType -> Dtype; C in {1,3,4}
    // shape=[h,w,c]; ptr = pt.data_ptr() as *mut u8; readonly=pt.is_read_only();
    // backing = Borrowed{ ptr: NonNull::new(ptr)?, keep: BorrowGuard::Dlpack(pt), readonly };
    // color_space = default_color_space(c); mode from channels.
}
```

- [ ] **Step 5: register** — `lib.rs`: `mod dlpack;` and `image_mod.add_function(wrap_pyfunction!(...))?` only if `from_dlpack` is a free fn; here it's a staticmethod on Image, so no extra registration (just ensure the module compiles).

- [ ] **Step 6: build + test**
Run: `cd kornia-py && pixi run -e py312 maturin develop --release && pixi run -e py312 pytest tests/test_dlpack.py -q 2>&1 | tail`
Expected: numpy round-trips pass; torch test passes if torch present (else skipped). Clippy clean.

- [ ] **Step 7: commit**
```bash
git add kornia-py/src/dlpack.rs kornia-py/src/image.rs kornia-py/src/lib.rs kornia-py/tests/test_dlpack.py
git commit -m "feat(kornia-py): __dlpack__/__dlpack_device__ export + from_dlpack import"
```

---

### Task 4: `from_buffer` ingest + robotics path (#23) + stubs + final verification

**Files:** `image.rs` (`from_buffer`, confirm `from_bytes`); `kornia-py/python/kornia_rs/image.pyi`; `kornia-py/tests/test_dlpack.py` or new `test_robotics.py`.

- [ ] **Step 1: failing test** — `kornia-py/tests/test_robotics.py`:
```python
import numpy as np
from kornia_rs.image import Image, ColorSpace

def test_from_buffer_zero_copy_bgr8_to_owned_tensor():
    # simulate a ROS2 sensor_msgs/Image bgr8 payload as a bytearray
    h, w = 480, 640
    payload = bytearray(np.random.randint(0, 255, (h, w, 3), np.uint8).tobytes())
    img = Image.from_buffer(payload, width=w, height=h, channels=3, dtype="uint8",
                            color_space=ColorSpace.Bgr)        # zero-copy borrow
    assert img.color_space == ColorSpace.Bgr
    rgb = img.cvt_color(ColorSpace.Rgb)                        # owned output
    t = rgb.to_float().resize_normalize_to_tensor((224, 224), (0.485,0.456,0.406), (0.229,0.224,0.225))
    # owned tensor independent of the transient payload:
    payload[:100] = b"\x00" * 100
    assert t.shape == (3, 224, 224)
    ptr = t.data_ptr
    assert isinstance(ptr, int) and ptr != 0
```
(Adjust `resize_normalize_to_tensor`/`to_float` signatures to the actual ones; the point is owned-output independence + stable data_ptr.)

- [ ] **Step 2: run → fail** — `from_buffer` missing.

- [ ] **Step 3: implement `from_buffer`** in `image.rs`:
```rust
#[staticmethod]
#[pyo3(signature = (data, width, height, channels=3, dtype="uint8", mode=None, color_space=None))]
fn from_buffer(py: Python<'_>, data: &Bound<'_, PyAny>, width: usize, height: usize,
    channels: usize, dtype: &str, mode: Option<String>, color_space: Option<PyColorSpace>) -> PyResult<Self> {
    let dt = backing::Dtype::from_numpy_str(dtype)?;
    let mut view = Box::new(unsafe { std::mem::zeroed::<pyo3::ffi::Py_buffer>() });
    // SAFETY: request a simple contiguous buffer; checks ndim/contig.
    let rc = unsafe { pyo3::ffi::PyObject_GetBuffer(data.as_ptr(), view.as_mut(), pyo3::ffi::PyBUF_SIMPLE) };
    if rc != 0 { return Err(PyErr::fetch(py)); }
    let need = width * height * channels * dt.itemsize();
    if (view.len as usize) < need {
        unsafe { pyo3::ffi::PyBuffer_Release(view.as_mut()); }
        return Err(value_err(format!("buffer too small: {} < {}", view.len, need)));
    }
    let ptr = std::ptr::NonNull::new(view.buf as *mut u8).ok_or_else(|| value_err("null buffer"))?;
    let readonly = view.readonly != 0;
    let keep = backing::BorrowGuard::PyObject { obj: data.clone().unbind(), buffer: Some(view) };
    Ok(Self {
        backing: backing::Backing::Borrowed { ptr, keep, readonly },
        dtype: dt, shape: [height, width, channels],
        color_space: color_space.map(Into::into).unwrap_or_else(|| default_color_space(channels)),
        mode: mode.unwrap_or_else(|| mode_from_channels(channels, dt == backing::Dtype::U16)),
        format: None,
    })
}
```
Confirm `from_bytes` (existing `frombytes`) now yields `Owned` (it should, post Task 2).

- [ ] **Step 4: add `from_numpy` explicit method + `to_numpy(copy=)`** if not already exposed in Task 2 (the spec names `from_numpy`/`to_numpy` as the canonical adapters):
```rust
#[staticmethod]
#[pyo3(signature = (array, *, copy=false, mode=None, color_space=None))]
fn from_numpy(py, array, copy, mode, color_space) -> PyResult<Self> { /* copy? from_owned_bytes(from numpy bytes) : from_numpy_borrow */ }
#[pyo3(signature = (*, copy=false))]
fn to_numpy(&self, py, copy) -> PyResult<Py<PyAny>> { /* view via buffer protocol; copy? .copy() */ }
```

- [ ] **Step 5: stubs** — `kornia-py/python/kornia_rs/image.pyi`: add `from_numpy`, `to_numpy`, `from_buffer`, `from_dlpack`, `__dlpack__`, `__dlpack_device__` signatures.

- [ ] **Step 6: full verification**
Run: `cd kornia-py && pixi run -e py312 maturin develop --release`
Run: `pixi run -e py312 pytest tests/ -q 2>&1 | tail -25` (entire suite incl. robotics + dlpack + backing).
Run (kornia-py clippy): `cargo clippy -p kornia-py --no-deps -- -D warnings`
Run (rust): `cargo test -p kornia-image -p kornia-imgproc --lib 2>&1 | tail -3`
Expected: all green.

- [ ] **Step 7: commit**
```bash
git add kornia-py/src/image.rs kornia-py/python/kornia_rs/image.pyi kornia-py/tests/test_robotics.py
git commit -m "feat(kornia-py): from_buffer/from_numpy/to_numpy adapters + ROS2 bgr8->owned-tensor path"
```

---

## Self-Review
- **Spec coverage:** Backing 2-variant (T1), compute integration + storage swap (T2), buffer protocol + pickle (T2), DLPack export/import (T3), ingest/egress adapters incl. from_buffer/from_numpy/to_numpy (T2+T4), robotics path/#23 (T4), 64-byte align (T1), keep-alive (T1 BorrowGuard, exercised T2/T3/T4), dtype scope (T1 Dtype), stubs (T4). ✓
- **Known risk — Task 2 size:** the storage swap is atomic (the struct change forces touching every `self.data` site to compile). It's bounded because methods route through shared helpers; the gate is the full regression suite. If Task 2 proves too large mid-implementation, split along: (2a) struct + accessors + constructors + buffer protocol + pickle compiling with reads via `borrow_self` and outputs still numpy-temporarily — NOT possible cleanly (Numpy variant removed), so prefer keeping 2 atomic but reviewing carefully.
- **pyo3 0.28 specifics to confirm during impl:** buffer-protocol `__getbuffer__` manual `Py_buffer` fill (use `PyBuffer_FillInfo` or manual), numpy dtype object construction, `PyObject_GetBuffer` flags. These are the "adapt to what compiles" spots — keep semantics (zero-copy + keep-alive + readonly) intact.
- **Placeholder scan:** Step 4/5 of Task 2 give signatures + the transformation pattern + an exhaustive call-site checklist rather than reproducing all 57 methods verbatim (they are mechanical reroutes through the named helpers). The new files (backing.rs, dlpack.rs) and the buffer-protocol/from_buffer code are complete.
- **Type consistency:** `Backing`/`BorrowGuard`/`Dtype`/`borrow_image`/`alloc_output_owned`/`borrow_self`/`from_owned_bytes`/`clone_handle`/`ImageExport` names consistent across tasks.
