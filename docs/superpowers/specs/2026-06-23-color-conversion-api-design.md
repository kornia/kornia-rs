# High-level color-conversion API — design spec

**Date:** 2026-06-23
**Status:** Approved (design)
**Target PR:** #944 (`feat/gray-neon-kernels`) — lands in the same PR as the color module.

## Goal

Add an ergonomic, compact, user-friendly top layer for color conversion that is
Rust-idiomatic and mirrors 1:1 into Python. The kernels, newtype color-space
wrappers, and the `ConvertColor` trait already exist; what is missing is the
*ergonomic surface*. Today a conversion in Rust requires pre-allocating the
destination:

```rust
let mut hsv = Hsvf32::from_size_val(rgb.size(), 0.0, alloc)?;
rgb.convert(&mut hsv)?;          // ConvertColor — type-safe but verbose
```

and Python exposes only loose free functions (`kornia_rs.imgproc.hsv_from_rgb(arr)`),
not methods on `Image`.

## Approved decisions

1. **Stateful, tagged Image** — the Image carries its `color_space`; conversions
   read the source automatically, validate the path, and return a tagged result.
2. **Hybrid dispatch** — a zero-cost typed `.cvt()` for Rust, plus a shared
   runtime `cvt_color(ColorSpace)` that Python binds 1:1. One `ColorSpace` enum
   and one `(from, to)` dispatch table feed every path.
3. **DynImage return** for the runtime path — a small `DynImage<T,A>` enum unifies
   the Rust and Python dynamic-Image model and supports channel-changing
   conversions (gray=1, rgb=3, rgba=4).
4. **Strict dtype policy** — converting a `u8` image into an f32-only space
   (HSV/HLS/Lab/Luv/XYZ/LinearRgb) raises a clear error; the user calls
   `.to_float()` first. Typed Rust `.cvt()` is always explicit (no implicit cast).
5. **Non-adjacent pairs error** — no silent auto-routing in v1 (e.g. `Hsv→Lab`
   errors with a hint to go via `Rgb`). Auto-routing can be added later.

Everything is **additive**: no existing kernel, free function, or binding changes
behavior. The new layer sits on top.

---

## Section 1 — Shared vocabulary: `ColorSpace` enum + conversion graph

A single `ColorSpace` enum in `kornia-image`, used by Rust and exported as a
`#[pyclass]` enum to Python — the shared vocabulary across the FFI bridge.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace { Rgb, Bgr, Gray, Rgba, Bgra, Hsv, Hls, Lab, Luv, Xyz, LinearRgb, YCbCr, Yuv }

impl ColorSpace {
    pub const fn channels(self) -> usize { /* Gray=1, Rgba/Bgra=4, else 3 */ }
    pub const fn requires_f32(self) -> bool { /* Hsv,Hls,Lab,Luv,Xyz,LinearRgb */ }
}
```

The conversion graph is **RGB-hub-centric**, matching the existing kernels. A
static table maps `(from, to) → kernel`. The legal set is exactly the conversions
with a direct kernel today (see the `ConvertColor` matrix in
`crates/kornia-imgproc/src/color/convert.rs`). Non-adjacent pairs error.

**Scope boundary (YAGNI):** packed/planar video formats (`Nv12`, `Yuyv8`,
`Bayer8`, …) keep their existing dedicated typed APIs — they have
non-`Image<T,3>` shapes and do not belong in a per-pixel color-space enum.
`ColorSpace` covers only true per-pixel color spaces.

---

## Section 2 — Rust typed layer (zero-cost): `.cvt()`

An extension trait gives every color newtype an allocating, type-inferred
conversion, built on the existing `ConvertColor`:

```rust
let hsv: Hsvf32<_> = rgb.cvt()?;          // inference picks the kernel
let lab            = rgb.cvt::<Labf32<_>>()?;
let gray           = rgb.cvt::<Gray8<_>>()?;   // channel change is natural — Dst encodes C
```

Mechanism:

```rust
pub trait NewColorImage: Sized {
    fn new_uninit<A: ImageAllocator>(size: ImageSize, alloc: A) -> Result<Self, ImageError>;
}

pub trait ConvertColorExt {
    fn cvt<Dst>(&self) -> Result<Dst, ImageError>
    where Self: ConvertColor<Dst>, Dst: NewColorImage;
}
```

`cvt()` allocates a correctly-sized **owned** output (`CpuAllocator`) via
`NewColorImage` (impl'd by the newtypes through their existing `from_size_val`),
then delegates to `ConvertColor::convert`. No new kernels — pure ergonomic sugar,
fully type-safe and zero-cost. The Rust typed path performs **no implicit dtype
cast**.

---

## Section 3 — Rust runtime layer + `DynImage`

A runtime `cvt_color(ColorSpace)` can return images of different channel counts,
so it returns a small owned dynamic enum that mirrors Python's dynamic Image:

```rust
pub enum DynImage<T, A: ImageAllocator> {
    C1(Image<T, 1, A>),
    C3(Image<T, 3, A>),
    C4(Image<T, 4, A>),
}

impl<T, A: ImageAllocator> DynImage<T, A> {
    pub fn color_space(&self) -> ColorSpace;
    pub fn size(&self) -> ImageSize;
    pub fn channels(&self) -> usize;
    pub fn as_slice(&self) -> &[T];
    pub fn try_into_c3(self) -> Result<Image<T, 3, A>, ImageError>;   // + c1/c4
}

// runtime entry — the same table the typed path and Python use.
// A plain Image carries no color-space tag, so `from` is explicit:
pub fn cvt_color<T, const C: usize, A>(
    src: &Image<T, C, A>, from: ColorSpace, to: ColorSpace,
) -> Result<DynImage<T, CpuAllocator>, ImageError>;

// A color newtype DOES encode its source space, so the method form infers `from`:
impl<A: ImageAllocator> Rgbf32<A> { pub fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<f32, CpuAllocator>, ImageError>; }
// (provided via a small `Tagged` trait blanket-impl'd for every newtype)
```

Usage:

```rust
let out = rgb.cvt_color(ColorSpace::Hsv)?;            // method on newtype — `from` inferred
let g   = rgb.cvt_color(ColorSpace::Gray)?;           // C1 variant
let out = cvt_color(&plain_img, ColorSpace::Rgb, ColorSpace::Hsv)?;  // free fn — `from` explicit
let hsv: Hsvf32<_> = out.try_into()?;                 // recover typed via TryFrom
```

`DynImage` carries the result's `ColorSpace`. `TryFrom<DynImage>` for each newtype
recovers the typed form, checking channel count and (where relevant) dtype.

The runtime function dispatches via the same `(from, to)` table as the typed
path, so the legal conversion set is identical everywhere.

---

## Section 4 — Python `Image` API (stateful, method-based)

The `Image` class (`PyImageApi`) gains a `color_space` field (defaults `Rgb` on
`from_numpy`) and method-based conversion — the recommended Python API:

```python
img  = Image.from_numpy(arr)                       # .color_space == ColorSpace.RGB
hsv  = img.to_float().cvt_color(ColorSpace.HSV)    # strict dtype: to_float() first
gray = img.cvt_color(ColorSpace.GRAY)              # u8 ok (not f32-only)
print(gray.color_space, gray.numpy().shape)        # GRAY, (H, W, 1)

img.to_gray(); img.to_hsv(); img.to_lab()          # thin sugar over cvt_color
img.to_float(); img.to_uint8()                     # dtype helpers
```

- `ColorSpace` exported as a `#[pyclass]` enum.
- Every conversion validates `(src.color_space, to)` against the shared table,
  releases the GIL, runs the monomorphized kernel into a fresh numpy array, and
  returns a **new tagged `Image`** (zero-copy in, single alloc out).
- Existing free functions remain for back-compat; methods are the headline API.
- Sugar methods (`to_gray`, `to_hsv`, …) are thin wrappers over `cvt_color`.
- `to_float()` casts `u8→f32` via `/255`; `to_uint8()` is the inverse (`*255`,
  saturating). These are the only dtype bridges the conversion layer needs.

---

## Section 5 — Error handling

Two precise errors in `kornia-image`, surfaced identically in Rust and Python
(mapped to `ValueError` in Python):

- `UnsupportedColorConversion { from: ColorSpace, to: ColorSpace }`
  → *"no direct Hsv→Lab path; convert via Rgb"*.
- `InvalidColorDtype { space: ColorSpace, expected: &str, got: &str }`
  → *"Hsv requires f32; call .to_float() first"*.

The static `(from, to)` table is the single source of truth, so the Rust typed
path, Rust runtime path, and Python all reject the same set.

---

## Section 6 — Testing

**Rust:**
- `.cvt()` round-trips for each bidirectional pair (`rgb.cvt::<Hsvf32<_>>()?.cvt::<Rgbf32<_>>()?` ≈ identity within kernel tolerance).
- `DynImage` variant + `color_space()` tag correctness; `try_into`/`TryFrom` typed recovery, including channel/dtype mismatch errors.
- `cvt_color` parity vs the direct free functions (identical output).
- `UnsupportedColorConversion` and `InvalidColorDtype` error cases.

**Python:**
- Color-space tag propagation through chained conversions.
- `cvt_color` parity vs the existing free functions (byte/oracle identical).
- Strict-dtype error + `to_float()` recovery; `to_uint8()` round-trip.
- Sugar methods and `.color_space` getter.

---

## Section 7 — File layout

| File | Change |
|---|---|
| `crates/kornia-image/src/color_space.rs` *(new)* | `ColorSpace` enum, `DynImage<T,A>`, channel/dtype metadata, `(from,to)` table, new error variants |
| `crates/kornia-image/src/error.rs` | add `UnsupportedColorConversion`, `InvalidColorDtype` |
| `crates/kornia-imgproc/src/color/convert.rs` | `NewColorImage`, `ConvertColorExt::cvt()`, runtime `cvt_color()`, `TryFrom<DynImage>` for newtypes |
| `crates/kornia-imgproc/src/color/mod.rs` | re-export the new surface |
| `kornia-py/src/image.rs` | `color_space` field, `cvt_color`/sugar/`to_float`/`to_uint8`, `ColorSpace` pyclass |
| `kornia-py/src/lib.rs` | register `ColorSpace` enum |
| `kornia-py/python/kornia_rs/*.pyi` | stubs for new methods + `ColorSpace` enum |

All additive; no behavior change to existing APIs.

## Verification

```
cargo test -p kornia-image --lib
cargo test -p kornia-imgproc --lib color
cargo clippy -p kornia-image -p kornia-imgproc --all-targets --features ci -- -D warnings
cd kornia-py && pixi run -e py312 maturin develop --release
pixi run -e py312 pytest -s tests/test_color.py
```
