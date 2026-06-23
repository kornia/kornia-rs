# High-level color-conversion API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ergonomic, stateful color-conversion API — zero-cost typed `.cvt()` for Rust plus a shared runtime `cvt_color(ColorSpace)` bound 1:1 in Python — on top of the existing kernels and `ConvertColor` trait.

**Architecture:** A single `ColorSpace` enum + static `(from,to)` legality table is the shared vocabulary. Rust gets a typed `.cvt::<Dst>()` extension and a runtime `Tagged::cvt_color` returning a new `DynImage<T,A>` enum (C1/C3/C4). Python's `Image` gains a `color_space` field and a `cvt_color` method that dispatches against the same table. Everything is additive over the current `ConvertColor` impls; no kernel changes.

**Tech Stack:** Rust (kornia-image, kornia-imgproc), pyo3/maturin (kornia-py), numpy, pytest.

## Global Constraints

- Naming convention `<output>_from_<input>` for kernels; do not introduce `rgb_to_*` names (existing repo rule, `SIMD.md`).
- New public Rust functions return named structs/enums, not tuples; new Python bindings are `#[pyclass]` where applicable.
- All work lands on branch `feat/gray-neon-kernels` (PR #944). Commit per task.
- Additive only: no existing kernel/free-function/binding changes behavior.
- Strict dtype policy: converting `u8` → an f32-only space (Hsv/Hls/Lab/Luv/Xyz/LinearRgb) is an error, not an implicit cast, in the runtime/Python layer. Typed `.cvt()` is always explicit.
- Legal conversions = exactly the pairs with a `ConvertColor` impl today (RGB-hub graph). Non-adjacent pairs error. No auto-routing in v1.
- Video/packed formats (Nv12, Yuyv8, Bayer8, …) are out of scope for `ColorSpace`.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/kornia-image/src/color_space.rs` *(new)* | `ColorSpace` enum, metadata (`channels`, `requires_f32`), `(from,to)` legality table, `DynImage<T,A>` + accessors + `TryFrom` into newtypes |
| `crates/kornia-image/src/error.rs` | add `UnsupportedColorConversion`, `InvalidColorDtype` variants |
| `crates/kornia-image/src/lib.rs` | `pub mod color_space;` + re-export `ColorSpace`, `DynImage` |
| `crates/kornia-imgproc/src/color/convert.rs` | `NewColorImage`, `ConvertColorExt::cvt()`, `Tagged::cvt_color()` runtime dispatch |
| `crates/kornia-imgproc/src/color/mod.rs` | re-export `ConvertColorExt`, `Tagged`, `NewColorImage` |
| `kornia-py/src/image.rs` | `color_space` field on `PyImageApi`, getter, `cvt_color`/sugar/`to_float`/`to_uint8` |
| `kornia-py/src/color_space.rs` *(new)* | `PyColorSpace` pyclass enum + `From`/`Into` `ColorSpace` |
| `kornia-py/src/lib.rs` | register `PyColorSpace` |
| `kornia-py/python/kornia_rs/image.pyi` | stubs: `ColorSpace`, new `Image` methods |
| `kornia-py/tests/test_cvt_color.py` *(new)* | Python regression tests |

---

### Task 1: `ColorSpace` enum + metadata + legality table + error variants

**Files:**
- Create: `crates/kornia-image/src/color_space.rs`
- Modify: `crates/kornia-image/src/error.rs` (after line 62), `crates/kornia-image/src/lib.rs:17`
- Test: inline `#[cfg(test)]` in `color_space.rs`

**Interfaces:**
- Produces:
  - `enum ColorSpace { Rgb, Bgr, Gray, Rgba, Bgra, Hsv, Hls, Lab, Luv, Xyz, LinearRgb, YCbCr, Yuv }` (derives `Clone, Copy, Debug, PartialEq, Eq, Hash`)
  - `ColorSpace::channels(self) -> usize`
  - `ColorSpace::requires_f32(self) -> bool`
  - `ColorSpace::supports(from: ColorSpace, to: ColorSpace) -> bool`
  - `ImageError::UnsupportedColorConversion { from: ColorSpace, to: ColorSpace }`
  - `ImageError::InvalidColorDtype { space: ColorSpace, expected: &'static str, got: &'static str }`

- [ ] **Step 1: Add error variants**

In `crates/kornia-image/src/error.rs`, after the `UnsupportedInterpolation` variant (line 62), add:

```rust
    /// No direct kernel exists for this color-space pair.
    #[error("no direct {from:?}->{to:?} color conversion; convert via Rgb")]
    UnsupportedColorConversion {
        from: crate::color_space::ColorSpace,
        to: crate::color_space::ColorSpace,
    },

    /// The color space requires a different element type than the image holds.
    #[error("{space:?} requires {expected} data, got {got}")]
    InvalidColorDtype {
        space: crate::color_space::ColorSpace,
        expected: &'static str,
        got: &'static str,
    },
```

- [ ] **Step 2: Write the failing test (create the file with tests first)**

Create `crates/kornia-image/src/color_space.rs` with ONLY the enum stub and tests so it compiles to a failing assertion:

```rust
//! Runtime color-space vocabulary shared by Rust and Python.

/// A per-pixel color space. The shared vocabulary for the high-level
/// conversion API (`.cvt()` typed path and `cvt_color` runtime path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Rgb, Bgr, Gray, Rgba, Bgra, Hsv, Hls, Lab, Luv, Xyz, LinearRgb, YCbCr, Yuv,
}

impl ColorSpace {
    /// Number of channels an image in this space has.
    pub const fn channels(self) -> usize {
        match self {
            ColorSpace::Gray => 1,
            ColorSpace::Rgba | ColorSpace::Bgra => 4,
            _ => 3,
        }
    }

    /// True for spaces whose kernels only operate on f32 data.
    pub const fn requires_f32(self) -> bool {
        matches!(
            self,
            ColorSpace::Hsv | ColorSpace::Hls | ColorSpace::Lab
                | ColorSpace::Luv | ColorSpace::Xyz | ColorSpace::LinearRgb
        )
    }

    /// Whether a direct kernel exists for `from -> to`. Mirrors the
    /// `ConvertColor` impls in kornia-imgproc (RGB-hub graph).
    pub const fn supports(from: ColorSpace, to: ColorSpace) -> bool {
        use ColorSpace::*;
        matches!(
            (from, to),
            (Rgb, Gray) | (Gray, Rgb)
                | (Rgb, Bgr) | (Bgr, Rgb)
                | (Rgb, Rgba) | (Rgba, Rgb)
                | (Rgb, Bgra) | (Bgra, Rgb)
                | (Rgb, Hsv) | (Hsv, Rgb)
                | (Rgb, Hls) | (Hls, Rgb)
                | (Rgb, Lab) | (Lab, Rgb)
                | (Rgb, Luv) | (Luv, Rgb)
                | (Rgb, Xyz) | (Xyz, Rgb)
                | (Rgb, LinearRgb) | (LinearRgb, Rgb)
                | (Rgb, YCbCr) | (YCbCr, Rgb)
                | (Rgb, Yuv) | (Yuv, Rgb)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ColorSpace;

    #[test]
    fn channels_and_dtype_metadata() {
        assert_eq!(ColorSpace::Gray.channels(), 1);
        assert_eq!(ColorSpace::Rgb.channels(), 3);
        assert_eq!(ColorSpace::Rgba.channels(), 4);
        assert!(ColorSpace::Hsv.requires_f32());
        assert!(!ColorSpace::Gray.requires_f32());
        assert!(!ColorSpace::Bgr.requires_f32());
    }

    #[test]
    fn legality_table_matches_rgb_hub() {
        assert!(ColorSpace::supports(ColorSpace::Rgb, ColorSpace::Hsv));
        assert!(ColorSpace::supports(ColorSpace::Hsv, ColorSpace::Rgb));
        assert!(ColorSpace::supports(ColorSpace::Rgb, ColorSpace::Gray));
        // non-adjacent pair is rejected
        assert!(!ColorSpace::supports(ColorSpace::Hsv, ColorSpace::Lab));
        assert!(!ColorSpace::supports(ColorSpace::Gray, ColorSpace::Hsv));
    }
}
```

- [ ] **Step 3: Register the module**

In `crates/kornia-image/src/lib.rs`, after `pub mod color_spaces;` (line 17) add:

```rust
pub mod color_space;
```

and after line 20 add:

```rust
pub use crate::color_space::{ColorSpace, DynImage};
```

(`DynImage` is added in Task 2; if running Task 1 in isolation, temporarily re-export only `ColorSpace` and add `DynImage` in Task 2.)

- [ ] **Step 4: Run tests**

Run: `cargo test -p kornia-image --lib color_space`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add crates/kornia-image/src/color_space.rs crates/kornia-image/src/error.rs crates/kornia-image/src/lib.rs
git commit -m "feat(color): ColorSpace enum, legality table, conversion errors"
```

---

### Task 2: `DynImage<T,A>` enum + accessors + typed recovery

**Files:**
- Modify: `crates/kornia-image/src/color_space.rs` (append)
- Test: inline `#[cfg(test)]` in `color_space.rs`

**Interfaces:**
- Consumes: `ColorSpace` (Task 1); `Image`, `ImageAllocator`, `ImageError`, `ImageSize`, color-space newtypes (`Rgbf32`, `Grayf32`, `Gray8`, …) from `kornia_image`.
- Produces:
  - `enum DynImage<T, A: ImageAllocator> { C1(ColorSpace, Image<T,1,A>), C3(ColorSpace, Image<T,3,A>), C4(ColorSpace, Image<T,4,A>) }`
  - `DynImage::color_space(&self) -> ColorSpace`
  - `DynImage::size(&self) -> ImageSize`
  - `DynImage::channels(&self) -> usize`
  - `DynImage::as_slice(&self) -> &[T]`
  - `TryFrom<DynImage<f32,A>> for Rgbf32<A>` and the same for each newtype (channel + space checked)

- [ ] **Step 1: Write the failing test**

Append to `crates/kornia-image/src/color_space.rs`'s `tests` module:

```rust
    use crate::allocator::CpuAllocator;
    use crate::color_spaces::{Grayf32, Rgbf32};
    use crate::{ColorSpace as CS, DynImage, ImageSize};
    use std::convert::TryFrom;

    #[test]
    fn dyn_image_tag_size_and_recovery() {
        let size = ImageSize { width: 2, height: 2 };
        let rgb = Rgbf32::from_size_val(size, 0.25, CpuAllocator).unwrap();
        let dynimg = DynImage::C3(CS::Rgb, rgb.into_inner());
        assert_eq!(dynimg.color_space(), CS::Rgb);
        assert_eq!(dynimg.channels(), 3);
        assert_eq!(dynimg.size(), size);
        // typed recovery succeeds for matching space+channels
        let back: Rgbf32<_> = Rgbf32::try_from(dynimg).unwrap();
        assert_eq!(back.as_slice()[0], 0.25);
    }

    #[test]
    fn dyn_image_recovery_rejects_wrong_space() {
        let size = ImageSize { width: 2, height: 2 };
        let gray = Grayf32::from_size_val(size, 0.0, CpuAllocator).unwrap();
        let dynimg = DynImage::C1(CS::Gray, gray.into_inner());
        // recovering as Rgbf32 must fail (channel mismatch C1 vs C3)
        assert!(Rgbf32::try_from(dynimg).is_err());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p kornia-image --lib color_space::tests::dyn_image`
Expected: FAIL — `DynImage` not found / `TryFrom` not implemented.

- [ ] **Step 3: Implement `DynImage` + recovery**

Append to `color_space.rs` (before the `tests` module):

```rust
use crate::{allocator::ImageAllocator, error::ImageError, image::Image, image::ImageSize};

/// Owned image whose channel count is known only at runtime. Mirrors Python's
/// dynamic numpy-backed Image; produced by the runtime `cvt_color` path.
pub enum DynImage<T, A: ImageAllocator> {
    /// 1-channel (e.g. Gray).
    C1(ColorSpace, Image<T, 1, A>),
    /// 3-channel (Rgb/Bgr/Hsv/...).
    C3(ColorSpace, Image<T, 3, A>),
    /// 4-channel (Rgba/Bgra).
    C4(ColorSpace, Image<T, 4, A>),
}

impl<T, A: ImageAllocator> DynImage<T, A> {
    /// The color space tag carried by this image.
    pub fn color_space(&self) -> ColorSpace {
        match self {
            DynImage::C1(s, _) | DynImage::C3(s, _) | DynImage::C4(s, _) => *s,
        }
    }

    /// Image dimensions.
    pub fn size(&self) -> ImageSize {
        match self {
            DynImage::C1(_, i) => i.size(),
            DynImage::C3(_, i) => i.size(),
            DynImage::C4(_, i) => i.size(),
        }
    }

    /// Channel count (1, 3, or 4).
    pub fn channels(&self) -> usize {
        match self {
            DynImage::C1(..) => 1,
            DynImage::C3(..) => 3,
            DynImage::C4(..) => 4,
        }
    }

    /// Contiguous (H, W, C) row-major data.
    pub fn as_slice(&self) -> &[T] {
        match self {
            DynImage::C1(_, i) => i.as_slice(),
            DynImage::C3(_, i) => i.as_slice(),
            DynImage::C4(_, i) => i.as_slice(),
        }
    }
}
```

Then add a recovery macro + invocations (covers every newtype the runtime path can emit):

```rust
macro_rules! impl_try_from_dyn {
    ($newtype:ident, $t:ty, C1, $space:expr) => {
        impl<A: ImageAllocator> std::convert::TryFrom<DynImage<$t, A>>
            for crate::color_spaces::$newtype<A>
        {
            type Error = ImageError;
            fn try_from(d: DynImage<$t, A>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C1(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::UnsupportedColorConversion {
                        from: other.color_space(), to: $space,
                    }),
                }
            }
        }
    };
    ($newtype:ident, $t:ty, C3, $space:expr) => {
        impl<A: ImageAllocator> std::convert::TryFrom<DynImage<$t, A>>
            for crate::color_spaces::$newtype<A>
        {
            type Error = ImageError;
            fn try_from(d: DynImage<$t, A>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C3(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::UnsupportedColorConversion {
                        from: other.color_space(), to: $space,
                    }),
                }
            }
        }
    };
    ($newtype:ident, $t:ty, C4, $space:expr) => {
        impl<A: ImageAllocator> std::convert::TryFrom<DynImage<$t, A>>
            for crate::color_spaces::$newtype<A>
        {
            type Error = ImageError;
            fn try_from(d: DynImage<$t, A>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C4(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::UnsupportedColorConversion {
                        from: other.color_space(), to: $space,
                    }),
                }
            }
        }
    };
}

impl_try_from_dyn!(Rgbf32, f32, C3, ColorSpace::Rgb);
impl_try_from_dyn!(Bgrf32, f32, C3, ColorSpace::Bgr);
impl_try_from_dyn!(Grayf32, f32, C1, ColorSpace::Gray);
impl_try_from_dyn!(Hsvf32, f32, C3, ColorSpace::Hsv);
impl_try_from_dyn!(Hlsf32, f32, C3, ColorSpace::Hls);
impl_try_from_dyn!(Labf32, f32, C3, ColorSpace::Lab);
impl_try_from_dyn!(Luvf32, f32, C3, ColorSpace::Luv);
impl_try_from_dyn!(Xyzf32, f32, C3, ColorSpace::Xyz);
impl_try_from_dyn!(LinearRgbf32, f32, C3, ColorSpace::LinearRgb);
impl_try_from_dyn!(YCbCrf32, f32, C3, ColorSpace::YCbCr);
impl_try_from_dyn!(Yuvf32, f32, C3, ColorSpace::Yuv);
impl_try_from_dyn!(Rgb8, u8, C3, ColorSpace::Rgb);
impl_try_from_dyn!(Bgr8, u8, C3, ColorSpace::Bgr);
impl_try_from_dyn!(Gray8, u8, C1, ColorSpace::Gray);
impl_try_from_dyn!(Rgba8, u8, C4, ColorSpace::Rgba);
impl_try_from_dyn!(Bgra8, u8, C4, ColorSpace::Bgra);
impl_try_from_dyn!(YCbCr8, u8, C3, ColorSpace::YCbCr);
impl_try_from_dyn!(Yuv8, u8, C3, ColorSpace::Yuv);
```

Add the necessary `use` for the newtype names at the top of `color_space.rs`:

```rust
use crate::color_spaces::{
    Bgr8, Bgrf32, Gray8, Grayf32, Hlsf32, Hsvf32, Labf32, LinearRgbf32, Luvf32,
    Rgb8, Rgba8, Bgra8, Rgbf32, Xyzf32, YCbCr8, YCbCrf32, Yuv8, Yuvf32,
};
```

Confirm `DynImage` is re-exported from `lib.rs` (Task 1, Step 3).

- [ ] **Step 4: Run tests**

Run: `cargo test -p kornia-image --lib color_space`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add crates/kornia-image/src/color_space.rs crates/kornia-image/src/lib.rs
git commit -m "feat(color): DynImage runtime image + typed recovery via TryFrom"
```

---

### Task 3: Rust typed layer — `NewColorImage` + `ConvertColorExt::cvt()`

**Files:**
- Modify: `crates/kornia-imgproc/src/color/convert.rs` (append after existing impls), `crates/kornia-imgproc/src/color/mod.rs` (re-export)
- Test: inline `#[cfg(test)]` in `convert.rs`

**Interfaces:**
- Consumes: `ConvertColor` (existing, convert.rs:34); newtypes; `CpuAllocator`; `ImageSize`.
- Produces:
  - `trait NewColorImage: Sized { fn new_zeroed(size: ImageSize) -> Result<Self, ImageError>; }` (impl for each newtype, `CpuAllocator`-backed)
  - `trait ConvertColorExt { fn cvt<Dst>(&self) -> Result<Dst, ImageError> where Self: ConvertColor<Dst>, Dst: NewColorImage + HasSize; }`
  - blanket `impl<Src> ConvertColorExt for Src`

- [ ] **Step 1: Write the failing test**

Append to `convert.rs` a test module:

```rust
#[cfg(test)]
mod cvt_ext_tests {
    use super::*;
    use crate::color::ConvertColorExt;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::color_spaces::{Grayf32, Hsvf32, Rgbf32};
    use kornia_image::ImageSize;

    #[test]
    fn cvt_allocates_and_converts_typed() {
        let size = ImageSize { width: 4, height: 3 };
        let rgb = Rgbf32::from_size_vec(size, vec![0.5f32; 4 * 3 * 3], CpuAllocator).unwrap();
        // typed, allocating: no manual dst construction
        let hsv: Hsvf32<_> = rgb.cvt().unwrap();
        assert_eq!(hsv.size(), size);
        // channel-changing conversion is natural — Dst encodes C
        let gray: Grayf32<_> = rgb.cvt().unwrap();
        assert_eq!(gray.num_channels(), 1);
    }

    #[test]
    fn cvt_round_trip_rgb_hsv() {
        let size = ImageSize { width: 8, height: 8 };
        let data: Vec<f32> = (0..8 * 8 * 3).map(|i| (i % 255) as f32 / 255.0).collect();
        let rgb = Rgbf32::from_size_vec(size, data.clone(), CpuAllocator).unwrap();
        let hsv: Hsvf32<_> = rgb.cvt().unwrap();
        let back: Rgbf32<_> = hsv.cvt().unwrap();
        for (a, b) in data.iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 1e-3, "round-trip drift {a} vs {b}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p kornia-imgproc --lib cvt_ext_tests`
Expected: FAIL — `cvt` / `ConvertColorExt` not found.

- [ ] **Step 3: Implement `NewColorImage` + `ConvertColorExt`**

Append to `convert.rs` (use the newtypes already imported at the top; add any missing to the `use` block):

```rust
use kornia_image::ImageSize;

/// Allocate a zeroed owned image of a color-space newtype at a given size.
/// Backed by `CpuAllocator` so `.cvt()` returns owned data.
pub trait NewColorImage: Sized {
    /// Allocate a zero-filled image of `size`.
    fn new_zeroed(size: ImageSize) -> Result<Self, ImageError>;
    /// Size of an already-constructed instance (used to size the output).
    fn size_of(&self) -> ImageSize;
}

macro_rules! impl_new_color_image {
    ($newtype:ident, $t:ty) => {
        impl NewColorImage for kornia_image::color_spaces::$newtype<kornia_image::allocator::CpuAllocator> {
            fn new_zeroed(size: ImageSize) -> Result<Self, ImageError> {
                kornia_image::color_spaces::$newtype::from_size_val(
                    size, <$t>::default(), kornia_image::allocator::CpuAllocator,
                )
            }
            fn size_of(&self) -> ImageSize { self.size() }
        }
    };
}

impl_new_color_image!(Rgbf32, f32);
impl_new_color_image!(Bgrf32, f32);
impl_new_color_image!(Grayf32, f32);
impl_new_color_image!(Hsvf32, f32);
impl_new_color_image!(Hlsf32, f32);
impl_new_color_image!(Labf32, f32);
impl_new_color_image!(Luvf32, f32);
impl_new_color_image!(Xyzf32, f32);
impl_new_color_image!(LinearRgbf32, f32);
impl_new_color_image!(YCbCrf32, f32);
impl_new_color_image!(Yuvf32, f32);
impl_new_color_image!(Rgb8, u8);
impl_new_color_image!(Bgr8, u8);
impl_new_color_image!(Gray8, u8);
impl_new_color_image!(Rgba8, u8);
impl_new_color_image!(Bgra8, u8);
impl_new_color_image!(YCbCr8, u8);
impl_new_color_image!(Yuv8, u8);

/// Ergonomic allocating conversion built on `ConvertColor`. Zero-cost sugar:
/// allocates the correctly-sized owned destination and delegates to the
/// existing kernel. The source must expose its size via `.size()` (all
/// newtypes do, through `Deref` to `Image`).
pub trait ConvertColorExt {
    /// Allocate and convert to `Dst`. `Dst` is chosen by inference or turbofish.
    fn cvt<Dst>(&self) -> Result<Dst, ImageError>
    where
        Self: ConvertColor<Dst> + SrcSize,
        Dst: NewColorImage;
}

/// Helper so `cvt` can size the destination from the source.
pub trait SrcSize {
    /// Source image size.
    fn src_size(&self) -> ImageSize;
}

macro_rules! impl_src_size {
    ($newtype:ident) => {
        impl<A: ImageAllocator> SrcSize for kornia_image::color_spaces::$newtype<A> {
            fn src_size(&self) -> ImageSize { self.size() }
        }
    };
}
impl_src_size!(Rgbf32); impl_src_size!(Bgrf32); impl_src_size!(Grayf32);
impl_src_size!(Hsvf32); impl_src_size!(Hlsf32); impl_src_size!(Labf32);
impl_src_size!(Luvf32); impl_src_size!(Xyzf32); impl_src_size!(LinearRgbf32);
impl_src_size!(YCbCrf32); impl_src_size!(Yuvf32);
impl_src_size!(Rgb8); impl_src_size!(Bgr8); impl_src_size!(Gray8);
impl_src_size!(Rgba8); impl_src_size!(Bgra8); impl_src_size!(YCbCr8); impl_src_size!(Yuv8);

impl<Src> ConvertColorExt for Src {
    fn cvt<Dst>(&self) -> Result<Dst, ImageError>
    where
        Self: ConvertColor<Dst> + SrcSize,
        Dst: NewColorImage,
    {
        let mut dst = Dst::new_zeroed(self.src_size())?;
        self.convert(&mut dst)?;
        Ok(dst)
    }
}
```

Note on `Rgba8`/`Bgra8` as `Dst`: `rgba_from_rgb` is `ConvertColor<Rgba8>` for `Rgb8` (convert.rs), so `rgb8.cvt::<Rgba8<_>>()` works. `Rgba8 -> Rgb8` uses `ConvertColorWithBackground` (not `ConvertColor`) so it is intentionally NOT reachable via `.cvt()` (background param required); leave it to the existing `rgb_from_rgba` free function.

- [ ] **Step 4: Re-export from `color/mod.rs`**

In `crates/kornia-imgproc/src/color/mod.rs`, find the line re-exporting `convert` items (the `pub use convert::{...}` near the `ConvertColor` export) and add `ConvertColorExt, NewColorImage`:

```rust
pub use convert::{ConvertColor, ConvertColorExt, ConvertColorWithBackground, NewColorImage};
```

(If the existing re-export lists items individually, append the two new names; if it is `pub use convert::*;`, no change needed.)

- [ ] **Step 5: Run tests**

Run: `cargo test -p kornia-imgproc --lib cvt_ext_tests`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add crates/kornia-imgproc/src/color/convert.rs crates/kornia-imgproc/src/color/mod.rs
git commit -m "feat(color): zero-cost typed .cvt() allocating conversion"
```

---

### Task 4: Rust runtime layer — `Tagged::cvt_color` → `DynImage`

**Files:**
- Modify: `crates/kornia-imgproc/src/color/convert.rs` (append), `crates/kornia-imgproc/src/color/mod.rs` (re-export)
- Test: inline `#[cfg(test)]` in `convert.rs`

**Interfaces:**
- Consumes: `ConvertColorExt::cvt` (Task 3), `DynImage` + `ColorSpace` (Tasks 1–2).
- Produces:
  - `trait Tagged<T> { fn space(&self) -> ColorSpace; fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<T, CpuAllocator>, ImageError>; }`
  - impls for `Rgbf32`, `Rgb8`, and the inverse source spaces back to RGB.

- [ ] **Step 1: Write the failing test**

Append to `convert.rs`:

```rust
#[cfg(test)]
mod cvt_color_tests {
    use super::*;
    use crate::color::Tagged;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::color_spaces::Rgbf32;
    use kornia_image::{ColorSpace, DynImage, ImageSize};

    #[test]
    fn runtime_cvt_color_returns_tagged_dynimage() {
        let size = ImageSize { width: 4, height: 4 };
        let rgb = Rgbf32::from_size_vec(size, vec![0.5f32; 4 * 4 * 3], CpuAllocator).unwrap();
        let hsv = rgb.cvt_color(ColorSpace::Hsv).unwrap();
        assert_eq!(hsv.color_space(), ColorSpace::Hsv);
        assert_eq!(hsv.channels(), 3);
        let gray = rgb.cvt_color(ColorSpace::Gray).unwrap();
        assert_eq!(gray.color_space(), ColorSpace::Gray);
        assert_eq!(gray.channels(), 1);
        assert!(matches!(gray, DynImage::C1(..)));
    }

    #[test]
    fn runtime_cvt_color_rejects_unsupported_pair() {
        let size = ImageSize { width: 2, height: 2 };
        let rgb = Rgbf32::from_size_vec(size, vec![0.0f32; 2 * 2 * 3], CpuAllocator).unwrap();
        // Rgb has no direct path to YCbCr? It does — pick a truly illegal target by
        // constructing from a non-Rgb source instead:
        let hsv = rgb.cvt_color(ColorSpace::Hsv).unwrap();
        let hsv_typed: kornia_image::color_spaces::Hsvf32<_> = hsv.try_into().unwrap();
        let err = hsv_typed.cvt_color(ColorSpace::Lab);
        assert!(err.is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p kornia-imgproc --lib cvt_color_tests`
Expected: FAIL — `Tagged` / `cvt_color` not found.

- [ ] **Step 3: Implement `Tagged` via a dispatch macro**

Append to `convert.rs`:

```rust
use kornia_image::{ColorSpace, DynImage};
use kornia_image::allocator::CpuAllocator;

/// Runtime, color-space-tagged conversion. The source newtype encodes its
/// space and dtype, so only the target `to` is supplied; the result is a
/// `DynImage` tagged with `to`. Same legal set as the typed `.cvt()` path.
pub trait Tagged<T> {
    /// This image's color space.
    fn space(&self) -> ColorSpace;
    /// Convert to `to`, returning an owned tagged `DynImage`.
    fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<T, CpuAllocator>, ImageError>;
}

/// Generates a `Tagged` impl for one source newtype. Each `to => Dst, Cn`
/// arm names the destination newtype and the DynImage channel constructor.
macro_rules! impl_tagged {
    ($src:ty, $t:ty, $space:expr, { $( $to:ident => $dst:ty , $ctor:ident );* $(;)? }) => {
        impl<A: ImageAllocator> Tagged<$t> for $src
        where Self: SrcSize {
            fn space(&self) -> ColorSpace { $space }
            fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<$t, CpuAllocator>, ImageError> {
                match to {
                    $( ColorSpace::$to => {
                        let out: $dst = self.cvt()?;
                        Ok(DynImage::$ctor(to, out.into_inner()))
                    } )*
                    _ => Err(ImageError::UnsupportedColorConversion { from: $space, to }),
                }
            }
        }
    };
}

// ---- f32 RGB source: all f32 targets ----
impl_tagged!(kornia_image::color_spaces::Rgbf32<A>, f32, ColorSpace::Rgb, {
    Gray      => kornia_image::color_spaces::Grayf32<CpuAllocator>, C1;
    Bgr       => kornia_image::color_spaces::Bgrf32<CpuAllocator>, C3;
    Hsv       => kornia_image::color_spaces::Hsvf32<CpuAllocator>, C3;
    Hls       => kornia_image::color_spaces::Hlsf32<CpuAllocator>, C3;
    Lab       => kornia_image::color_spaces::Labf32<CpuAllocator>, C3;
    Luv       => kornia_image::color_spaces::Luvf32<CpuAllocator>, C3;
    Xyz       => kornia_image::color_spaces::Xyzf32<CpuAllocator>, C3;
    LinearRgb => kornia_image::color_spaces::LinearRgbf32<CpuAllocator>, C3;
    YCbCr     => kornia_image::color_spaces::YCbCrf32<CpuAllocator>, C3;
    Yuv       => kornia_image::color_spaces::Yuvf32<CpuAllocator>, C3;
});

// ---- f32 inverse sources back to RGB ----
impl_tagged!(kornia_image::color_spaces::Hsvf32<A>, f32, ColorSpace::Hsv, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Hlsf32<A>, f32, ColorSpace::Hls, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Labf32<A>, f32, ColorSpace::Lab, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Luvf32<A>, f32, ColorSpace::Luv, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Xyzf32<A>, f32, ColorSpace::Xyz, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::LinearRgbf32<A>, f32, ColorSpace::LinearRgb, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::YCbCrf32<A>, f32, ColorSpace::YCbCr, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Yuvf32<A>, f32, ColorSpace::Yuv, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgrf32<A>, f32, ColorSpace::Bgr, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Grayf32<A>, f32, ColorSpace::Gray, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});

// ---- u8 RGB source: u8-valid targets ----
impl_tagged!(kornia_image::color_spaces::Rgb8<A>, u8, ColorSpace::Rgb, {
    Gray  => kornia_image::color_spaces::Gray8<CpuAllocator>, C1;
    Bgr   => kornia_image::color_spaces::Bgr8<CpuAllocator>, C3;
    Rgba  => kornia_image::color_spaces::Rgba8<CpuAllocator>, C4;
    YCbCr => kornia_image::color_spaces::YCbCr8<CpuAllocator>, C3;
    Yuv   => kornia_image::color_spaces::Yuv8<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Gray8<A>, u8, ColorSpace::Gray, {
    Rgb => kornia_image::color_spaces::Rgb8<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgr8<A>, u8, ColorSpace::Bgr, {
    Rgb => kornia_image::color_spaces::Rgb8<CpuAllocator>, C3;
});
```

Note: `self.cvt()` here requires `Self: ConvertColor<Dst>` for each arm's `Dst`; those impls exist in this file. The output allocator is `CpuAllocator` because `NewColorImage`/`.cvt()` produce owned `CpuAllocator` images.

- [ ] **Step 4: Re-export `Tagged`**

In `crates/kornia-imgproc/src/color/mod.rs`, add `Tagged` to the `convert` re-export:

```rust
pub use convert::{ConvertColor, ConvertColorExt, ConvertColorWithBackground, NewColorImage, Tagged};
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p kornia-imgproc --lib cvt_color_tests`
Expected: PASS.

Then full color suite + clippy:
Run: `cargo test -p kornia-imgproc --lib color`
Run: `cargo clippy -p kornia-image -p kornia-imgproc --all-targets --features ci -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/kornia-imgproc/src/color/convert.rs crates/kornia-imgproc/src/color/mod.rs
git commit -m "feat(color): runtime Tagged::cvt_color returning tagged DynImage"
```

---

### Task 5: Python `PyColorSpace` enum + registration

**Files:**
- Create: `kornia-py/src/color_space.rs`
- Modify: `kornia-py/src/lib.rs` (module list + register), `kornia-py/src/image.rs` (top `mod`/`use` if needed)
- Test: covered in Task 7 (Python).

**Interfaces:**
- Produces:
  - `#[pyclass(name = "ColorSpace", eq, eq_int, module = "kornia_rs.image")] pub enum PyColorSpace { Rgb, Bgr, Gray, Rgba, Bgra, Hsv, Hls, Lab, Luv, Xyz, LinearRgb, YCbCr, Yuv }`
  - `From<PyColorSpace> for ColorSpace` and `From<ColorSpace> for PyColorSpace`

- [ ] **Step 1: Create the pyclass enum**

Create `kornia-py/src/color_space.rs`:

```rust
//! Python `ColorSpace` enum mirroring `kornia_image::ColorSpace`.

use kornia_image::ColorSpace;
use pyo3::prelude::*;

/// Per-pixel color space tag used by `Image.cvt_color`.
#[pyclass(name = "ColorSpace", eq, eq_int, module = "kornia_rs.image")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyColorSpace {
    Rgb, Bgr, Gray, Rgba, Bgra, Hsv, Hls, Lab, Luv, Xyz, LinearRgb, YCbCr, Yuv,
}

impl From<PyColorSpace> for ColorSpace {
    fn from(v: PyColorSpace) -> Self {
        match v {
            PyColorSpace::Rgb => ColorSpace::Rgb,
            PyColorSpace::Bgr => ColorSpace::Bgr,
            PyColorSpace::Gray => ColorSpace::Gray,
            PyColorSpace::Rgba => ColorSpace::Rgba,
            PyColorSpace::Bgra => ColorSpace::Bgra,
            PyColorSpace::Hsv => ColorSpace::Hsv,
            PyColorSpace::Hls => ColorSpace::Hls,
            PyColorSpace::Lab => ColorSpace::Lab,
            PyColorSpace::Luv => ColorSpace::Luv,
            PyColorSpace::Xyz => ColorSpace::Xyz,
            PyColorSpace::LinearRgb => ColorSpace::LinearRgb,
            PyColorSpace::YCbCr => ColorSpace::YCbCr,
            PyColorSpace::Yuv => ColorSpace::Yuv,
        }
    }
}

impl From<ColorSpace> for PyColorSpace {
    fn from(v: ColorSpace) -> Self {
        match v {
            ColorSpace::Rgb => PyColorSpace::Rgb,
            ColorSpace::Bgr => PyColorSpace::Bgr,
            ColorSpace::Gray => PyColorSpace::Gray,
            ColorSpace::Rgba => PyColorSpace::Rgba,
            ColorSpace::Bgra => PyColorSpace::Bgra,
            ColorSpace::Hsv => PyColorSpace::Hsv,
            ColorSpace::Hls => PyColorSpace::Hls,
            ColorSpace::Lab => PyColorSpace::Lab,
            ColorSpace::Luv => PyColorSpace::Luv,
            ColorSpace::Xyz => PyColorSpace::Xyz,
            ColorSpace::LinearRgb => PyColorSpace::LinearRgb,
            ColorSpace::YCbCr => PyColorSpace::YCbCr,
            ColorSpace::Yuv => PyColorSpace::Yuv,
        }
    }
}
```

- [ ] **Step 2: Register the module + class**

In `kornia-py/src/lib.rs`: add `mod color_space;` near the other `mod` declarations, and in the module init where `PyImageApi`/`PyImageSize` are added to the `image` submodule, add:

```rust
    image_mod.add_class::<crate::color_space::PyColorSpace>()?;
```

(Place it next to the existing `image_mod.add_class::<PyImageApi>()?;` line. If the image submodule is built in `image.rs`, add it there instead, matching the existing pattern.)

- [ ] **Step 3: Build**

Run: `cd kornia-py && pixi run -e py312 maturin develop --release`
Expected: builds cleanly.

- [ ] **Step 4: Smoke-check the enum exists**

Run: `cd kornia-py && pixi run -e py312 python -c "import kornia_rs; print(kornia_rs.image.ColorSpace.HSV)"`
Expected: prints `ColorSpace.Hsv` (or similar repr) without error.

- [ ] **Step 5: Commit**

```bash
git add kornia-py/src/color_space.rs kornia-py/src/lib.rs
git commit -m "feat(kornia-py): ColorSpace pyclass enum"
```

---

### Task 6: Python `Image.cvt_color` + dtype helpers + color_space field

**Files:**
- Modify: `kornia-py/src/image.rs` — add `color_space` field to `PyImageApi`, default in `wrap`/`wrap_u16`/`wrap_f32`, getter, `cvt_color`, `to_float`, `to_uint8`, sugar methods.
- Test: Task 7.

**Interfaces:**
- Consumes: `numpy_as_image::<C>`, `numpy_as_image_f32::<C>`, `alloc_output_pyarray::<C>`, `alloc_output_pyarray_f32::<C>` (image.rs), the `*_from_*` free functions in `kornia_imgproc::color`, `ColorSpace` legality (`ColorSpace::supports`, `requires_f32`), `PyColorSpace` (Task 5).
- Produces (Python): `Image.color_space -> ColorSpace`, `Image.cvt_color(to: ColorSpace) -> Image`, `Image.to_float() -> Image`, `Image.to_uint8() -> Image`, `Image.to_gray()/to_hsv()/to_lab()/...` sugar.

- [ ] **Step 1: Add the `color_space` field + default-by-channels**

In `kornia-py/src/image.rs`, add to `PyImageApi` (after `format` at line 916):

```rust
    /// The per-pixel color space this Image is interpreted as. Defaults by
    /// channel count on construction (1->Gray, 3->Rgb, 4->Rgba) and is updated
    /// by `cvt_color`.
    color_space: kornia_image::ColorSpace,
```

Add a helper near `mode_from_channels`:

```rust
fn default_color_space(channels: usize) -> kornia_image::ColorSpace {
    match channels {
        1 => kornia_image::ColorSpace::Gray,
        4 => kornia_image::ColorSpace::Rgba,
        _ => kornia_image::ColorSpace::Rgb,
    }
}
```

Update every `Self { data, mode, format }` constructor (in `wrap`, `wrap_u16`, `wrap_f32`, and any others — grep `format: None`) to also set `color_space`. For `wrap`:

```rust
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(channels, false));
        Self {
            data: ImageData::U8(data),
            mode,
            format: None,
            color_space: default_color_space(channels),
        }
```

Apply the analogous change in `wrap_u16` and `wrap_f32` (channels read from their array shape). For any other struct-literal construction of `PyImageApi`, set `color_space: default_color_space(<channels>)`.

- [ ] **Step 2: Write the failing test (Python)**

Create `kornia-py/tests/test_cvt_color.py` (full content in Task 7); for now add the minimal first test and run it:

```python
import numpy as np
import kornia_rs
from kornia_rs.image import Image, ColorSpace

def test_color_space_defaults_and_getter():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image.from_numpy(arr)
    assert img.color_space == ColorSpace.Rgb
```

Run: `cd kornia-py && pixi run -e py312 pytest tests/test_cvt_color.py::test_color_space_defaults_and_getter -v`
Expected: FAIL — `color_space` getter not present.

- [ ] **Step 3: Add the getter**

In the `#[pymethods] impl PyImageApi` block (near the `data_ptr`/`numpy` getters around line 2108), add:

```rust
    /// The color space this Image is interpreted as.
    #[getter]
    fn color_space(&self) -> crate::color_space::PyColorSpace {
        self.color_space.into()
    }
```

Rebuild + rerun Step 2's test → PASS.

- [ ] **Step 4: Implement `to_float` / `to_uint8`**

Add to the `#[pymethods]` block:

```rust
    /// Cast u8 -> f32 by dividing by 255 (range [0,1]). f32 input is returned
    /// unchanged (cloned handle). Required before converting to f32-only spaces.
    fn to_float(&self, py: Python<'_>) -> PyResult<Self> {
        match &self.data {
            ImageData::F32(_) => Ok(self.clone_handle(py)),
            ImageData::U8(a) => {
                let arr = a.bind(py);
                let out = arr.cast::<f32>(false)?; // numpy cast to f32
                let out: Py<PyArray3<f32>> = out.unbind();
                // scale by 1/255 in-place
                scale_f32_inplace(py, &out, 1.0 / 255.0);
                let mut img = Self::wrap_f32(py, out, Some(self.mode.clone()));
                img.color_space = self.color_space;
                Ok(img)
            }
            ImageData::U16(_) => Err(u16_imgproc_unsupported("to_float")),
        }
    }

    /// Cast f32 [0,1] -> u8 by multiplying by 255 (saturating round). u8 input
    /// is returned unchanged.
    fn to_uint8(&self, py: Python<'_>) -> PyResult<Self> {
        match &self.data {
            ImageData::U8(_) => Ok(self.clone_handle(py)),
            ImageData::F32(a) => {
                let out = scale_clamp_f32_to_u8(py, a, 255.0)?;
                let mut img = Self::wrap(py, out, Some(self.mode.clone()));
                img.color_space = self.color_space;
                Ok(img)
            }
            ImageData::U16(_) => Err(u16_imgproc_unsupported("to_uint8")),
        }
    }
```

Add the small helpers (free functions in image.rs). Implement `scale_f32_inplace` and `scale_clamp_f32_to_u8` using numpy element access via slices:

```rust
fn clone_handle(&self, py: Python<'_>) -> Self { /* clone_ref data, same mode/format/color_space */ }
```

Implementation detail for `clone_handle`: construct `Self { data: <clone of self.data variant>, mode: self.mode.clone(), format: self.format, color_space: self.color_space }` using `ImageData`'s inner `clone_ref(py)`.

For `scale_f32_inplace`:

```rust
fn scale_f32_inplace(py: Python<'_>, arr: &Py<PyArray3<f32>>, k: f32) {
    let bound = arr.bind(py);
    let len = bound.shape().iter().product::<usize>();
    // SAFETY: freshly-cast contiguous f32 array we own.
    let s = unsafe { std::slice::from_raw_parts_mut(bound.data(), len) };
    for v in s.iter_mut() { *v *= k; }
}
```

For `scale_clamp_f32_to_u8`:

```rust
fn scale_clamp_f32_to_u8(
    py: Python<'_>, arr: &Py<PyArray3<f32>>, k: f32,
) -> PyResult<Py<PyArray3<u8>>> {
    let bound = arr.bind(py);
    let shape = bound.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let len = h * w * c;
    let src = unsafe { std::slice::from_raw_parts(bound.data(), len) };
    let out = unsafe { PyArray::<u8, _>::new(py, [h, w, c], false) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), len) };
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = (s * k).round().clamp(0.0, 255.0) as u8;
    }
    Ok(out.unbind())
}
```

- [ ] **Step 5: Implement `cvt_color` dispatch**

Add the dispatcher to the `#[pymethods]` block. It validates legality + dtype, then routes to the existing `*_from_*` free functions:

```rust
    /// Convert to another color space, returning a new tagged Image. Strict
    /// dtype: f32-only spaces require a float image (call `to_float()` first).
    fn cvt_color(&self, py: Python<'_>, to: crate::color_space::PyColorSpace) -> PyResult<Self> {
        use kornia_image::ColorSpace as CS;
        let from = self.color_space;
        let to: CS = to.into();
        if from == to {
            return Ok(self.clone_handle(py));
        }
        if !CS::supports(from, to) {
            return Err(value_err(format!(
                "no direct {from:?}->{to:?} color conversion; convert via Rgb"
            )));
        }
        // strict dtype: f32-only target (or source) needs f32 storage
        let needs_f32 = to.requires_f32() || from.requires_f32();
        if needs_f32 && !self.data.is_f32() {
            return Err(value_err(format!(
                "{to:?} requires float32; call img.to_float() first"
            )));
        }
        if self.data.is_u16() {
            return Err(u16_imgproc_unsupported("cvt_color"));
        }
        let out = dispatch_cvt(py, &self.data, from, to)?;
        let mut img = out;
        img.color_space = to;
        Ok(img)
    }
```

Implement the `dispatch_cvt` free function mapping `(from, to, dtype)` to the existing kernels. It is the Python-side analog of `Tagged`. Cover the legal set:

```rust
fn dispatch_cvt(
    py: Python<'_>,
    data: &ImageData,
    from: kornia_image::ColorSpace,
    to: kornia_image::ColorSpace,
) -> PyResult<PyImageApi> {
    use kornia_image::ColorSpace as CS;
    use kornia_imgproc::color as kc;

    macro_rules! f32_3to3 {
        ($func:path) => {{
            let ImageData::F32(a) = data else { unreachable!() };
            let src = unsafe { numpy_as_image_f32::<3>(py, a)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, src.size())? };
            py.detach(|| $func(&src, &mut dst)).map_err(to_pyerr)?;
            Ok(PyImageApi::wrap_f32(py, out, None))
        }};
    }
    macro_rules! u8_3to3 {
        ($func:path) => {{
            let ImageData::U8(a) = data else { unreachable!() };
            let src = unsafe { numpy_as_image::<3>(py, a)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
            py.detach(|| $func(&src, &mut dst)).map_err(to_pyerr)?;
            Ok(PyImageApi::wrap(py, out, None))
        }};
    }

    match (from, to) {
        // u8 channel/space conversions
        (CS::Rgb, CS::Bgr) | (CS::Bgr, CS::Rgb) if !data.is_f32() => u8_3to3!(kc::bgr_from_rgb),
        (CS::Rgb, CS::Gray) if !data.is_f32() => {
            let ImageData::U8(a) = data else { unreachable!() };
            let src = unsafe { numpy_as_image::<3>(py, a)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
            py.detach(|| kc::gray_from_rgb_u8(&src, &mut dst)).map_err(to_pyerr)?;
            Ok(PyImageApi::wrap(py, out, None))
        }
        (CS::Gray, CS::Rgb) if !data.is_f32() => {
            let ImageData::U8(a) = data else { unreachable!() };
            let src = unsafe { numpy_as_image::<1>(py, a)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
            py.detach(|| kc::rgb_from_gray(&src, &mut dst)).map_err(to_pyerr)?;
            Ok(PyImageApi::wrap(py, out, None))
        }
        // f32 perceptual/cylindrical conversions (3->3)
        (CS::Rgb, CS::Hsv) => f32_3to3!(kc::hsv_from_rgb),
        (CS::Hsv, CS::Rgb) => f32_3to3!(kc::rgb_from_hsv),
        (CS::Rgb, CS::Hls) => f32_3to3!(kc::hls_from_rgb),
        (CS::Hls, CS::Rgb) => f32_3to3!(kc::rgb_from_hls),
        (CS::Rgb, CS::Lab) => f32_3to3!(kc::lab_from_rgb),
        (CS::Lab, CS::Rgb) => f32_3to3!(kc::rgb_from_lab),
        (CS::Rgb, CS::Luv) => f32_3to3!(kc::luv_from_rgb),
        (CS::Luv, CS::Rgb) => f32_3to3!(kc::rgb_from_luv),
        (CS::Rgb, CS::Xyz) => f32_3to3!(kc::xyz_from_rgb),
        (CS::Xyz, CS::Rgb) => f32_3to3!(kc::rgb_from_xyz),
        (CS::Rgb, CS::LinearRgb) => f32_3to3!(kc::linear_rgb_from_rgb),
        (CS::LinearRgb, CS::Rgb) => f32_3to3!(kc::rgb_from_linear_rgb),
        (CS::Rgb, CS::YCbCr) => f32_3to3!(kc::ycbcr_from_rgb),
        (CS::YCbCr, CS::Rgb) => f32_3to3!(kc::rgb_from_ycbcr),
        (CS::Rgb, CS::Yuv) => f32_3to3!(kc::yuv_from_rgb),
        (CS::Yuv, CS::Rgb) => f32_3to3!(kc::rgb_from_yuv),
        _ => Err(value_err(format!(
            "no direct {from:?}->{to:?} color conversion; convert via Rgb"
        ))),
    }
}
```

Note: `to_pyerr` is the existing helper in `color.rs`; import it (`use crate::color::to_pyerr;` or replicate). If `to_pyerr` is private to `color.rs`, make it `pub(crate)`.

- [ ] **Step 6: Add sugar methods**

Add to the `#[pymethods]` block:

```rust
    /// Convert to grayscale.
    fn to_gray(&self, py: Python<'_>) -> PyResult<Self> { self.cvt_color(py, crate::color_space::PyColorSpace::Gray) }
    /// Convert to HSV (requires float input).
    fn to_hsv(&self, py: Python<'_>) -> PyResult<Self> { self.cvt_color(py, crate::color_space::PyColorSpace::Hsv) }
    /// Convert to Lab (requires float input).
    fn to_lab(&self, py: Python<'_>) -> PyResult<Self> { self.cvt_color(py, crate::color_space::PyColorSpace::Lab) }
    /// Convert to BGR.
    fn to_bgr(&self, py: Python<'_>) -> PyResult<Self> { self.cvt_color(py, crate::color_space::PyColorSpace::Bgr) }
    /// Convert to RGB.
    fn to_rgb(&self, py: Python<'_>) -> PyResult<Self> { self.cvt_color(py, crate::color_space::PyColorSpace::Rgb) }
```

- [ ] **Step 7: Build + run the smoke test**

Run: `cd kornia-py && pixi run -e py312 maturin develop --release`
Run: `cd kornia-py && pixi run -e py312 pytest tests/test_cvt_color.py -v`
Expected: PASS for the tests present so far.

- [ ] **Step 8: Commit**

```bash
git add kornia-py/src/image.rs
git commit -m "feat(kornia-py): Image.cvt_color, dtype helpers, color_space field"
```

---

### Task 7: Python regression tests (full suite)

**Files:**
- Modify: `kornia-py/tests/test_cvt_color.py` (expand)

**Interfaces:**
- Consumes: `Image`, `ColorSpace`, the existing free functions `kornia_rs.imgproc.hsv_from_rgb` etc. for parity checks.

- [ ] **Step 1: Write the full test file**

Replace `kornia-py/tests/test_cvt_color.py` with:

```python
import numpy as np
import pytest
import kornia_rs
from kornia_rs.image import Image, ColorSpace


def _rgb_u8(h=8, w=8):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_color_space_defaults_and_getter():
    assert Image.from_numpy(np.zeros((4, 4, 3), np.uint8)).color_space == ColorSpace.Rgb
    assert Image.from_numpy(np.zeros((4, 4, 1), np.uint8)).color_space == ColorSpace.Gray
    assert Image.from_numpy(np.zeros((4, 4, 4), np.uint8)).color_space == ColorSpace.Rgba


def test_cvt_color_tag_propagates():
    img = Image.from_numpy(_rgb_u8())
    g = img.cvt_color(ColorSpace.Gray)
    assert g.color_space == ColorSpace.Gray
    assert g.numpy().shape[2] == 1


def test_cvt_color_strict_dtype_error_then_to_float():
    img = Image.from_numpy(_rgb_u8())
    with pytest.raises(ValueError, match="float32"):
        img.cvt_color(ColorSpace.Hsv)
    hsv = img.to_float().cvt_color(ColorSpace.Hsv)
    assert hsv.color_space == ColorSpace.Hsv


def test_cvt_color_unsupported_pair():
    img = Image.from_numpy(_rgb_u8()).to_float().cvt_color(ColorSpace.Hsv)
    with pytest.raises(ValueError, match="no direct"):
        img.cvt_color(ColorSpace.Lab)


def test_cvt_color_parity_with_free_function():
    arr = _rgb_u8().astype(np.float32) / 255.0
    img = Image.from_numpy(arr)  # f32 RGB
    via_method = img.cvt_color(ColorSpace.Hsv).numpy()
    via_free = kornia_rs.imgproc.hsv_from_rgb(arr)
    np.testing.assert_allclose(via_method, via_free, rtol=0, atol=0)


def test_to_uint8_round_trip():
    arr = _rgb_u8()
    img = Image.from_numpy(arr)
    back = img.to_float().to_uint8().numpy()
    np.testing.assert_array_equal(back, arr)


def test_sugar_methods():
    img = Image.from_numpy(_rgb_u8())
    assert img.to_gray().color_space == ColorSpace.Gray
    assert img.to_bgr().color_space == ColorSpace.Bgr
```

- [ ] **Step 2: Run the suite**

Run: `cd kornia-py && pixi run -e py312 pytest tests/test_cvt_color.py -v`
Expected: all PASS. If `test_to_uint8_round_trip` drifts by 1 LSB, that is acceptable rounding — adjust to `assert_allclose(atol=1)` only if it fails (document the change).

- [ ] **Step 3: Commit**

```bash
git add kornia-py/tests/test_cvt_color.py
git commit -m "test(kornia-py): cvt_color tag propagation, strict dtype, parity"
```

---

### Task 8: `.pyi` stubs for `ColorSpace` + new `Image` methods

**Files:**
- Modify: `kornia-py/python/kornia_rs/image.pyi`

**Interfaces:**
- Consumes: the methods added in Tasks 5–6.

- [ ] **Step 1: Add the `ColorSpace` enum stub**

In `kornia-py/python/kornia_rs/image.pyi`, add at module scope:

```python
from enum import Enum

class ColorSpace(Enum):
    Rgb: int
    Bgr: int
    Gray: int
    Rgba: int
    Bgra: int
    Hsv: int
    Hls: int
    Lab: int
    Luv: int
    Xyz: int
    LinearRgb: int
    YCbCr: int
    Yuv: int
```

- [ ] **Step 2: Add the new `Image` methods to the stub**

Inside the `class Image:` block, add:

```python
    @property
    def color_space(self) -> ColorSpace: ...
    def cvt_color(self, to: ColorSpace) -> "Image": ...
    def to_float(self) -> "Image": ...
    def to_uint8(self) -> "Image": ...
    def to_gray(self) -> "Image": ...
    def to_hsv(self) -> "Image": ...
    def to_lab(self) -> "Image": ...
    def to_bgr(self) -> "Image": ...
    def to_rgb(self) -> "Image": ...
```

- [ ] **Step 3: Verify stub import resolves**

Run: `cd kornia-py && pixi run -e py312 python -c "from kornia_rs.image import Image, ColorSpace; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add kornia-py/python/kornia_rs/image.pyi
git commit -m "feat(kornia-py): .pyi stubs for ColorSpace and Image.cvt_color"
```

---

### Task 9: Final verification + docs touch

**Files:**
- Modify: none required (verification); optionally `crates/kornia-imgproc/src/color/convert.rs` doc example.

- [ ] **Step 1: Full Rust test + clippy**

Run: `cargo test -p kornia-image -p kornia-imgproc --lib`
Run: `cargo clippy -p kornia-image -p kornia-imgproc --all-targets --features ci -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 2: Rust doctests**

Run: `cargo test -p kornia-imgproc --doc`
Expected: PASS.

- [ ] **Step 3: Full Python suite**

Run: `cd kornia-py && pixi run -e py312 pytest tests/test_cvt_color.py tests/test_color.py -v`
Expected: PASS.

- [ ] **Step 4: Commit any doc tweaks**

```bash
git add -A
git commit -m "docs(color): note .cvt()/cvt_color in convert.rs"
```

(Skip if no changes.)

---

## Self-Review

**Spec coverage:**
- §1 ColorSpace enum + graph → Task 1. ✓
- §2 typed `.cvt()` → Task 3. ✓
- §3 DynImage + runtime `cvt_color` → Tasks 2, 4. ✓
- §4 Python stateful Image + methods → Tasks 5, 6. ✓
- §5 error handling → Task 1 (variants), Tasks 4/6 (surfaced). ✓
- §6 testing → Tasks 3/4 (Rust), 7 (Python). ✓
- §7 file layout → matches the File Structure table. ✓

**Deviations from spec (intentional, noted):**
- Spec §3 showed a free `cvt_color(src, from, to)` for plain `Image`. The plan implements the runtime path as the `Tagged::cvt_color` trait method (source space known from the newtype) plus Python's `dispatch_cvt`. A plain-`Image` free function is omitted (YAGNI: Rust callers with a plain `Image` already know its type and can wrap it in one line). If wanted later, it is a thin match over `from` that wraps into the newtype then delegates — additive.
- `NewColorImage` uses `new_zeroed` (concrete name) rather than the spec's `new_uninit` (zeroed is safe and matches `from_size_val`).

**Type consistency:** `ColorSpace`, `DynImage::{C1,C3,C4}(space, img)`, `.cvt()`, `Tagged::cvt_color`, `dispatch_cvt`, `to_pyerr`, `default_color_space`, `clone_handle` are used consistently across tasks. The DynImage variants carry `(ColorSpace, Image)` in both Task 2 (definition) and Task 4 (construction). ✓

**Open implementation risk to watch:** the `numpy` `cast::<f32>` API name in `to_float` (Step 4, Task 6) — if `rust-numpy`'s method differs, allocate a new f32 array and copy/scale in one loop (same pattern as `scale_clamp_f32_to_u8`). This is a local fallback, not a design change.
