
<b>Pattern 1: Replace hard-coded numeric literals used as algorithm parameters, border padding, thresholds, or loop bounds with well-named `const` values (or configuration fields) so behavior is self-documenting and easy to tune consistently across the codebase.
</b>

Example code before:
```
for x in 3..cols - 3 {
    if score > 7.0 { return true; }
}
```

Example code after:
```
const BORDER: usize = 3;
const SCORE_THRESH: f32 = 7.0;

for x in BORDER..cols - BORDER {
    if score > SCORE_THRESH { return true; }
}
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/656#discussion_r2714126179
- https://github.com/kornia/kornia-rs/pull/656#discussion_r2714131044
- https://github.com/kornia/kornia-rs/pull/622#discussion_r2683063561
</details>


___

<b>Pattern 2: Do not use `unwrap()`, `expect()`, or unchecked assumptions in library and binding code; propagate errors with `Result`/`?` and make constructors fallible when initialization can fail, including in tests when they touch fallible APIs.
</b>

Example code before:
```
let cfg = DecodeTagsConfig::new(families);
let family = TagFamily::tag36_h11();
let header = decoder.read_info().unwrap();
```

Example code after:
```
let cfg = DecodeTagsConfig::new(families)?;
let family = TagFamily::tag36_h11()?;
let header = decoder.read_info()?;
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/626#discussion_r2671962058
- https://github.com/kornia/kornia-rs/pull/626#discussion_r2678890224
- https://github.com/kornia/kornia-rs/pull/491#discussion_r2371449237
</details>


___

<b>Pattern 3: Remove AI/agentic/prompted commentary and keep comments/docstrings focused on invariants, rationale, and user-facing documentation; prefer concise, project-consistent language rather than meta commentary about how code was generated.
</b>

Example code before:
```
// Copilot suggestion: this should be faster
// As an AI model, I recommend...
// prompted comment: do X
```

Example code after:
```
// Invariant: pixel_idx + offsets[i] is in-bounds because we exclude BORDER pixels.
// Rationale: use small-angle Taylor expansion for numerical stability.
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/633#discussion_r2678811089
- https://github.com/kornia/kornia-rs/pull/622#discussion_r2679732813
- https://github.com/kornia/kornia-rs/pull/554#discussion_r2527034575
</details>


___

<b>Pattern 4: Use semantically meaningful API return types instead of tuples of loosely related vectors, and prefer `Option`/domain structs to represent empty or missing results clearly.
</b>

Example code before:
```
fn correspondences(...) -> (Vec<Point>, Vec<Point>, Vec<f64>) {
    if input.is_empty() { return (vec![], vec![], vec![]); }
    // ...
}
```

Example code after:
```
struct Correspondence { src: Point, dst: Point, dist: f64 }

fn correspondences(...) -> Option<Vec<Correspondence>> {
    if input.is_empty() { return None; }
    Some(corrs)
}
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/533#discussion_r2493522424
</details>


___

<b>Pattern 5: Standardize metadata/layout APIs across formats and layers (Rust core and bindings): use a shared `ImageLayout` (or similarly named) struct and consistent naming such as `*_layout` rather than `*_info`, enabling fast dispatch without repeated decoding.
</b>

Example code before:
```
pub fn decode_image_png_info(src: &[u8]) -> Result<(ImageSize, u8, u8), IoError> { ... }
pub fn decode_image_jpeg_info(src: &[u8]) -> Result<(ImageSize, u8), IoError> { ... }
```

Example code after:
```
pub fn decode_image_png_layout(src: &[u8]) -> Result<ImageLayout, IoError> { ... }
pub fn decode_image_jpeg_layout(src: &[u8]) -> Result<ImageLayout, IoError> { ... }
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/571#discussion_r2544949850
- https://github.com/kornia/kornia-rs/pull/571#discussion_r2556624483
- https://github.com/kornia/kornia-rs/pull/571#discussion_r2556592665
</details>


___

<b>Pattern 6: Avoid avoidable overhead in hot paths and dispatchers: match on enums directly rather than converting to intermediate strings, and avoid decoding the same file twice by extracting metadata first or reusing already-read buffers.
</b>

Example code before:
```
let mode = match color_type { ColorType::Rgb => "rgb", _ => "mono" };
match mode {
    "rgb" => decode_rgb(...),
    "mono" => decode_mono(...),
    _ => unreachable!(),
}
```

Example code after:
```
match (color_type, bit_depth) {
    (ColorType::Rgb, BitDepth::Eight) => decode_rgb8(...),
    (ColorType::Grayscale, BitDepth::Sixteen) => decode_mono16(...),
    _ => return Err(Error::UnsupportedFormat),
}
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/571#discussion_r2541238970
- https://github.com/kornia/kornia-rs/pull/571#discussion_r2540145812
- https://github.com/kornia/kornia-rs/pull/571#discussion_r2540171481
</details>


___

<b>Pattern 7: Keep public APIs minimal and idiomatic: avoid unnecessary lifetimes and avoid passing cheap parameter structs by reference; prefer simpler signatures and names that align with Rust conventions.
</b>

Example code before:
```
pub fn solve(..., params: &Params) -> Result<Out, Err> { ... }
struct Helper<'a> { kernel: &'a [f32] }
```

Example code after:
```
pub fn solve(..., params: Params) -> Result<Out, Err> { ... }
struct Helper { kernel: Vec<f32> }
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia-rs/pull/612#discussion_r2660897952
- https://github.com/kornia/kornia-rs/pull/480#discussion_r2376077209
</details>


___

<b>Pattern 8: Name color-space conversion functions `<output>_from_<input>`, not `<input>_to_<output>`. The output type always comes first.
</b>

Example code before:
```rust
pub fn rgb_to_gray(...) { ... }
pub fn gray_to_rgb(...) { ... }
pub fn rgb_to_bgr(...) { ... }
```

Example code after:
```rust
pub fn gray_from_rgb(...) { ... }
pub fn rgb_from_gray(...) { ... }
pub fn bgr_from_rgb(...) { ... }
```

This mirrors how Rust's `From`/`Into` traits read: `Gray::from(rgb)`. Consistent naming makes the conversion direction unambiguous at every call site without reading the signature.

<details><summary>Examples for relevant past discussions:</summary>

- PR #944 — `gray_from_rgb`, `gray_from_rgb_u8`, `gray_from_rgb_f32`, `rgb_from_gray`, `bgr_from_rgb`
</details>


___

<b>Pattern 9: When a SIMD op has meaningfully different kernel implementations per pixel dtype, split the module into a subdir (`mod.rs` + `kernels.rs`) and route dtype dispatch through a sealed trait — not via separate public functions per type.
</b>

Example code before:
```rust
// color/gray.rs — 900-line monolith mixing public API with SIMD internals
pub fn gray_from_rgb<T: Float>(...) { /* scalar */ }
pub fn gray_from_rgb_u8(...)  { /* NEON/AVX2 u8 */ }
pub fn gray_from_rgb_f32(...) { /* NEON/AVX2 f32 */ }
```

Example code after:
```
color/gray/
  mod.rs      — public Image API, sealed trait, thin compat wrappers
  kernels.rs  — par_strip_dispatch, NEON/AVX2/scalar kernel impls
```

```rust
// mod.rs
mod sealed { pub trait Sealed {} }
pub trait GrayFromRgb: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn gray_from_rgb_impl<A1, A2>(...) -> Result<(), ImageError>;
}
impl sealed::Sealed for u8 {}
impl GrayFromRgb for u8 { /* → kernels::rgb_to_gray_u8 */ }
impl sealed::Sealed for f32 {}
impl GrayFromRgb for f32 { /* → kernels::rgb_to_gray_f32 */ }
impl sealed::Sealed for f64 {}
impl GrayFromRgb for f64 { /* → scalar */ }

pub fn gray_from_rgb<T: GrayFromRgb, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 3, A1>, dst: &mut Image<T, 1, A2>,
) -> Result<(), ImageError> {
    T::gray_from_rgb_impl(src, dst)
}
// gray_from_rgb_u8 / gray_from_rgb_f32 → thin wrappers for backward compat only
```

The sealed trait provides compile-time dtype dispatch with zero runtime overhead. No `TypeId`, no branching on dtype at runtime, no code duplication. `mod sealed` blocks external impls. Follow `color/gray/` as the canonical example; see `SIMD.md` for the full pattern including the `check_size` helper and `par_strip_dispatch`.

<details><summary>Examples for relevant past discussions:</summary>

- PR #944 — `color/gray/{mod,kernels}.rs` + `GrayFromRgb` sealed trait
</details>


___
