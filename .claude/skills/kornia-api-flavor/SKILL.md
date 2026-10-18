---
name: kornia-api-flavor
description: Use before designing a new public function, trait, config struct or binding in kornia-rs — shapes of signatures (pre-allocated &mut dst, generic Image<T, C, A>, sealed-trait dtype dispatch, fallible constructors, shared layout structs). Read before sketching any new public API.
---

# kornia-rs API flavor

1. **Output buffers**: image ops are `fn op(src: &Image<T, C, A1>, dst: &mut Image<T, C2, A2>) -> Result<(), E>`.
   Validate `src`/`dst` sizes first and return a descriptive error on mismatch.
2. **Naming**: `<output>_from_<input>` (`gray_from_rgb`).
3. **Dtype dispatch**: one generic entry point bound by a sealed trait
   (e.g. `GrayFromRgb`) with explicit impls for `u8`, `f32`, `f64`. No blanket
   `impl<T: Float>` (conflicts without specialization). See `kornia-simd`.
4. **Generics over concrete**: keep `Image<T, C, A>` generic over allocator
   and channels (const generics); do not hard-code `Vec`.
5. **Errors**: each crate has a `thiserror` enum; wrap lower-crate errors with
   `#[from]`. Fallible constructors (`new(..) -> Result<Self, _>`).
6. **Config structs** document units and coordinate frames (pixels vs
   normalized coordinates, radians vs degrees).
7. **Return types**: domain structs/`Option`, not tuples of vectors.
8. **Bindings parity**: a Rust API exposed to Python (`kornia-py/src`) or C++
   (`kornia-cpp/src`, `include/`) must keep the same naming and semantics;
   add tests on the binding side.
9. **Feature gating**: hardware-specific code (`cuda`, `gstreamer`, `v4l`,
   `turbojpeg`) stays behind features; the default build must compile without.
10. **Docs**: new public API is not done until it has rustdoc with Arguments,
    Returns, Errors, Example, and an entry in the crate README/CHANGELOG
    `[Unreleased]` when user-visible.
