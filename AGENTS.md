# AGENTS.md

Guidance for AI AGENTS and human contributors working in this repository.

## Project Overview

**kornia-rs** is a low-level 3D computer vision library written in Rust, with Python bindings (PyO3/Maturin) and C++ bindings. Part of the [Kornia](https://github.com/kornia/kornia) ecosystem.

---

## Build & Development Commands

The project uses [pixi](https://pixi.sh) for environment management.

### Rust

```bash
pixi run rust-test                    # Run all Rust tests (with CI features)
pixi run rust-test-package <package>  # Test a specific crate
pixi run rust-lint                    # Format + clippy + check (run before PRs)
pixi run rust-clippy                  # Clippy with -D warnings, all targets, CI features
pixi run rust-fmt                     # Format with rustfmt
pixi run rust-check                   # Check compilation

# Without pixi (direct cargo)
cargo test --features ci
cargo test -p kornia-imgproc
cargo clippy --workspace --no-deps --all-targets --features ci -- -D warnings
```

### Python Bindings

```bash
pixi run py-build   # Build kornia-py (dev mode via maturin)
pixi run py-test    # Run pytest (auto-builds first)
```

### C++ Bindings

```bash
pixi run cpp-test   # Build and run C++ tests
```

---

## Architecture

### Workspace Structure

```
Cargo.toml          ← workspace root
crates/*            ← core library crates (default members)
examples/*          ← standalone example binaries
kornia-py/          ← Python bindings (PyO3/Maturin)
kornia-cpp/         ← C++ bindings
```

### Core Crates

| Crate | Re-exported as | Purpose |
|---|---|---|
| `kornia-tensor` | `kornia::tensor` | N-dimensional tensor type |
| `kornia-tensor-ops` | `kornia::tensor_ops` | Tensor operations |
| `kornia-image` | `kornia::image` | Image type built on tensors (`Image<T, C, _>`) |
| `kornia-imgproc` | `kornia::imgproc` | Image processing (resize, color, filters) |
| `kornia-io` | `kornia::io` | Image/video I/O (turbojpeg, gstreamer, v4l) |
| `kornia-3d` | `kornia::k3d` | 3D vision (ICP, point clouds) |
| `kornia-algebra` | `kornia::linalg` | Linear algebra |
| `kornia-vlm` | (direct) | Visual language model integration (candle) |
| `kornia-apriltag` | (standalone) | AprilTag detection |
| `kornia-bow` | (standalone) | Bag of words |

### Feature Flags

| Flag | Description | Notes |
|---|---|---|
| `turbojpeg` | libjpeg-turbo I/O | Requires `nasm` |
| `gstreamer` | GStreamer video | — |
| `v4l` | Video4Linux camera | Requires `clang` |
| `cuda` | CUDA for VLM | — |
| `ci` | All CI-compatible features | Excludes `cuda` |
| `serde`, `bincode`, `arrow` | Serialization formats | — |

### Key Patterns

- **Image type**: `Image<T, C, Alloc>` — `T` is pixel type (`u8`, `f32`), `C` is channel count (`1`, `3`)
- **Operations**: Many `imgproc` functions take `&source` and `&mut destination` (pre-allocated output)
- **I/O**: `kornia::io::functional` provides top-level read/write functions

---

## Code Conventions

### General

- Rust edition **2021**, MSRV **1.82**
- Run `rustfmt` and `clippy` before every commit — **warnings are denied**
- Prefer **borrowing over cloning**, especially for images and tensors
- No `unwrap()` or `expect()` in library code — always propagate errors with `?`
- Tests live in `#[cfg(test)]` modules; run per-crate with `cargo test -p <crate>`
- Use [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

### Documentation

Every public item **must** have a doc comment. Follow this structure:

```rust
/// One-line summary of what this does.
///
/// Longer explanation if needed — describe the algorithm, coordinate system,
/// units, or any non-obvious behavior. Keep it factual, not conversational.
///
/// # Arguments
///
/// * `src` - Input image. Must be RGB with `u8` pixels.
/// * `dst` - Pre-allocated output image. Must match `src` dimensions.
///
/// # Returns
///
/// `Ok(())` on success, or a [`KorniaError`] if dimensions mismatch.
///
/// # Errors
///
/// Returns [`KorniaError::InvalidChannels`] if `src` does not have 3 channels.
/// Returns [`KorniaError::SizeMismatch`] if `src` and `dst` dimensions differ.
///
/// # Example
///
/// ```rust
/// use kornia::image::Image;
/// use kornia::imgproc;
///
/// let src = Image::<u8, 3>::new(...)?;
/// let mut dst = Image::<u8, 1>::new(...)?;
/// imgproc::color::rgb_to_grayscale(&src, &mut dst)?;
/// ```
pub fn rgb_to_grayscale(src: &Image<u8, 3>, dst: &mut Image<u8, 1>) -> Result<(), KorniaError> {
    // ...
}
```

**Rules:**
- Document all `pub` functions, structs, enums, and traits — no exceptions
- `# Arguments`, `# Returns`, and `# Errors` sections are **required** on all public functions
- Include at least one `# Example` for non-trivial public APIs
- Keep examples compilable (they run as doctests)

### Safety

- Avoid `unsafe` unless strictly necessary
- Every `unsafe` block **must** be preceded by a `// SAFETY:` comment explaining exactly why it is sound:

```rust
// SAFETY: `ptr` is non-null and aligned, and `len` bytes have been
// initialised by the preceding call to `init_buffer`. No other references
// to this memory exist at this point.
let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
```

- Never cast away `const` or use `transmute` without an explicit soundness proof in the comment
- Prefer safe abstractions (e.g., `bytemuck`, `zerocopy`) over raw pointer arithmetic where possible

### Error Handling

- Use `thiserror` for all library error types — no `anyhow` in library crates
- Never use `.unwrap()`, `.expect()`, or `panic!()` in library code
- In test code, `.unwrap()` is acceptable but prefer `?` with `#[test] -> Result<(), Box<dyn Error>>`
- Error variants should be descriptive and carry context:

```rust
// Bad
Err(KorniaError::Generic("failed".into()))

// Good
Err(KorniaError::SizeMismatch {
    expected: src.size(),
    got: dst.size(),
})
```

### Performance

- Avoid allocations in hot paths — use pre-allocated `&mut` output buffers (established pattern)
- Prefer iterators over index-based loops for cache-friendly access
- Do not clone images or tensors unnecessarily — pass references
- Benchmark before and after any change to a core algorithm using `cargo bench`
- SIMD or platform-specific optimisations must be gated behind a feature flag and have a scalar fallback

---

## Contributing

### Pull Requests

1. **Keep PRs focused** — one concern per PR. Multiple features → multiple PRs.
2. **Test locally first** — all of the following must pass before opening a PR:

   ```bash
   pixi run rust-lint   # formatting + clippy
   pixi run rust-test   # full test suite
   pixi run py-test     # Python binding tests (if py code changed)
   pixi run cpp-test    # C++ binding tests (if C++ code changed)
   ```

3. **Update documentation** — any new or changed public API must have updated doc comments. New features need an entry in the relevant crate's `README.md` or module-level doc.

4. **Write tests** — new functionality requires unit tests. Bug fixes require a regression test that would have caught the bug.

5. **PR description** — include:
   - What changed and why
   - Any breaking changes
   - Benchmarks or profiling data for performance-sensitive changes
   - What use case does this pr address which you use yourself

### Contributing to Documentation

- Doc comments live alongside the code — edit the `.rs` source file directly
- Verify doctests compile and pass: `cargo test --doc -p <crate>`
- For larger documentation (guides, architecture docs), add or update Markdown files under `docs/`

### Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org)

---

## Quick Reference Checklist (before every PR)

- [ ] `pixi run rust-lint` passes with zero warnings
- [ ] `pixi run rust-test` passes
- [ ] All new/changed `pub` items have complete doc comments (args, errors, panics, example)
- [ ] Every `unsafe` block has a `// SAFETY:` comment
- [ ] No `unwrap()`/`expect()` added to library code
- [ ] New functionality has unit tests; bug fixes have regression tests
- [ ] Commit messages follow Conventional Commits