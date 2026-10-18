---
name: kornia-coding-style
description: Use when writing or reviewing Rust in kornia-rs — naming (<output>_from_<input>), error handling with thiserror, no unwrap/expect in library code, rustdoc structure, SAFETY comments, const instead of magic numbers, test layout. Trigger on any Rust edit under crates/.
---

# kornia-rs coding style

Run `cargo fmt` and clippy (`-D warnings`) before every commit.

## Naming

- Functions/variables `snake_case`, types `PascalCase`.
- Conversions are `<output>_from_<input>`: `gray_from_rgb`, `rgb_from_gray`,
  `bgr_from_rgb`. **Never** `rgb_to_gray` / `gray_to_rgb`.
- Metadata APIs use `*_layout` (e.g. `ImageLayout`), not `*_info`.
- Domain-specific names (`image_src`, `homography_dst`), not `data`, `helper`.

## Errors

- Library crates use `thiserror` error enums (e.g. `ImageError`); no `anyhow`.
- No `unwrap()`, `expect()`, `panic!()` in library code; propagate with `?`.
  Constructors that can fail return `Result`.
- Variants carry context (`SizeMismatch { expected, got }`), never
  `Generic("failed")`.
- Tests may `unwrap`, but prefer `-> Result<(), Box<dyn Error>>` with `?`.

## Documentation

Every `pub` item has `///`; every crate has `//!`. Public functions require
`# Arguments`, `# Returns`, `# Errors` sections, plus a compilable `# Example`
for non-trivial APIs (runs as a doctest: `cargo test --doc -p <crate>`).
Comments state invariants/rationale only — no obvious comments, no AI/prompt
commentary, English only.

## Safety

Avoid `unsafe`. Every `unsafe` block is preceded by `// SAFETY:` explaining
why it is sound. Prefer `bytemuck`/`zerocopy` over raw pointers. No
`transmute`/const-casting without a soundness argument.

## Other conventions

- Replace numeric literals used as thresholds, borders or loop bounds with
  named `const`s.
- Return domain structs, not tuples of loosely related `Vec`s; use `Option`
  for empty results.
- Borrow instead of `.clone()`; never clone images/tensors needlessly.
- Pre-allocated `&mut` outputs, no allocation in hot paths, iterators over
  index loops, match on enums rather than strings.
- Tests live in `#[cfg(test)]` modules, deterministic and focused.
- Reuse existing kornia-rs utilities instead of reimplementing them.
- New dependencies: minimal, mind build time and binary size.
