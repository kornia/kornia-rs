---
name: kornia-bindings
description: Use when changing Python (kornia-py, PyO3/Maturin) or C++ (kornia-cpp) bindings in kornia-rs, or Rust APIs they wrap. Covers build/test commands, parity rules and free-threaded Python.
---

# Python and C++ bindings

## Python (`kornia-py/`)

- Build: `pixi run py-build` (maturin develop); test: `pixi run py-test`
  (pytest in `kornia-py/tests`, auto-builds). Threaded stress:
  `pixi run py-test-threaded`.
- Free-threaded 3.14t environment uses `scripts/build-py314t.sh`.
- Stubs/python sources live under `kornia-py/python`; update them with Rust
  changes. Read `kornia-py/CONTRIBUTING.md`.
- Release wheels are built with `--features cuda` (dynamic loading, CPU
  fallback); do not break import on machines without CUDA.
- Errors map to Python exceptions; never `unwrap()` in binding code.

## C++ (`kornia-cpp/`)

- Build/test: `pixi run cpp-test`; format: `pixi run cpp-fmt` / `cpp-fmt-check`;
  sanitizers: `pixi run cpp-test-sanitizers`.
- Headers in `include/`, Rust FFI in `src/`, tests in `tests/`.

## Rules

- Public Rust API changes that are exposed must be reflected in both bindings
  (or explicitly noted as not exposed in the PR).
- Add binding-side tests for every new exposed function.
