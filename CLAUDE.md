# CLAUDE.md

## Project Overview
Kornia‑rs is a low‑level 3D computer‑vision library written in Rust, with bindings for Python (`kornia-py`) and C++ (`kornia-cpp`). It provides a collection of vision primitives, tensor operations, and utilities for building higher‑level vision pipelines.

## Workspace Structure
- `crates/` – core Rust crates (e.g., `kornia`, `kornia-image`, `kornia-tensor`).
- `examples/` – standalone example applications demonstrating crate usage.
- `kornia-cpp/` – C++ bindings and examples.
- `kornia-py/` – Python bindings (via `pyo3`).

## Build & Test Commands
```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Generate documentation
cargo doc --no-deps
```

## Conventions
- **Code style**: `cargo fmt` and `cargo clippy` must pass.
- **Documentation**: Every crate must have a `README.md` following the standard template defined in the implementation plan.
- **AI Guidance**: When generating code or documentation, adhere to the unified README format and include the badge shields, installation snippet, usage example, and licensing information.

## License
Apache‑2.0
