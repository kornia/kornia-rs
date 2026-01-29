## Project Overview
`kornia-rs` is a low-level computer vision library written in Rust. It serves as a backend for image I/O, visualization, and operations, designed to be thread-safe and efficient. It provides bindings for Python (`kornia-py`) and C++ (`kornia-cpp`).

**Key Technologies:**
*   **Rust:** Core implementation (workspace with multiple crates).
*   **Python:** Bindings via `PyO3` and `maturin`.
*   **C++:** Bindings via `CMake`.
*   **Pixi:** Development environment and task management.

## Project Structure
*   `crates/`: Contains the core Rust crates (`kornia-image`, `kornia-io`, `kornia-tensor`, etc.).
*   `examples/`: Standalone Rust examples demonstrating various capabilities.
*   `kornia-py/`: Python bindings and tests.
*   `kornia-cpp/`: C++ bindings and tests.
*   `pixi.toml`: Defines the development environment and tasks.
*   `Cargo.toml`: Rust workspace configuration.

## Crates Overview

The functionality is distributed across multiple crates in the `crates/` directory:

*   **`kornia`**: The main umbrella crate that re-exports functionality from sub-crates. Use this for general usage.
*   **`kornia-image`**: Defines core image types (`Image`, `ImageSize`) and traits for image manipulation.
*   **`kornia-io`**: Handles Image and Video I/O. Supports `turbojpeg`, `gstreamer`, and `v4l` (Video4Linux) via feature flags.
*   **`kornia-tensor`**: A lightweight tensor library designed for computer vision tasks.
*   **`kornia-tensor-ops`**: Provides operations and kernels for `kornia-tensor`.
*   **`kornia-imgproc`**: Image processing algorithms (color conversion, resizing, warping, filtering, edge detection, etc.).
*   **`kornia-3d`**: 3D computer vision algorithms and point cloud processing.
*   **`kornia-algebra`**: Algebraic types and utilities, interfacing with `nalgebra` and `glam`.
*   **`kornia-apriltag`**: Implementation of AprilTag marker detection and decoding.
*   **`kornia-bow`**: High-performance Hierarchical Bag of Words implementation.
*   **`kornia-vlm`**: Integration with Vision Language Models (VLM) using `candle`.

## Core Architecture & Concepts

### Image vs. Tensor
*   **`Tensor<T, N, A>` (`kornia-tensor`)**: The foundational data structure. It represents an N-dimensional array with owned storage, shape, and strides. It is generic over:
    *   `T`: Element type (e.g., `u8`, `f32`).
    *   `N`: Number of dimensions (const generic).
    *   `A`: Allocator (e.g., `CpuAllocator`).
    *   **Layout:** Defaults to row-major (C-contiguous).
*   **`Image<T, C, A>` (`kornia-image`)**: A wrapper around `Tensor<T, 3, A>`. It strictly enforces a 3D structure: `(Height, Width, Channels)`.
    *   **`ImageSize`**: Helper struct for `width` and `height`.
    *   **`ImageLayout`**: Metadata including size, channels, and pixel format.
    *   **Data Access:** Provides convenience methods like `get_pixel(x, y, ch)` but underlying data is accessible via `.as_slice()` or the inner tensor.

### Memory Management
*   **Allocators**: The library uses an `Allocator` trait. `CpuAllocator` is the standard default.
*   **Ownership**: `Tensor` and `Image` own their data.
*   **Views**: `TensorView` provides a non-owning window into tensor data, useful for reshaping or permuting without copying.

### Image Processing (`kornia-imgproc`)
*   Algorithms are organized by domain (e.g., `color`, `resize`, `features`).
*   **API Style**: Functions typically take input and mutable output buffers (destination passing style) to minimize allocations.
    *   Example: `imgproc::resize::resize_native(&src, &mut dst, ...)`


## Testing & Benchmarking

### Testing
*   **Unit Tests:** Located within `src/` files inside `#[cfg(test)]` modules. These focus on individual function logic.
*   **Integration Tests:** Located in the `tests/` directory at the root or within crates. These test higher-level workflows and API interactions.
*   **Commands:**
    *   Run all tests: `pixi run rust-test`
    *   Run tests for a package: `pixi run rust-test-package <package_name>`
    *   Run Python tests: `pixi run py-test`
    *   Run C++ tests: `pixi run cpp-test`

### Benchmarking
*   The project uses `criterion` for benchmarking performance.
*   **Location:** Benchmark files are found in `benches/` directories within each crate (e.g., `crates/kornia-imgproc/benches/`).
*   **Implementation:** Benchmarks compare `kornia-rs` performance against other libraries (like `image` crate or `ndarray`) and different implementation strategies (e.g., sequential vs. parallel).
*   **Commands:**
    *   Run all benchmarks: `cargo bench`
    *   Run benchmarks for a specific crate: `cargo bench -p <package_name>`
    *   Run a specific benchmark: `cargo bench --bench <bench_name>`

## Building and Development

### Environment Setup
The project uses `pixi` for environment management.
```bash
pixi install           # Install default dependencies
pixi install -e dev    # Install dev tools
pixi install -e cuda   # Install CUDA dependencies (Linux only)
```

### Rust (Core)
*   **Test:** `pixi run rust-test` (or `cargo test --features ci`)
*   **Lint:** `pixi run rust-clippy` (or `cargo clippy --workspace --no-deps --all-targets --features ci -- -D warnings`)
*   **Format:** `pixi run rust-fmt`
*   **Build Examples:** `pixi run rust-build-examples`

### Python Bindings
*   **Build (Dev):** `pixi run py-build` (runs `maturin develop` in `kornia-py`)
*   **Test:** `pixi run py-test` (runs `pytest`)

### C++ Bindings
*   **Build:** `pixi run cpp-build`
*   **Test:** `pixi run cpp-test`

## Development Conventions

### Coding Standards
*   **Rust Edition:** 2021.
*   **Formatting:** Enforced via `rustfmt`. Run `pixi run rust-fmt` before committing.
*   **Linting:** Enforced via `clippy`. No warnings allowed. Run `pixi run rust-clippy`.
*   **Error Handling:** Use `Result<T, E>` with descriptive errors. Avoid `unwrap()`/`expect()` in library code.
*   **Testing:**
    *   Unit tests in `#[cfg(test)]` modules within the same file.
    *   Integration tests in `tests/` directories.
    *   **Verification:** PRs *must* include proof of local test execution (logs).

### AI Policy (Strict)
*   **Authorship:** Contributors must be the sole responsible author. AI-generated code where the submitter acts merely as a proxy is **rejected**.
*   **Verification:** You must understand and be able to explain every line of code.
*   **Redundancy:** Use existing utilities; do not reinvent the wheel.
*   **See `AI_POLICY.md` for full details.**

### Workflow
1.  **Issue:** Every PR must be linked to an approved and assigned GitHub issue.
2.  **Branches:** Create feature branches (e.g., `feat/foo`, `fix/bar`) from `main`.
3.  **Commits:** Use clear, conventional commit messages.
4.  **PRs:** Keep them small and focused.
