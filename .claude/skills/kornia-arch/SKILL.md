---
name: kornia-arch
description: Use first when learning or navigating kornia-rs — workspace layout, crate responsibilities, feature flags, and core types (Tensor, Image<T, C, A>). Trigger on "where does X live", "which crate", "how is the workspace organised", or before touching an unfamiliar crate.
---

# kornia-rs architecture

Low-level 3D computer vision library in Rust (edition 2021, MSRV 1.89), with
PyO3/Maturin Python bindings and C++ bindings.

## Layout

- `crates/*` — core library crates (workspace default members)
- `examples/*` — standalone example binaries
- `kornia-py/` — Python bindings (PyO3/Maturin, `python/`, `tests/`)
- `kornia-cpp/` — C++ bindings (CMake, `include/`, `tests/`)
- `scripts/` — release/version helpers (`release_rust.sh`, `check_version_pins.py`)

## Crates

| Crate | Purpose |
|---|---|
| `kornia-tensor` | N-dimensional tensor |
| `kornia-tensor-ops` | Tensor operations |
| `kornia-image` | `Image<T, C, Alloc>` built on tensors (`ImageError`) |
| `kornia-imgproc` | resize, color, filters, warp, features, SIMD kernels |
| `kornia-io` | image/video I/O (turbojpeg, gstreamer, v4l) |
| `kornia-3d` | ICP, point clouds, stereo |
| `kornia-algebra` | linear algebra |
| `kornia-calib` | multi-camera extrinsic calibration (AprilTag BA) |
| `kornia-apriltag`, `kornia-bow`, `kornia-vlm` | detection, bag of words, VLM (candle) |
| `kornia` | umbrella crate that re-exports the others |

The publish set is `PUBLISH_ORDER` / `NOT_PUBLISHED` in `scripts/release_rust.sh`
(crates with `publish = false` are skipped). Read it there; do not restate it.

## Feature flags

`ci` = `arrow`, `bincode`, `gstreamer`, `serde`, `turbojpeg` (needs `nasm`),
`v4l` (needs `clang`). `cuda` is **not** in `ci` (needs hardware). OpenCV
comparison code in `kornia-imgproc` is behind `opencv_bench` and is not built
by the lint workflow.

## Key patterns

- `Image<T, C, Alloc>`: `T` pixel type (`u8`, `f32`), `C` channel count.
- imgproc ops take `&src` and a pre-allocated `&mut dst`.
- I/O entry points live in `kornia::io::functional`.
- Calibration thresholds (`max_reprojection_error`, `robust_scale_sq`) are in
  **normalized image coordinates**, not pixels.

See also: `kornia-workflow`, `kornia-coding-style`, `kornia-api-flavor`.
