# Changelog

All notable changes to kornia-rs are recorded here, newest first. Each entry is
written for users of the Rust crates and the `kornia-rs` Python wheels — what
changed and why it matters, not a raw commit dump.

Pre-releases (`-rc.N`) are cut before a stable tag so the community can try
changes early: `cargo add kornia-imgproc@0.1.15-rc.1` or `pip install --pre kornia-rs`.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Reference the diff range and prior tag
     so the changelog stays navigable (see the "Full changelog" links below). -->

## [0.1.15-rc.3] — 2026-07-06 (pre-release)

Two fixes on top of rc.2 (which brought the crates.io publish):

- **AprilTag**: bias the adaptive binarization toward white (`threshold_split`,
  default `0.33`) so small / glary / low-contrast Tag36H11 markers decode instead
  of returning zero detections; the tag's black border no longer merges with
  adjacent dark regions (#985).
- **imgproc (GPU/CUDA)**: texture-object warp-affine kernels — a `CudaTexObject`
  RAII wrapper with `CU_TR_ADDRESS_MODE_BORDER` replaces the per-pixel OOB
  bounds-check branch, cutting warp divergence on rotated images (~30–40% of
  output pixels map outside the source); optional `block_dim` tuning (#979).

**Full changelog:** `v0.1.15-rc.2...v0.1.15-rc.3`

## [0.1.15-rc.2] — 2026-07-06 (pre-release)

Same contents as rc.1; unblocks the **crates.io** publish. The `dlpack-rs`
dependency moved from a git tag to a published crate (`0.3.3` on crates.io, with
its `dlpack-sys` internals inlined), so `kornia-tensor` / `kornia-image` no
longer carry a git dependency and the workspace can publish to crates.io. Python
wheels are unchanged from rc.1.

**Full changelog:** `v0.1.15-rc.1...v0.1.15-rc.2`

## [0.1.15-rc.1] — 2026-07-06 (pre-release)

First pre-release since 0.1.14 (2026-05-19), bundling ~247 commits. Headline is
**GPU acceleration landing across the stack** plus a **CPU warp-affine rewrite**.

### CUDA / GPU
- **CUDA color conversions** with residency-aware dispatch and fused camera
  preprocessing — faster than OpenCV + VPI on the tested paths (#966).
- **`kornia_rs.cuda` Python module** — GPU color conversions + fused camera
  preprocess from Python (#969), with a pinned-staging preprocessor and
  zero-copy DLPack import into torch / TensorRT (#970, #972).
- **cudarc device-memory integration** — `CudaResource` / `CudaAllocator`,
  `Tensor::to_cuda`, `CudaKernel` (#950).
- **GPU resize kernels** (nearest + bilinear) via CubeCL (#946).
- **GPU image→tensor Preprocessor** + cudarc kernel ergonomics (#960).
- **DLPack interop** for `Tensor` / `Image` and kornia-py — zero-copy,
  bidirectional with torch (#951).

### imgproc
- **CPU warp-affine rewrite** — incremental coordinates, analytical valid-range
  skip, 16-row Rayon chunks; 2–2.5× faster nearest, bilinear now beats cv2 CPU
  (#971).
- **Preprocessor builder** — RGBA source, mean/std normalize, configurable pad
  (#964).

### AprilTag
- **3D pose estimation** + SIMD-accelerated detector (#959).
- Dedup no longer drops repeated tag ids — spatial-overlap dedup (#973).

### Core / breaking
- **Removed the compile-time allocator type param** on `Tensor` / `Image`;
  runtime `AllocHandle` + host-default constructors (#955). Migration:
  `Image<f32, 3, CpuAllocator>` → `Image<f32, 3>`.
- New `AllocationFailed` error variant preserving the backend message.

**Full changelog:** `v0.1.14...v0.1.15-rc.1`

## [0.1.14] — 2026-05-19

turbojpeg is finally reachable from the Python API. 0.1.13 shipped wheels with
libjpeg-turbo compiled in, but the Python-facing symbols were silently absent
because the `#[cfg(feature = "turbojpeg")]` gates checked kornia-py's own flag,
not the kornia-io dep's. Fixed with `default = ["turbojpeg"]`. Also: Windows CI
switched to `CMAKE_GENERATOR=Ninja` + `ilammy/msvc-dev-cmd` (VS-version-agnostic).

**Full changelog:** `v0.1.13...v0.1.14`

## [0.1.13] — 2026-05-18

- turbojpeg enabled by default in published wheels (2–4× faster JPEG vs pure-Rust).
- Windows CI fixed (pin `CMAKE_GENERATOR=Visual Studio 17 2022`).
- All turbojpeg paths gated behind `#[cfg(feature = "turbojpeg")]`; pure-Rust
  fallback still works for source builds without cmake.

**Full changelog:** `v0.1.12...v0.1.13`

## [0.1.12] — 2026-05-18

- 3D / BA: Schur-complement bundle adjustment, SE(3) pose-graph optimization,
  cheirality-robust BA, IRLS Huber + Cauchy loss; `solve_pnp_ransac` → `k3d`.
- Generic RANSAC framework (NEON + AVX2 scorer, LO-RANSAC) + homography estimator.
- Segmentation + depth: COCO RLE ↔ mask, `depth.sample_depth` (19× vs Python).

**Full changelog:** `v0.1.11...v0.1.12`

## [0.1.11] — 2026-05-04

First release in 6 months, 122 commits. New crates: kornia-vlm, kornia-pnp,
kornia-lie, kornia-nn, kornia-apriltag, kornia-algebra, v4l2 camera capture.
Python: PIL-parity Image API with zero-copy numpy, u16/f32 dtypes, WebP/TIFF,
AprilTag bindings, Python 3.14t free-threaded builds.

**Full changelog:** `v0.1.10...v0.1.11`

## [0.1.10] — 2025-11-08

Video reader, custom `Allocator` for `Image`, kornia-lie on glam-rs, SIMD DNN
linear layer / kornia-nn, kornia-apriltag, zero-copy gstreamer images. See the
[GitHub release](https://github.com/kornia/kornia-rs/releases/tag/v0.1.10) for
the full per-PR list.

[0.1.15-rc.3]: https://github.com/kornia/kornia-rs/compare/v0.1.15-rc.2...v0.1.15-rc.3
[0.1.15-rc.2]: https://github.com/kornia/kornia-rs/compare/v0.1.15-rc.1...v0.1.15-rc.2
[0.1.15-rc.1]: https://github.com/kornia/kornia-rs/compare/v0.1.14...v0.1.15-rc.1
[0.1.14]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.14
[0.1.13]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.13
[0.1.12]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.12
[0.1.11]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.11
[0.1.10]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.10
