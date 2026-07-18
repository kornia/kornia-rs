# Changelog

All notable changes to kornia-rs are recorded here, newest first. Each entry is
written for users of the Rust crates and the `kornia-rs` Python wheels — what
changed and why it matters, not a raw commit dump.

Pre-releases (`-rc.N`) are cut before a stable tag so the community can try
changes early: `cargo add kornia-imgproc@0.1.15-rc.1` or `pip install --pre kornia-rs`.

<!-- When cutting a release, move the curated items from [Unreleased] into a new
     dated section and reset [Unreleased]. Reference the diff range and prior tag
     so the changelog stays navigable (see the "Full changelog" links below). -->

## [Unreleased]

**GPU histogram + `equalize_hist` (CPU and CUDA), byte-for-byte with
OpenCV.** New `equalize_hist` for u8 single-channel images on CPU and
device (`kornia_rs.imgproc.equalize_hist` in Python), matching
`cv2.equalizeHist` exactly — including its f32 round-half-to-even LUT
rounding and identity behavior on constant images. `compute_histogram`
now routes device images to a CUDA shared-memory-bins kernel whose counts
are exactly equal to the CPU's for any bin count. CPU and GPU equalize
outputs are byte-identical.

**GPU Gaussian pyramids + `PyramidPlan`.** `pyrdown_f32` (one fused
5×5-Gaussian+subsample kernel), `pyrup_f32` (polyphase pair), `pyrdown_u8`
and `pyrup_u8` (separable `[1,4,6,4,1]` integer pairs) now route device
images to CUDA kernels that are bit-identical (f32) / byte-identical (u8)
to the CPU paths, including the `reflect_101` borders and 1-pixel-wide
degenerate sources. `PyramidPlan` preallocates every level buffer so a
steady-state pyramid build launches kernels only (CUDA-Graph-capturable);
its levels are bit-identical to `build_pyramid`. Measured 1080p: pyrdown
u8 RGB 0.62 ms (4.1× OpenCV CPU), 4-level f32 plan 0.68 ms.

**Faster Lanczos-3 warps on CUDA; sub-ULP output change in Lanczos warp/remap
sampling.** The warp kernels now derive all six tap weights per axis from four
`sin(πx)` evaluations instead of twelve, using the sinc lattice's periodicity
(`sin(π(frac+n)) = ±sin(π·frac)`); the CPU sampler (`lanczos_sample`, used by
`warp_affine`/`warp_perspective`/`remap` with `InterpolationMode::Lanczos`) is
refactored identically, so CPU and GPU remain bit-exact to each other. Absolute
values shift by ≤1e-6 relative to previous releases (weights are identical in
real arithmetic, rounding differs). Measured on Jetson AGX Orin (locked
clocks, 1080p→1080p 3-channel f32): warp-affine lanczos 5.49→4.26 ms,
warp-perspective lanczos 5.93→4.28 ms sustained.

**GPU separable filters: gaussian, box, sobel, scharr.** New CUDA separable
engine (`cuda/filter.rs`) with residency dispatch on `gaussian_blur`,
`box_blur`, `sobel`, `scharr` (f32) and `gaussian_blur_u8`, `box_blur_u8` —
f32 paths bit-identical to the CPU engine (skip-zero border, sequential
taps, fmad-free), u8 paths byte-identical to the Q8 striped blur (replicate
borders) and the 3×3 binomial fast path, with the u8 kernel/sigma routing
decided by one selector shared with the CPU. Python `gaussian_blur` /
`box_blur` accept device images; new `imgproc.sobel` (f32, numpy + device).
Measured 1080p vs OpenCV CPU: gaussian 5×5 u8 RGB 9.2×, box 5×5 9.5×,
sobel-3 f32 13×. The u8 binomial tails/edges now use the same nested
halving-add rounding as the SIMD bulk (≤1 LSB shift on border bytes vs
previous releases).

**u8 color conversions are now byte-for-byte with OpenCV.** Three alignments
(CPU and CUDA changed together, so device output stays `assert_eq!`-equal to
host): u8 grayscale now uses cv2's exact Q14 formula
(`(4899·R + 9617·G + 1868·B + 8192) >> 14`; ±1 LSB vs previous releases);
`yuv_from_rgb`/`rgb_from_yuv` switch to the classic Y'UV chroma constants
cv2 and kornia-python use (0.492/0.877 forward, 1.140/2.032/−0.395/−0.581
inverse) instead of Cb/Cr in YUV order — a semantic change for chroma
values; and Bayer demosaic adopts cv2's border rule (the 1-px frame is the
interior neighbour's result). Verified byte-exact against cv2 for
yuv (both directions), bayer (all pixels), ycbcr, bgr, rgba and colormap;
gray matches cv2's documented formula exactly (cv2's own NEON build deviates
from its spec on ~0.25% of pixels). BGR/YCbCr/colormap already matched.

**Python: every color conversion with a CUDA kernel now accepts a device
`Image`.** Previously ~10 GPU color paths existed in the Rust crates but the
Python surface raised "no GPU kernel for a device Image": HLS, Luv, XYZ,
linear-RGB (f32 pairs), planar YUV (u8 **and** f32 pairs), plus the f32 arms
of YCbCr, sepia and gray (`imgproc.gray_from_rgb` on an f32 device image now
returns a device gray f32 image). u8 arms stay bit-exact with the CPU path;
f32 arms follow the existing tolerance contract. Covered by device-vs-CPU
parity tests and a memory-leak loop over the new arms.

**u8 CUDA morphology (dilate / erode), bit-identical to the CPU ops, plus
first-time Python bindings.** `dilate` / `erode` route u8 device images
(1/3/4-channel) to a CUDA kernel that samples the exact tap multiset the CPU
does — same active structuring-element offsets, same border index mapping
for all five padding modes — so min/max output is byte-for-byte equal
(`assert_eq!` parity tests over box/cross/ellipse and even-sized kernels).
The structuring element is cached on-device per geometry (Jetson pageable
H2D tail; warm path is CUDA-Graph-capturable). New
`kornia_rs.imgproc.dilate` / `erode` Python functions cover the numpy u8 CPU
path and u8 device images alike. Non-u8 device pairs error with the typed
no-GPU-kernel message instead of touching device memory from the CPU.

**u8 CUDA warps (bilinear), bit-identical to the CPU fast paths — which are
now bit-identical to each other too.** `warp_affine_u8` and
`warp_perspective_u8` route u8 device images (1/3/4-channel) to new
integer-only CUDA kernels that mirror the CPU's per-row span math and
Q16/Q10 fixed-point sampling exactly; `kornia_rs.imgproc.warp_affine` /
`warp_perspective` accept u8 device images including `out=`. Along the way
the CPU u8 perspective warp switched from incremental coordinate updates
(`nx += dnx` per column, with a SIMD reciprocal estimate) to direct
per-column evaluation with exact IEEE division — scalar, NEON, and AVX2 now
produce identical bytes instead of agreeing "up to sub-ULP noise". Output
shifts by at most one Q10 quantum on some pixels relative to earlier
releases; benchmarks show no measurable CPU cost (the division latency hides
behind the 4-lane structure).

**u8 CUDA resize — every mode, bit-identical to the CPU fast paths.** Camera
data is u8; until now the GPU only resized f32 (4× the memory traffic once you
count the conversion). `resize_fast_u8_aa` (Rust) and `kornia_rs.imgproc.resize`
(Python, including `out=`) now route u8 device images — 1/3/4-channel — to new
integer-only CUDA kernels covering the full CPU cascade: exact-2× pyramid fast
paths, nearest, Q14 bilinear, and antialiased/plain separable bicubic and
Lanczos-3. The coordinate and weight tables are built by the same host
functions the CPU uses and uploaded once per geometry (a device-side LUT
cache — Jetson pageable H2D uploads have a ~250 µs average latency tail that
otherwise dominates, and the warm path is CUDA-Graph-capturable), so device
output is byte-for-byte equal to host output (asserted with `assert_eq!`
across modes, channel counts, odd sizes, and extreme downscales). Measured on
Jetson Orin at 1080p→720p u8 RGB (median, out= reuse): nearest 0.22 ms,
bilinear 0.38 ms, bicubic 1.20 ms — faster than VPI-CUDA on every comparable
op (5.4× / 3.4× / 1.1×) and 3–6× ahead of OpenCV CPU. Kernel compile failures
now surface as errors instead of panics across all resize launchers.

**Python: `out=` and CUDA Graphs for allocation-free frame loops.** The device
geometry ops accept a preallocated device `out=` image (torch-style, returned
back); `kornia_rs.cuda.Stream.new()` creates a capturable non-default stream;
and `kornia_rs.cuda.Graph.capture(f, retain, stream)` records an
allocation-free op sequence for microsecond-overhead `replay()`. Measured on
Jetson Orin (1080p→720p f32 resize, per frame with sync): 1.31 ms → 0.62 ms
with `out=` on a created stream → **0.42 ms** with graph replay — 20× the
naive pattern of two days ago, byte-exact output preserved through capture.

**Jetson/CUDA: per-call device allocations no longer stall frame loops.** The
default CUDA memory pool releases freed blocks back to the OS at every stream
synchronization (release threshold 0), so a per-frame `op(); sync()` pattern
re-paid the full allocation cost each frame — ~8.5 ms/frame of alloc/free churn
for 1080p f32 on Jetson Orin. kornia-tensor now raises the pool's release
threshold once per device at first allocation: the same loop drops to 1.3
ms/frame (and pipelined use improves 1.08 → 0.60 ms). Applies to every device
op (color conversions included).

**Bicubic and Lanczos-3 for the f32 geometry ops, byte-exact CPU↔GPU.**
`resize`, `warp_affine`, `warp_perspective`, and `remap` accept
`InterpolationMode::Bicubic` and `::Lanczos` on the CPU, and device images
route to the previously launcher-only CUDA kernels — all six now reachable
from the public API, with bit-identical output between backends. Lanczos
weights use a shared deterministic sin(πx) polynomial (libm/CUDA `sinf`
rounding differs between platforms); resize-lanczos builds its per-axis weight
tables on the host once per call, shared verbatim with the GPU launcher —
which also made the GPU kernels up to 27% faster than the old `__sinf` path.
Hot-loop interpolation dispatch is hoisted per mode, making CPU f32 resize
~30% faster and warp-perspective ~7% faster than before.

**Python: GPU resize and warps.** `kornia_rs.imgproc.resize` / `warp_affine` /
`warp_perspective` now accept a device `Image` (f32, 3-channel) and run the
CUDA kernels, returning a device `Image` — output bit-identical to the CPU f32
path. numpy u8 inputs behave exactly as before; a device image with an
unsupported dtype/channel count raises instead of silently copying to the
host. `out=` is unsupported on the device path.

**`resize_native` is renamed to `resize`** (clean rename, no alias).

**GPU resize and warps from the high-level API.** `resize` (né `resize_native`),
`warp_affine`, and `warp_perspective` now run on the GPU when called with
device-resident images (`Image::to_cuda` / `zeros_cuda`), with **bit-identical
output to the CPU path**. Same rules as the color conversions: both operands
must be on the same device (mixed pairs are a typed error, no implicit
transfers), and unsupported dtype/channel combinations error instead of
silently falling back (the GPU kernels are 3-channel f32).


**OpenCV-compatible resize (opt-in).** New `resize_opencv_u8` / `resize_opencv_f32`
reproduce OpenCV's reference `INTER_LINEAR` (u8: 11-bit fixed point; f32:
separable float) and floor-based `INTER_NEAREST` semantics, for migrating from
or cross-validating against cv2 pipelines. Validated against 72 reference
vectors generated by `cv2.resize` on Jetson Orin. Notable empirical finding
encoded in the tests: a shipped cv2 wheel is not internally byte-consistent for
`INTER_LINEAR` — the aarch64 wheel routes different channel counts and scales
through different backends (OpenCV, Carotene, Arm KleidiCV) whose bits disagree
with each other and with OpenCV's own reference arithmetic (u8 up to 2 LSB, f32
up to 3 ulp), so the tests assert exact bytes for nearest and a measured ≤2 LSB
/ ≤4 ulp corridor for linear; our reference arithmetic itself is pinned
byte-exactly by unit vectors.


**CPU and CUDA warp-affine / warp-perspective are now byte-exact** (f32,
bilinear + nearest): the CPU evaluates the source coordinate per pixel with the
same expression tree as the kernels (no more per-row accumulation drift), the
perspective kernels divide by `w` directly instead of multiplying by a
reciprocal, and the interpolation expression shapes match — parity tests assert
`f32::to_bits` equality on rotations, shears, and projective homographies.
Fixes a real GPU artifact along the way: `cos(90°)` is an ulp of noise rather
than zero, and the old strict border test zero-filled edge pixels of exact
right-angle rotations (splitting rows into half-warped, half-black); validity
of a near-constant axis is now judged on its row constant on both CPU and GPU.
No measured throughput change on Jetson Orin (CPU and GPU within run-to-run
jitter).

**CPU and CUDA resize are now byte-exact.** NVRTC kernels compile with
`--fmad=false` (no automatic FP contraction), and the resize coordinate and
bilinear arithmetic use the identical expression shapes on both sides, so
`resize_native` and the CUDA resize kernels produce bit-identical f32 output —
asserted by parity tests across dyadic and non-dyadic sizes. Kernels that want
fused multiply-adds keep them via explicit `fmaf()` (unaffected by the flag);
no measured throughput change on Jetson Orin. The `bilinear_interpolation`
restructure (weights formed before the tap products) shifts f32 bilinear
output by up to 1 ulp in every consumer — `warp_perspective`, `remap`, and
the optical-flow border path, not just resize.

**BREAKING — `resize_native` now samples on the half-pixel grid.** The f32
resize previously used align-corners (`sx = x * (src-1)/(dst-1)`); it now uses
the pixel-center convention (`sx = (x + 0.5) * src/dst - 0.5`) shared by
OpenCV, Pillow, ONNX `Resize` (`half_pixel` default), PyTorch
(`align_corners=False`), and NVIDIA VPI. Pixel values change for every
non-identity resize — typically a sub-pixel content shift. The u8 fast paths
(`resize_fast_*`) are unaffected. This also makes the CPU and CUDA resize
sample the same grid (`resize_native` was the sole align-corners holdout —
the fused CPU preprocess path and the CUDA kernels already sampled
half-pixel), and CPU/GPU parity is now tested.

- `kornia_imgproc::cuda::resize`: launchers take a new `PixelMapping`
  parameter (`HalfPixel` — the default convention — or `AlignCorners` to
  reproduce pre-change output); kernels generalized to per-axis affine
  coefficients with no measured throughput change on Jetson Orin.

## [0.1.15-rc.4] — 2026-07-06 (pre-release)

**GPU wheels.** Linux wheels now build with `--features cuda`, so
`pip install --pre kornia-rs` ships the `kornia_rs.cuda` module (GPU color
conversions, `CudaImage`/`CudaPreprocessor`, zero-copy DLPack). The wheel loads
CUDA lazily (cudarc `fallback-dynamic-loading`): it runs on CPU when no GPU is
present and activates CUDA at runtime when an NVIDIA driver + `nvrtc` are
available. Same crate/library contents as rc.3 otherwise.

- `python_release.yml`: linux job builds `--features cuda` (macOS/Windows stay
  CPU-only — no NVIDIA hardware).
- README: added a **GPU / CUDA** usage section with an `upload → op → download`
  snippet and the runtime requirements.

**Full changelog:** `v0.1.15-rc.3...v0.1.15-rc.4`

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

[0.1.15-rc.4]: https://github.com/kornia/kornia-rs/compare/v0.1.15-rc.3...v0.1.15-rc.4
[0.1.15-rc.3]: https://github.com/kornia/kornia-rs/compare/v0.1.15-rc.2...v0.1.15-rc.3
[0.1.15-rc.2]: https://github.com/kornia/kornia-rs/compare/v0.1.15-rc.1...v0.1.15-rc.2
[0.1.15-rc.1]: https://github.com/kornia/kornia-rs/compare/v0.1.14...v0.1.15-rc.1
[0.1.14]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.14
[0.1.13]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.13
[0.1.12]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.12
[0.1.11]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.11
[0.1.10]: https://github.com/kornia/kornia-rs/releases/tag/v0.1.10
