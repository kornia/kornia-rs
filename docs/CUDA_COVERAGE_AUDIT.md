# 📘 CUDA Backend Coverage Handbook
> **Reference Guide & Gap Analysis for `kornia-imgproc`**  
> *Tracking Issue: [Phase 1 Coverage Audit (#1135)](https://github.com/kornia/kornia-rs/issues/1135)*

---

## 🧭 Executive Dashboard

This handbook provides an easy-to-read, comprehensive comparison between `kornia-imgproc`'s **CPU implementation** and its **CUDA GPU backend**. It serves as an actionable roadmap for contributors looking to implement missing GPU kernels or verify numerical correctness.

### Coverage at a Glance

> *Note: Percentages include both full parity (✅) and partial support (🟡).*

```
1. Filter Operations     [████████████████░░░░] 83%  (10 of 12 supported: 8 full, 2 partial)
2. Geometric Operations  [███████████░░░░░░░░░] 57%  (4 of 7 supported: 3 full, 1 partial)
3. Color / Hist / CLAHE  [████████████████░░░░] 82%  (18 of 22 supported: 16 full, 2 partial)
4. Feature Operations    [███░░░░░░░░░░░░░░░░░] 17%  (1 of 6 supported: 1 full)
```

### Status Legend
* ✅ **Full Parity**: Implemented on CUDA with matching parameters, types, and precision.
* 🟡 **Partial / Type Gap**: Implemented on CUDA, but limited to certain types (e.g. `u8` only, single-channel only, or fixed kernel sizes).
* ❌ **CPU-Only (Missing)**: No CUDA kernel exists today. High priority for new contributions.

---

## 🔍 Chapter 1: Filter Operations

Filter operations are located in [`crates/kornia-imgproc/src/filter/`](../crates/kornia-imgproc/src/filter/). Most separable filters are accelerated through a shared 2-pass launcher (`separable_filter_f32_cuda` and `separable_blur_u8_cuda`).

### Quick Status Matrix

| Operation | CPU Types | CUDA Status | CUDA Types | Notes |
|---|---|:---:|---|---|
| **`box_blur`** | `f32`, any `C` | ✅ | `f32`, any `C` | Full parity via separable horizontal + vertical filter. |
| **`box_blur_u8`** | `u8`, any `C` | ✅ | `u8`, any `C` | Optimized Q8 fixed-point separable blur. |
| **`box_blur_fast`** | `f32`, any `C` | ❌ | *None* | Fast 3-pass box approximation of Gaussian. |
| **`gaussian_blur`** | `f32`, any `C` | ✅ | `f32`, any `C` | Full parity with arbitrary kernel sizes & sigmas. |
| **`gaussian_blur_u8`** | `u8`, any `C` | ✅ | `u8`, any `C` | Routes to 3×3 binomial or Q8 two-pass automatically. |
| **`sobel`** | `f32`, any `C` | ✅ | `f32`, any `C` | Computes gradient magnitude $\sqrt{g_x^2 + g_y^2}$. |
| **`scharr`** | `f32`, any `C` | ✅ | `f32`, any `C` | Higher accuracy 3×3 derivative filter. |
| **`spatial_gradient`** | `f32`, any `C` | ❌ | *None* | Returns `(gx, gy)` pair. Computed on CUDA internally, but no public API. |
| **`laplacian_u8`** | `u8` $\to$ `i16` | ✅ | `u8` $\to$ `i16` | 3×3 second-order derivative filter. |
| **`bilateral_filter`** | `u8` (`C=1`) | 🟡 | `u8` (`C=1` only) | Byte-for-byte OpenCV parity. Multi-channel (RGB) and `f32` missing on both. |
| **`median_blur`** | `u8`, any `C` | 🟡 | `u8` (3×3 & 5×5) | Fast sorting networks. Kernels $\ge 7\times 7$ and `f32` missing on CUDA. |
| **`integral_image`** | `f32`, `u8` | ✅ | `f32`, `u8` | Summed Area Table computation. |

> [!TIP]
> **High-Impact Filter Issue to Pick Up**:
> **Expose CUDA `spatial_gradient_f32`**: The CUDA Sobel/Scharr pipelines already allocate scratch buffers for `gx` and `gy` internally. Creating a public function that returns `(Image<f32, C>, Image<f32, C>)` on device memory is an easy, high-value PR!

---

## 📐 Chapter 2: Geometric Operations

Geometric operations are in [`crates/kornia-imgproc/src/warp/`](../crates/kornia-imgproc/src/warp/), [`resize/`](../crates/kornia-imgproc/src/resize/), and [`interpolation/`](../crates/kornia-imgproc/src/interpolation/).

### Quick Status Matrix

| Operation | Interpolation Modes | CUDA Status | Supported Dtypes | Notes |
|---|---|:---:|---|---|
| **`resize`** | Nearest, Bilinear, Bicubic, Lanczos | ✅ | `u8`, `f32` (any `C`) | Full parity across all 4 modes. |
| **`warp_affine`** | Nearest, Bilinear, Bicubic, Lanczos | ✅ | `u8`, `f32` (any `C`) | Byte-exact contract with CPU implementation. |
| **`warp_perspective`** | Nearest, Bilinear, Bicubic, Lanczos | ✅ | `u8`, `f32` (any `C`) | Full parity with 3×3 homography transform. |
| **`remap`** | Nearest, Bilinear | 🟡 | `u8`, `f32` (any `C`) | Bicubic and Lanczos modes are missing on both CPU and CUDA. |
| **`crop`** | Rectangular RoI | ❌ | *None* | Extracting a sub-image bounding box is CPU-only. |
| **`pad`** | Constant, Reflect, Replicate | ❌ | *None* | Standalone padding is CPU-only (only exists fused inside `preprocess`). |
| **`flip`** | Horizontal, Vertical, Both, Transpose | ❌ | *None* | Mirroring and 90° rotations are completely CPU-only. |

> [!TIP]
> **Easiest PR in the Entire Roadmap**:
> **`feat(cuda): implement CUDA flip and transpose`**:
> Flipping an image requires no interpolation math—just swapping index coordinates in a 2D grid:
> * Horizontal: `src_x = (width - 1) - dst_x`
> * Vertical: `src_y = (height - 1) - dst_y`

---

## 🎨 Chapter 3: Color, Histogram & CLAHE Operations

Color space transformations live in [`crates/kornia-imgproc/src/color/`](../crates/kornia-imgproc/src/color/). This is the most complete CUDA subsystem in the library.

### Quick Status Matrix

| Subsystem | Operations | CUDA Status | Notes |
|---|---|:---:|---|
| **Basic Color** | `gray_from_rgb`, `rgb_from_gray` | ✅ | Supports `u8`, `f32`, and `f64`. |
| **Swizzle** | `bgr_from_rgb`, `rgba_from_rgb`, `bgra_from_rgb` | ✅ | Supports `u8` and `f32`. Fast memory swizzling. |
| **Perceptual Spaces** | `hsv`, `hls` | 🟡 | Full parity for `f32` and `f64`. Integer `u8` missing. |
| **CIE Standards** | `linear_rgb`, `xyz`, `lab`, `luv` | ✅ | Full parity for `f32` and `f64`. |
| **Video / Broadcast** | `yuv`, `ycbcr` | ✅ | Full parity for `u8` and `f32` across chroma formats. |
| **Sensor Demosaicing** | `rgb_from_bayer` | ✅ | Supports RGGB, BGGR, GBRG, GRBG patterns. |
| **Histogramming** | `compute_histogram`, `equalize_hist` | ✅ | Single-channel `u8` full parity. |
| **Contrast** | `clahe` | ✅ | Contrast Limited Adaptive Histogram Equalization. |
| **Colormaps** | `apply_colormap` | ✅ | Full parity for `u8` across all 21 OpenCV colormaps. |
| **Color Matrix** | `transform_color` | ❌ | Custom 3×3 matrix color transform is CPU-only. |
| **Thresholding** | `threshold_binary`, `truncate`, `otsu` | ❌ | All thresholding ops in `threshold.rs` are CPU-only. |

> [!TIP]
> **Recommended First Color PR**:
> **`feat(cuda): implement CUDA threshold_binary`**:
> Thresholding is embarrassingly parallel and trivial to write:
> ```cuda
> int idx = blockIdx.x * blockDim.x + threadIdx.x;
> if (idx < npixels) {
>     dst[idx] = (src[idx] > thresh) ? max_val : 0;
> }
> ```

---

## 🎯 Chapter 4: Feature Detection & Matching

Feature algorithms live in [`crates/kornia-imgproc/src/features/`](../crates/kornia-imgproc/src/features/).

### Quick Status Matrix

| Module | Description | CUDA Status | Notes |
|---|---|:---:|---|
| **`sift`** | SIFT detector, orientation & descriptors | ✅ | **World-class CUDA implementation**: Scale-space pyramid, DoG extrema, orientation histogram, 128D descriptors, and GPU matcher. |
| **`fast`** | FAST-9 / FAST-12 corner detector | ❌ | CPU-only (Planned for Phase 4). |
| **`orb`** | ORB detector, FAST keypoints, rBRIEF | ❌ | CPU-only (Planned for Phase 4). |
| **`responses`** | Harris & Shi-Tomasi corner scores | ❌ | CPU-only. |
| **`match`** | Brute-force & Hamming distance matcher | ❌ | Only SIFT matcher is on GPU; generic matcher is CPU-only. |
| **`cells`** | Spatial grid distribution / binning | ❌ | CPU-only. |

---

## 🔄 Maintenance & Verification Protocol (For Agents & Contributors)

To prevent this audit from becoming stale and to guide future contributors and AI agents when implementing or extending CUDA kernels, follow this standardized verification protocol.

### The 4-Layer Verification Rubric

Before updating an operation's status in this document, verify its implementation across all four layers:

1. **Rust Device Kernel**:
   * Inspect [`crates/kornia-imgproc/src/cuda/`](../crates/kornia-imgproc/src/cuda/) and subsystem folders (e.g. `color/`, `sift/`).
   * Verify the NVRTC kernel string or device launch function (e.g. `crates/kornia-imgproc/src/<module>/cuda.rs`) is compiled and registered under `#[cfg(feature = "cuda")]`.
2. **Residency Dispatch**:
   * Inspect the public API entry points in [`crates/kornia-imgproc/src/`](../crates/kornia-imgproc/src/).
   * Verify that device memory triggers the GPU launcher (via `try_device!` residency macro or device tensor dispatch) instead of falling back to CPU host copies.
3. **Python Bindings & Type Stubs**:
   * Inspect [`kornia-py/src/cuda_ext/`](../kornia-py/src/cuda_ext/) for PyO3 module bindings.
   * Verify type annotations exist in [`kornia-py/python/kornia_rs/cuda.pyi`](../kornia-py/python/kornia_rs/cuda.pyi).
4. **Numerical Parity Tests**:
   * Run device tests: `pixi run rust-test-cuda` or `cargo test --features cuda`.
   * Check Python parity test suites in [`kornia-py/tests/`](../kornia-py/tests/) (`test_cuda_*.py`) comparing device results with CPU implementations or OpenCV within precision tolerances (`atol` / `rtol`).

### Automated Verification Script

Run the automated verification script to cross-reference this document with the repository's codebase:

```bash
python scripts/verify_cuda_coverage.py --check
```

This script asserts that:
* Every operation marked as ✅ or 🟡 has valid device implementations and matching symbols in `crates/kornia-imgproc/src/cuda/`.
* Operations marked as ❌ (missing) undergo heuristic validation to check for unmapped implementation files or declarations in `crates/kornia-imgproc/src/cuda/` (supplementing manual source audits).
* All top-level CUDA modules declared in `crates/kornia-imgproc/src/cuda/mod.rs` are represented.

### Update Checklist for PRs Touching CUDA

Whenever a PR adds or modifies a CUDA kernel:

- [ ] **Update Matrix Status**: Change status symbols (`❌` $\to$ `🟡` $\to$ `✅`) and document supported types/channels in the respective Chapter table.
- [ ] **Update Progress Bars**: Recalculate supported counts and percentage bars in the [Executive Dashboard](#-executive-dashboard).
- [ ] **Update Playbook**: If the PR closes an issue in the [Implementation Playbook](#%EF%B8%8F-implementation-playbook-top-4-gap-issues-to-file--solve), mark it completed and nominate the next candidate gap.
- [ ] **Run Linter**: Run `python scripts/verify_cuda_coverage.py --check` to ensure no drift (and update `OP_SRC_MAP` in `scripts/verify_cuda_coverage.py` when adding new ops).

---

## 🛠️ Implementation Playbook: Top 4 Gap Issues to File & Solve

If you want to contribute code after this audit, here are the 4 best bite-sized issues ranked by difficulty:

### 1. `feat(cuda): implement CUDA flip operations`
* **Crate**: `kornia-imgproc`
* **Difficulty**: 🟢 Easy (1–2 days)
* **What to do**:
  1. Add NVRTC kernel in `crates/kornia-imgproc/src/cuda/` mapping `dst[y, x] = src[y, (w - 1) - x]` (horizontal) and `dst[(h - 1) - y, x]` (vertical).
  2. Add `try_device!` residency branch in [`crates/kornia-imgproc/src/flip.rs`](../crates/kornia-imgproc/src/flip.rs).
  3. Add parity tests comparing CPU and CUDA outputs.

### 2. `feat(cuda): implement binary threshold operations`
* **Crate**: `kornia-imgproc`
* **Difficulty**: 🟢 Easy (1–2 days)
* **What to do**:
  1. Add element-wise kernel in `crates/kornia-imgproc/src/cuda/` for `threshold_binary` and `threshold_truncate`.
  2. Connect to `crates/kornia-imgproc/src/threshold.rs`.

### 3. `feat(cuda): implement standalone CUDA pad operations`
* **Crate**: `kornia-imgproc`
* **Difficulty**: 🟡 Medium (2–3 days)
* **What to do**:
  1. Add NVRTC kernel in `crates/kornia-imgproc/src/cuda/` for constant, replicate, and reflect padding modes.
  2. Expose a standalone public `pad` function in `crates/kornia-imgproc/src/pad.rs` with `try_device!` residency dispatch.
  3. Add parity unit tests comparing CPU and CUDA outputs.

### 4. `feat(cuda): expose spatial_gradient_f32`
* **Crate**: `kornia-imgproc`
* **Difficulty**: 🟡 Medium (2–3 days)
* **What to do**:
  1. Create a public function `spatial_gradient` that returns `(Image<f32, C>, Image<f32, C>)`.
  2. Reuse the existing separable filter launches already present in `gradient_magnitude_f32_cuda`.
