# kornia-py Benchmarks

## Run Info

| Field | Value |
|-------|-------|
| Date | 2026-04-16 |
| Commit | `cc63e51` (antialias flag + separable NEON bicubic/lanczos) |
| pyo3 | 0.28 |
| numpy (crate) | 0.28 |
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| Rust | 1.93.0, opt-level=2 (LLVM bug workaround for faer) |
| Python | 3.10.12 |
| numpy | 2.2.6 |
| OpenCV | 4.13.0 |
| albumentations | 2.0.8 |
| Iterations | 200 (10 warmup) |

## Results — 640×480

| Operation | kornia-rs (ms) | albumentations (ms) | OpenCV (ms) | vs best competitor |
|-----------|---------------:|--------------------:|------------:|-------------------:|
| ColorJitter (b+c+s+h) | **4.54** | 7.00 | 9.80 | **1.5× faster** |
| Brightness | **0.093** | 0.54 | 0.92 | **5.8× faster** |
| Horizontal Flip | **0.20** | 0.37 | 0.35 | **1.8× faster** |
| Vertical Flip | **0.076** | 0.11 | 0.087 | **1.1× faster** |
| Crop 224×224 | 0.025 | 0.040 | **0.023** | ~tie |
| Grayscale | **0.064** | — | 0.15 | **2.4× faster** |
| Resize (half, bilinear) | **0.086** | — | 0.37 | **4.4× faster** |
| Gaussian Blur 5×5 | 0.76 | — | **0.22** | 3.5× slower |
| Rotation ±30° | **1.14** | 2.15 | 1.57 | **1.4× faster** |
| Normalize | **0.67** | — | 10.7 | **16× faster** |

## Results — 1920×1080

| Operation | kornia-rs (ms) | albumentations (ms) | OpenCV (ms) | vs best competitor |
|-----------|---------------:|--------------------:|------------:|-------------------:|
| ColorJitter (b+c+s+h) | **30.9** | 43.5 | 64.2 | **1.4× faster** |
| Brightness | **0.87** | 3.62 | 6.03 | **4.2× faster** |
| Horizontal Flip | **0.96** | 2.62 | 2.47 | **2.6× faster** |
| Vertical Flip | **0.76** | 1.14 | 1.08 | **1.4× faster** |
| Crop 224×224 | 0.063 | 0.076 | **0.025** | 2.5× slower |
| Grayscale | **0.44** | — | 0.49 | **1.1× faster** |
| Resize (half, bilinear) | **0.35** | — | 0.71 | **2.0× faster** |
| Gaussian Blur 5×5 | 3.68 | — | **1.05** | 3.5× slower |
| Rotation ±30° | **8.06** | 9.91 | 8.75 | **1.1× faster** |
| Normalize | **4.26** | — | 75.0 | **17.6× faster** |

## Resize matrix — all modes, multiple shapes

Resize results across interpolation modes, `antialias` flag, and source→dest shape combinations. `antialias=True` matches PIL / torchvision semantics (kernel widens by scale; no aliasing on strong downscale). `antialias=False` matches OpenCV INTER_CUBIC / INTER_LANCZOS4 semantics (fixed 4/8-tap). `Bilinear` and `Nearest` are unaffected by the flag.

| Shape | Mode | k(aa=True) | k(aa=False) | OpenCV | PIL | `aa=False` vs OpenCV |
|---|---|---:|---:|---:|---:|---:|
| 1080p → 540p | bilinear | 0.49 | 0.42 | 0.82 | 19.0 | **1.97× faster** |
| 1080p → 540p | bicubic  | 4.90 | 2.38 | 4.18 | 31.6 | **1.75× faster** |
| 1080p → 540p | lanczos  | 8.63 | 4.15 | 6.80 | 45.4 | **1.64× faster** |
| 1080p → 224² | bilinear | 0.54 | 0.25 | 0.72 | 10.4 | **2.84× faster** |
| 1080p → 224² | bicubic  | 4.14 | 0.78 | 2.29 | 19.9 | **2.93× faster** |
| 1080p → 224² | lanczos  | 7.13 | 1.85 | 4.57 | 30.1 | **2.47× faster** |
| 640² → 320²  | bilinear | 0.07 | 0.09 | 0.29 |  3.79 | **3.09× faster** |
| 640² → 320²  | bicubic  | 1.43 | 0.93 | 1.48 |  6.22 | **1.60× faster** |
| 640² → 320²  | lanczos  | 2.90 | 1.52 | 2.53 |  9.07 | **1.66× faster** |
| 1080p → 2160p (up) | bilinear | 20.7 | 20.6 | **9.59** | 181 | 2.14× slower |
| 1080p → 2160p (up) | bicubic  | 19.1 | 19.1 | 24.2 | 229 | **1.27× faster** |
| 1080p → 2160p (up) | lanczos  | 42.2 | 42.5 | 45.6 | 283 | **1.07× faster** |

## Improvement log (2026-04-02 → 2026-04-16)

| Operation | Before | After | Speedup |
|-----------|-------:|------:|--------:|
| Resize half, bilinear (640×480) | 0.92 ms | **0.086 ms** | **11×** |
| Resize half, bilinear (1080p)   | 3.76 ms | **0.35 ms**  | **11×** |
| Rotation ±30° (640×480) | 10.9 ms | **1.14 ms** | **9.6×** |
| Rotation ±30° (1080p)   | 58.2 ms | **8.06 ms** | **7.2×** |
| Grayscale (1080p)       | 0.74 ms | **0.44 ms** | **1.7×** |
| Horizontal Flip (1080p) | 1.32 ms | **0.96 ms** | 1.4× |
| Gaussian Blur (640×480) | 14.2 ms → 0.44 ms (initial landing) | 0.76 ms | small regression on latest tree |

## Summary

**Faster than best competitor — 8/10 ops at 640×480, 8/10 at 1080p:**
- ColorJitter, Brightness, HFlip, VFlip, Grayscale, Resize, Rotation, Normalize.
- Normalize is the largest absolute win: **16–17× faster** than OpenCV's `astype(f32) − mean / std` path.

**Still slower than OpenCV:**
- **Gaussian Blur 5×5** (3.5× slower). Regressed vs previous best landing; worth a look.
- **Crop at 1080p** (2.5× slower). Memory-bound; OpenCV's memcpy path is already well-tuned and the kornia NEON crop helps at larger crop ratios.

**New this cycle:**
- **Resize flipped from bottleneck to category leader.** Own u8 bilinear/bicubic/lanczos separable NEON kernel; `antialias=False` fast path beats OpenCV on every downscale shape (up to 2.9×). `antialias=True` matches PIL quality.
- **Rotation flipped from bottleneck to category leader.** Previously 6.4× slower than albumentations; now 1.1–1.4× *faster* than both OpenCV and albumentations.

**Upscale note:** OpenCV wins bilinear upscale (2160p target) by 2.1×. Bicubic/lanczos upscale are in kornia's favor. An integer-ratio or gather-free upscale fast path could close the bilinear gap.

## Techniques used

| Technique | Used in | Benefit |
|-----------|---------|---------|
| Q14 separable u8 resize (H then V, i16 intermediate) | Bicubic, Lanczos | Halves bandwidth vs i32; single H + single V pass |
| Compact coefficient LUTs (`u16 xsrc`, `i16 xw`) | Resize (both axes) | Smaller L1 footprint per dst pixel |
| 4-row NEON horizontal kernel | Resize bicubic/lanczos C=3 | Coefficient loads shared across 4 src rows |
| 4-rolling-accumulator vertical pass | Resize bicubic/lanczos | Hides 4c `vmlal` latency (Cortex-A78AE) |
| `antialias=False` fixed-tap path | Resize bicubic/lanczos | 4×–5× fewer MACs at strong downscale (1080p→224²) |
| 2× box-average fast path | Resize bilinear (exact 2:1) | Dedicated NEON pyrdown; replaces general bilinear |
| Nearest-neighbor LUT dispatch | Resize nearest | Branch-free row/col index map |
| Strip-mined separable filter | Gaussian Blur | u16 temp stays in L1 (~48KB strips) |
| NEON `vmull_u8`/`vmlal_u8` | Gaussian Blur (horizontal) | Integer convolution, 16 bytes/iter |
| NEON `vmull_u16`/`vmlal_u16` | Gaussian Blur (vertical) | 16 u16→u8 values/iter |
| Pre-padded border rows | Gaussian Blur | No per-tap boundary checks |
| Rayon parallel strips | Blur, Resize, Normalize | All heavy passes parallelized |
| NEON `vld3q_u8` + `vmlal_u8` | Grayscale | Deinterleave RGB + widening MAC |
| NEON `vld3_u8` + `vfmaq_f32` | Normalize | Fused u8→f32 cast + channel-wise scale+offset |
| 256-byte LUT | ColorJitter (brightness+contrast) | L1-resident, 1 lookup vs float math |
| Saturating add (`uqadd`/`uqsub`) | Brightness | 16 bytes/instruction, no clamp needed |
| Rodrigues rotation matrix | ColorJitter (hue) | Branchless 9 muls vs HSV round-trip |
| NEON 32-byte prefetched row copy | Crop | L1 streaming prefetch, 2× `vld1q_u8` per iter |
| Integer-coeff affine warp | Rotation | Replaces f32 bilinear sampling loop |
| Zero-copy `ForeignAllocator` | All ops | Wrap numpy pointer, no memcpy |
| Direct PyArray output | All ops | Write into numpy buffer, skip intermediate Vec |

## NEON kernel locations (upstream in `kornia-imgproc`)

| Kernel | File | Function |
|--------|------|----------|
| Grayscale | `color/gray.rs` | `rgb_to_gray_u8` |
| Normalize | `normalize.rs` | `normalize_rgb_u8` |
| Gaussian Blur | `filter/ops.rs` | `gaussian_blur_u8` (strip-mined separable) |
| Resize (bilinear, N-ch) | `resize.rs` | `resize_bilinear_u8_nch` |
| Resize (bicubic / lanczos) | `resize.rs` | `resize_separable_u8`, `horizontal_row_c3_x4_neon`, `vertical_single_row` |
| Resize (2× pyrdown) | `resize.rs` | `pyrdown_2x_rgb_u8` |
| Crop (row copy, aarch64) | `crop.rs` | `copy_row_neon` |
| Rotation / warp affine | `warp/affine.rs` | `warp_affine_u8` |
