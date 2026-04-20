# kornia-py Benchmarks

## Run Info

| Field | Value |
|-------|-------|
| Date | 2026-04-20 |
| Commit | f1830ab (warp: NEON 4-wide reciprocal for perspective; extract per-arch kernels module) |
| pyo3 | 0.28 |
| numpy (crate) | 0.28 |
| Platform | Jetson Orin (aarch64), Linux 5.15.148-tegra |
| Rust | 1.93.0, opt-level=2 (LLVM bug workaround for faer) |
| Python | 3.10.12 |
| numpy | 2.2.6 |
| OpenCV | 4.13.0 |
| albumentations | 2.0.8 |
| Iterations | 200 (10 warmup), `taskset -c 0-5` |

## Results — 640×480

| Operation | kornia-rs (ms) | albumentations (ms) | OpenCV (ms) | vs OpenCV |
|-----------|---------------:|--------------------:|------------:|----------:|
| ColorJitter (b+c+s+h) | **3.085** | 5.94 | 8.70 | **2.82× faster** ✓ |
| Brightness | **0.079** | 0.46 | 0.77 | **9.80× faster** ✓ |
| Horizontal Flip | **0.072** | 0.32 | 0.31 | **4.22× faster** ✓ |
| Vertical Flip | **0.078** | 0.11 | 0.098 | **1.25× faster** ✗ (<2×) |
| Crop 224×224 | **0.019** | 0.035 | 0.023 | **1.24× faster** ✗ (<2×) |
| Grayscale | **0.083** | — | 0.102 | **1.23× faster** ✗ (<2×) |
| Resize (half, bilinear) | **0.058** | — | 0.314 | **5.43× faster** ✓ |
| Gaussian Blur 5×5 | **0.146** | — | 0.368 | **2.53× faster** ✓ |
| Rotation ±30° | **0.919** | 1.67 | 1.234 | **1.34× faster** ✗ (<2×; noisy) |
| Warp Perspective | **0.901** | — | 1.533 | **1.70× faster** ✗ (<2×) |
| Normalize | **0.553** | — | 9.294 | **16.8× faster** ✓ |

## Results — 1920×1080

| Operation | kornia-rs (ms) | albumentations (ms) | OpenCV (ms) | vs OpenCV |
|-----------|---------------:|--------------------:|------------:|----------:|
| ColorJitter (b+c+s+h) | **20.81** | 40.55 | 57.88 | **2.78× faster** ✓ |
| Brightness | **0.821** | 3.25 | 5.22 | **6.36× faster** ✓ |
| Horizontal Flip | **0.626** | 2.40 | 2.285 | **3.65× faster** ✓ |
| Vertical Flip | **0.945** | 1.10 | 1.049 | **1.11× faster** ✗ (<2×) |
| Crop 224×224 | **0.050** | 0.074 | 0.063 | **1.26× faster** ✗ (<2×, flipped) |
| Grayscale | **0.365** | — | 0.401 | **1.10× faster** ✗ (<2×) |
| Resize (half, bilinear) | **0.329** | — | 0.597 | **1.81× faster** ✗ (<2×; flips around 2× run-to-run) |
| Gaussian Blur 5×5 | **0.734** | — | 0.954 | **1.30× faster** ✗ (<2×) |
| Rotation ±30° | **5.520** | 8.29 | 7.462 | **1.35× faster** ✗ (<2×) |
| Warp Perspective | **4.821** | — | 9.627 | **2.00× faster** ✓ |
| Normalize | **3.698** | — | 67.32 | **18.2× faster** ✓ |

## Target: every op ≥2× faster than OpenCV

| Op | 640×480 | 1080p | Status |
|---|---|---|---|
| ColorJitter | 2.82× ✓ | 2.78× ✓ | ✓ (Phase 4: NEON saturation + src→dst orchestration) |
| Brightness | 9.80× ✓ | 6.36× ✓ | ✓ |
| Horizontal Flip | 4.22× ✓ | 3.65× ✓ | ✓ (NEON `vld3q`/`vrev64q` pair-reverse, 4b82ef3) |
| Vertical Flip | 1.25× ✗ | 1.11× ✗ | memcpy-bound (single core saturates LPDDR ~15 GB/s); 1080p noisy run-to-run |
| Crop 224×224 | 1.24× ✗ | 1.26× ✗ | flipped from slower → faster after NEON threshold bump (≤672B rows now take plain memcpy) |
| Grayscale | 1.23× ✗ | 1.10× ✗ | 1080p at BW floor (8 MB traffic / 0.4 ms = 20 GB/s) |
| Resize (½) | 5.43× ✓ | 1.81× ✗ | pyrdown_2x + rayon 8-row groups; 1080p flips around 2× line from DRAM jitter |
| Gaussian Blur 5×5 | 2.53× ✓ | 1.30× ✗ | binomial 5×5 NEON fast path (f0d2047); 640 cleared 2×, 1080p wants fused H+V strip |
| Rotation ±30° | 1.34× ✗ | 1.35× ✗ | gather-bound — NEON 4-wide regresses vs scalar OoO |
| Warp Perspective | 1.70× ✗ | 2.00× ✓ | ✓ at 1080p (NEON 4-wide `vrecpeq_f32` + NR for per-pixel divide, f1830ab) |
| Normalize | 16.8× ✓ | 18.2× ✓ | ✓ (NEON u8→f32 fused scale+offset, bb711e4) |

**All 11 ops now faster than OpenCV.** Five clear the 2× bar at both sizes (ColorJitter, Brightness, HFlip, Resize at 640, Normalize); two more clear it at one size (Blur at 640, Perspective at 1080p); the rest win by <2×. Remaining gaps organized by why they stall:
- **Bandwidth-bound** (irreducible without better cache blocking): vflip, grayscale, crop 1080p. A78AE single-core hits ~15 GB/s which is the LPDDR5 ceiling; rayon adds spawn cost without headroom. Crop at 1080p used to be 2.33× slower; after lifting the NEON-dispatch threshold to 4KB so small (≤672B) rows take LLVM's tuned `copy_from_slice` memcpy (avoiding a `prfm pldl1strm, [src, #2048]` hint that was landing 1.4KB past the 672-byte crop row), it now wins by 1.26×.
- **Gather-bound** (NEON provably slower than scalar): rotation — bilinear samples 4 scattered corners per output pixel, and scalar OoO hides the latency better than NEON's `vsetq_lane` lane-inserts. A 4-wide NEON version was tried (2026-04-18) and regressed 70%.
- **Close to 2× but not over**: Gaussian Blur 1080p. Binomial 5×5 NEON fast path (f0d2047) lifted 1080p from 0.29× to 1.30× and 640 from 0.38× to 2.53× — last step at 1080p likely needs fused horizontal+vertical strip to drop one pass over the 6 MB image.
- **Noise-dominated around the 2× line**: Resize 1080p (1.81–2.03× run-to-run), Blur 1080p (1.30–1.37×). These are bandwidth-sensitive enough that DRAM jitter moves the ratio across runs.

## Resize matrix — all modes, multiple shapes

Resize results across interpolation modes, `antialias` flag, and source→dest shape combinations. `antialias=True` matches PIL / torchvision semantics (kernel widens by scale; no aliasing on strong downscale). `antialias=False` matches OpenCV INTER_CUBIC / INTER_LANCZOS4 semantics (fixed 4/8-tap). `Bilinear` and `Nearest` are unaffected by the flag.

**Note on Pillow-SIMD:** Pillow-SIMD (`pillow-simd` 9.5.0) gates all accelerated code paths on `__SSE4_2__` / `__AVX2__` in `src/libImaging/ImagingSIMD.h` — it is x86-only and ships no NEON path. On this Jetson Orin (aarch64) it would either fail to build or fall back to vanilla Pillow's scalar path, so it is not included in the matrix below. The vanilla PIL column is the only "Pillow" result available on ARM; on x86 hardware, Pillow-SIMD is typically 3–5× faster than vanilla PIL on resize, so the PIL column here overstates kornia's lead vs the best Python imaging baseline an x86 user would see.

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
| 1080p → 2160p (up) | bilinear | 1.78 | 1.78 | 9.80 | 181 | **5.52× faster** (exact-2× NEON `vrhaddq_u8` pair) |
| 1080p → 2160p (up) | bicubic  | 17.9 | 17.9 | 21.6 | 229 | **1.21× faster** |
| 1080p → 2160p (up) | lanczos  | 40.6 | 40.6 | 43.3 | 283 | **1.07× faster** |

## Improvement log (2026-04-02 → 2026-04-20)

| Operation | Before | After | Speedup |
|-----------|-------:|------:|--------:|
| Resize half, bilinear (640×480) | 0.92 ms | **0.071 ms** | **13×** |
| Resize half, bilinear (1080p)   | 3.76 ms | **0.330 ms** | **11.4×** |
| Rotation ±30° (640×480) | 10.9 ms | **0.743 ms** | **14.7×** |
| Rotation ±30° (1080p)   | 58.2 ms | **6.17 ms** | **9.4×** |
| Grayscale (1080p)       | 0.74 ms | **0.418 ms** | **1.77×** |
| Horizontal Flip (1080p) | 1.06 ms | **0.520 ms** | **2.04×** (NEON `vld3q`/`vrev64q`, f0d2047) |
| Gaussian Blur (640×480) | 0.60 ms | **0.099 ms** | **6.06×** (binomial 5×5 fast path, f0d2047) |
| Gaussian Blur (1080p)   | 3.36 ms | **0.708 ms** | **4.75×** (same kernel; flipped 3.46× slower → 1.37× faster vs OpenCV) |
| Normalize (1080p)       | 4.28 ms | **3.705 ms** | 1.15× (NEON u8→f32 fused scale+offset, bb711e4) |
| ColorJitter (1080p)     | 30.8 ms | **20.81 ms** | **1.48×** (src→dst orchestration, afa1f08) |
| Crop 224² (1080p)       | 0.063 ms | **0.050 ms** | **1.26×** (raise NEON threshold to 4KB; small rows now use LLVM memcpy, 2026-04-20) |
| Crop 224² (640×480)     | 0.024 ms | **0.019 ms** | **1.26×** (same change; flipped from 1.04× slower to 1.24× faster vs OpenCV) |
| Warp Perspective (1080p) | 6.800 ms | **4.821 ms** | **1.41×** (NEON 4-wide reciprocal for per-pixel `nx/nd` + `ny/nd`, f1830ab) — ratio vs OpenCV: 1.41× → **2.00×** |
| Resize 1080p→2160p (bilinear upscale) | 20.7 ms | **1.78 ms** | **11.6×** (exact-2× NEON `vrhaddq_u8` pair — fixed {0.25, 0.75} weights, no LUT, no float) — ratio vs OpenCV: **0.46× → 5.52×** |

## Summary

**Faster than OpenCV — 11/11 ops at 640×480, 11/11 at 1080p.** Clean sweep. Breakdown by win margin:
- **≥2× at both sizes (5 ops)**: ColorJitter, Brightness, HFlip, Resize (640), Normalize.
- **≥2× at one size (2 ops)**: Blur (640 only), Perspective (1080p only), Resize (1080p noisy at line).
- **<2× but still winning (5 ops)**: VFlip, Crop, Grayscale, Rotation, Blur 1080p.

Normalize is the largest absolute win: **17–18× faster** than OpenCV's `astype(f32) − mean / std` path.

**New this cycle (bb711e4 → crop threshold fix → perspective NEON → pyrup 2× 2026-04-20):**
- **Bilinear 2× upscale flipped from 2.14× loss to 5.52× win at 1080p.** The exact-2× case has fixed `{0.25, 0.75}` weights, which is exactly what `vrhaddq_u8(a, vrhaddq_u8(a, b))` computes (first rhadd is the 50/50 midpoint, second biases it back toward `a`). No LUT, no fractional arithmetic, no gather. The new `pyrup_2x_rgb_u8` kernel feeds each src row through a horizontal `vld3q_u8`/`vst3q_u8` upscale once; each upscaled row is then reused by two dst rows via a byte-wise vertical `blend_75_25_row`. At 1080p→2160p: kornia 20.7ms → **1.78ms** (internal 11.6×), vs OpenCV 9.80ms (ratio **0.46× → 5.52×**). This was the only bilinear case OpenCV was winning — clean sweep for bilinear now.
- **Perspective crossed the 2× line at 1080p.** Commit f1830ab adds a NEON 4-wide reciprocal (`vrecpeq_f32` + one `vrecpsq_f32` Newton-Raphson refine) for the per-pixel `xf = nx/nd`, `yf = ny/nd` divides, combined with an analytical branch-free valid-range split (4 linear constraints intersected against `[0, src_w) × [0, src_h)`). OpenCV pays a serial scalar divide per pixel; amortizing across 4 lanes collapses the critical path. Side benefit: `vrecpeq` + 1 NR yields ~17-bit precision that happens to match OpenCV's Q-math closer than exact f32, so correctness vs the scalar reference improved (large-diff pixels: 20.13% → 4.85%). The same commit also extracted `warp/kernels.rs` with `_scalar` reference impls + `_neon` variants behind `cfg(target_arch = "aarch64")`, giving a stable seam for future AVX / WASM-SIMD / SVE backends.
- **Crop flipped from worst loss to a win.** Raised the NEON-dispatch threshold from 128 to 4096 bytes (`crop.rs:69`). The hand-rolled path issued a `prfm pldl1strm, [src, #2048]` prefetch tuned for long contiguous reads, but 224-col crops have only 672-byte rows — the prefetch was landing on irrelevant source pixels outside the crop window, polluting L1. Small rows now take `copy_from_slice` which LLVM lowers to a vectorized memcpy without the misfiring prefetch. Result: 1080p 0.063→0.050ms (2.33× slower → 1.26× faster vs OpenCV), 640 0.024→0.019ms (1.04× slower → 1.24× faster).

**Prior cycle (afa1f08 → bb711e4):**
- **Gaussian Blur flipped from bottleneck to winner.** Binomial 5×5 NEON fast path (f0d2047) brings 1080p from 3.36ms to 0.71ms — 4.75× improvement, and flips the OpenCV ratio from 3.46× slower to 1.30× faster.
- **Horizontal Flip doubled.** NEON `vld3q`/`vrev64q` pair-reverse (f0d2047).
- **Normalize fused u8→f32.** `vfmaq_f32(bias, vcvtq_f32_u32(...), scale)` in a single pass (bb711e4).
- **Resize 1080p→540p around the 2× line.** pyrdown_2x now groups 8 output rows per rayon task (c68000b).

**Upscale note:** With the exact-2× bilinear upscale fast path (`pyrup_2x_rgb_u8`), kornia now wins bilinear upscale at 1080p→2160p by **5.5×** (1.78ms vs 9.80ms) — previously a 2.14× loss. The path uses a `vrhaddq_u8(a, vrhaddq_u8(a, b))` pair for the fixed {0.25, 0.75} weights (no LUT, no gather, no float math). Bicubic/lanczos upscales still win but by small margins; a similar integer-ratio path would help there too.

## Techniques used

| Technique | Used in | Benefit |
|-----------|---------|---------|
| Q14 separable u8 resize (H then V, i16 intermediate) | Bicubic, Lanczos | Halves bandwidth vs i32; single H + single V pass |
| Compact coefficient LUTs (`u16 xsrc`, `i16 xw`) | Resize (both axes) | Smaller L1 footprint per dst pixel |
| 4-row NEON horizontal kernel | Resize bicubic/lanczos C=3 | Coefficient loads shared across 4 src rows |
| 4-rolling-accumulator vertical pass | Resize bicubic/lanczos | Hides 4c `vmlal` latency (Cortex-A78AE) |
| `antialias=False` fixed-tap path | Resize bicubic/lanczos | 4×–5× fewer MACs at strong downscale (1080p→224²) |
| 2× box-average fast path | Resize bilinear (exact 2:1) | Dedicated NEON pyrdown; replaces general bilinear |
| 2× bilinear upscale fast path (`vrhaddq_u8` pair) | Resize bilinear (exact 1:2) | `vrhaddq_u8(a, vrhaddq_u8(a, b))` gives 0.75·a + 0.25·b with rounding — no LUT, no float, no gather. One horizontal-interp row feeds two dst rows via byte-wise vertical blend. |
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
| Dual-path row copy (NEON+prfm ≥4KB rows, LLVM memcpy otherwise) | Crop | Prefetch helps wide contiguous reads; mis-tunes for sub-2KB rows (pollutes L1) so small crops route to plain memcpy |
| NEON `vld3q_u8` + `vrev64q_u8` pair-reverse | Horizontal Flip | 16 RGB pixels reversed per iter via `vrev64q` + `vextq_u8(hi,lo,8)` |
| Binomial 5×5 fast path (power-of-2 divisor) | Gaussian Blur | `[1,4,6,4,1]` → `>>8` replaces float divide; kernel-agnostic path kept as fallback |
| Integer-coeff affine warp | Rotation | Replaces f32 bilinear sampling loop |
| NEON 4-wide reciprocal (`vrecpeq_f32` + 1 NR) | Warp Perspective | Amortizes per-pixel `nx/nd` divide across 4 lanes; ~17-bit precision |
| Analytical branch-free valid-range | Warp Perspective | 4 linear-constraint intersections give safe `[x_lo, x_hi)` per row; no per-pixel bounds check in hot path |
| Per-arch kernel dispatch (`warp/kernels.rs`) | Affine + Perspective | `_scalar` reference + `_neon` behind `cfg`; stable seam for future AVX / WASM-SIMD / SVE |
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
| Resize (2× pyrup, bilinear) | `resize.rs` | `pyrup_2x_rgb_u8`, `hinterp_row_rgb_u8_neon`, `blend_75_25_row_neon` |
| Crop (row copy, aarch64) | `crop.rs` | `copy_row_neon` |
| Horizontal Flip (RGB u8) | `flip.rs` | `hflip_rgb_u8_neon` |
| Rotation / warp affine | `warp/affine.rs` | `warp_affine_u8` (dispatches to `kernels::process_affine_span`) |
| Warp perspective | `warp/perspective.rs` + `warp/kernels.rs` | `warp_perspective_u8` + `process_perspective_span_neon` (NEON 4-wide reciprocal) |
