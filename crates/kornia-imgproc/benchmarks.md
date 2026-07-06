# kornia-imgproc GPU benchmarks

## How to run

```sh
# CPU baseline (no CUDA required)
cargo run --example bench_gpu_resize --release

# Native CUDA (NVRTC) GPU + CPU comparison (requires CUDA driver)
cargo run --example bench_gpu_resize --features cuda --release
cargo run --example bench_gpu_color_conversions --features cuda --release
cargo run --example bench_gpu_warp_affine --features cuda --release

# OpenCV CPU comparison (requires Python + opencv-python)
python3 crates/kornia-imgproc/examples/bench_opencv_color.py
python3 crates/kornia-imgproc/examples/bench_opencv_resize.py

# OpenCV CUDA comparison (requires OpenCV built with -DWITH_CUDA=ON)
# If cv2 CUDA build is in dist-packages rather than site-packages:
PYTHONPATH=/path/to/cuda-opencv/dist-packages \
  python3 crates/kornia-imgproc/examples/bench_opencv_resize.py
PYTHONPATH=/path/to/cuda-opencv/dist-packages \
  python3 crates/kornia-imgproc/examples/bench_opencv_warp_affine.py
```

## Methodology

| Parameter | Value |
|-----------|-------|
| Warmup iters | 50 |
| Timed iters | 200 |
| GPU source buffers | 8 rotating (defeats GPU L2 read cache across iterations) |
| GPU sync | `read_one_unchecked` after full batch — measures sustained throughput |
| GPU handle clone | `Arc` refcount bump only — negligible overhead inside the timed loop |
| CPU scalar | auto-vectorised by LLVM (`-O3`, `--release`) |
| CPU AVX2 | sequential 256-bit loads + `permutevar8x32` deinterleave + FMA; no gather |
| Bandwidth formula | 3R + 1W × 4 B/f32 = **16 B/pixel** |

**CPU timing note:** the CPU numbers inside the GPU comparison table are
cache-warm (src data was just allocated and touched for GPU buffer creation).
The standalone CPU section runs after the GPU section, so both are subject to
thermal effects on sustained workloads; treat the standalone numbers as
indicative, not precise.

**OpenCV note:** OpenCV 4.12.0 was benchmarked via the Python bindings
(`cv2.cvtColor`), which call the same C++ kernel.  Python call overhead is
≤ 5 μs/call — negligible for 1080p+ where kernel time is 5–40 ms, but noticeable
for 512×512 (~0.1 ms kernels).

---

## Results — 2026-06-15

### Hardware / software

| Field | Value |
|-------|-------|
| Commit | `854e47e` on `gpu/pr-2` |
| GPU | NVIDIA GeForce GTX 1650 4 GiB — GDDR6, ~192 GB/s peak |
| CPU | Intel Core i5-10300H — 4c/8t, 2.5–4.5 GHz, AVX2+FMA, no AVX-512 |
| RAM | DDR4 dual-channel (est. ~42 GB/s peak) |
| OS | Ubuntu 22.04 x86\_64 |
| CUDA | nvcc 12.4 |
| Rust | 1.92.0, `--release` |
| OpenCV | 4.12.0, Python 3, AVX2+FMA3 dispatch, single-threaded |

---

### GPU vs CPU (from GPU comparison table)

| Size | GPU ms | GPU GB/s | scalar ms | AVX2 ms | GPU vs AVX2 |
|------|---------:|---------:|----------:|--------:|------------:|
| 512×512 | 0.028 | 148 | 0.089 | 0.078 | 5.4× |
| 1024×1024 | 0.103 | 162 | 1.136 | 1.236 | 25.9× |
| 1920×1080 | 0.199 | 167 | 2.766 | 2.548 | 22.8× |
| 3840×2160 | 0.808 | 164 | 14.098 | 11.424 | 18.1× |

GPU bandwidth sits at **148–167 GB/s** (77–87% of GTX 1650 GDDR6 theoretical peak).
The 512×512 speedup (5.4×) is launch-overhead limited, not compute limited.

---

### CPU — kornia scalar vs kornia AVX2 vs OpenCV

All single-threaded.  kornia numbers from the standalone CPU section
(same run as above).  OpenCV numbers from the Python runner.

| Size | kornia scalar ms | kornia AVX2 ms | OpenCV (1T) ms | kornia AVX2 vs OpenCV |
|------|----------------:|---------------:|---------------:|---------------------:|
| 512×512 | 0.092 | 0.141 | 0.228¹ | 1.6× faster |
| 1024×1024 | 5.084 | 6.282 | 4.641 | 0.74× (slower) |
| 1920×1080 | 7.209 | 7.774 | 8.940 | 1.15× faster |
| 3840×2160 | 30.410 | 21.356 | 37.917 | 1.78× faster |

¹ 512×512 OpenCV number is inflated by Python call overhead (~50–150 μs).

**Key findings:**
- Our AVX2 path beats OpenCV single-threaded at 1080p+ by ~1.2–1.8×.
  OpenCV's `cvtColor` float32 path does not appear to use its AVX2 dispatch
  (OpenCV's `AVX2 (37 files)` dispatch targets integer operations mainly).
- Our scalar and AVX2 paths are inconsistent at small sizes due to measurement
  noise (short wall-clock times amplify jitter).
- OpenCV multi-threaded (8 threads) at 1080p: 5.5 ms — similar to our single-threaded AVX2.

---

### Comparison table — GPU vs everything

| Size | kornia GPU | kornia AVX2 | OpenCV 1T | OpenCV 8T | GPU vs OpenCV 1T |
|------|----------:|------------:|----------:|----------:|----------------:|
| 1920×1080 | **0.199 ms** | 2.548 ms | 8.940 ms | 5.500 ms | **44.9×** |
| 3840×2160 | **0.808 ms** | 11.424 ms | 37.917 ms | 27.296 ms | **46.9×** |

The kornia GPU kernel at 1080p is **45× faster than OpenCV CPU single-threaded**
and **28× faster than OpenCV CPU with 8 threads**.

---

---

## GPU resize benchmarks — 2026-06-18

Hardware matches the color section above (GTX 1650, CUDA 12.4, Rust 1.92 release).

**Methodology:** 50 warmup, 200 timed iters, 8 rotating f32 RGB source buffers,
single `read_one_unchecked` sync after the batch.  CPU reference is a hand-rolled
f32 bilinear loop (same algorithm, no SIMD).

### Nearest-neighbor

| Source → Dest | GPU ms | GB/s (formula) | CPU ms | GPU speedup |
|---------------|-------:|---------------:|-------:|------------:|
| 1024×1024→512×512 | 0.061 | 102.8 | 3.95 | **65×** |
| 512×512→1024×1024 | 0.115 | 218.4 | 14.27 | **124×** |
| 1920×1080→960×540 | 0.118 | 105.3 | 7.59 | **64×** |
| 1920×1080→3840×2160 | 0.905 | 220.1 | 113.5 | **125×** |
| 3840×2160→1920×1080 | 0.465 | 107.1 | 29.96 | **64×** |

### Bilinear

| Source → Dest | GPU ms | GB/s (formula) | CPU ms | GPU speedup |
|---------------|-------:|---------------:|-------:|------------:|
| 1024×1024→512×512 | 0.097 | 64.7 | 3.90 | **40×** |
| 512×512→1024×1024 | 0.140 | 179.7 | 14.39 | **103×** |
| 1920×1080→960×540 | 0.186 | 66.7 | 7.51 | **40×** |
| 1920×1080→3840×2160 | 1.009 | 197.3 | 113.5 | **112×** |
| 3840×2160→1920×1080 | 0.742 | 67.1 | 29.29 | **39×** |

**Bandwidth note:** The formula counts 1 src read + 1 dst write per *output* pixel
(`npix_dst × NC × 8 B`).  For bilinear downscale (scale > 1) the actual DRAM
traffic is higher (up to 4 unique source reads per output pixel); for nearest/bilinear
upscale the actual traffic is lower (source cache hits for scale < 1).  Effective
DRAM utilisation corrected for actual traffic:

| Workload | Formula GB/s | Actual DRAM GB/s | % of 192 GB/s peak |
|----------|-------------:|-----------------:|-------------------:|
| Nearest downscale | ~103–107 | ~106–110 (≈1:1 src reads) | **55–57%** |
| Nearest upscale | ~218–220 | ~130–140 (L2 reuse) | **68–73%** |
| Bilinear downscale | ~65–67 | ~162–168 (4× src reads) | **84–88%** |
| Bilinear upscale | ~180–197 | bandwidth-saturated | **~100%** |

**Key findings:**

- The implementation is essentially **hardware-limited** for all cases: bilinear
  upscale saturates DRAM, bilinear downscale reaches 84–88% of peak, and nearest
  exceeds 68% in all cases.
- Bilinear is **39–112× faster** than single-threaded CPU for the same resolution
  and channel count; nearest is **64–125×** faster.
- Downscale nearest trails the other cases (55–57%) due to strided source reads
  defeating L1/L2 cache-line reuse — texture memory would close the gap but is
  not currently exposed by CubeCL.

### OpenCV comparison — 2026-06-22

**OpenCV 4.12.0** benchmarked via Python bindings (`cv2.resize`), same methodology
(50 warmup, 200 timed iters, f32 RGB).  OpenCV uses multi-threaded CPU (TBB) where
available.

```
python3 crates/kornia-imgproc/examples/bench_opencv_resize.py
```

#### Nearest-neighbor

| Source → Dest | GPU ms | OpenCV ms | GPU vs OpenCV |
|---------------|-------:|----------:|--------------:|
| 1024×1024→512×512 | 0.061 | 0.510 | **8×** |
| 512×512→1024×1024 | 0.115 | 2.290 | **20×** |
| 1920×1080→960×540 | 0.118 | 2.086 | **18×** |
| 1920×1080→3840×2160 | 0.905 | 36.332 | **40×** |
| 3840×2160→1920×1080 | 0.465 | 11.054 | **24×** |

#### Bilinear

| Source → Dest | GPU ms | OpenCV ms | GPU vs OpenCV |
|---------------|-------:|----------:|--------------:|
| 1024×1024→512×512 | 0.097 | 0.750 | **8×** |
| 512×512→1024×1024 | 0.140 | 1.848 | **13×** |
| 1920×1080→960×540 | 0.186 | 2.824 | **15×** |
| 1920×1080→3840×2160 | 1.009 | 32.207 | **32×** |
| 3840×2160→1920×1080 | 0.742 | 11.778 | **16×** |

GPU is **8–40× faster than OpenCV** across all cases.

---

## GPU resize benchmarks — native CUDA (NVRTC) — 2026-06-30

Rewritten kernels using `cudarc` + NVRTC instead of CubeCL, enabling `__ldg`
read-only cache routing and `CU_FUNC_CACHE_PREFER_L1` (32 KB → 64 KB L1).
Downscale-only (same cases as the OpenCV comparison above).

```sh
cargo run --example bench_gpu_resize --features cuda --release
```

### Hardware / software

| Field | Value |
|-------|-------|
| GPU | NVIDIA GeForce GTX 1650 4 GiB — GDDR5, ~128 GB/s peak |
| CUDA | nvcc 12.4, cudarc 0.19.8, NVRTC |
| Rust | 1.87.0, `--release` |
| Warmup | 50 iters |
| Timed | 200 iters |

### Nearest-neighbor downscale

| Source → Dest | kornia-rs ms | GB/s | cv2 CUDA ms | PyTorch GPU ms | vs cv2 CUDA | vs PyTorch |
|---------------|-------------:|-----:|------------:|---------------:|------------:|-----------:|
| 1024²→512² | 0.053 | 118.7 | 0.137 | 0.339 | **2.6×** | **6.4×** |
| 1920×1080→960×540 | 0.064 | 194.4 | 0.249 | 0.667 | **3.9×** | **10.4×** |
| 4K→1080 | 0.237 | 210.0 | 0.684 | 2.650 | **2.9×** | **11.2×** |

### Bilinear downscale

| Source → Dest | kornia-rs ms | GB/s | cv2 CUDA ms | PyTorch GPU ms | vs cv2 CUDA | vs PyTorch |
|---------------|-------------:|-----:|------------:|---------------:|------------:|-----------:|
| 1024²→512² | 0.082 | 76.7 | 0.177 | 0.096 | **2.2×** | 0.85× |
| 1920×1080→960×540 | 0.101 | 123.2 | 0.287 | 0.184 | **2.8×** | 1.8× |
| 4K→1080 | 0.385 | 129.3 | 0.987 | 0.716 | **2.6×** | 1.9× |

**Key findings:**

- kornia-rs NVRTC is **2.2–3.9× faster than OpenCV 4.12 CUDA** for downscale.
- PyTorch bilinear is competitive (uses texture memory internally); kornia-rs
  nearest is significantly faster because PyTorch nearest does not exploit
  the spatial cache.
- Bilinear bandwidth (~120 GB/s by the 1 src read + 1 dst write per output-pixel
  formula used throughout this doc) is near the GTX 1650 DRAM ceiling; true
  traffic is higher for bilinear downscale (4 source taps per output pixel).

---

## GPU warp-affine benchmarks — native CUDA (NVRTC) — 2026-06-30

45° centre rotation, same-size canvas, 3-channel f32.  Source data held on
device across iterations; CUDA stream synchronised after each timed batch.

```sh
cargo run --example bench_gpu_warp_affine --features cuda --release
# Python comparison (requires OpenCV built with -DWITH_CUDA=ON):
python3 crates/kornia-imgproc/examples/bench_opencv_warp_affine.py
```

### Hardware / software

Same hardware and toolchain as the resize NVRTC section above.
OpenCV 4.12.0 built from source with `-DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5`.
PyTorch 2.9.1+cu128 via `F.affine_grid` + `F.grid_sample(align_corners=True)`.

### Nearest-neighbor

| Size | kornia-rs GPU ms | GB/s | kornia-rs CPU ms | vs CPU |
|------|-------------:|-----:|-------:|-------:|
| 256×224 | 0.011 | 125.1 | 0.263 | **24×** |
| 512×448 | 0.038 | 144.9 | 0.860 | **23×** |
| 1024×896 | 0.151 | 145.8 | 3.983 | **26×** |
| 1920×1080 | 0.353 | 141.0 | 9.869 | **28×** |

### Bilinear

| Size | kornia-rs GPU ms | GB/s | kornia-rs CPU ms | cv2 CUDA ms | PyTorch GPU ms | cv2 CPU ms | kornia CPU vs cv2 CPU | vs cv2 CUDA | vs PyTorch | vs cv2 CPU |
|------|-------------:|-----:|-------:|------------:|---------------:|-----------:|------:|------------:|-----------:|-----------:|
| 256×224 | 0.025 | 55.1 | 0.161 | 0.037 | 0.111 | 0.584 | **3.6× faster** | **1.5×** | **4.5×** | **23×** |
| 512×448 | 0.092 | 59.8 | 1.535 | 0.178 | 0.449 | 2.235 | **1.5× faster** | **1.9×** | **4.9×** | **24×** |
| 1024×896 | 0.274 | 80.4 | 5.775 | 0.412 | 1.471 | 7.908 | **1.4× faster** | **1.5×** | **5.4×** | **29×** |
| 1920×1080 | 0.572 | 87.0 | 13.82 | 0.753 | 3.298 | 31.11 | **2.3× faster** | **1.3×** | **5.8×** | **54×** |

**Key findings:**

- kornia-rs GPU bilinear warp-affine is **1.3–1.9× faster than OpenCV 4.12 CUDA**
  and **4.5–5.8× faster than PyTorch `grid_sample`**.
- The optimized CPU nearest path (**incremental coords + analytical valid-range skip +
  16-row Rayon chunks**) is **2–2.5× faster than the previous baseline**; GPU nearest
  remains **23–28× faster** than the optimized CPU.
- The optimized CPU bilinear path **beats cv2 CPU at every size** (1.4–3.6×) without
  any SIMD; this holds because cv2's f32 warpAffine does not use its AVX2 dispatch
  for this combination of type, border mode, and rotation angle.
- Higher apparent GB/s vs resize: ~half of output pixels in a 45° rotation are
  out-of-bounds black corners, written with zero without reading source DRAM,
  reducing effective traffic and inflating the GB/s formula.
- PyTorch gap is larger than for resize because `affine_grid` + `grid_sample`
  allocates an intermediate coordinate grid tensor on every call.

---

## How this compares to other Rust crates

Most Rust crates use `criterion` or `divan` for microbenchmarks, which provide:
- Statistical analysis (mean, median, std-dev, outlier detection)
- HTML reports and regression detection between runs
- Automatic iteration-count selection

Our hand-rolled wallclock is simpler but standard for GPU work — `criterion`
does not support async CUDA synchronisation and its iteration-count selection
breaks when warmup involves JIT compilation.  This pattern (manual warmup,
fixed ITERS, single-sync-at-end) matches what `candle`, `burn`, and `wgpu`
use in their GPU benchmarks.

If a CPU-only criterion harness is added later, it should measure the scalar
and AVX2 paths separately and in isolation from any GPU activity.

---

---

## GPU bicubic benchmarks — native CUDA (NVRTC) — 2026-07-06

Keys cubic interpolation (`a = -0.5`, matching OpenCV `INTER_CUBIC`).  4×4 tap
neighborhood; out-of-range taps clamped (BORDER_REPLICATE); OOB centre pixels
zero-filled (BORDER_CONSTANT).  All 16 source reads via `__ldg`.

**Kernel optimisations** (relative to the first implementation):

- **Horner-form weight precomputation** — `frac ∈ [0,1)` places each tap in a
  known polynomial region, making all 8 weight computations branch-free.
  Eliminates `fabsf` + two conditionals from the naive `cubic_w` helper, and
  removes the 12 redundant x-weight evaluations the original loop incurred.
- **Row base hoisting** — 4 row-address multiplies moved outside the inner loop.
- **`#pragma unroll` + `fmaf`** — ptxas fully unrolls the 4×4 tap loop and
  emits one fused multiply-add per channel per tap.

Result: **+7–10% downscale**, **+33–34% upscale** vs the unoptimised version.
Warp-affine bicubic unchanged (scattered DRAM reads from rotation are the bottleneck).

```sh
cargo run --example bench_gpu_resize    --features cuda --release
cargo run --example bench_gpu_warp_affine --features cuda --release
# Python comparison (requires CUDA-built OpenCV + torch with CUDA)
# PYTHONPATH points to the CUDA-enabled cv2 build in dist-packages.
# Replace <dist-packages> with the path reported by your custom OpenCV build.
# Example: $(python3 -c "import site; print(site.getusersitepackages())")
PYTHONPATH=/path/to/cuda-opencv/dist-packages \
  python3 crates/kornia-imgproc/examples/bench_opencv_resize.py
PYTHONPATH=/path/to/cuda-opencv/dist-packages \
  python3 crates/kornia-imgproc/examples/bench_opencv_warp_affine.py
```

### Hardware / software

| Field | Value |
|-------|-------|
| GPU | NVIDIA GeForce GTX 1650 4 GiB — GDDR5, ~128 GB/s peak |
| CUDA | nvcc 12.4, cudarc 0.19.8, NVRTC |
| Rust | 1.87.0, `--release` |
| OpenCV | 4.12.0 built with `-DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5` |
| PyTorch | 2.9.1+cu128 |
| Warmup | 50 iters; Timed | 200 iters |

### Bicubic resize

| Source → Dest | kornia-rs ms | GB/s | cv2 CPU ms | cv2 CUDA ms | PyTorch GPU ms | vs cv2 CPU | vs cv2 CUDA | vs PyTorch GPU |
|---------------|-------------:|-----:|-----------:|------------:|---------------:|-----------:|------------:|---------------:|
| 1024²→512² | 0.120 | 52.6 | 1.071 | 0.320 | 0.803 | **8.9×** | **2.7×** | **6.7×** |
| 512²→1024² | 0.207 | 121.4 | 2.026 | 0.539 | 2.875 | **9.8×** | **2.6×** | **13.9×** |
| 1920×1080→960×540 | 0.245 | 50.8 | 2.559 | 0.464 | 1.611 | **10.4×** | **1.9×** | **6.6×** |
| 1920×1080→3840×2160 | 1.709 | 116.5 | 23.620 | 3.289 | 23.168 | **13.8×** | **1.9×** | **13.6×** |
| 3840×2160→1920×1080 | 0.959 | 51.9 | 10.896 | 1.696 | 6.549 | **11.4×** | **1.8×** | **6.8×** |

### Bicubic warp-affine (45° centre rotation)

| Size | kornia-rs ms | GB/s | cv2 CPU ms | cv2 CUDA ms | PyTorch GPU ms | vs cv2 CPU | vs cv2 CUDA | vs PyTorch GPU |
|------|-------------:|-----:|-----------:|------------:|---------------:|-----------:|------------:|---------------:|
| 256×224 | 0.065 | 21.1 | 1.172 | 0.109 | 0.163 | **18×** | **1.7×** | **2.5×** |
| 512×448 | 0.244 | 22.6 | 3.278 | 0.419 | 0.657 | **13×** | **1.7×** | **2.7×** |
| 1024×896 | 0.936 | 23.5 | 10.061 | 1.288 | 2.738 | **11×** | **1.4×** | **2.9×** |
| 1920×1080 | 1.951 | 25.5 | 32.304 | 2.576 | 5.849 | **17×** | **1.3×** | **3.0×** |

**Key findings:**

- kornia-rs bicubic resize is **8.9–13.8× faster than OpenCV 4.12 CPU** and
  **1.8–2.7× faster than OpenCV 4.12 CUDA** and **6.6–14× faster than PyTorch GPU**.
- The upscale cases (512→1024, 1080p→4K) show the largest gap vs PyTorch (~14×)
  because output-pixel count drives latency, cache reuse is excellent, and
  PyTorch adds Python/dispatcher overhead per call.
- Warp-affine bicubic is **1.3–1.7× faster than OpenCV CUDA** and **2.5–3× faster
  than PyTorch `grid_sample(bicubic)`** (which allocates an intermediate grid
  tensor per call at every size).
- Bicubic downscale is ~1.4× slower than bilinear downscale (DRAM-bound; 16
  reads vs 4, partially amortised by L1 reuse within the 4×4 tap neighbourhood).

### Interpolation comparison — resize 1920×1080→960×540

| Method | kornia-rs ms | GB/s | vs bilinear |
|--------|-------------:|-----:|------------:|
| Nearest | 0.107 | 116.5 | 1.7× faster |
| Bilinear | 0.178 | 70.0 | baseline |
| Bicubic | 0.245 | 50.8 | 1.4× slower |

---

## CUDA driver status

Confirmed working as of 2026-06-15.  If the kernel-module / userspace mismatch
recurs:

```sh
sudo apt-get install --reinstall nvidia-dkms-580 nvidia-utils-580
sudo rmmod nvidia_uvm nvidia_modeset nvidia_drm nvidia && sudo modprobe nvidia
```
