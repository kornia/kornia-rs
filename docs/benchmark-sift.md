# SIFT benchmark

Covers both backends of `kornia_imgproc`'s SIFT: the CUDA pipeline and the NEON
CPU one. Numbers are medians unless stated; the machine is shared, so re-measure
before trusting a difference under ~5%.

## System

| | |
|---|---|
| Date (UTC) | 2026-07-26 18:40 |
| Host | nvidia-orin00 |
| Machine | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| Kernel / arch | 5.15.148-tegra aarch64 |
| CPU | Cortex-A78AE x6 |
| L4T | # R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025 |
| GPU | nvidia,ga10b |
| CUDA | 12.6.68 |
| Power mode | MAXN_SUPER |
| rustc | rustc 1.93.0 (254b59607 2026-01-19) |
| OpenCV (py) | 4.13.0 |
| OpenCV (C++) | 5.0.0 (`/mnt/data/ocv5build/install`) |
| Git commit | 1623345e60 |

## End to end, mh01_frame1 (752x480)

Detection plus descriptors, median of 9, each engine in its **own process** with
`RAYON_NUM_THREADS` and `cv2.setNumThreads` set to the same value. OpenCV 4.13
and 5.0 are within 1% of each other here, so the baseline column is either.

Two baselines, because OpenCV parallelises too and quoting only the first is
misleading:

| backend | config | ms | vs cv2 1-thread | vs cv2 6-thread |
|---|---|---|---|---|
| CUDA | `fo=-1` (OpenCV's default) | **19.7** | 11.4x | **5.3x** |
| CUDA | `fo=-1`, fast descriptor | 13.9 | 16.2x | 7.5x |
| CUDA | `fo=0`, 4 octaves | **7.7** | 29x | 13.5x |
| CPU (NEON) | `fo=-1` | 95 | 2.4x | **1.14x** |
| CPU (NEON) | `fo=0`, 4 octaves | 40 | 5.6x | 2.6x |
| OpenCV 5.0.0 | default, 1 thread | 224.7 | 1.0x | — |
| OpenCV 5.0.0 | default, 6 threads | 104 | 2.15x | 1.0x |

Only the `fo=-1` rows are like-for-like: `fo=0, 4 octaves` skips the 2x upsample
and two octaves, producing 933 keypoints instead of 2515. It is a different
amount of work, not a faster way to do the same work, and its ratio should not
be read as a speedup.

At `fo=-1` the CUDA path's output is **identical to cv2 on every column** of the
matching audit — keypoint count, homography matches, fundamental-matrix inliers,
median epipolar error.

### Threading

| threads | cv2 | kornia NEON | ratio |
|---|---|---|---|
| 1 | 224.5 | 296.0 | 0.76x |
| 2 | 146.6 | 156.9 | 0.93x |
| 4 | 111.8 | 92.7 | 1.21x |
| 6 | ~108 | 95.0 | **~1.15x** |

Per core the NEON path is still 1.3x slower than OpenCV; on all six cores it is
ahead. Earlier revisions of this document claimed 2.3x-7x on CPU; those compared
our six-thread figure against `setNumThreads(1)` and were wrong. The CUDA rows
are unaffected in kind — a GPU is being compared against a CPU either way — but
their ratios are restated above against the faster baseline.

The single-thread column is the kernel-quality diagnostic and the one to
optimise against: it is where the remaining deficit lives, and it is not a
scheduler artefact. It has come down from 389.6 ms (0.58x) — see below.

### Single-thread stage split

Measured with `KORNIA_SIFT_STAGES=1` at `RAYON_NUM_THREADS=1` on apriltags
(752x480, ~1800 kp), against cv2's own split from `detect` / `compute` /
`detectAndCompute` timings:

| stage | cv2 | was | now |
|---|---|---|---|
| pyramid / blur | 61.1 | 73.7 | 74.4 |
| gradients | inline | 31.1 | 30.9 |
| extrema + orient | 38.7 | 75.8 | 36.3 |
| descriptors | 78.6 | 112.3 | 84.3 |
| **total** | **178.4** | **311.0** | **241** |

Three changes got there, each held to the bitwise oracle:

* **Extrema scan** — 34.1% of pixels clear the contrast threshold and enter the
  26-neighbour test, but only 0.09% are extrema. A NEON prefilter, staged by
  plane so the previous and next planes' 18 loads are usually never issued,
  cut it 58.4 -> 19.5 ms. `val >= max(neighbours)` is implied by the strict
  test, so it cannot drop a candidate; the exact scalar test still decides.
* **Descriptor trilinear weights** vectorised, 98.0 -> 84.3 ms. Every op up to
  the scatter is elementwise and so bit-identical lane-wise; only the histogram
  accumulation is order-sensitive and stays scalar. This is the split
  `calcSIFTDescriptor` itself uses.
* **Descriptor sample loop** narrowed, 112.3 -> 98.0 ms. `rbin` and `cbin` are
  affine in `j`, so each row's accepted samples form one contiguous run —
  5.50M iterations were being run for 2.63M accepted samples.

The remaining gap is `gradients` (31 ms), which cv2 does not have as a separate
pass: it computes gradients inline per keypoint patch, over ~3.6M samples
against our 5.63M whole-layer pixels. Closing that means moving the magnitude
and angle evaluation into the descriptor and orientation loops.

### Falsified, do not retry

* **Interleaving the magnitude and angle planes** into one `(mag, ang)` plane:
  +5% (descriptor 98.0 -> 101.6, gradients 30.9 -> 32.2). The pair is still two
  load instructions — adjacent addresses do not merge — while `vst2q_f32` adds
  an interleaving shuffle two plain stores avoid.
* **X-tiling the vertical blur** so its `ksize`-row window fits L1 (162 KB at
  the base octave, so it does not): 74.4 -> 79.0 ms at a 128-column strip, even
  with the reflected row bases hoisted out of the strip loop. The full-width
  walk feeds the hardware prefetcher long sequential streams, which is worth
  more than L1 residency.

**Measure each engine in a separate process.** Running both in one process
inflates the NEON figure to ~118-127 ms, because OpenCV's worker pool stays
alive and competes for cores.

## Matching quality (homography + epipolar)

`python3 kornia-py/benchmarks/bench_sift_quality.py`, 2026-07-26. Matching is
cv2's `BFMatcher` for every engine, so these columns compare **descriptors**,
not matchers.

| engine | kp | ms | H match | H ok | F match | F inl | inl% | sed |
|---|---|---|---|---|---|---|---|---|
| opencv (1 thread) | 2515 | 225.2 | 5293 | 5232 | 816 | 533 | 65.3% | 0.29 |
| opencv (all cores) | 2515 | 103.8 | 5293 | 5232 | 816 | 533 | 65.3% | 0.29 |
| cuda `fo=-1` | 2515 | 19.6 | 5293 | 5232 | 816 | 533 | 65.3% | 0.29 |
| cuda `fo=-1` fast | 2515 | 13.9 | 5313 | 5252 | 819 | 496 | 60.6% | 0.26 |
| cuda `fo=0` 4oct | 933 | 7.7 | 1911 | 1884 | 346 | 207 | 59.8% | 0.28 |
| neon `fo=-1` | 2515 | 117.8 | 5293 | 5232 | 816 | 533 | 65.3% | 0.29 |
| neon `fo=0` 4oct | 933 | 39.6 | 1911 | 1884 | 346 | 207 | 59.8% | 0.28 |

The `ms` column here is **in-process** and both engines' pools are live at once,
so the CPU rows run slower than the dedicated-process table above. Use this
table for the quality columns and that one for speed.

Both `fo=-1` backends match OpenCV on **every** column. The fast descriptor
trades a little epipolar inlier ratio for a slightly better homography count
and median error — it samples coarse octaves more evenly than the reference's
pixel walk does.

### Why two geometric tests

Keypoint counts and wall time say nothing about descriptor quality, and a
detector can be made arbitrarily fast by being wrong.

* **Homography** warps a planar image by a known `H`, so every correct match is
  consistent with it by construction. Measures invariance to rotation and
  scale; says nothing about 3D parallax.
* **Epipolar** uses a real stereo-motion pair where no homography exists, scored
  by symmetric epipolar distance under a RANSAC fundamental matrix — the
  geometry SfM and VO actually rely on.

A backend can look fine on one and fail the other. An early version of this port
scored a healthy median epipolar error while returning a fifth of OpenCV's
homography matches: descriptors for every octave but the first were zero, and a
zero descriptor is equidistant from everything, so the ratio test *rejected*
rather than mismatched. Reporting both is what made that visible.

The script also checks the matcher implementations against cv2's `BFMatcher` on
identical descriptors — CUDA, NEON and OpenCV return the same 816-pair set.

## CUDA kernel breakdown (nsys, per frame)

| kernel | exact | fast descriptor |
|---|---|---|
| descriptor | 7.71 ms | 2.64 ms |
| orientation | 2.39 | 2.39 |
| blur_v_dog | 2.06 | 2.06 |
| blur_h_tiled | 1.97 | 1.97 |
| find_extrema | 1.37 | 1.37 |
| upsample + blur_v | 0.40 | 0.40 |
| **GPU total** | **15.9** | **10.8** |

## Matching

Brute force, 128-D, Lowe ratio 0.8 with mutual nearest neighbour. All three
implementations return the **identical 816-pair set** on mh01 frame 1 against
frame 2.

| | ms |
|---|---|
| CUDA (descriptors stay on device) | 51.8 |
| CPU NEON | 320 |
| CPU scalar fallback | 1240 |

The criterion group measures the two CPU kernels in isolation on 2515x2515
descriptors: NEON 55.1 ms, scalar 212.8 ms — **3.9x**.

## How to run

### Matching quality

```bash
python3 kornia-py/benchmarks/bench_sift_quality.py
```

Speed plus the homography and epipolar tables above, for every backend, against
OpenCV at one thread and at all cores. Run this after any change that could move
descriptors — a speedup that costs matches is not a speedup.

For the threading table, sweep the rayon pool (it is fixed at process start, so
one run only produces one valid row):

```bash
for n in 1 2 4 6; do
  RAYON_NUM_THREADS=$n python3 kornia-py/benchmarks/bench_sift_quality.py
done
```

**Compare CPU engines in separate processes.** Both pools live in one process
costs the NEON path ~15%, and pinning OpenCV to one thread while ours uses six
is not a comparison — that mistake produced a bogus 2.3x in this document.

### Rust benchmarks (criterion, CPU)

```bash
cargo bench -p kornia-imgproc --bench bench_features -- Sift
```

Groups: `Sift/detect/*` for the four detector configurations, `Sift/match/neon`
and `Sift/match/scalar`. Criterion tracks regressions between runs, so run it
before and after a change rather than eyeballing one number.

### CUDA end to end

The CUDA path needs a device, a stream and a warm kernel cache, so it is
benchmarked through the Python binding rather than criterion:

```python
import kornia_rs as K, numpy as np, cv2, time
st = K.cuda.Stream.default()
g = cv2.imread("tests/data/mh01_frame1.png", 0).astype(np.float32)
d = K.image.Image.from_numpy(np.ascontiguousarray(g[..., None])).to_cuda(st)
s = K.imgproc.Sift()
for _ in range(5): s.detect_and_compute(d)      # warm the NVRTC cache
t = time.perf_counter(); s.detect_and_compute(d); print((time.perf_counter()-t)*1e3)
```

**Always warm up.** The first call compiles every kernel — that is ~1.2 s, and
attributing it to whatever stage runs first is a mistake this project has
already made once.

### Per-stage timing

`KORNIA_SIFT_STAGES=1` prints a per-stage breakdown from either backend:

```
stages: blur=5.0 detect=1.8 orient=2.7 descriptor=8.2 copyback=1.0 (ms)
```

On CUDA each probe synchronises, so the total is inflated — read the ratios, not
the absolutes. Compare the stage sum against wall time: a large gap is host-side
overhead (allocation, transfers), which has been the dominant cost three
separate times in this module.

### Per-kernel truth (nsys)

`ncu` needs `NVreg_RestrictProfilingToAdminUsers=0` plus a reboot on this host.
`nsys` does not — it uses CUPTI tracing and works as a normal user:

```bash
nsys profile -t cuda --force-overwrite true -o /tmp/sift python3 prof.py
nsys stats --report cuda_gpu_kern_sum /tmp/sift.nsys-rep   # per-kernel ms
nsys stats --report cuda_api_sum      /tmp/sift.nsys-rep   # host API / sync cost
```

Divide the kernel totals by the iteration count. `cuda_api_sum` is how the 41
blocking `cuMemcpyDtoHAsync` calls per frame were found.

## How to debug numerics

The contract is **bitwise** equality with `cv::SIFT`, so a tolerance is never the
answer — a mismatch is a bug to root-cause.

### Oracle harness

Dumps of the reference's own internals live in the scratchpad and are selected by
a colon-separated list:

```bash
ALL=$S/or_dog:$S/or_mh01:$S/or_tags:$S/or_tiny:$S/or_sat:\
$S/or_flat:$S/or_noise:$S/or_checker:$S/or_grad:$S/or_frame2
KORNIA_SIFT_ORACLE="$ALL" KORNIA_SIFT_HALREF=$S/halref.bin \
  cargo test --release -p kornia-imgproc --features cuda --lib sift
```

Ten images, including adversarial ones (saturated, flat, pure noise,
checkerboard, gradient, all with odd dimensions). Every bitwise test **skips
silently** without `KORNIA_SIFT_ORACLE` — so a green run proves nothing unless
the variable is set. Tests that must run unconditionally exist too; that
distinction is deliberate.

### Debug knobs

| variable | effect |
|---|---|
| `KORNIA_SIFT_ORACLE` | oracle dirs for the bitwise tests |
| `KORNIA_SIFT_HALREF` | reference vectors for `exp`/`atan2`/`magnitude` |
| `KORNIA_SIFT_STAGES=1` | per-stage timing |
| `KORNIA_SIFT_ORIDBG=1` | print mismatching orientation cases |
| `KORNIA_SIFT_ORIDUMP=layer,cc,rr` | dump one keypoint's 36 histogram bins |
| `KORNIA_SIFT_DESC=exact` | the sequential, bit-exact descriptor kernel |
| `KORNIA_SIFT_FASTMATH=1` | approximate HAL primitives (measured: no gain) |
| `KORNIA_SIFT_DUMP_SRC=<path>` | write generated NVRTC source to disk |

### The technique that works

Component-wise isolation has repeatedly reported "exact" for every part of a
chain that was collectively wrong — it happened with `atan2`, with the Cramer
solve, and with the orientation histogram. What has resolved every one of them is
a **side-by-side chain diff**: dump every intermediate for one failing element
from both sides and read down the two columns until they diverge.

Two traps worth knowing before generating a reference:

* A hand-written oracle compiled `-O2` contracts float expressions across
  statements, and *across NEON intrinsics*. Build reference generators with
  `-ffp-contract=off` and cross-check against numpy, which does not contract.
* `.cargo/config.toml` sets `-Copt-level=2` for aarch64, which turns
  `debug_assertions` **off** in dev and test builds on this host. Every
  `debug_assert!` is inert here. Use `RUSTFLAGS="-C debug-assertions=on"` when
  that matters.

### Cross-compilation

Bit-exactness is an aarch64 property — off aarch64 the estimate instructions the
reference backend composes do not exist, and `magnitude` falls back to a real
square root. The scalar paths must still *compile*:

```bash
cargo clippy -p kornia-imgproc --tests --target x86_64-unknown-linux-gnu -- -D warnings
```

They did not, until this was checked.
