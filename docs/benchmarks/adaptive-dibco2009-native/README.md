# Adaptive mean threshold: reusable scratch versus native C++ OpenCV

The caller-owned workspace implementation beats native C++ OpenCV in all
**70 image/window cases**, with **zero differing pixels over 44,014,824
comparisons**. This is a local feature implementation, not a released Kornia version.

## Results

Native Apple Silicon, macOS 26.6.2 arm64, Rust 1.98.0 and OpenCV 4.11.0
built in Release mode with Apple Clang 21. Both use NEON-capable builds.
All ten original-resolution [DIBCO 2009 test documents](https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/)
were used, five handwritten and five printed. There is no resizing or cropping.

Each value below is the arithmetic mean over images of each image's median
of five rounds of 200 native calls. Ratios divide those aggregate means.

| Window | Kornia ms/image | C++ OpenCV ms/image | Kornia speedup |
|---|---:|---:|---:|
| 3 × 3 | 0.127057 | 0.319728 | 2.52× |
| 5 × 5 | 0.181325 | 0.328126 | 1.81× |
| 15 × 15 | 0.357683 | 0.499953 | 1.40× |
| 31 × 31 | 0.372276 | 0.672401 | 1.81× |
| 51 × 51 | 0.382102 | 0.683599 | 1.79× |
| 67 × 67 | 0.389689 | 0.689108 | 1.77× |
| 101 × 101 | 0.408502 | 0.709571 | 1.74× |

A second full run also won all 70 cases with zero differing pixels. Its
aggregate speedups were 2.58×, 1.81×, 1.40×, 1.82×, 1.79×, 1.77× and 1.72×
respectively; the raw samples are in [repeat-results.json](repeat-results.json).
Some individual rounds show scheduling variation, so the reported ratios use
the documented medians rather than selecting the fastest samples.

Both implementations use grayscale u8, adaptive mean, binary output, offset 10,
maximum 255 and replicated borders. Exact output equality establishes reference
compatibility for these inputs and parameters, not a binarization quality score.

## Workspace API

```rust
use kornia_imgproc::threshold::{adaptive_threshold_mean, AdaptiveThresholdMeanWorkspace};

let mut workspace = AdaptiveThresholdMeanWorkspace::new();
// Retain workspace between images; src and dst have matching dimensions.
adaptive_threshold_mean(&src, &mut dst, &mut workspace, 31, 10, 255)?;
```

`new()` allocates nothing. Each buffer grows only when its required length
exceeds retained capacity. Repeated calls within that capacity reuse the
allocations, including the u64 large-window fallback. The 3 × 3 path needs no
scratch. Source images are borrowed and the caller supplies the destination.
Each concurrent operation needs its own workspace.

Active scratch at width 1920/window 101 is about 773 KiB. Vec growth can retain
more capacity than that, and switching to smaller shapes/windows does not release
the high-water allocations. The streaming algorithm, exact arithmetic and
specializations are unchanged; ownership moved from per-call locals to the workspace.

The new test `adaptive_mean_reuses_workspace_across_shapes_and_kernels` checks
buffer addresses/capacities remain stable on repeated calls, the 3 × 3 path
allocates no buffers, and state resets correctly after changing image shape,
contents and kernel path. Other tests retain independent exact-sum oracles.

## Timing boundaries and allocation costs

- Rust and C++ each run a persistent native worker. Python only orchestrates
  rounds, decodes images and compares masks; **no Python binding or IPC time
  is included in either timed region**.
- Both see the same decoded grayscale bytes and reuse input/output storage.
- Warmup is at least 10 calls and 250 ms per library/image/window. Rust's
  workspace is populated during warmup; these are steady-state reuse timings.
- OpenCV's public `adaptiveThreshold` API does not accept external scratch.
  Its normal internal temporary allocations remain included in native timings.
  This compares the public APIs as used, not two allocation-free inner kernels.
  First-use Rust workspace allocation and destruction are not timed.
- Alternate which native worker runs first each round. Five rounds of 200 calls
  per implementation are recorded. No compilation or tests run concurrently.
- Decoding, process startup, file I/O, command transport and mask comparison are
  outside timed sections. Worker source/binary hashes and raw samples are saved.
- Rust is single-threaded. C++ requests `setNumThreads(1)` and disables OpenCL.
  GCD's getter reports CPU count (8), even though this setting bypasses OpenCV's
  standard parallel dispatch; see the upstream
  [threading implementation](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp#L514-L624).
  Actual worker-thread activity was not profiled.

## Relation to the earlier Python comparison

The [earlier report](../adaptive-dibco2009-optimized/README.md) used a Rust API
that allocated scratch per call and an OpenCV Python wheel built with Apple
Clang 15. This native reference is a separate OpenCV 4.11 Release build with
Apple Clang 21, standard CPU dispatch and Carotene HAL enabled. Full build
information is in [opencv-build.txt](opencv-build.txt).

In particular, this C++ build is slower than that wheel on the larger windows.
The changed ratios must not be attributed to removing Python overhead or to
workspace reuse alone: the OpenCV build also changed. A compiler/build ablation
was not performed. The table above is the contemporary native-to-native result.

## Reproduce

The source images and archive hashes are documented in the
[original dataset report](../adaptive-dibco2009/README.md#reproduce).
NumPy and opencv-python are needed only for decoding and orchestration.

Download and extract the official OpenCV source archive:

```text
https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz
SHA-256: 9a7c11f924eff5f8d8070e297b322ee68b9227e003fd600d4b8122198091665f
```

With CMake and Ninja available, from the repository root:

```sh
cmake -S /private/tmp/opencv-4.11.0 -B /private/tmp/opencv-4.11.0-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_LIST=core,imgproc -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=ON \
  -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DWITH_GCD=ON \
  -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DOPENCV_GENERATE_PKGCONFIG=ON
cmake --build /private/tmp/opencv-4.11.0-build \
  --target opencv_core opencv_imgproc --parallel 6

clang++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Werror \
  -I/private/tmp/opencv-4.11.0/modules/core/include \
  -I/private/tmp/opencv-4.11.0/modules/imgproc/include \
  -I/private/tmp/opencv-4.11.0-build \
  crates/kornia-imgproc/benches/bench_adaptive_opencv.cpp \
  -L/private/tmp/opencv-4.11.0-build/lib \
  -Wl,-rpath,/private/tmp/opencv-4.11.0-build/lib \
  -lopencv_imgproc -lopencv_core -o /private/tmp/kornia-opencv-worker

cargo build --release -p kornia-imgproc --example bench_adaptive_dataset
python3 crates/kornia-imgproc/benches/bench_adaptive_dataset.py \
  --images /path/to/extracted-dibco-images \
  --output /tmp/adaptive-native-results \
  --opencv-binary /private/tmp/kornia-opencv-worker
```

Omit `--opencv-binary` to time the installed Python binding instead. The harness
was smoke-tested in both modes; 14 native outputs on small/narrow images were
also compared with Python OpenCV. Both native workers reject invalid input.

[results.json](results.json) stores provenance and per-round samples;
[results.csv](results.csv) stores individual median timings and mismatch counts.
[native-build.json](native-build.json) records source/archive and shared-library
hashes, compiler/CMake versions and linkage. [cmake-options.txt](cmake-options.txt)
records the native library's build options. CMake 4.4.4 and Ninja were used;
both OpenCV libraries and the benchmark worker remain in temporary directories.

## Validation and limits

`cargo test -p kornia-imgproc --offline --quiet`: **527 passed, 5 ignored**,
including 65 doctests. Formatting and diff checks pass. All-target package
Clippy passes with warnings denied except the pre-existing
`chunks_exact_to_as_chunks` and `int_plus_one` categories in unchanged files.
The C++ worker compiles with `-Wall -Wextra -Werror`.

No MSRV, complete workspace/CI gate, native x86, GPU, parallel thresholding or
ground-truth quality check is claimed. Results apply to this Apple Silicon
machine, these builds, these windows and ten scanned documents.
