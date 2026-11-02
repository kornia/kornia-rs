# Adaptive mean thresholding versus native C++ OpenCV

This benchmark compares the Rust CPU `AdaptiveThresholdMean` operation with
native C++ OpenCV 4.11 `adaptiveThreshold` (`ADAPTIVE_THRESH_MEAN_C`,
`THRESH_BINARY`). Both use grayscale u8, offset 10, maximum 255 and replicated
borders. Pixel equality checks compatibility, not quality against ground truth.

## Results

Native Apple Silicon, macOS 26.6.2 arm64, Rust 1.98.0 and OpenCV 4.11.0
Release with Apple Clang 21. All ten original-resolution
[DIBCO 2009 test documents](https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/)
are used, five handwritten and five printed, without resizing or cropping.

For each image/window, record five rounds of 200 native calls. The table reports
arithmetic means over images of the per-image median and minimum (best-of-five)
round times. Speedups divide OpenCV's aggregate time by Rust's aggregate time;
the two minima need not occur in the same round. Raw rounds, per-image results,
source/binary/input hashes and build information are in [results.json](results.json).

| Window | Rust median ms | OpenCV median ms | Speedup | Rust best ms | OpenCV best ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 3 × 3 | 0.242515 | 0.615696 | 2.54× | 0.220622 | 0.555480 | 2.52× |
| 5 × 5 | 0.316364 | 0.568691 | 1.80× | 0.315713 | 0.567286 | 1.80× |
| 15 × 15 | 0.621216 | 0.868408 | 1.40× | 0.618732 | 0.866052 | 1.40× |
| 31 × 31 | 0.645189 | 1.167659 | 1.81× | 0.643927 | 1.164640 | 1.81× |
| 51 × 51 | 0.677054 | 1.213215 | 1.79× | 0.673570 | 1.183887 | 1.76× |
| 67 × 67 | 0.676901 | 1.196212 | 1.77× | 0.674161 | 1.193741 | 1.77× |
| 101 × 101 | 0.720646 | 1.274039 | 1.77× | 0.704285 | 1.229585 | 1.75× |

All **70 image/window cases** match OpenCV pixel for pixel across
**44,014,824 comparisons**. Rust is faster in all 70 cases using either statistic.


## API and timing boundaries

```rust
use kornia_imgproc::threshold::AdaptiveThresholdMean;

let mut threshold = AdaptiveThresholdMean::new(31, 10, 255)?;
// Retain the instance between images; src and dst have matching dimensions.
threshold.apply(&src, &mut dst)?;
```

The constructor validates configuration without allocating. `apply` grows its
private scratch only when required capacity increases; the 3 × 3 path uses none.
Active scratch at width 1920/window 101 is about 773 KiB. Capacity is retained
when processing smaller images. Each concurrent operation needs its own instance.

- Rust and C++ use persistent native workers and preallocated input/output.
  Python decodes and orchestrates; decoding, IPC, process startup, mask comparison
  and file I/O are outside both timers.
- Configuration is constructed outside the timed loop. Warmup is at least ten
  calls and 250 ms per library/image/window, populating Rust's internal scratch.
  Initial allocation/destruction is excluded. OpenCV's internal temporary
  allocations remain included because its public API does not expose scratch.
- Worker order alternates each round. No builds or tests run concurrently.
  Rust uses `black_box`; C++ consumes the output mask after timed rounds.
- Rust is single-threaded. C++ requests `setNumThreads(1)` and disables OpenCL.
  GCD reports CPU count (8) even though this setting bypasses standard parallel
  dispatch; see OpenCV's [threading implementation](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp#L514-L624).
  Actual worker activity was not profiled.

## Reproduce

Download both archives from the official DIBCO index above and extract into
one image directory. Source images are not redistributed.

| Archive | SHA-256 |
|---|---|
| `DIBC02009_Test_images-handwritten.rar` | `d522e6e658e1aedb57c5cfdc1e4717a0e678568a73383f69321580245b5af39e` |
| `DIBCO2009_Test_images-printed.rar` | `10c78a4beb71abe70ed7151d2e448fde01ed07bf0b17c17253c6256960c7ad53` |

Extract [OpenCV 4.11.0](https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz)
(SHA-256 `9a7c11f924eff5f8d8070e297b322ee68b9227e003fd600d4b8122198091665f`)
into `/tmp/opencv-4.11.0`. With CMake, Ninja, NumPy and opencv-python installed,
from the repository root:

```sh
cmake -S /tmp/opencv-4.11.0 -B /tmp/opencv-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_LIST=core,imgproc -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=ON \
  -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DWITH_GCD=ON \
  -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DOPENCV_GENERATE_PKGCONFIG=ON
cmake --build /tmp/opencv-build --target opencv_core opencv_imgproc --parallel 6
clang++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Werror \
  -I/tmp/opencv-4.11.0/modules/core/include \
  -I/tmp/opencv-4.11.0/modules/imgproc/include -I/tmp/opencv-build \
  crates/kornia-imgproc/benches/bench_adaptive_opencv.cpp \
  -L/tmp/opencv-build/lib -Wl,-rpath,/tmp/opencv-build/lib \
  -lopencv_imgproc -lopencv_core -o /tmp/opencv-worker
cargo build --release -p kornia-imgproc --example bench_adaptive_dataset
python3 crates/kornia-imgproc/benches/bench_adaptive_dataset.py \
  --images /path/to/extracted-dibco-images --output /tmp/adaptive-results \
  --opencv-binary /tmp/opencv-worker
```

Use the source revision recorded in `results.json` to reproduce that measurement.
The recorded tree is clean; results are copied into the repository afterward.
The runner also generates a CSV for convenience; only JSON is committed.
Omitting `--opencv-binary` measures the installed Python binding instead and
includes its binding/loop overhead, so those timings are not native-to-native.

Results apply to this machine, these builds and the tested inputs. No native
x86, GPU, parallel-threshold or ground-truth quality measurement is claimed.
