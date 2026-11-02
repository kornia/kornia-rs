# Adaptive mean threshold: optimized CPU implementation

Historical measurement of the API that allocated scratch per call. The current
API uses a reusable workspace; see the
[native C++ OpenCV comparison](../adaptive-dibco2009-native/README.md).

The optimized local implementation beats the installed OpenCV in all **70
image/window cases**, with **zero differing pixels across 44,014,824 comparisons**.
These are local changes, not a published Kornia release.

## Results

Measured on native Apple Silicon, macOS 26.6.2 arm64, Rust 1.98.0,
Python 3.13.3 and OpenCV 4.11.0. All ten original-resolution
[DIBCO 2009 test documents](https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/)
were used: five handwritten and five printed, 286,344–1,292,236 pixels each.
No resizing or cropping. Both implementations use grayscale u8, adaptive mean,
binary output, offset 10, maximum 255 and replicated borders.

Each entry is the arithmetic mean across images of the median of five timed
rounds, each containing 200 calls. The speedup divides the two aggregate means.

| Window | Kornia ms/image | OpenCV ms/image | Kornia speedup |
|---|---:|---:|---:|
| 3 × 3 | 0.126840 | 0.310547 | 2.45× |
| 5 × 5 | 0.181971 | 0.316731 | 1.74× |
| 15 × 15 | 0.357772 | 0.489183 | 1.37× |
| 31 × 31 | 0.373468 | 0.512959 | 1.37× |
| 51 × 51 | 0.383717 | 0.524995 | 1.37× |
| 67 × 67 | 0.392445 | 0.530231 | 1.35× |
| 101 × 101 | 0.411497 | 0.546143 | 1.33× |

An earlier full paired run also won all 70 cases: aggregate speedups were
2.45×, 1.74×, 1.37×, 1.37×, 1.37×, 1.35× and 1.33× respectively. Its raw
rounds are saved in [repeat-results.json](repeat-results.json). The final
source adds fallback cleanup, comments and tests without changing the measured
fast kernel. The [historical implementation](../adaptive-dibco2009/README.md)
was slower than OpenCV for every tested window except 3 × 3. That run used
shorter, unpaired rounds, so the contemporary comparison above is the primary result.

## What changed

- Keep the existing allocation-free, independent 3 × 3 sums.
- Specialize the 5-wide horizontal sum with independent fixed-size windows.
- For larger windows, compute each source row's prefix sums once. Each interior
  horizontal window is then a difference of two prefixes; output elements are
  independent and can be vectorized by the compiler.
- Cache horizontal sums in a ring containing at most `min(window, height)` rows.
  Subtract the outgoing row and add the entering row to maintain vertical sums.
- Use u32 accumulators when the bounds prove every intermediate fits. Replace
  rounded division with the equivalent integer comparison
  `clamp(pixel + offset, 0, 256) * area > sum + area / 2`.
- Retain the exact u64 rolling fallback for windows larger than 4095. No new
  dependencies, unsafe code, GPU kernels or parallel dispatch were added.

Both old and new algorithms are O(width × height). The gain comes from removing
serial rolling-sum dependencies in the classification loop, allowing independent
work, and reducing accumulator width. The prefix scan itself still has one
dependent addition per input pixel. Larger windows increase ring size, border
work and initialization cost, explaining the remaining modest rise in runtime.

The tradeoff is scratch memory: at width 1920 and window 101, the ring, columns
and prefix need about **773 KiB**, versus **15 KiB** for the old u64 column
buffer. These buffers are allocated inside each measured call. This is a row
cache, not a full-image integral image. When the window exceeds image height,
the cache can contain the whole image's horizontal sums.

## Experiments

[experiments.json](experiments.json) preserves the screening samples, source and
binary hashes, and OpenCV mismatch counts on H02 (946 × 1366).

| Approach tested | Decision |
|---|---|
| Division-free comparison alone in the u64 rolling loop | No useful improvement; discard as a standalone change. |
| u32 rolling sums | Small or inconsistent improvement; serial recurrence remains. |
| Full-frame separable horizontal/vertical sums | Faster, but more scratch memory; prefer a row cache. |
| Streaming sums with rolling horizontal interiors and isolated borders | Improvement, but prefix differences gave better final results. |
| Explicit four-lane SIMD prefix of rolling deltas | Regressed in screening; discard. |
| Extra ring slot to fuse subtraction, addition and classification | No consistent gain; discard. |
| Row prefix differences, streaming cache and specialized 5 × 5 | Retain; verify against OpenCV on the full dataset twice. |

Early screens used five rounds of 20 calls and showed substantial scheduling
noise. Their individual samples are exploratory, not a reliable ranking or
controlled attribution of the gain to each change. Full 2D integral images,
GPU acceleration and multithreading were not benchmarked.

## Method and limits

- Warm each library for at least 10 calls and 250 ms per image/window.
- Alternate Rust/OpenCV execution order each round; no builds or tests run
  concurrently. Input and destination buffers are reused.
- Decode once with OpenCV and pass exactly the same contiguous grayscale bytes
  to Rust. Decoding, process startup, IPC, file I/O and correctness comparisons
  are excluded. Each implementation's internal allocations are included.
- Rust timing surrounds direct native calls inside a persistent worker. OpenCV
  timing includes its Python binding and loop overhead. This is not an identical
  C++/Rust native-call harness, particularly relevant for the smallest images.
- Rust is single-threaded (`RAYON_NUM_THREADS=1`, with no parallel loop in this
  function). OpenCV receives `setNumThreads(1)`, but its **GCD build reports 8**
  immediately afterward and throughout the run. This is a getter limitation,
  not evidence that the setting was ignored: OpenCV 4.11's
  [parallel.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp#L514-L624)
  returns CPU count for GCD, while `numThreads == 1` bypasses parallel dispatch.
  Moreover, the upstream adaptive-mean CPU fallback uses a serial box-filter
  engine and serial lookup loop. Backend overrides may differ; actual worker
  activity in the installed binary was not profiled. OpenCL is disabled and
  NumPy host buffers are used.
- Pixel equality checks OpenCV compatibility for these inputs and parameters,
  not binarization quality against DIBCO ground truth. Other offsets, shapes,
  output maxima and very large windows are covered by unit tests.
- Results apply to this machine and OpenCV build. No GPU or x86 measurements.

## Reproduce and validation

Download/extract the two archives linked in the
[original dataset report](../adaptive-dibco2009/README.md#reproduce), which records
their SHA-256 hashes. Images are not redistributed here.

```sh
cargo build --release -p kornia-imgproc --example bench_adaptive_dataset
python3 crates/kornia-imgproc/benches/bench_adaptive_dataset.py \
  --images /path/to/extracted-images \
  --output /tmp/adaptive-dibco2009-optimized
```

NumPy and opencv-python must be installed. Per-image timings and raw rounds are
in [results.csv](results.csv) and [results.json](results.json). The JSON records
input file/grayscale hashes, source and binary hashes, compiler and base commit.
The source hash identifies the benchmark-time `threshold.rs`. Later work added
nonuniform-image tests and changed the API to use reusable scratch; this report
does not measure that newer implementation.
OpenCV configuration is saved in [opencv-build.txt](opencv-build.txt).

Validation: `cargo test -p kornia-imgproc --offline --quiet` passed **526 tests**
(including doctests), with 5 ignored. `cargo fmt --all -- --check` and
`git diff --check` passed. Package Clippy passed for all targets with warnings
denied except the pre-existing `chunks_exact_to_as_chunks` and `int_plus_one`
categories in unchanged code. Full workspace/pre-PR and cross-architecture
checks have not passed for this experiment. A workspace check with `ci` and
`--offline` stops because `RustyXML v0.3.0` is not cached; strict package Clippy
on Rust 1.98 fails on the two existing lint categories above. The per-call
scratch concern found at this stage was subsequently resolved with a workspace.
