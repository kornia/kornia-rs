# Adaptive mean threshold: DIBCO 2009 versus OpenCV

Historical results before the larger-window optimization. See the
[workspace implementation and native C++ benchmark](../adaptive-dibco2009-native/README.md)
for the current results. The current runner uses paired rounds of 200 calls;
this historical run used 20 calls per round.

Measured 2026-11-01 on native Apple Silicon, macOS 26.6.2 arm64,
Rust 1.98.0, Python 3.13 and OpenCV 4.11.0. This measures the local,
uncommitted adaptive threshold implementation, not a published Kornia release.

## Dataset and correctness

All ten original-resolution DIBCO 2009 test images: five handwritten and
five machine-printed documents, downloaded from the
[official dataset index](https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/).
Images contain 286,344–1,292,236 pixels each. No resizing, cropping or
synthetic augmentation was applied.

OpenCV decodes each source to contiguous grayscale u8 once; exactly those
bytes are passed to Rust. All 70 image/window outputs match OpenCV byte for
byte: **zero differing pixels across 44,014,824 pixel comparisons**.
This checks parity with OpenCV, not binarization accuracy against DIBCO
ground truth; no ground-truth score is claimed.

Both use adaptive **mean** thresholding, binary output, offset 10,
maximum 255 and replicated borders. OpenCV was requested to use one thread;
its GCD getter reports eight CPUs even though the setting serializes OpenCV's
standard parallel loop. See the source explanation in the
[optimized report](../adaptive-dibco2009-optimized/README.md#method-and-limits).
Actual worker usage was not profiled. OpenCL is disabled; Rust is single-threaded.

## Results

For each library/image/window, the measured value is the median of five
rounds of 20 calls. Each table entry is the arithmetic mean of those
per-image medians over all ten images; ratios divide those aggregate means.

| Window | Kornia mean ms/image | OpenCV mean ms/image | Kornia relative to OpenCV |
|---|---:|---:|---|
| 3 × 3 | 0.126338 | 0.310756 | 2.46× faster |
| 5 × 5 | 0.767961 | 0.319918 | 2.40× slower |
| 15 × 15 | 0.781703 | 0.489662 | 1.60× slower |
| 31 × 31 | 0.795517 | 0.512657 | 1.55× slower |
| 51 × 51 | 0.819717 | 0.526268 | 1.56× slower |
| 67 × 67 | 0.840451 | 0.530993 | 1.58× slower |
| 101 × 101 | 0.886073 | 0.547180 | 1.62× slower |

The 3 × 3 specialization wins on every image. OpenCV wins on every image
for every other tested window. The generic rolling-sum path remains the
optimization target; its runtime grows only modestly with window size.

## Method and limits

- Warmup: at least 10 calls **and 250 ms** per library/image/window. An
  initial ten-call-only run showed first-image startup drift and was replaced.
- Input and output buffers are reused. Each implementation's internal
  scratch allocations remain included in its timings.
- Decoding, raw-buffer transfer to the worker, process startup, file I/O and
  output comparisons are outside the timed sections.
- Alternate which library runs first on successive images. No compilation
  or other validation jobs were run concurrently with the final benchmark.
- Rust timing surrounds direct native Rust calls; OpenCV timing surrounds
  its Python binding calls. Python binding/loop overhead is included for
  OpenCV. This is not an identical C++/Rust native-call harness.
- Results apply to this hardware, build and ten scanned documents. They
  do not establish GPU performance or performance on other CPU architectures.

## Reproduce

Download these two archives from the official index and extract both into
one directory. The benchmark does not redistribute the source images.

| Archive | SHA-256 |
|---|---|
| `DIBC02009_Test_images-handwritten.rar` | `d522e6e658e1aedb57c5cfdc1e4717a0e678568a73383f69321580245b5af39e` |
| `DIBCO2009_Test_images-printed.rar` | `10c78a4beb71abe70ed7151d2e448fde01ed07bf0b17c17253c6256960c7ad53` |

From the repository root, with NumPy and opencv-python installed:

```sh
cargo build --release -p kornia-imgproc --example bench_adaptive_dataset
python3 crates/kornia-imgproc/benches/bench_adaptive_dataset.py \
  --images /path/to/extracted-images \
  --output /tmp/adaptive-dibco2009-results
```

The tested source is based on commit
`a183d34f45d71d6880c05b7c8c46e2751869af8d` plus local changes. Exact
threshold-source and benchmark-binary hashes, input hashes and all raw
per-round samples are in [results.json](results.json). Individual timings
and mismatch counts are in [results.csv](results.csv), and OpenCV's build
configuration is in [opencv-build.txt](opencv-build.txt).
