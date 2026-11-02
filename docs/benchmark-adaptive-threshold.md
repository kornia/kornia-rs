# Adaptive mean threshold: 3 × 3 CPU investigation

For current real-image comparisons across seven window sizes, see the
[native C++ OpenCV benchmark](benchmarks/adaptive-dibco2009-native/README.md).
The measurements below predate optimization of windows larger than 3 × 3.

Measured locally on 2026-11-01, native Apple Silicon macOS, Rust 1.98.0,
OpenCV Python 4.11.0. Both Rust implementations are uncommitted changes on
`a183d34f45d71d6880c05b7c8c46e2751869af8d`. No GPU measurements were made.

Single-channel u8 images, offset 10, maximum 255, preallocated destination.
Input pixel i is `(i * 73 + (i / width) * 11) % 256`.

| Size | Initial rolling sums | Direct 3 × 3 path | OpenCV CPU |
|---|---:|---:|---:|
| 640 × 480 | 0.3793 ms | 0.0624 ms | 0.1544 ms |
| 1920 × 1080 | 2.5288 ms | 0.4165 ms | 1.1591 ms |

Rust: Criterion, 30 samples, 1-second warmup, 1-second measurement target
(Criterion extends this as necessary), saved baseline `adaptive-initial`.
The optimized run was repeated, giving 0.0624 / 0.4169 ms and
0.0624 / 0.4165 ms. OpenCV: one thread requested (the installed GCD getter
reports CPU count; see the optimized report's explanation), 10 warmups, median of five rounds
of 50 calls, reused output. These use different timing harnesses; the Rust
before/after comparison uses the same harness. Earlier quick measurements
taken while other checks were active substantially overstated both libraries' runtimes.

The direct path removes scratch allocation and the rolling-sum dependency.
Inspecting the timed binary with `otool -tvV` confirms compiler-generated
NEON widening adds, multiplies and comparisons. Merely specializing division
by nine in the rolling loop gave no improvement and was discarded. At the
time of this run, windows larger than 3 × 3 retained the generic rolling-sum path.

The 12 threshold tests pass, including pinned OpenCV results and direct-window
oracle checks for narrow images, borders, wide row tails and extreme offsets.
The new doctest also passes. Clippy passes with the existing
`chunks_exact_to_as_chunks` and `int_plus_one` lint categories allowed.

The function remains CPU-only and returns `UnsupportedDevice` for CUDA
images. A CUDA implementation would need its own dispatch and parity tests.
GPU kernel time and total upload/launch/execute/download time must be measured
separately. This checkout has no Metal backend for Apple GPUs.
