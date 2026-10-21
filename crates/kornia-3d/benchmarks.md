# kornia-3d benchmarks

Timings and accuracy of the PnP solvers, logged with the machine they were measured on, so a later
run can spot regressions. Add a new dated **Results** section when re-measuring on another machine
or after a change.

## How to run

```sh
# Speed: EPnP, EPnP + LM and SQPnP, plus solve_pnp_ransac with each kernel
cargo bench -p kornia-3d --features test-utils --bench bench_pnp

# Accuracy and failure rates, also inside solve_pnp_ransac (prints the tables below)
cargo test --release -p kornia-3d --lib -- --ignored --nocapture pnp_accuracy_report
```

## Methodology

| Parameter | Value |
|-----------|-------|
| Scenes (`bench_pnp`) | random points in a 2 m cube, or a 2 m square for `planar`, seen from 6 m; Gaussian pixel noise with σ = 1 px; random poses; fixed seed |
| Batch | each `pnp` iteration solves 20 scenes, each `pnp_ransac` iteration 5 (100 inliers + 100 outliers); times below are per single solve |
| Failing cases | every scene is solved once before timing; a case that fails is reported as skipped, never timed |
| Timing | criterion defaults (3 s warm-up, 5 s measurement, 100 samples); the time shown is criterion's point estimate divided by the batch size |
| Accuracy (`pnp_accuracy_report`) | 200 random scenes per row, fixed seeds; Gaussian pixel noise with the σ given in each table; failure = rotation error above 5° or no pose. The RANSAC rows add random outlier correspondences and use a 3 px threshold |

**Note:** plain EPnP still has #1181 (wrong pose on noise-free data), which is why it fails in
most accuracy rows and likely why the EPnP RANSAC kernel finds too few inliers on one bench scene. Expect EPnP and
EPnP + LM numbers to change when that is fixed.

---

## Results — 2026-10-22 (follow-up review)

Compared the first corrected-objective revision below with the follow-up that shares one scaled
3×3 SVD between decomposition and rotation projection. Projection skips the former second
normalization and does not restore unused singular values. The shared fixtures now validate
inputs, bound rejection sampling and propagate typed errors; valid seeded scenes use the same
random draws and coordinates as before. Setup remains outside the timed loop.

Hardware, compiler and batch size match the first review run. Three alternating rounds used
20 samples, 0.5 s warm-up, 1 s measurement and 10,000 bootstrap resamples. Each executable was
captured separately after building: SHA-256 prefixes `c0ea9582a17acfbb` (first correction) and
`9678dddced572314` (follow-up). Both ran from this checkout; no shared-target rebuilds occurred
between timing the two versions. Values are the lowest round mean per solve and its 95% interval.

**Timing limitation:** observed one-minute load averages at round starts were approximately
101–119. Competing workloads severely limit interpretation of these results; do not infer a
speedup or compare their absolute times with the lower-load run below. Repeat on an idle machine
before making performance claims. The 100-point cases were slightly slower in these samples.

| Points | First correction, non-planar (µs) | Follow-up, non-planar (µs) | First correction, planar (µs) | Follow-up, planar (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 50.9 [45.9, 57.0] | 45.0 [42.1, 48.9] | 131.3 [116.0, 150.8] | 113.5 [109.1, 118.5] |
| 20 | 32.5 [31.2, 34.0] | 31.1 [28.4, 34.6] | 126.0 [115.9, 138.7] | 116.8 [104.4, 132.7] |
| 100 | 39.0 [35.3, 43.1] | 39.6 [37.0, 42.5] | 108.5 [99.9, 118.8] | 109.6 [105.0, 114.6] |
| 500 | 59.0 [55.5, 63.6] | 55.3 [50.6, 62.3] | 111.9 [108.6, 115.3] | 105.8 [102.4, 110.0] |

Use the command below with `--nresamples 10000` and distinct `--save-baseline` names to repeat
these measurement settings. No updated full accuracy or downstream SLAM evaluation is claimed.

## Results — 2026-10-21 (first objective correction)

Compared PR head `34f5519a` with the local review revision that corrects the algebraic objective,
validates parameters and uses borrowed constraint views plus the shared SVD backend. Both used
the **same current scene generator**, Cargo.lock, Rust 1.99.0, Criterion 0.8.2, and MacBook Air
Mac17,3 (Apple M5, 16 GB RAM, macOS 27.0.1). The old build received only the shared fixtures and
benchmark feature wiring. Separate captured executables were checked to distinguish the two
implementations; each round ran the original and corrected version sequentially.

Three alternating rounds, 20 samples per case, 0.5 s warm-up, 1 s measurement. Values below
are the **lowest round mean per solve**, dividing the 20-scene batch by 20. Brackets are that
round's 95% confidence interval. The machine was shared with other workloads: observed
one-minute load averages were approximately 18–30, and some round means varied by more than 2×.
These are preliminary measurements, not evidence of a general speedup. The 100- and 500-point
planar cases were slower in this run; an idle-machine profile is needed to attribute that change.

```sh
cargo bench -p kornia-3d --features test-utils --bench bench_pnp -- \
  'pnp/sqpnp' --sample-size 20 --warm-up-time 0.5 --measurement-time 1 \
  --save-baseline <version-and-round> --noplot
```

| Points | Original, non-planar (µs) | Corrected, non-planar (µs) | Original, planar (µs) | Corrected, planar (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 36.3 [35.2, 37.4] | 27.9 [27.4, 28.4] | 88.0 [85.7, 90.3] | 83.2 [81.5, 84.9] |
| 20 | 20.0 [19.6, 20.3] | 20.8 [18.3, 24.9] | 80.5 [76.1, 86.8] | 71.9 [69.3, 74.7] |
| 100 | 25.8 [25.0, 26.8] | 23.9 [23.7, 24.1] | 74.9 [73.6, 76.5] | 96.0 [91.6, 100.8] |
| 500 | 40.5 [38.6, 42.5] | 38.2 [36.5, 40.5] | 72.7 [71.2, 74.3] | 91.9 [86.7, 98.5] |

The corrected solver passes a fixed noisy OpenCV 5.0.0 SQPnP reference case that the original
fails, and an independently evaluated algebraic-residual test. The historical accuracy tables
below have not been regenerated for the corrected objective and must not be cited as its results.

## Historical results — 2026-10-20 (superseded)

These numbers describe commit `d12c8045`, whose objective was perpendicular distance to rays.
That objective differs from the algebraic reprojection objective in SQPnP's paper and OpenCV.
The speed harness also used a separate generator with a fixed pose and uniform ±1 px noise.
Keep these results only as historical measurements; they do not validate the corrected solver
or the current shared-fixture benchmark.

### Hardware / software

| Field | Value |
|-------|-------|
| Commit | `d12c8045` on `feat/sqpnp-1182` (#1183) |
| Machine | MacBook Air (Mac17,3), Apple M5 (4P+6E cores), 16 GB RAM |
| OS | macOS 27.0.1 (arm64) |
| Rust | 1.99.0 (b940084d7 2026-09-28), `--release` (cargo bench) |
| Criterion | 0.8.2 |
| Power | on AC power, load average ~3 (laptop; expect a few % to ~20% run-to-run noise); the `solve_pnp_ransac` rows are a separate run (`-- pnp_ransac`) at the same load |

### Speed — one solve

| Points | EPnP | EPnP + LM | SQPnP | EPnP, planar | EPnP + LM, planar | SQPnP, planar |
|-------:|-----:|----------:|------:|-------------:|------------------:|--------------:|
| 6 | 10.0 µs | 100.1 µs | 13.9 µs | 8.6 µs | 105.9 µs | 46.0 µs |
| 20 | 10.7 µs | 278.2 µs | 10.8 µs | 9.5 µs | skipped (LM's linear solve fails on one scene) | 39.6 µs |
| 100 | 14.3 µs | 1.18 ms | 12.2 µs | 12.8 µs | 1.26 ms | 40.6 µs |
| 500 | 33.9 µs | 6.09 ms | 19.2 µs | 34.9 µs | 6.94 ms | 50.7 µs |

### `solve_pnp_ransac` — one run (100 inliers + 100 outliers, 3 px threshold, fixed seed)

| Kernel | Time |
|--------|-----:|
| EPnP | skipped (too few inliers on one scene; EPnP still has #1181) |
| AP3P | 1.31 ms |
| SQPnP | 13.12 ms |

### Accuracy

Median rotation error (°) and median relative translation error (%) over the scenes that returned a pose, and failure rate (%).

#### Correspondence count (1 px noise, points in a 2 m cube 6 m away)

| n | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2.629 | 1.785 | 42.5 | 0.881 | 0.498 | 11.5 | 0.754 | 0.445 | 0.5 |
| 6 | 32.069 | 9.529 | 80.5 | 0.539 | 0.346 | 5.5 | 0.515 | 0.353 | 0.0 |
| 10 | 36.286 | 8.631 | 86.5 | 0.374 | 0.242 | 1.5 | 0.367 | 0.238 | 0.0 |
| 20 | 45.629 | 7.594 | 83.5 | 0.243 | 0.141 | 0.5 | 0.239 | 0.140 | 0.0 |
| 50 | 36.761 | 6.243 | 82.0 | 0.153 | 0.089 | 0.0 | 0.151 | 0.099 | 0.0 |
| 100 | 41.065 | 7.791 | 81.0 | 0.099 | 0.057 | 0.0 | 0.101 | 0.054 | 0.0 |
| 500 | 31.401 | 7.005 | 82.5 | 0.047 | 0.029 | 0.0 | 0.048 | 0.035 | 0.0 |

#### Pixel noise (20 points in a 2 m cube 6 m away)

| noise px | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 38.669 | 7.001 | 81.5 | 0.000 | 0.000 | 0.5 | 0.000 | 0.000 | 0.0 |
| 0.5 | 42.629 | 8.081 | 87.0 | 0.122 | 0.067 | 1.0 | 0.126 | 0.066 | 0.0 |
| 1 | 48.118 | 6.627 | 84.0 | 0.252 | 0.157 | 0.5 | 0.252 | 0.155 | 0.0 |
| 2 | 46.882 | 7.400 | 81.5 | 0.470 | 0.264 | 0.0 | 0.480 | 0.276 | 0.0 |
| 5 | 37.627 | 6.491 | 83.0 | 1.098 | 0.701 | 0.0 | 1.131 | 0.740 | 0.0 |

#### Harder configurations (1 px noise)

| case | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| planar, n=20 | 78.625 | 23.255 | 99.5 | 0.639 | 0.274 | 24.5 | 0.411 | 0.211 | 0.0 |
| small object far (0.5 m cube at 30 m), n=20 | 34.747 | 10.236 | 91.5 | 8.840 | 8.322 | 81.5 | 5.147 | 7.121 | 52.0 |
| close range (2 m cube at 2.5 m), n=20 | 52.409 | 14.725 | 67.0 | 0.093 | 0.070 | 0.5 | 0.109 | 0.073 | 0.0 |
| minimum, n=4 (0.5 px noise) | 1.009 | 0.806 | 30.0 | 0.409 | 0.255 | 6.0 | 0.395 | 0.224 | 0.0 |

#### Inside solve_pnp_ransac (50 inliers with 1 px noise, plus random outliers)

| case | EPnP rot° | EPnP trans % | EPnP fail % | AP3P rot° | AP3P trans % | AP3P fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| no outliers | 0.587 | 0.401 | 0.0 | 0.205 | 0.106 | 0.0 | 0.169 | 0.104 | 0.0 |
| 25% outliers | 0.587 | 0.408 | 0.0 | 0.183 | 0.110 | 0.0 | 0.160 | 0.097 | 0.0 |
| 50% outliers | 1.379 | 1.161 | 59.0 | 0.180 | 0.104 | 0.0 | 0.214 | 0.144 | 0.0 |
| planar, 25% outliers | 30.507 | 5.959 | 100.0 | 0.425 | 0.196 | 0.5 | 0.266 | 0.148 | 0.0 |
