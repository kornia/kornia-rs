# kornia-3d benchmarks

Timings of the PnP solvers, logged with the machine they were measured on, so a later run can
spot regressions. Add a new dated **Results** section when re-measuring on another machine or
after a change; do not overwrite old ones.

## How to run

```sh
# Speed: EPnP, EPnP + LM and SQPnP, plus solve_pnp_ransac with each kernel
cargo bench -p kornia-3d --bench bench_pnp

# Accuracy and failure rates (prints the Markdown tables below)
cargo bench -p kornia-3d --bench bench_pnp_accuracy
```

## Methodology

| Parameter | Value |
|-----------|-------|
| Scenes (`bench_pnp`) | random points in a 2 m cube, or a 2 m square for `planar`, seen from 6 m; 1 px Gaussian noise; fixed seed |
| Batch | each `pnp` iteration solves 20 scenes, each `pnp_ransac` iteration 5 (100 inliers + 100 outliers); times below are per single solve |
| Failing cases | every scene is solved once before timing; a case that fails is reported as skipped, never timed |
| Timing | criterion defaults (3 s warm-up, 5 s measurement, 100 samples); the median estimate is reported |
| Accuracy (`bench_pnp_accuracy`) | 200 random scenes per row, fixed seeds; failure = rotation error above 5° or no pose |

**Note:** plain EPnP still has #1181 (wrong pose on noise-free data), which is why it fails in
most accuracy rows and why the EPnP RANSAC kernel fails on one bench scene. Expect EPnP and
EPnP + LM numbers to change when that is fixed.

---

## Results — 2026-10-14

### Hardware / software

| Field | Value |
|-------|-------|
| Commit | `ce39c868` on `feat/sqpnp-1182` (#1183) |
| Machine | MacBook Air (Mac17,3), Apple M5 (4P+6E cores), 16 GB RAM |
| OS | macOS 27.0.1 (arm64) |
| Rust | 1.99.0 (b940084d7 2026-09-28), `--release` (cargo bench) |
| Criterion | 0.7.0 |
| Power | on battery (laptop; expect a few % to ~20% run-to-run noise) |

### Speed — one solve

| Points | EPnP | EPnP + LM | SQPnP | EPnP, planar | EPnP + LM, planar | SQPnP, planar |
|-------:|-----:|----------:|------:|-------------:|------------------:|--------------:|
| 6 | 9.6 µs | 89.0 µs | 14.8 µs | 8.2 µs | 109.8 µs | 45.3 µs |
| 20 | 10.4 µs | 255.9 µs | 11.5 µs | 8.9 µs | 278.9 µs | 40.8 µs |
| 100 | 14.2 µs | 1.19 ms | 14.4 µs | 12.8 µs | 1.39 ms | 41.5 µs |
| 500 | 33.2 µs | 5.58 ms | 23.6 µs | 31.7 µs | 6.23 ms | 50.5 µs |

### `solve_pnp_ransac` — one run (100 inliers + 100 outliers, 3 px threshold, fixed seed)

| Kernel | Time |
|--------|-----:|
| EPnP | skipped (fails on one scene, #1181) |
| AP3P | 1.27 ms |
| SQPnP | 12.52 ms |

### Accuracy (`bench_pnp_accuracy`, same commit)

Median rotation error (°), median relative translation error (%) and failure rate (%).

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
| minimum, n=4 | 1.009 | 0.806 | 30.0 | 0.409 | 0.255 | 6.0 | 0.395 | 0.224 | 0.0 |
