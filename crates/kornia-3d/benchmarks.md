# kornia-3d benchmarks

Timings and accuracy of the PnP solvers, logged with the machine they were measured on, so a later
run can spot regressions. Add a new dated **Results** section when re-measuring on another machine
or after a change.

## How to run

```sh
# Speed: EPnP, EPnP + LM and SQPnP, plus solve_pnp_ransac with each kernel
cargo bench -p kornia-3d --bench bench_pnp

# Accuracy and failure rates, also inside solve_pnp_ransac (prints the tables below)
cargo test --release -p kornia-3d --lib -- --ignored --nocapture pnp_accuracy_report
```

## Methodology

| Parameter | Value |
|-----------|-------|
| Scenes (`bench_pnp`) | random points in a 2 m cube, or a 2 m square for `planar`, seen from 6 m; uniform noise of up to 1 px; fixed seed |
| Batch | each `pnp` iteration solves 20 scenes, each `pnp_ransac` iteration 5 (100 inliers + 100 outliers); times below are per single solve |
| Failing cases | every scene is solved once before timing; a case that fails is reported as skipped, never timed |
| Timing | criterion defaults (3 s warm-up, 5 s measurement, 100 samples); the time shown is criterion's point estimate divided by the batch size |
| Accuracy (`pnp_accuracy_report`) | 200 random scenes per row, fixed seeds; Gaussian pixel noise with the σ given in each table; failure = rotation error above 5° or no pose. The RANSAC rows add random outlier correspondences and use a 3 px threshold |

---

## Results — 2026-11-26

### Hardware / software

| Field | Value |
|-------|-------|
| Code | #1183 merged with main `4ada5a75` |
| Machine | MacBook Air (Mac17,3), Apple M5 (4P+6E cores), 16 GB RAM |
| OS | macOS 27.2.1 (arm64) |
| Rust | 1.99.0 (b940084d7 2026-09-28), `--release` |

### Speed

`cargo bench -p kornia-3d --bench bench_pnp`, run twice. Each value is the faster of the two
criterion point estimates, per single solve, in µs. The machine was not idle (load average 3 to 8), so
read these as rough sizes, not exact numbers.

| Solver | n = 6 | n = 20 | n = 100 | n = 500 |
|--------|-------|--------|---------|---------|
| EPnP | 9.9 | 10.5 | 14.1 | 33.7 |
| EPnP + LM | 77 | 218 | 939 | 3623 |
| SQPnP | 13.6 | 10.2 | 12.0 | 18.6 |
| EPnP, planar | 7.9 | 8.2 | 11.7 | 31.1 |
| EPnP + LM, planar | skipped | skipped | skipped | skipped |
| SQPnP, planar | 44.2 | 39.0 | 40.4 | 45.0 |

EPnP + LM, planar is skipped because `refine_pose_lm` returns an error ("Point behind or too close
to camera") on one scene of each size.

`solve_pnp_ransac` (100 inliers + 100 outliers), per scene: EPnP 1050 µs, AP3P 872 µs, SQPnP
12647 µs.

### Accuracy

`cargo test --release -p kornia-3d --lib -- --ignored --nocapture pnp_accuracy_report`. Median
rotation error (°) and median relative translation error (%) over the scenes that returned a pose,
and failure rate (%).

#### Correspondence count (1 px noise, points in a 2 m cube 6 m away)

| n | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.272 | 1.004 | 26.0 | 0.880 | 0.478 | 13.5 | 0.762 | 0.442 | 0.5 |
| 6 | 0.580 | 0.397 | 0.0 | 0.515 | 0.347 | 0.0 | 0.515 | 0.354 | 0.0 |
| 10 | 0.443 | 0.263 | 0.0 | 0.370 | 0.235 | 0.0 | 0.365 | 0.240 | 0.0 |
| 20 | 0.275 | 0.183 | 0.0 | 0.236 | 0.129 | 0.0 | 0.240 | 0.139 | 0.0 |
| 50 | 0.170 | 0.103 | 0.0 | 0.151 | 0.091 | 0.0 | 0.151 | 0.098 | 0.0 |
| 100 | 0.120 | 0.068 | 0.0 | 0.099 | 0.055 | 0.0 | 0.101 | 0.055 | 0.0 |
| 500 | 0.051 | 0.035 | 0.0 | 0.046 | 0.028 | 0.0 | 0.048 | 0.035 | 0.0 |

#### Pixel noise (20 points in a 2 m cube 6 m away)

| noise px | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 | 0.0 |
| 0.5 | 0.146 | 0.075 | 0.0 | 0.121 | 0.066 | 0.0 | 0.126 | 0.067 | 0.0 |
| 1 | 0.300 | 0.192 | 0.0 | 0.247 | 0.162 | 0.0 | 0.253 | 0.155 | 0.0 |
| 2 | 0.551 | 0.365 | 0.0 | 0.477 | 0.262 | 0.0 | 0.479 | 0.274 | 0.0 |
| 5 | 1.402 | 0.890 | 0.0 | 1.121 | 0.685 | 0.0 | 1.128 | 0.739 | 0.0 |

#### Harder configurations (1 px noise)

| case | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| planar, n=20 | 64.518 | 100.000 | 98.0 | 0.497 | 0.240 | 53.0 | 0.412 | 0.210 | 0.0 |
| small object far (0.5 m cube at 30 m), n=20 | 6.153 | 3.112 | 65.5 | 5.469 | 3.419 | 59.0 | 5.147 | 7.123 | 52.0 |
| close range (2 m cube at 2.5 m), n=20 | 0.131 | 0.096 | 0.0 | 0.093 | 0.071 | 0.0 | 0.106 | 0.075 | 0.0 |
| minimum, n=4 (0.5 px noise) | 0.946 | 0.674 | 32.0 | 0.455 | 0.285 | 18.5 | 0.395 | 0.226 | 0.0 |

#### Inside solve_pnp_ransac (50 inliers with 1 px noise, plus random outliers)

| case | EPnP rot° | EPnP trans % | EPnP fail % | AP3P rot° | AP3P trans % | AP3P fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| no outliers | 0.208 | 0.110 | 0.0 | 0.206 | 0.111 | 0.0 | 0.170 | 0.105 | 0.0 |
| 25% outliers | 0.187 | 0.120 | 0.0 | 0.177 | 0.113 | 0.0 | 0.160 | 0.098 | 0.0 |
| 50% outliers | 0.239 | 0.144 | 2.5 | 0.178 | 0.105 | 0.0 | 0.213 | 0.144 | 0.0 |
| planar, 25% outliers | 13.913 | 2.349 | 94.5 | 0.662 | 0.287 | 2.0 | 0.268 | 0.153 | 0.0 |
