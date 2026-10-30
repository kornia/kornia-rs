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

**Note:** plain EPnP still has #1181 (wrong pose on noise-free data), which is why it fails in
most accuracy rows and likely why the EPnP RANSAC kernel finds too few inliers on one bench scene. Expect EPnP and
EPnP + LM numbers to change when that is fixed.

---

## Results — 2026-10-22

### Hardware / software

| Field | Value |
|-------|-------|
| Code | #1183 with SQPnP's algebraic objective (`Q_i = A_iᵀA_i`, as OpenCV's `computeOmega`) |
| Machine | MacBook Air (Mac17,3), Apple M5 (4P+6E cores), 16 GB RAM |
| OS | macOS 27.0.1 (arm64) |
| Rust | 1.99.0 (b940084d7 2026-09-28), `--release` |

### Speed

Not measured yet for this code: the machine was under heavy load (load average above 100). Run
`cargo bench -p kornia-3d --bench bench_pnp` on an idle machine and add the tables here.

### Accuracy

Median rotation error (°) and median relative translation error (%) over the scenes that returned a pose, and failure rate (%).

#### Correspondence count (1 px noise, points in a 2 m cube 6 m away)

| n | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2.629 | 1.785 | 42.5 | 0.881 | 0.498 | 11.5 | 0.762 | 0.442 | 0.5 |
| 6 | 32.069 | 9.529 | 80.5 | 0.539 | 0.346 | 5.5 | 0.515 | 0.354 | 0.0 |
| 10 | 36.286 | 8.631 | 86.5 | 0.374 | 0.242 | 1.5 | 0.365 | 0.240 | 0.0 |
| 20 | 45.629 | 7.594 | 83.5 | 0.243 | 0.141 | 0.5 | 0.240 | 0.139 | 0.0 |
| 50 | 36.761 | 6.243 | 82.0 | 0.153 | 0.089 | 0.0 | 0.151 | 0.098 | 0.0 |
| 100 | 41.065 | 7.791 | 81.0 | 0.099 | 0.057 | 0.0 | 0.101 | 0.055 | 0.0 |
| 500 | 31.401 | 7.005 | 82.5 | 0.047 | 0.029 | 0.0 | 0.048 | 0.035 | 0.0 |

#### Pixel noise (20 points in a 2 m cube 6 m away)

| noise px | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 38.669 | 7.001 | 81.5 | 0.000 | 0.000 | 0.5 | 0.000 | 0.000 | 0.0 |
| 0.5 | 42.629 | 8.081 | 87.0 | 0.122 | 0.067 | 1.0 | 0.126 | 0.067 | 0.0 |
| 1 | 48.118 | 6.627 | 84.0 | 0.252 | 0.157 | 0.5 | 0.253 | 0.155 | 0.0 |
| 2 | 46.882 | 7.400 | 81.5 | 0.470 | 0.264 | 0.0 | 0.479 | 0.274 | 0.0 |
| 5 | 37.627 | 6.491 | 83.0 | 1.098 | 0.701 | 0.0 | 1.128 | 0.739 | 0.0 |

#### Harder configurations (1 px noise)

| case | EPnP rot° | EPnP trans % | EPnP fail % | EPnP+LM rot° | EPnP+LM trans % | EPnP+LM fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| planar, n=20 | 78.625 | 23.255 | 99.5 | 0.639 | 0.274 | 24.5 | 0.412 | 0.210 | 0.0 |
| small object far (0.5 m cube at 30 m), n=20 | 34.747 | 10.236 | 91.5 | 8.840 | 8.322 | 81.5 | 5.147 | 7.123 | 52.0 |
| close range (2 m cube at 2.5 m), n=20 | 52.409 | 14.725 | 67.0 | 0.093 | 0.070 | 0.5 | 0.106 | 0.075 | 0.0 |
| minimum, n=4 (0.5 px noise) | 1.009 | 0.806 | 30.0 | 0.409 | 0.255 | 6.0 | 0.395 | 0.226 | 0.0 |

#### Inside solve_pnp_ransac (50 inliers with 1 px noise, plus random outliers)

| case | EPnP rot° | EPnP trans % | EPnP fail % | AP3P rot° | AP3P trans % | AP3P fail % | SQPnP rot° | SQPnP trans % | SQPnP fail % |
|---|---|---|---|---|---|---|---|---|---|
| no outliers | 0.587 | 0.401 | 0.0 | 0.205 | 0.106 | 0.0 | 0.170 | 0.105 | 0.0 |
| 25% outliers | 0.587 | 0.408 | 0.0 | 0.183 | 0.110 | 0.0 | 0.160 | 0.098 | 0.0 |
| 50% outliers | 1.379 | 1.161 | 59.0 | 0.180 | 0.104 | 0.0 | 0.213 | 0.144 | 0.0 |
| planar, 25% outliers | 30.507 | 5.959 | 100.0 | 0.425 | 0.196 | 0.5 | 0.268 | 0.153 | 0.0 |
