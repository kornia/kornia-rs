"""
benchmark_pnp.py
================
Synthetic benchmark comparing kornia_rs.k3d.solve_pnp_ransac against
cv2.solvePnPRansac(SOLVEPNP_AP3P) at varying inlier ratios.

Setup
-----
    pip install opencv-python-headless numpy

    # kornia-rs must be built from your patched branch or installed via:
    pip install kornia-rs

Usage
-----
    python benchmark_pnp.py

    # run a single quick smoke-test (1 trial, fixed seed):
    python benchmark_pnp.py --quick

    # save results to CSV:
    python benchmark_pnp.py --csv results.csv

What it measures
----------------
For each (inlier_ratio, lo_every) combination:
  - Inliers recovered  (vs ground truth)
  - Rotation error     (degrees, vs ground-truth pose)
  - Translation error  (L2, normalised units)
  - Wall-clock time    (seconds)

The synthetic scene
-------------------
N 3D points are drawn uniformly in a cube in front of the camera.
They are projected through a known pose to get clean 2D observations.
Gaussian pixel noise is added to all 2D points.
(1 - inlier_ratio) * N points are replaced with random pixel coordinates
to simulate outliers (wrong matches from a wide-baseline matcher).
"""

import argparse
import csv
import time
import sys
import os
from typing import NamedTuple

import cv2
import numpy as np

pixi_site_packages = "/home/adarsh_gupta/Dev/kornia-rs/.pixi/envs/default/lib/python3.14/site-packages/kornia_rs/"
if os.path.exists(pixi_site_packages):
    sys.path.append(pixi_site_packages)

# ---------------------------------------------------------------------------
# Try to import kornia-rs.  If not installed, the OpenCV-only columns still
# run so you can verify the test harness is working.
# ---------------------------------------------------------------------------
try:
    import kornia_rs
    # If it was imported as a direct layout file, alias k3d safely
    k3d = getattr(kornia_rs, 'k3d', None) or kornia_rs
    KORNIA_AVAILABLE = True
    print("[info] Successfully hooked up kornia_rs binary.")
except ImportError:
    KORNIA_AVAILABLE = False
    print("[warn] kornia_rs not found — only OpenCV columns will run.\n")

# ── Scene parameters ────────────────────────────────────────────────────────

N_POINTS        = 600          # total correspondences per trial
PIXEL_NOISE_STD = 1.5          # Gaussian noise on inlier observations (px)
IMG_W, IMG_H    = 1280, 720    # synthetic image resolution

# Camera intrinsics — generic 50 mm-ish lens
FX, FY = 800.0, 800.0
CX, CY = IMG_W / 2, IMG_H / 2
K = np.array([[FX,  0, CX],
              [ 0, FY, CY],
              [ 0,  0,  1]], dtype=np.float64)

# Ground-truth pose: camera sits 5 units back, slight tilt
R_GT_ROD = np.array([0.05, -0.03, 0.02], dtype=np.float64)   # Rodrigues
R_GT, _  = cv2.Rodrigues(R_GT_ROD)
T_GT     = np.array([[0.1], [-0.05], [5.0]], dtype=np.float64)


# ── Benchmark grid ──────────────────────────────────────────────────────────

INLIER_RATIOS  = [0.10, 0.20, 0.30, 0.50]
LO_EVERY_VALS  = [0, 1, 2, 3]          # 0 = LO disabled (vanilla RANSAC)
N_TRIALS       = 5                      # independent random seeds per cell
MAX_ITERATIONS = 10_000
THRESHOLD_PX   = 6.0
CONFIDENCE     = 0.999


# ── Helpers ─────────────────────────────────────────────────────────────────

class Result(NamedTuple):
    solver:        str
    inlier_ratio:  float
    lo_every:      int        # -1 for OpenCV (not applicable)
    n_inliers:     int
    recall:        float      # recovered / ground-truth inlier count
    rot_err_deg:   float
    trans_err:     float
    wall_sec:      float


def make_scene(inlier_ratio: float, seed: int):
    """Return (world_pts, image_pts, true_inlier_mask)."""
    rng = np.random.default_rng(seed)

    n_inliers  = max(4, int(round(N_POINTS * inlier_ratio)))
    n_outliers = N_POINTS - n_inliers

    # 3-D points in a 4×4×2 box, 2–6 units in front of the camera
    world = rng.uniform([-2, -2, 2], [2, 2, 6],
                        size=(n_inliers, 3)).astype(np.float64)

    # Project through ground-truth pose
    image_clean, _ = cv2.projectPoints(
        world, R_GT_ROD, T_GT, K, distCoeffs=None
    )
    image_clean = image_clean.reshape(-1, 2)

    # Add pixel noise to inlier observations
    image_clean += rng.normal(0, PIXEL_NOISE_STD, image_clean.shape)

    # Random outlier correspondences (wrong matches)
    world_out  = rng.uniform([-2, -2, 2], [2, 2, 6],
                             size=(n_outliers, 3)).astype(np.float64)
    image_out  = rng.uniform([0, 0], [IMG_W, IMG_H],
                             size=(n_outliers, 2)).astype(np.float64)

    world_all  = np.vstack([world, world_out])
    image_all  = np.vstack([image_clean, image_out])
    true_mask  = np.array([True]*n_inliers + [False]*n_outliers)

    # Shuffle so inliers aren't conveniently at the front
    perm = rng.permutation(N_POINTS)
    return world_all[perm], image_all[perm], true_mask[perm]


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    R_rel = R_est @ R_gt.T
    # clamp to [-1, 1] for numerical safety
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_est.ravel() - t_gt.ravel()))


# ── Solvers ──────────────────────────────────────────────────────────────────

def run_opencv(world, image):
    """cv2.solvePnPRansac with AP3P."""
    t0 = time.perf_counter()
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        world.astype(np.float32),
        image.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        iterationsCount=MAX_ITERATIONS,
        reprojectionError=THRESHOLD_PX,
        confidence=CONFIDENCE,
        flags=cv2.SOLVEPNP_AP3P,
    )
    wall = time.perf_counter() - t0
    if not ok or rvec is None:
        return None, wall
    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec, inliers), wall


def run_kornia(world, image, lo_every: int):
    """kornia_rs.k3d.solve_pnp_ransac with configurable lo_every."""
    t0 = time.perf_counter()
    R, t, mask, n_in = k3d.solve_pnp_ransac(
        world,
        image,
        K,
        threshold=THRESHOLD_PX,
        max_iterations=MAX_ITERATIONS,
        confidence=CONFIDENCE,
        lo_every=lo_every,
    )
    wall = time.perf_counter() - t0
    return (R, t, mask, n_in), wall


# ── Main benchmark loop ──────────────────────────────────────────────────────

def run_benchmark(quick: bool = False) -> list[Result]:
    trials   = 1 if quick else N_TRIALS
    ratios   = [0.10] if quick else INLIER_RATIOS
    lo_vals  = [1]    if quick else LO_EVERY_VALS
    results  = []

    total = len(ratios) * (1 + (len(lo_vals) if KORNIA_AVAILABLE else 0)) * trials
    done  = 0

    for ratio in ratios:
        true_n = max(4, int(round(N_POINTS * ratio)))

        for seed_offset in range(trials):
            seed = 42 + seed_offset
            world, image, true_mask = make_scene(ratio, seed)

            # ── OpenCV ──────────────────────────────────────────────────────
            ret, wall = run_opencv(world, image)
            done += 1
            print(f"  [{done}/{total}] opencv  ratio={ratio:.0%} seed={seed}", end="")
            if ret is None:
                print(" → NO MODEL")
                results.append(Result("opencv_ap3p", ratio, -1, 0, 0.0,
                                      180.0, float("inf"), wall))
            else:
                R_est, t_est, cv_inliers = ret
                n_in = len(cv_inliers) if cv_inliers is not None else 0
                recall   = n_in / true_n
                rot_err  = rotation_error_deg(R_est, R_GT)
                t_err    = translation_error(t_est, T_GT)
                print(f" → {n_in} inliers  recall={recall:.2f}"
                      f"  R_err={rot_err:.3f}°  t_err={t_err:.4f}"
                      f"  {wall*1000:.1f} ms")
                results.append(Result("opencv_ap3p", ratio, -1, n_in,
                                      recall, rot_err, t_err, wall))

            # ── kornia-rs ───────────────────────────────────────────────────
            if KORNIA_AVAILABLE:
                for lo in lo_vals:
                    ret_k, wall_k = run_kornia(world, image, lo)
                    done += 1
                    R_k, t_k, mask_k, n_k = ret_k
                    recall_k  = n_k / true_n
                    rot_err_k = rotation_error_deg(R_k, R_GT)
                    t_err_k   = translation_error(t_k, T_GT)
                    label = f"kornia lo={lo}"
                    print(f"  [{done}/{total}] {label:<14} ratio={ratio:.0%} seed={seed}"
                          f" → {n_k} inliers  recall={recall_k:.2f}"
                          f"  R_err={rot_err_k:.3f}°  t_err={t_err_k:.4f}"
                          f"  {wall_k*1000:.1f} ms")
                    results.append(Result(f"kornia_lo{lo}", ratio, lo, n_k,
                                          recall_k, rot_err_k, t_err_k, wall_k))

    return results


def print_summary(results: list[Result]):
    """Aggregate results by (solver, inlier_ratio) and print a table."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        buckets[(r.solver, r.inlier_ratio)].append(r)

    header = (f"{'Solver':<18} {'Ratio':>6} {'Inliers':>8} "
              f"{'Recall':>7} {'R_err°':>8} {'t_err':>8} {'ms':>8}")
    print("\n" + "═"*len(header))
    print(header)
    print("─"*len(header))

    prev_ratio = None
    for (solver, ratio), rows in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0])):
        if prev_ratio is not None and ratio != prev_ratio:
            print("─"*len(header))
        prev_ratio = ratio
        avg = lambda f: np.mean([getattr(r, f) for r in rows])
        print(f"{solver:<18} {ratio:>6.0%} {avg('n_inliers'):>8.1f} "
              f"{avg('recall'):>7.2f} {avg('rot_err_deg'):>8.3f} "
              f"{avg('trans_err'):>8.4f} {avg('wall_sec')*1000:>8.1f}")
    print("═"*len(header))


def save_csv(results: list[Result], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(Result._fields)
        for r in results:
            w.writerow(r)
    print(f"\nResults saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PnP RANSAC benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Single trial, ratio=10%%, lo_every=1 only")
    parser.add_argument("--csv", metavar="FILE",
                        help="Save results to CSV file")
    args = parser.parse_args()

    print(f"Scene: {N_POINTS} points, noise={PIXEL_NOISE_STD}px, "
          f"threshold={THRESHOLD_PX}px, max_iters={MAX_ITERATIONS}\n")

    results = run_benchmark(quick=args.quick)
    print_summary(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()