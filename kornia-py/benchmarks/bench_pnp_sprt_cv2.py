"""Benchmark kornia-rs PnP (AP3P / EPnP, with and without SPRT) vs OpenCV solvePnPRansac.

Synthetic scene: fixed `total = 600` 3D-2D correspondences generated from a
known pose, with the inlier count varying as a fraction of `total` at
10/20/30/50 % outlier ratios. Each measurement reports the inlier count,
recall, rotation error in degrees, translation error in metres, and
wall-clock time in milliseconds.

The output is grouped by inlier ratio (one block per ratio, with all
solver variants stacked within the block) — matching the reference
table format.

LO levels: 0 (plain RANSAC), 1, 2, 3 (LO every 1/2/3 accepted hypotheses).
SPRT: enable / disable Wald's Sequential Probability Ratio Test.

Run with:
    /tmp/pnp_bench_venv/bin/python benchmarks/bench_pnp_sprt_cv2.py
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

import kornia_rs
from kornia_rs.k3d import PnPSolverMethod, solve_pnp_ransac


# ----------------------------------------------------------------------------
# Scene generation
# ----------------------------------------------------------------------------

def make_scene(n_inliers: int, n_outliers: int, seed: int):
    """Generate a synthetic 3D-2D scene with ground-truth pose.

    Returns ``(world, image, K, R_gt, t_gt)`` as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    fx = fy = 800.0
    cx, cy = 640.0, 480.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Ground-truth rotation: a small generic orientation.
    ax, ay, az = np.deg2rad([10.0, -15.0, 30.0])
    sx, cx_ = np.sin(ax), np.cos(ax)
    sy, cy_ = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_gt = Rz @ Ry @ Rx
    t_gt = np.array([0.7, -0.4, 5.0], dtype=np.float64)

    # Inlier world points in a 0.6 m cube.
    world_in = rng.uniform(-0.3, 0.3, size=(n_inliers, 3)).astype(np.float64)
    world_in[:, 2] = rng.uniform(0.5, 1.5, size=n_inliers)

    # Project inliers with light noise.
    pc_in = world_in @ R_gt.T + t_gt
    image_in = np.column_stack(
        [
            fx * pc_in[:, 0] / pc_in[:, 2] + cx,
            fy * pc_in[:, 1] / pc_in[:, 2] + cy,
        ]
    ) + rng.normal(scale=0.5, size=(n_inliers, 2))

    if n_outliers == 0:
        return world_in, image_in, K, R_gt, t_gt

    # Outliers: random world points with random pixel projections that are
    # very unlikely to be consistent with any rigid motion.
    world_out = np.column_stack(
        [
            rng.uniform(-0.5, 0.5, size=n_outliers),
            rng.uniform(-0.5, 0.5, size=n_outliers),
            rng.uniform(0.5, 1.5, size=n_outliers),
        ]
    )
    image_out = np.column_stack(
        [
            rng.uniform(0, 1280, size=n_outliers),
            rng.uniform(0, 960, size=n_outliers),
        ]
    )

    world = np.vstack([world_in, world_out])
    image = np.vstack([image_in, image_out])
    return world, image, K, R_gt, t_gt


# ----------------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------------

def pose_error(R_est, t_est, R_gt, t_gt):
    r_err = np.arccos(
        np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    )
    r_err_deg = float(np.degrees(r_err))
    t_err = float(np.linalg.norm(np.asarray(t_est).ravel() - t_gt))
    return r_err_deg, t_err


def inlier_count(mask):
    if mask is None:
        return 0
    m = np.asarray(mask)
    if m.ndim == 1:
        return int(m.sum())
    return int(np.sum(m > 0))


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------

@dataclass
class Result:
    method: str
    outlier_ratio: float
    n_inliers: int
    n_outliers: int
    inliers_found: int
    total: int
    recall: float
    r_error_deg: float
    t_error: float
    time_ms: float
    success: bool


def run_kornia(world, image, K, method, lo_every, use_sprt, seed, inlier_ratio):
    R, t, mask, _ = solve_pnp_ransac(
        world,
        image,
        K,
        method=method,
        threshold=4.0,
        max_iterations=1000,
        confidence=0.999,
        lo_every=lo_every,
        seed=seed,
        use_sprt=use_sprt,
        sprt_epsilon=inlier_ratio,
        sprt_delta=0.01,
    )
    return R, t, mask


def run_opencv(world, image, K, method_flag):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        world.astype(np.float32),
        image.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        iterationsCount=1000,
        reprojectionError=4.0,
        confidence=0.999,
        flags=method_flag,
    )
    if not ok or rvec is None:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, (inliers if inliers is not None else np.empty((0, 1)))


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


def evaluate_kornia(world, image, K, R_gt, t_gt, n_inliers, n_outliers,
                    method, lo_every, use_sprt, seed, inlier_ratio):
    total = n_inliers + n_outliers
    R, t, mask = run_kornia(world, image, K, method, lo_every, use_sprt, seed, inlier_ratio)

    if R is None:
        return Result(
            method=method_name(method, lo_every, use_sprt),
            outlier_ratio=inlier_ratio,
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            inliers_found=0,
            total=total,
            recall=0.0,
            r_error_deg=float("nan"),
            t_error=float("nan"),
            time_ms=0.0,
            success=False,
        )
    # Pass inlier_ratio to the timed call too
    _, elapsed_ms = time_call(run_kornia, world, image, K, method, lo_every, use_sprt, seed, inlier_ratio)
    r_err, t_err = pose_error(R, t, R_gt, t_gt)
    n_inliers_found = inlier_count(mask)
    return Result(
        method=method_name(method, lo_every, use_sprt),
        outlier_ratio=inlier_ratio,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        inliers_found=n_inliers_found,
        total=total,
        recall=n_inliers_found / total,
        r_error_deg=r_err,
        t_error=t_err,
        time_ms=elapsed_ms,
        success=True,
    )


def evaluate_cv2(world, image, K, R_gt, t_gt, n_inliers, n_outliers, method_flag):
    total = n_inliers + n_outliers
    inlier_ratio = n_inliers / total

    t0 = time.perf_counter()
    R, t, mask = run_opencv(world, image, K, method_flag)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if R is None:
        return Result(
            method=cv2_method_name(method_flag),
            outlier_ratio=inlier_ratio,
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            inliers_found=0,
            total=total,
            recall=0.0,
            r_error_deg=float("nan"),
            t_error=float("nan"),
            time_ms=0.0,
            success=False,
        )

    r_err, t_err = pose_error(R, t, R_gt, t_gt)
    return Result(
        method=cv2_method_name(method_flag),
        outlier_ratio=inlier_ratio,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        inliers_found=inlier_count(mask),
        total=total,
        recall=inlier_count(mask) / total,
        r_error_deg=r_err,
        t_error=t_err,
        time_ms=elapsed_ms,
        success=True,
    )


def method_name(method, lo_every, use_sprt):
    base = "ap3p" if method == PnPSolverMethod.AP3P else "epnp"
    flags = []
    if lo_every > 0:
        flags.append(f"lo{lo_every}")
    if use_sprt:
        flags.append("sprt")
    return f"k_{base}" + (("_" + "_".join(flags)) if flags else "")


def cv2_method_name(method_flag):
    if method_flag == cv2.SOLVEPNP_AP3P:
        return "opencv_ap3p"
    if method_flag == cv2.SOLVEPNP_EPNP:
        return "opencv_epnp"
    if method_flag == cv2.SOLVEPNP_SQPNP:
        return "opencv_sqpnp"
    return f"opencv_{method_flag}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=600,
                        help="fixed total correspondences per scene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/adarsh_gupta/Dev/kornia-rs/kornia-py/benchmarks/bench_pnp_sprt_results.json"),
    )
    args = parser.parse_args()

    total = args.total
    ratios = [0.10, 0.20, 0.30, 0.50]
    seeds = [args.seed + i for i in range(3)]

    # LO levels: 0 (plain), 1, 2, 3; SPRT on/off; AP3P / EPnP.
    settings = []
    for method in (PnPSolverMethod.EPnP, PnPSolverMethod.AP3P):
        for lo_every in (0, 1, 2, 3):
            for use_sprt in (False, True):
                settings.append((method, lo_every, use_sprt))

    cv2_flags = [
        (cv2.SOLVEPNP_EPNP, "epnp"),
        (cv2.SOLVEPNP_AP3P, "ap3p"),
    ]
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        cv2_flags.append((cv2.SOLVEPNP_SQPNP, "sqpnp"))

    results = []
    for ratio in ratios:
        n_inliers = int(round(total * ratio))
        n_outliers = total - n_inliers
        for seed in seeds:
            world, image, K, R_gt, t_gt = make_scene(n_inliers, n_outliers, seed)
            for method, lo_every, use_sprt in settings:
                results.append(
                    asdict(
                        evaluate_kornia(
                            world, image, K, R_gt, t_gt,
                            n_inliers, n_outliers,
                            method, lo_every, use_sprt, seed, ratio
                        )
                    )
                )
            for flag, name in cv2_flags:
                results.append(
                    asdict(
                        evaluate_cv2(
                            world, image, K, R_gt, t_gt,
                            n_inliers, n_outliers, flag,
                        )
                    )
                )

    # Aggregate by (method, outlier_ratio): mean and std.
    agg = {}
    for r in results:
        key = (r["method"], round(r["outlier_ratio"], 2))
        agg.setdefault(key, []).append(r)

    table = []
    for (method, ratio), runs in sorted(agg.items()):
        succ = [r for r in runs if r["success"]]
        inl = [r["inliers_found"] for r in succ]
        rec = [r["recall"] for r in succ]
        r_errs = [r["r_error_deg"] for r in succ]
        t_errs = [r["t_error"] for r in succ]
        tms = [r["time_ms"] for r in succ]
        table.append({
            "method": method,
            "outlier_ratio": ratio,
            "total": runs[0]["total"] if runs else 0,
            "inliers_mean": float(np.mean(inl)) if inl else 0.0,
            "recall_mean": float(np.mean(rec)) if rec else 0.0,
            "r_error_deg_mean": float(np.mean(r_errs)) if r_errs else float("nan"),
            "r_error_deg_std": float(np.std(r_errs)) if r_errs else 0.0,
            "t_error_mean": float(np.mean(t_errs)) if t_errs else float("nan"),
            "t_error_std": float(np.std(t_errs)) if t_errs else 0.0,
            "time_ms_mean": float(np.mean(tms)) if tms else 0.0,
            "time_ms_std": float(np.std(tms)) if tms else 0.0,
            "n_runs": len(runs),
        })

    # Print in the user's reference format: grouped by inlier ratio,
    # one header per ratio block, methods stacked within the block.
    header = (
        f"{'Solver':<16} {'Ratio':>6} {'Inliers/Tot':>14} "
        f"{'Recall':>7} {'R_err°':>9} {'t_err':>9} {'ms':>9}"
    )
    sep = "─" * 78

    # Group rows by ratio, then order within each ratio:
    #   k_ap3p_lo0, k_ap3p_lo1, ... k_ap3p_lo3_sprt
    #   k_epnp_lo0, ...
    #   opencv_ap3p, opencv_epnp, opencv_sqpnp
    within_ratio_order = {
        "k_ap3p_lo0": 0,
        "k_ap3p_lo1": 1,
        "k_ap3p_lo2": 2,
        "k_ap3p_lo3": 3,
        "k_ap3p_lo1_sprt": 4,
        "k_ap3p_lo2_sprt": 5,
        "k_ap3p_lo3_sprt": 6,
        "k_ap3p_sprt": 7,
        "k_epnp_lo0": 8,
        "k_epnp_lo1": 9,
        "k_epnp_lo2": 10,
        "k_epnp_lo3": 11,
        "k_epnp_lo1_sprt": 12,
        "k_epnp_lo2_sprt": 13,
        "k_epnp_lo3_sprt": 14,
        "k_epnp_sprt": 15,
        "opencv_ap3p": 16,
        "opencv_epnp": 17,
        "opencv_sqpnp": 18,
    }
    rows_by_ratio = {}
    for row in table:
        rows_by_ratio.setdefault(row["outlier_ratio"], []).append(row)
    for ratio in sorted(rows_by_ratio):
        rows = sorted(
            rows_by_ratio[ratio],
            key=lambda r: within_ratio_order.get(r["method"], 99),
        )
        print(sep)
        print(f"  outlier ratio = {ratio*100:.0f}%  (total = {rows[0]['total']})")
        print(header)
        for row in rows:
            print(
                f"{row['method']:<16} {row['outlier_ratio']*100:>5.0f}% "
                f"{row['inliers_mean']:>7.1f}/{row['total']:<5} "
                f"{row['recall_mean']:>7.3f} "
                f"{row['r_error_deg_mean']:>9.3f} {row['t_error_mean']:>9.4f} "
                f"{row['time_ms_mean']:>9.2f}"
            )
    print(sep)

    args.out.write_text(json.dumps({"raw": results, "summary": table}, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
