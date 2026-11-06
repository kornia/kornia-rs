"""Compare the affine Schur prototype to oblique-ug's alternating aerial BA.

Inputs are a local NPZ of pre-filtering tracks (cam_idx, pt_idx, xy, n_views).
Run from kornia-rs with the prototype wheel installed:
  OPENBLAS_NUM_THREADS=1 python scripts/benchmark_affine_ba.py \
    --oblique-root /path/to/oblique-ug --tracks /path/to/tracks.npz
No private captures are included. Timings include Python/Rust copies and exclude initialization.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import kornia_rs
import numpy as np
from scipy.linalg import qr
from kornia_rs import k3d


def prepare(A, X, tr):
    registered = np.flatnonzero(np.isfinite(A[:, 0, 0]))
    use = np.isin(tr.cam_idx, registered) & np.isfinite(X[tr.pt_idx]).all(1)
    tracks, old = tr.subset(use)
    x = np.ascontiguousarray(X[old], dtype=np.float64)
    if len(x) < 4:
        raise ValueError("Affine gauge needs four noncoplanar points")
    xn = x - x.mean(0)
    xn /= np.sqrt(np.mean(xn * xn))
    _, _, pivot = qr(np.c_[xn, np.ones(len(x))].T, pivoting=True)
    fixed = pivot[:4].tolist()
    lookup = np.full(len(A), -1, dtype=np.int64)
    lookup[registered] = np.arange(len(registered))
    return (
        (
            np.ascontiguousarray(A[registered], dtype=np.float64),
            x,
            np.ascontiguousarray(lookup[tracks.cam_idx]),
            np.ascontiguousarray(tracks.pt_idx, dtype=np.int64),
            np.ascontiguousarray(tracks.xy, dtype=np.float64),
            fixed,
        ),
        registered,
        old,
    )


def adjust_oblique(A, X, tr, f_scale=2.0, max_iters=300):
    """Drop-in experimental adapter for affine.bundle_adjust; unregistered rows stay unchanged."""
    args, registered, old = prepare(A, X, tr)
    result = k3d.bundle_adjust_affine(
        *args, robust_scale=f_scale, max_iterations=max_iters
    )
    cameras, points = A.copy(), X.copy()
    cameras[registered] = result["cameras"]
    points[old] = result["points"]
    return cameras, points, result["iterations"]


def objective(a, x, ci, pi, xy, scale=2.0):
    r = np.einsum("nij,nj->ni", a[ci, :, :3], x[pi]) + a[ci, :, 3] - xy
    return float(0.5 * scale**2 * np.log1p(np.sum(r * r, axis=1) / scale**2).sum())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--oblique-root", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-iters", type=int, default=300)
    parser.add_argument("--reconstruct", choices=["weak-perspective", "plan-oblique"])
    parser.add_argument(
        "--directions",
        nargs="+",
        help="One heading per input view; use none for unknown",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.oblique_root.resolve()))
    from oblique_ug import affine
    from oblique_ug.tracks import Tracks

    with np.load(args.tracks) as data:
        tr = Tracks(data["cam_idx"], data["pt_idx"], data["xy"], int(data["n_views"]))
    start = time.perf_counter()
    a = affine.initialize(tr)
    x = affine.triangulate(a, tr)
    inputs, registered, old = prepare(a, x, tr)
    init_seconds = time.perf_counter() - start
    # Both solvers see exactly the same active cameras, points and observations.
    aa, xx, ci, pi, xy, fixed = inputs
    tracks = Tracks(ci, pi, xy, len(aa))
    report = {
        "points": len(xx),
        "observations": len(xy),
        "cameras": len(aa),
        "initialization_seconds": init_seconds,
        "fixed_points": fixed,
        "initial_cost": objective(aa, xx, ci, pi, xy),
    }
    report["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    report["source_dirty"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    )
    report["extension_path"] = kornia_rs.kornia_rs.__file__
    report["python"] = sys.version
    report["numpy_version"] = np.__version__
    report["thread_env"] = {
        k: os.environ.get(k)
        for k in ("OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS")
    }
    start = time.perf_counter()
    k3d.bundle_adjust_affine(*inputs, max_iterations=1)
    report["first_rust_call_one_iteration_seconds"] = time.perf_counter() - start
    affine.bundle_adjust(aa, xx, tracks, max_iters=1)
    baseline_times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        ab, xb, it = affine.bundle_adjust(
            aa, xx, tracks, max_iters=args.reference_iters
        )
        baseline_times.append(time.perf_counter() - start)
    baseline_cost = objective(ab, xb, ci, pi, xy)
    report["numpy"] = {
        "seconds": baseline_times,
        "median_seconds": float(np.median(baseline_times)),
        "iterations": it,
        "cost": baseline_cost,
    }
    print(json.dumps(report), flush=True)
    for cap in (5, 10, 20, 50, 100):
        elapsed = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = k3d.bundle_adjust_affine(*inputs, max_iterations=cap)
            elapsed.append(time.perf_counter() - start)
        row = {k: v for k, v in result.items() if k not in ("cameras", "points")}
        row.update(
            seconds=elapsed,
            median_seconds=float(np.median(elapsed)),
            speedup=float(np.median(baseline_times) / np.median(elapsed)),
            cost_ratio=result["cost"] / baseline_cost,
        )
        # Align the baseline to the prototype by least-squares affine registration.
        transform = np.linalg.lstsq(
            np.c_[xb, np.ones(len(xb))], result["points"], rcond=None
        )[0]
        delta = np.linalg.norm(
            np.c_[xb, np.ones(len(xb))] @ transform - result["points"], axis=1
        )
        row["aligned_point_delta_median"] = float(np.median(delta))
        row["aligned_point_delta_p95"] = float(np.quantile(delta, 0.95))
        row["scene_rms_radius"] = float(
            np.sqrt(np.mean(np.sum((xx - xx.mean(0)) ** 2, axis=1)))
        )
        report[f"rust_{cap}"] = row
        print(json.dumps({f"rust_{cap}": row}), flush=True)
    if args.reconstruct:
        directions = (
            [None if d == "none" else d for d in args.directions]
            if args.directions
            else [None] * tr.n_views
        )
        if len(directions) != tr.n_views:
            raise ValueError("directions must match the number of input views")
        report["reconstruction"] = {}
        for name, fn in [("numpy", affine.bundle_adjust), ("rust", adjust_oblique)]:
            elapsed = []
            for _ in range(args.repeats):
                with patch.object(affine, "bundle_adjust", fn):
                    start = time.perf_counter()
                    rec = affine.reconstruct_affine(
                        tr, args.reconstruct, directions, 45.0, None
                    )
                    elapsed.append(time.perf_counter() - start)
            row = dict(
                seconds=elapsed,
                median_seconds=float(np.median(elapsed)),
                points=len(rec.X),
                observations=len(rec.tracks.xy),
                median_residual=float(np.nanmedian(rec.residuals())),
                p95_residual=float(np.nanquantile(rec.residuals(), 0.95)),
                log=rec.log,
            )
            report["reconstruction"][name] = row
            print(json.dumps({"reconstruction_" + name: row}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
