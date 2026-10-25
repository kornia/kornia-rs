"""Baselines for `bench_voxel.rs` on the same inputs.

Times, when the package is importable (each is skipped otherwise):

* Open3D legacy `geometry.PointCloud`: `voxel_down_sample`,
  `remove_radius_outlier`, `remove_statistical_outlier` (CPU).
* Open3D tensor `t.geometry.PointCloud`: `voxel_down_sample`,
  `remove_radius_outliers`, `remove_statistical_outliers` on CPU and, if
  Open3D was built with CUDA, `CUDA:0`.
* scipy `cKDTree` radius counts (`query_ball_point(return_length=True)`)
  and k-NN mean distances, and a numpy `np.unique` voxel centroid, as
  library-independent references (CPU).

Parameter mapping: kornia's `min_neighbours` excludes the query point.
Open3D's legacy `remove_radius_outlier` keeps points with more than
`nb_points` points in the radius (itself included), so it runs with
`nb_points = min_neighbours`; the tensor `remove_radius_outliers` keeps at
least `nb_points` (itself included), so it runs with `min_neighbours + 1`.
Open3D's `voxel_down_sample` puts the voxel grid at
`min_bound - voxel_size / 2`; the timing does not depend on it.

Usage: [OMP_NUM_THREADS=1] python3 bench_voxel_baselines.py [--rounds N] [--workers W]
`--workers` is the scipy `cKDTree` worker count (-1: all cores); pair
`--workers 1` with `OMP_NUM_THREADS=1` for single-thread Open3D numbers.
Prints best-of-N milliseconds per op and size.
"""

import argparse
import time

import numpy as np

SIZES = [100_000, 1_000_000]
EXTENT_M = 4.0
VOXEL_M = 0.1
RADIUS_M = 0.05
RADIUS_MIN_NEIGHBOURS = 2
STAT_K = 16
STAT_STD_RATIO = 1.0


def points(n):
    """Mirror of `points` in bench_voxel.rs (xorshift64, same constants)."""
    mask64 = (1 << 64) - 1
    s = 0x2545F4914F6CDD1D
    out = np.empty(3 * n, dtype=np.float64)
    two53 = float(1 << 53)
    for i in range(3 * n):
        s ^= (s << 13) & mask64
        s ^= s >> 7
        s ^= (s << 17) & mask64
        out[i] = (s >> 11) / two53 * EXTENT_M
    return out.reshape(n, 3)


def best_ms(fn, rounds):
    fn()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def report(name, n, ms):
    print(f"{name:<44} {n // 1000:>5}k  {ms:9.2f} ms")


def numpy_centroids(p):
    keys = np.floor(p / VOXEL_M).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, inv, p)
    return sums / counts[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=-1)
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        o3d = None
        print("open3d not importable: skipping Open3D baselines")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
        print("scipy not importable: skipping scipy baselines")

    for n in SIZES:
        p = points(n)
        report("numpy_unique_centroid_cpu", n, best_ms(lambda: numpy_centroids(p), args.rounds))

        if cKDTree is not None:

            def radius_counts():
                tree = cKDTree(p)
                c = tree.query_ball_point(p, RADIUS_M, return_length=True, workers=args.workers)
                return np.nonzero(c - 1 >= RADIUS_MIN_NEIGHBOURS)[0]

            def stat_means():
                tree = cKDTree(p)
                d, _ = tree.query(p, k=STAT_K + 1, workers=args.workers)
                m = d[:, 1:].mean(axis=1)
                return np.nonzero(m <= m.mean() + STAT_STD_RATIO * m.std(ddof=1))[0]

            report("scipy_ckdtree_radius_outliers_cpu", n, best_ms(radius_counts, args.rounds))
            report("scipy_ckdtree_statistical_outliers_cpu", n, best_ms(stat_means, args.rounds))

        if o3d is None:
            continue
        legacy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        report(
            "open3d_legacy_voxel_down_sample_cpu",
            n,
            best_ms(lambda: legacy.voxel_down_sample(VOXEL_M), args.rounds),
        )
        report(
            "open3d_legacy_remove_radius_outlier_cpu",
            n,
            best_ms(
                lambda: legacy.remove_radius_outlier(RADIUS_MIN_NEIGHBOURS, RADIUS_M),
                args.rounds,
            ),
        )
        report(
            "open3d_legacy_remove_statistical_outlier_cpu",
            n,
            best_ms(
                lambda: legacy.remove_statistical_outlier(STAT_K, STAT_STD_RATIO),
                args.rounds,
            ),
        )

        devices = [("cpu", o3d.core.Device("CPU:0"))]
        if o3d.core.cuda.is_available():
            devices.append(("cuda", o3d.core.Device("CUDA:0")))
        for tag, dev in devices:
            t = o3d.t.geometry.PointCloud(dev)
            t.point.positions = o3d.core.Tensor(p, o3d.core.float64, dev)

            def sync():
                if tag == "cuda":
                    o3d.core.cuda.synchronize()

            def down():
                t.voxel_down_sample(VOXEL_M)
                sync()

            def radius():
                t.remove_radius_outliers(RADIUS_MIN_NEIGHBOURS + 1, RADIUS_M)
                sync()

            def stat():
                t.remove_statistical_outliers(STAT_K, STAT_STD_RATIO)
                sync()

            report(f"open3d_tensor_voxel_down_sample_{tag}", n, best_ms(down, args.rounds))
            report(f"open3d_tensor_remove_radius_outliers_{tag}", n, best_ms(radius, args.rounds))
            if tag == "cpu":
                report(
                    f"open3d_tensor_remove_statistical_outliers_{tag}",
                    n,
                    best_ms(stat, args.rounds),
                )


if __name__ == "__main__":
    main()
