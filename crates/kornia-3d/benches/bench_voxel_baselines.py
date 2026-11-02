"""Baselines for `bench_voxel.rs` on the same inputs.

Times, when the package is importable (each is skipped otherwise):

* Open3D legacy `geometry.PointCloud`: `voxel_down_sample` (CPU).
* Open3D tensor `t.geometry.PointCloud`: `voxel_down_sample` on CPU and,
  if Open3D was built with CUDA, `CUDA:0`.
* A numpy `np.unique` voxel centroid, as a library-independent reference
  (CPU).

Open3D's `voxel_down_sample` puts the voxel grid at
`min_bound - voxel_size / 2`; the timing does not depend on it.

Usage: [OMP_NUM_THREADS=1] python3 bench_voxel_baselines.py [--rounds N]
       python3 bench_voxel_baselines.py --write-fixtures DIR
Pass `OMP_NUM_THREADS=1` for single-thread Open3D numbers.
Prints best-of-N milliseconds per op and size. `--write-fixtures` (needs
Open3D) writes the parity fixture read by the kornia-3d tests,
`voxel_downsample_open3d.bin`, into DIR (the repository's `tests/data`)
instead of timing anything.
"""

import argparse
import pathlib
import struct
import time

import numpy as np

SIZES = [100_000, 1_000_000]
EXTENT_M = 4.0
VOXEL_M = 0.1


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


def triplets_le(p):
    """`u64` count, then the `x, y, z` triplets as little-endian `f64`."""
    p = np.ascontiguousarray(p, dtype="<f8")
    return struct.pack("<Q", len(p)) + p.tobytes()


def write_fixtures(out_dir):
    """A small synthetic cloud and Open3D's result on it (legacy API)."""
    import open3d as o3d

    out = pathlib.Path(out_dir)
    rng = np.random.default_rng(20240607)

    # Voxel downsampling: uniform points in a unit cube, a power-of-two
    # voxel so the voxel boundaries are exact.
    voxel = 0.125
    p = rng.random((1000, 3))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    down = np.asarray(pc.voxel_down_sample(voxel).points)
    origin = p.min(axis=0) - voxel / 2
    blob = struct.pack("<d", voxel) + np.asarray(origin, dtype="<f8").tobytes()
    blob += triplets_le(p) + triplets_le(down)
    (out / "voxel_downsample_open3d.bin").write_bytes(blob)
    print(f"wrote {len(p)} -> {len(down)} voxels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--write-fixtures", metavar="DIR")
    args = ap.parse_args()
    if args.write_fixtures:
        write_fixtures(args.write_fixtures)
        return

    try:
        import open3d as o3d
    except ImportError:
        o3d = None
        print("open3d not importable: skipping Open3D baselines")

    for n in SIZES:
        p = points(n)
        report("numpy_unique_centroid_cpu", n, best_ms(lambda: numpy_centroids(p), args.rounds))

        if o3d is None:
            continue
        legacy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        report(
            "open3d_legacy_voxel_down_sample_cpu",
            n,
            best_ms(lambda: legacy.voxel_down_sample(VOXEL_M), args.rounds),
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

            report(f"open3d_tensor_voxel_down_sample_{tag}", n, best_ms(down, args.rounds))


if __name__ == "__main__":
    main()
