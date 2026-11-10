"""Refine a generated affine scene; no images or downloaded data are required.

Run after building a release wheel:
    python kornia-py/examples/affine_bundle_adjustment.py --points 10000
Optional small independent reference (requires SciPy):
    OPENBLAS_NUM_THREADS=1 python kornia-py/examples/affine_bundle_adjustment.py --points 24 --scipy

Four tetrahedron vertices fix the affine gauge. The remaining points form a
synthetic scene with four views, pixel noise and occasional large outliers.
This is an unconstrained affine reconstruction, not metric camera calibration.
"""

import argparse
import time

import numpy as np
import kornia_rs
from kornia_rs import k3d


def scene(count):
    rng = np.random.default_rng(17)
    truth = rng.uniform(-2.0, 2.0, (count, 3))
    truth[:4] = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    cameras = []
    for angle in np.arange(4) * 1.5:
        c, s = np.cos(angle), np.sin(angle)
        cameras.append([[3 * c, 3 * s, 0, 100], [-2 * s, 2 * c, -2, 200]])
    cameras = np.array(cameras, dtype=np.float64)
    ci = np.repeat(np.arange(4, dtype=np.int64), count)
    pi = np.tile(np.arange(count, dtype=np.int64), 4)
    pixels = np.einsum("nij,nj->ni", cameras[ci, :, :3], truth[pi]) + cameras[ci, :, 3]
    pixels += rng.normal(0, 0.01, pixels.shape)
    # Anchors retain accurate observations; corrupt a few free-point measurements.
    outliers = np.flatnonzero(pi >= 4)[::31]
    pixels[outliers] += [3, -2]
    points = truth.copy()
    points[4:] += rng.normal(0, 0.05, points[4:].shape)
    cameras[:, 0, 2] += 0.03
    cameras[:, 1, 3] += 0.2
    return cameras, points, ci, pi, pixels, truth


def scipy_reference(cameras, points, ci, pi, pixels):
    from scipy.optimize import least_squares

    ncam = cameras.size

    def residual(q):
        a = q[:ncam].reshape(cameras.shape)
        x = np.concatenate([points[:4], q[ncam:].reshape(-1, 3)])
        r = np.einsum("nij,nj->ni", a[ci, :, :3], x[pi]) + a[ci, :, 3] - pixels
        s2 = np.sum(r * r, axis=1) / 4.0
        # Transform residuals so linear least_squares has the same vector-norm
        # Cauchy objective. SciPy loss="cauchy" would robustify each coordinate.
        factor = np.sqrt(
            np.divide(np.log1p(s2), s2, out=np.ones_like(s2), where=s2 > 0)
        )
        return (r * factor[:, None]).ravel()

    return least_squares(
        residual,
        np.r_[cameras.ravel(), points[4:].ravel()],
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument(
        "--scipy",
        action="store_true",
        help="dense finite-difference reference; use a small scene",
    )
    args = parser.parse_args()
    if args.points < 5:
        parser.error("--points must be at least 5")
    a, x, ci, pi, xy, truth = scene(args.points)
    print(f"Extension: {kornia_rs.__file__}")
    print(f"Synthetic scene: 4 cameras, {len(x)} points, {len(xy)} observations")
    # Warm up the extension; report the full second call including array copies.
    k3d.bundle_adjust_affine(a, x, ci, pi, xy, [0, 1, 2, 3], max_iterations=1)
    start = time.perf_counter()
    result = k3d.bundle_adjust_affine(
        a, x, ci, pi, xy, [0, 1, 2, 3], cost_tolerance=1e-10
    )
    elapsed = time.perf_counter() - start
    print(f"Rust: {elapsed:.6f} s; cost {result.initial_cost:.9g} -> {result.cost:.9g}")
    print(
        f"Iterations: {result.iterations}; accepted: {result.accepted_steps}; converged: {result.converged}"
    )
    error = np.linalg.norm(result.points - truth, axis=1)
    print(f"Median point error: {np.median(error):.6g} world units")
    np.testing.assert_array_equal(result.points[:4], x[:4])
    assert result.cost < result.initial_cost
    if args.scipy:
        # Warm import before timing, as for the Rust extension above.
        import scipy.optimize  # noqa: F401

        start = time.perf_counter()
        reference = scipy_reference(a, x, ci, pi, xy)
        reference_elapsed = time.perf_counter() - start
        print(
            f"SciPy dense finite differences: {reference_elapsed:.6f} s; cost {reference.cost:.9g}; success: {reference.success}"
        )
        print(
            "These methods use different linear solvers and stopping rules; timings describe only this generated scene."
        )
        assert reference.success
        np.testing.assert_allclose(result.cost, reference.cost, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    main()
