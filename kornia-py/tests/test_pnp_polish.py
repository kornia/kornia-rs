"""Pins the terminal LM polish of `kornia_rs.k3d.solve_pnp_ransac`.

The binding runs generic RANSAC, then refines the pose with LM on the
RANSAC inliers and returns the refined pose with the RANSAC mask (the pre-LM
mask). These tests record that behaviour so a later change to it is
deliberate (kornia-rs#1171, phase B).

The generic driver's final refit already runs LM (EPnP with `refine_lm`) once
there are more inliers than a minimal sample, so on a typical scene the
binding's polish starts at a minimum and changes little. Only the four-point
EPnP test isolates the polish. The pre-LM mask is documented, not pinned:
pinning it needs a point whose error crosses the threshold under LM, and that
depends on the ISA's float rounding.
"""

import numpy as np
import pytest

from kornia_rs.k3d import PnPSolverMethod, solve_pnp_ransac

K = np.array([[800.0, 0, 640.0], [0, 800.0, 480.0], [0, 0, 1]], dtype=np.float64)


def _rotate(R, axis, angle):
    k = np.zeros(3)
    k[axis] = 1.0
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx) @ R


def _project(world, R, t):
    pc = world @ R.T + t
    return pc[:, :2] / pc[:, 2:] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]


def _scene(seed: int, n: int = 80, n_outliers: int = 20, noise: float = 0.7):
    rng = np.random.default_rng(seed)
    angle = np.deg2rad(12.0)
    R = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    t = np.array([0.2, -0.1, 4.0])
    world = rng.uniform(-1.0, 1.0, size=(n, 3))
    image = _project(world, R, t)
    image += rng.normal(scale=noise, size=image.shape)
    outliers = rng.choice(n, size=n_outliers, replace=False)
    image[outliers] = rng.uniform([0, 0], [1280, 960], size=(n_outliers, 2))
    return world, image


def _reprojection(world, image, R, t):
    return np.linalg.norm(_project(world, R, t) - image, axis=1)


def _cost(world, image, R, t):
    return float(np.sum(_reprojection(world, image, R, t) ** 2))


def _minimum_cost(world, image, R, t, iterations=50):
    """Reprojection cost at the minimum nearest `(R, t)`, by float64 Gauss-Newton."""

    def step(R_, t_, d):
        angle = np.linalg.norm(d[:3])
        if angle > 0:
            k = d[:3] / angle
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_ = (np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx) @ R_
        return R_, t_ + d[3:]

    def residuals(R_, t_):
        return (_project(world, R_, t_) - image).ravel()

    for _ in range(iterations):
        r = residuals(R, t)
        J = np.empty((r.size, 6))
        for j in range(6):
            d = np.zeros(6)
            d[j] = 1e-7
            J[:, j] = (residuals(*step(R, t, d)) - residuals(*step(R, t, -d))) / 2e-7
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]
        R, t = step(R, t, delta)
        if np.linalg.norm(delta) < 1e-12:
            break
    return _cost(world, image, R, t)


def test_four_point_epnp_pose_is_polished_by_the_binding():
    # The driver refits only on more inliers than its four-point EPnP sample,
    # so with exactly four inliers the binding's LM is the only non-linear
    # step. The bare minimal EPnP pose costs about 6.7x the minimum here.
    world, image = _scene(seed=1, n=4, n_outliers=0, noise=1.0)
    R, t, mask, count = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.EPnP, threshold=20.0, seed=1
    )
    assert count == 4 and mask.all()
    # The LM runs in f32 for at most 20 iterations, so it stops within about
    # 1.5% of the float64 minimum.
    assert _cost(world, image, R, t) <= 1.1 * _minimum_cost(world, image, R, t)


@pytest.mark.parametrize("method", [PnPSolverMethod.EPnP, PnPSolverMethod.AP3P])
def test_returned_pose_is_a_local_minimum_on_its_inliers(method):
    # Either the driver's LM refit or the binding's polish puts it there; this
    # test cannot tell which.
    world, image = _scene(seed=3)
    threshold = 4.0
    R, t, mask, count = solve_pnp_ransac(world, image, K, method=method, threshold=threshold, seed=11)
    inliers = mask.astype(bool)
    assert count == inliers.sum() >= 50

    def cost(R_, t_):
        return _cost(world[inliers], image[inliers], R_, t_)

    # No small rotation or translation lowers the inliers' reprojection error
    # (the LM runs in f32, hence the relative tolerance).
    base = cost(R, t)
    for axis in range(3):
        for step in (-2e-4, 2e-4):
            assert cost(_rotate(R, axis, step), t) >= base * (1 - 1e-4)
            dt = np.zeros(3)
            dt[axis] = step * 0.1
            assert cost(R, t + dt) >= base * (1 - 1e-4)


def test_mask_agrees_with_the_returned_pose_away_from_the_threshold():
    # The mask is classified before the LM polish, so it may disagree with
    # the returned pose for points close to the threshold. A mask recomputed
    # from the polished pose would pass too.
    world, image = _scene(seed=5)
    threshold = 4.0
    R, t, mask, count = solve_pnp_ransac(world, image, K, threshold=threshold, seed=2)
    errors = _reprojection(world, image, R, t)
    clear = (errors < 0.8 * threshold) | (errors > 1.25 * threshold)
    assert np.array_equal(mask.astype(bool)[clear], errors[clear] < threshold)
    assert count == int(mask.sum())
