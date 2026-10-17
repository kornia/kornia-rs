"""Pins the terminal LM polish of `kornia_rs.k3d.solve_pnp_ransac`.

The binding runs generic RANSAC, then refines the pose with LM on the
RANSAC inliers and returns the refined pose with the RANSAC mask (the pre-LM
mask). These tests record that behaviour so a later change to it is
deliberate (kornia-rs#1171, phase B).
"""

import numpy as np
import pytest

from kornia_rs.k3d import PnPSolverMethod, solve_pnp_ransac


def _scene(seed: int):
    rng = np.random.default_rng(seed)
    K = np.array([[800.0, 0, 640.0], [0, 800.0, 480.0], [0, 0, 1]], dtype=np.float64)
    angle = np.deg2rad(12.0)
    R = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    t = np.array([0.2, -0.1, 4.0])
    world = rng.uniform(-1.0, 1.0, size=(80, 3))
    pc = world @ R.T + t
    image = pc[:, :2] / pc[:, 2:] * 800.0 + np.array([640.0, 480.0])
    image += rng.normal(scale=0.7, size=image.shape)
    outliers = rng.choice(len(world), size=20, replace=False)
    image[outliers] = rng.uniform([0, 0], [1280, 960], size=(20, 2))
    return world, image, K


def _reprojection(world, image, K, R, t):
    pc = world @ R.T + t
    proj = pc[:, :2] / pc[:, 2:] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return np.linalg.norm(proj - image, axis=1)


def _rotate(R, axis, angle):
    k = np.zeros(3)
    k[axis] = 1.0
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx) @ R


@pytest.mark.parametrize("method", [PnPSolverMethod.EPnP, PnPSolverMethod.AP3P])
def test_returned_pose_is_lm_polished_on_the_ransac_inliers(method):
    world, image, K = _scene(seed=3)
    threshold = 4.0
    R, t, mask, count = solve_pnp_ransac(world, image, K, method=method, threshold=threshold, seed=11)
    inliers = mask.astype(bool)
    assert count == inliers.sum() >= 50

    def cost(R_, t_):
        return float(np.sum(_reprojection(world[inliers], image[inliers], K, R_, t_) ** 2))

    # A polished pose is a local minimum of the inliers' reprojection error:
    # no small rotation or translation lowers it (the LM runs in f32, hence
    # the relative tolerance).
    base = cost(R, t)
    for axis in range(3):
        for step in (-2e-4, 2e-4):
            assert cost(_rotate(R, axis, step), t) >= base * (1 - 1e-4)
            dt = np.zeros(3)
            dt[axis] = step * 0.1
            assert cost(R, t + dt) >= base * (1 - 1e-4)


def test_mask_is_the_pre_polish_ransac_mask():
    world, image, K = _scene(seed=5)
    threshold = 4.0
    R, t, mask, count = solve_pnp_ransac(world, image, K, threshold=threshold, seed=2)
    errors = _reprojection(world, image, K, R, t)
    # The mask was classified before the LM polish, so it may disagree with
    # the polished pose only for points close to the threshold.
    clear = (errors < 0.8 * threshold) | (errors > 1.25 * threshold)
    assert np.array_equal(mask.astype(bool)[clear], errors[clear] < threshold)
    assert count == int(mask.sum())
