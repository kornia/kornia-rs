"""Tests for SPRT functionality in kornia-rs PnP bindings."""

import numpy as np
import pytest

import kornia_rs
from kornia_rs.k3d import PnPSolverMethod, solve_pnp_ransac


def _make_scene(n_inliers: int, n_outliers: int, seed: int):
    """Generate a synthetic 3D-2D scene with ground-truth pose."""
    rng = np.random.default_rng(seed)
    fx = fy = 800.0
    cx, cy = 640.0, 480.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    ax, ay, az = np.deg2rad([10.0, -15.0, 30.0])
    sx, cx_ = np.sin(ax), np.cos(ax)
    sy, cy_ = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_gt = Rz @ Ry @ Rx
    t_gt = np.array([0.7, -0.4, 5.0], dtype=np.float64)

    world_in = rng.uniform(-0.3, 0.3, size=(n_inliers, 3)).astype(np.float64)
    world_in[:, 2] = rng.uniform(0.5, 1.5, size=n_inliers)
    pc_in = world_in @ R_gt.T + t_gt
    image_in = np.column_stack(
        [
            fx * pc_in[:, 0] / pc_in[:, 2] + cx,
            fy * pc_in[:, 1] / pc_in[:, 2] + cy,
        ]
    ) + rng.normal(scale=0.5, size=(n_inliers, 2))

    if n_outliers == 0:
        return world_in, image_in, K, R_gt, t_gt

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


def _rotation_error(R_est, R_gt):
    return float(np.degrees(np.arccos(np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0))))


# ----------------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene_clean():
    return _make_scene(50, 0, seed=42)


@pytest.fixture(scope="module")
def scene_30pct_outliers():
    return _make_scene(70, 30, seed=42)


@pytest.fixture(scope="module")
def scene_50pct_outliers():
    return _make_scene(50, 50, seed=42)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_solve_pnp_ransac_clean_data_ap3p(scene_clean):
    """AP3P on clean data should recover the pose with sub-degree rotation error."""
    world, image, K, R_gt, t_gt = scene_clean
    R, t, mask, inliers = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.AP3P, seed=42,
    )
    assert inliers >= 50
    assert _rotation_error(R, R_gt) < 1.0


def test_solve_pnp_ransac_clean_data_epnp(scene_clean):
    """EPnP on clean data should recover the pose with sub-degree rotation error."""
    world, image, K, R_gt, t_gt = scene_clean
    R, t, mask, inliers = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.EPnP, seed=42,
    )
    assert inliers >= 50
    assert _rotation_error(R, R_gt) < 1.0


def test_solve_pnp_ransac_with_sprt_recovers_pose(scene_30pct_outliers):
    """SPRT-enabled run on 30% outliers should recover the pose."""
    world, image, K, R_gt, t_gt = scene_30pct_outliers
    R, t, mask, inliers = solve_pnp_ransac(
        world,
        image,
        K,
        method=PnPSolverMethod.EPnP,
        seed=42,
        use_sprt=True,
        sprt_epsilon=0.7,
        sprt_delta=0.01,
    )
    assert inliers >= 60  # should keep most of the 70 inliers
    assert _rotation_error(R, R_gt) < 1.0


def test_solve_pnp_ransac_sprt_off_vs_on_equivalent_quality(scene_30pct_outliers):
    """SPRT on/off should produce comparable accuracy on the same data."""
    world, image, K, R_gt, t_gt = scene_30pct_outliers

    R_off, t_off, mask_off, inliers_off = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.EPnP, seed=42, use_sprt=False,
    )
    R_on, t_on, mask_on, inliers_on = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.EPnP, seed=42, use_sprt=True,
    )
    # Both should be reasonably close to ground truth.
    err_off = _rotation_error(R_off, R_gt)
    err_on = _rotation_error(R_on, R_gt)
    assert err_off < 1.0
    assert err_on < 1.0


def test_solve_pnp_ransac_sprt_wrong_prior_recovers_pose():
    """A wildly optimistic SPRT epsilon must not starve the run.

    Regression test: the pre-acceptance epsilon cap (min(prior, 0.3)) and
    the grace period keep good hypotheses alive even when the caller's
    prior is far above the true inlier ratio.
    """
    world, image, K, R_gt, t_gt = _make_scene(60, 140, seed=7)  # 30% inliers
    R, t, mask, inliers = solve_pnp_ransac(
        world,
        image,
        K,
        method=PnPSolverMethod.AP3P,
        seed=7,
        max_iterations=2000,
        use_sprt=True,
        sprt_epsilon=0.95,
        sprt_delta=0.01,
    )
    assert inliers >= 45  # should recover most of the 60 inliers
    assert _rotation_error(R, R_gt) < 1.0


def test_solve_pnp_ransac_sprt_invalid_epsilon_rejected():
    """A non-finite or out-of-range sprt_epsilon should raise ValueError."""
    world, image, K, _R_gt, _t_gt = _make_scene(20, 0, seed=1)
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            solve_pnp_ransac(
                world, image, K, method=PnPSolverMethod.EPnP, seed=1,
                use_sprt=True, sprt_epsilon=bad, sprt_delta=0.01,
            )


def test_solve_pnp_ransac_sprt_invalid_delta_rejected():
    """A non-finite or out-of-range sprt_delta should raise ValueError."""
    world, image, K, _R_gt, _t_gt = _make_scene(20, 0, seed=1)
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            solve_pnp_ransac(
                world, image, K, method=PnPSolverMethod.EPnP, seed=1,
                use_sprt=True, sprt_epsilon=0.5, sprt_delta=bad,
            )


def test_solve_pnp_ransac_sprt_with_lo_refit(scene_50pct_outliers):
    """SPRT + LO should both refine the pose and reject bad hypotheses."""
    world, image, K, R_gt, t_gt = scene_50pct_outliers
    R, t, mask, inliers = solve_pnp_ransac(
        world,
        image,
        K,
        method=PnPSolverMethod.AP3P,
        seed=42,
        lo_every=5,
        use_sprt=True,
        sprt_epsilon=0.5,
        sprt_delta=0.01,
    )
    assert inliers >= 40
    assert _rotation_error(R, R_gt) < 1.0


def test_solve_pnp_ransac_deterministic_with_seed(scene_30pct_outliers):
    """Same seed and same settings → identical results, with or without SPRT."""
    world, image, K, _R_gt, _t_gt = scene_30pct_outliers

    R1, t1, mask1, inliers1 = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.EPnP, seed=42, use_sprt=True,
    )
    R2, t2, mask2, inliers2 = solve_pnp_ransac(
        world, image, K, method=PnPSolverMethod.EPnP, seed=42, use_sprt=True,
    )
    np.testing.assert_array_equal(R1, R2)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(mask1, mask2)
    assert inliers1 == inliers2


def test_solve_pnp_ransac_sprt_50pct_outliers_ap3p(scene_50pct_outliers):
    """AP3P + SPRT on 50% outliers should still recover the pose."""
    world, image, K, R_gt, t_gt = scene_50pct_outliers
    R, t, mask, inliers = solve_pnp_ransac(
        world,
        image,
        K,
        method=PnPSolverMethod.AP3P,
        seed=42,
        use_sprt=True,
        sprt_epsilon=0.5,
        sprt_delta=0.01,
    )
    assert inliers >= 40
    assert _rotation_error(R, R_gt) < 2.0


def test_solve_pnp_ransac_input_validation_shape_mismatch():
    """Wrong-shape world/image arrays should raise ValueError."""
    K = np.eye(3, dtype=np.float64)
    world = np.zeros((10, 3), dtype=np.float64)
    image = np.zeros((9, 2), dtype=np.float64)  # mismatched N
    with pytest.raises(ValueError):
        solve_pnp_ransac(world, image, K, method=PnPSolverMethod.EPnP, seed=0)


def test_solve_pnp_ransac_input_validation_insufficient_points():
    """EPnP with fewer than 4 points should raise."""
    K = np.eye(3, dtype=np.float64)
    world = np.zeros((3, 3), dtype=np.float64)
    image = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        solve_pnp_ransac(world, image, K, method=PnPSolverMethod.EPnP, seed=0)
