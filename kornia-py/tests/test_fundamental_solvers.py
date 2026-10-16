"""Focused correctness checks for seven/eight-point fundamental RANSAC."""
import numpy as np
import pytest

import kornia_rs as kr


def correspondences(n=80):
    rng = np.random.default_rng(42)
    points = rng.uniform([-2, -1.5, 3], [2, 1.5, 8], (n, 3))
    angle = 0.1
    rotation = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    moved = points @ rotation.T + [0.5, 0.1, 0.2]
    a = np.ascontiguousarray(points[:, :2] / points[:, 2:] * 800 + [320, 240])
    b = np.ascontiguousarray(moved[:, :2] / moved[:, 2:] * 800 + [320, 240])
    return a, b


def residuals(f, a, b):
    ah, bh = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    fa, fb = ah @ f.T, bh @ f
    return np.sum(fa * bh, axis=1) ** 2 / (
        np.sum(fa[:, :2] ** 2, axis=1) + np.sum(fb[:, :2] ** 2, axis=1)
    )


@pytest.mark.parametrize("solver", ["7point", "8point"])
@pytest.mark.parametrize("engine", ["generic", "k3d"])
def test_recovers_clean_inliers_among_outliers(solver, engine):
    a, b = correspondences()
    b[60:] = np.random.default_rng(3).uniform([0, 0], [640, 480], (20, 2))
    if engine == "generic":
        result = kr.ransac.fundamental(
            np.c_[a, b], threshold=1e-6, max_iters=1000, seed=0, solver=solver
        )
        assert result.model is not None
        f, mask = np.array(result.model).reshape(3, 3), np.array(result.inliers)
    else:
        f, mask = kr.k3d.find_fundamental(
            a, b, method=8, ransac_threshold=1e-3, max_iterations=1000,
            seed=0, solver=solver,
        )
    assert np.all(np.asarray(mask[:60], bool))
    assert np.count_nonzero(mask[60:]) == 0
    assert np.max(residuals(f, a[:60], b[:60])) < 1e-6
    singular = np.linalg.svd(f, compute_uv=False)
    assert singular[-1] / singular[0] < 1e-10


def test_seven_match_boundary_and_default():
    a, b = correspondences(7)
    matches = np.c_[a, b]
    default = kr.ransac.fundamental(matches, threshold=1e-6, seed=0)
    explicit = kr.ransac.fundamental(matches, threshold=1e-6, seed=0, solver="7point")
    assert default.model == explicit.model
    assert sum(default.inliers) == 7
    assert kr.ransac.fundamental(matches, solver="8point").model is None
    f, mask = kr.k3d.find_fundamental(
        a, b, method=8, min_inliers=7, ransac_threshold=1e-3, seed=0
    )
    assert np.all(mask)
    assert np.max(residuals(f, a, b)) < 1e-6
    with pytest.raises(ValueError, match="at least 8"):
        kr.k3d.find_fundamental(a, b, method=8, min_inliers=7, solver="8point")


def test_invalid_solver_and_degenerate_input():
    a, b = correspondences()
    with pytest.raises(ValueError, match="solver"):
        kr.ransac.fundamental(np.c_[a, b], solver="unknown")
    with pytest.raises(ValueError, match="solver"):
        kr.k3d.find_fundamental(a, b, method=8, solver="unknown")
    assert kr.ransac.fundamental(np.zeros((7, 4)), solver="7point").model is None
    with pytest.raises(ValueError, match="RANSAC failed"):
        kr.k3d.find_fundamental(np.zeros((7, 2)), np.zeros((7, 2)), method=8, min_inliers=7)


@pytest.mark.parametrize("estimator", ["fundamental", "essential", "homography"])
def test_generic_ransac_validates_confidence(estimator):
    a, b = correspondences()
    estimate = getattr(kr.ransac, estimator)
    for confidence in [0.0, 1.0, -0.5, 1.5, np.nan, np.inf]:
        with pytest.raises(ValueError, match="confidence"):
            estimate(np.c_[a, b], confidence=confidence)


@pytest.mark.parametrize("solver", ["7point", "8point"])
def test_find_fundamental_accepts_and_validates_confidence(solver):
    a, b = correspondences()
    f, mask = kr.k3d.find_fundamental(
        a,
        b,
        method=8,
        ransac_threshold=1e-3,
        max_iterations=1000,
        seed=0,
        solver=solver,
        confidence=0.999999,
    )
    assert np.all(np.isfinite(f))
    assert np.count_nonzero(mask) == len(a)
    for confidence in [0.0, 1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="confidence"):
            kr.k3d.find_fundamental(a, b, method=8, solver=solver, confidence=confidence)


def test_direct_dlt_retains_eight_point_fit():
    a, b = correspondences()
    f7, mask7 = kr.k3d.find_fundamental(a, b, method=0, solver="7point")
    f8, mask8 = kr.k3d.find_fundamental(a, b, method=0, solver="8point")
    np.testing.assert_array_equal(f7, f8)
    np.testing.assert_array_equal(mask7, mask8)
    assert np.all(mask7)
    f_unused, mask_unused = kr.k3d.find_fundamental(a, b, method=0, confidence=np.nan)
    np.testing.assert_array_equal(f7, f_unused)
    np.testing.assert_array_equal(mask7, mask_unused)
