"""Binding parity, input validation and affine gauge invariance."""

import numpy as np
import pytest
from kornia_rs import k3d


def scene():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(80, 3))
    a = rng.normal(size=(4, 2, 4))
    ci = np.repeat(np.arange(4, dtype=np.int64), len(x))
    pi = np.tile(np.arange(len(x), dtype=np.int64), 4)
    xy = np.einsum("nij,nj->ni", a[ci, :, :3], x[pi]) + a[ci, :, 3]
    start = x.copy()
    start[4:] += rng.normal(0, 0.05, start[4:].shape)
    return a, start, ci, pi, xy, x


def test_recovers_points_and_does_not_mutate_inputs():
    a, x, ci, pi, xy, truth = scene()
    originals = [v.copy() for v in (a, x, ci, pi, xy)]
    result = k3d.bundle_adjust_affine(a, x, ci, pi, xy, [0, 1, 2, 3])
    assert result["cost"] < 1e-12
    assert result["cost"] < result["initial_cost"]
    np.testing.assert_allclose(result["points"], truth, atol=1e-5)
    np.testing.assert_array_equal(result["points"][:4], x[:4])
    for before, after in zip(originals, (a, x, ci, pi, xy)):
        np.testing.assert_array_equal(before, after)


def test_large_world_translation_and_scaling():
    a, x, ci, pi, xy, truth = scene()
    scale = 1000.0
    offset = np.array([1e6, -2e6, 5e6])
    transformed = a.copy()
    transformed[:, :, :3] /= scale
    transformed[:, :, 3] -= transformed[:, :, :3] @ offset
    result = k3d.bundle_adjust_affine(
        transformed, np.ascontiguousarray(x * scale + offset), ci, pi, xy, [0, 1, 2, 3]
    )
    np.testing.assert_allclose((result["points"] - offset) / scale, truth, atol=1e-5)


@pytest.mark.parametrize(
    "mode",
    [
        "negative",
        "out_of_bounds",
        "duplicate",
        "nonfinite",
        "planar",
        "strided",
        "bad_scale",
    ],
)
def test_invalid_input(mode):
    a, x, ci, pi, xy, _ = scene()
    kwargs = {}
    if mode == "negative":
        ci[0] = -1
    if mode == "out_of_bounds":
        pi[0] = len(x)
    if mode == "duplicate":
        pi[1] = pi[0]
    if mode == "nonfinite":
        xy[0, 0] = np.nan
    if mode == "planar":
        x[:4, 2] = 0
    if mode == "strided":
        x = x[::-1]
    if mode == "bad_scale":
        kwargs["robust_scale"] = 0
    with pytest.raises(ValueError):
        k3d.bundle_adjust_affine(a, x, ci, pi, xy, [0, 1, 2, 3], **kwargs)


def test_matches_independent_scipy_objective():
    """SciPy finite-difference least squares on the same vector-norm Cauchy objective."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(19)
    truth = rng.normal(size=(12, 3))
    a = rng.normal(size=(3, 2, 4))
    ci = np.repeat(np.arange(3, dtype=np.int64), len(truth))
    pi = np.tile(np.arange(len(truth), dtype=np.int64), 3)
    xy = np.einsum("nij,nj->ni", a[ci, :, :3], truth[pi]) + a[ci, :, 3]
    xy += rng.normal(0, 0.02, xy.shape)
    initial = truth.copy()
    initial[4:] += rng.normal(0, 0.03, initial[4:].shape)

    def residual(q):
        cameras = q[:24].reshape(3, 2, 4)
        points = np.concatenate([initial[:4], q[24:].reshape(-1, 3)])
        r = (
            np.einsum("nij,nj->ni", cameras[ci, :, :3], points[pi])
            + cameras[ci, :, 3]
            - xy
        )
        s2 = np.sum(r * r, axis=1) / 4.0
        factor = np.sqrt(
            np.divide(np.log1p(s2), s2, out=np.ones_like(s2), where=s2 > 0)
        )
        return (r * factor[:, None]).ravel()

    q0 = np.r_[a.ravel(), initial[4:].ravel()]
    reference = scipy_opt.least_squares(
        residual, q0, ftol=1e-12, xtol=1e-12, gtol=1e-12
    )
    result = k3d.bundle_adjust_affine(
        a, initial, ci, pi, xy, [0, 1, 2, 3], cost_tolerance=1e-11
    )
    assert reference.success
    assert result["cost"] == pytest.approx(reference.cost, rel=1e-7, abs=1e-12)
    np.testing.assert_allclose(
        result["points"][4:], reference.x[24:].reshape(-1, 3), atol=1e-6
    )


def test_rejects_disconnected_graph():
    a, x, ci, pi, xy, _ = scene()
    use = ((ci < 2) & (pi < 40)) | ((ci >= 2) & (pi >= 40))
    with pytest.raises(ValueError, match="disconnected"):
        k3d.bundle_adjust_affine(a, x, ci[use], pi[use], xy[use], [0, 1, 2, 3])


def test_rejects_nearly_planar_anchors():
    a, x, ci, pi, xy, _ = scene()
    x[:4, 2] *= 1e-10
    with pytest.raises(ValueError, match="noncoplanar"):
        k3d.bundle_adjust_affine(a, x, ci, pi, xy, [0, 1, 2, 3])


def test_camera_only_refinement():
    a, _, ci, pi, xy, truth = scene()
    initial = a.copy()
    initial[:, :, 3] += 0.2
    result = k3d.bundle_adjust_affine(
        initial, truth, ci, pi, xy, list(range(len(truth)))
    )
    np.testing.assert_array_equal(result["points"], truth)
    np.testing.assert_allclose(result["cameras"], a, atol=1e-8)
