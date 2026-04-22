"""Tests for the Python-exposed feature utilities:
- `kornia_rs.features.match_descriptors` — brute-force Hamming matcher.
- `kornia_rs.features.find_homography` — DLT / RANSAC+refit homography solver.
- `kornia_rs.features.ransac_homography` — legacy 4-point RANSAC (kept for back-compat).

The goal is to pin the public contract: input shapes, dtypes, error paths, and
rough numerical behavior against known-good synthetic data. We don't compare
byte-for-byte with OpenCV here — that happens in the e2e script. These are
fast unit tests that run under pytest.
"""

import numpy as np
import pytest

import kornia_rs as K


# ---------- match_descriptors ----------


def _make_descriptor_pair(n=20, seed=0):
    """Return `(desc_a, desc_b)` where row i of desc_b equals row i of desc_a
    with a handful of random bits flipped — so the ideal matcher returns
    the identity permutation."""
    rng = np.random.default_rng(seed)
    desc_a = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    # Flip 3 bits per row (tiny Hamming distance — always the nearest neighbor).
    desc_b = desc_a.copy()
    for i in range(n):
        bit_positions = rng.integers(0, 256, size=3)
        for bp in bit_positions:
            desc_b[i, bp // 8] ^= np.uint8(1 << (bp % 8))
    return desc_a, desc_b


def test_match_descriptors_identity_permutation():
    desc_a, desc_b = _make_descriptor_pair(n=32, seed=0)
    matches = K.features.match_descriptors(desc_a, desc_b)
    # Every query should find its own row as the nearest neighbor.
    assert matches.shape == (32, 2)
    assert matches.dtype == np.int64
    np.testing.assert_array_equal(matches[:, 0], np.arange(32))
    np.testing.assert_array_equal(matches[:, 1], np.arange(32))


def test_match_descriptors_cross_check():
    """With cross_check=True, only symmetric best-matches survive — row i of
    desc_a's NN in desc_b must also have desc_a[i] as its own NN."""
    desc_a, desc_b = _make_descriptor_pair(n=16, seed=1)
    matches = K.features.match_descriptors(desc_a, desc_b, cross_check=True)
    # Our synthetic pair has unique nearest neighbors, so all 16 survive.
    assert len(matches) == 16


def test_match_descriptors_max_ratio():
    """Lowe's ratio: accept match iff best_dist < ratio * second_best_dist.
    We build queries with one clearly-close candidate (dist=8) and one
    clearly-far candidate (dist=160). Ratio = 8/160 = 0.05:
    - max_ratio=0.5 → 8 < 0.5*160 = 80 → accept all 4.
    - max_ratio=0.04 → 8 < 0.04*160 = 6.4 → reject all 4."""
    rng = np.random.default_rng(7)
    query = rng.integers(0, 256, size=(4, 32), dtype=np.uint8)
    close = query.copy()
    close[:, 0] ^= np.uint8(0xFF)  # 8 flips → dist 8.
    far = query.copy()
    far[:, :20] ^= np.uint8(0xFF)  # 160 flips → dist 160.
    candidates = np.concatenate([close, far])
    accepted = K.features.match_descriptors(query, candidates, max_ratio=0.5)
    assert len(accepted) == 4
    rejected = K.features.match_descriptors(query, candidates, max_ratio=0.04)
    assert len(rejected) == 0


def test_match_descriptors_rejects_wrong_descriptor_size():
    bad = np.zeros((4, 16), dtype=np.uint8)  # must be 32 bytes.
    with pytest.raises(ValueError):
        K.features.match_descriptors(bad, bad)


def test_match_descriptors_empty_input():
    empty = np.zeros((0, 32), dtype=np.uint8)
    matches = K.features.match_descriptors(empty, empty)
    assert matches.shape == (0, 2)


# ---------- find_homography ----------


def _make_known_homography_pair(n=50, seed=0, noise_px=0.0):
    """Generate src/dst point pairs that lie exactly on a known homography,
    with optional gaussian pixel noise on dst."""
    rng = np.random.default_rng(seed)
    # Realistic H: rotate 5°, scale 1.1, translate, small perspective.
    a = np.deg2rad(5.0)
    H_gt = np.array([
        [1.1 * np.cos(a), -1.1 * np.sin(a), 10.0],
        [1.1 * np.sin(a),  1.1 * np.cos(a), -5.0],
        [1e-4,             2e-4,             1.0],
    ], dtype=np.float64)
    src = rng.uniform(100.0, 500.0, size=(n, 2)).astype(np.float64)
    homo = np.concatenate([src, np.ones((n, 1))], axis=1)
    proj = homo @ H_gt.T
    dst = proj[:, :2] / proj[:, 2:3]
    if noise_px > 0:
        dst = dst + rng.normal(0.0, noise_px, size=dst.shape)
    return src, dst, H_gt


def _normalize_h(h):
    """Scale H so its last entry is 1 (homographies are defined up to scale)."""
    return h / h[2, 2]


def test_find_homography_dlt_recovers_known_h():
    src, dst, H_gt = _make_known_homography_pair(n=50, seed=0)
    H_est, mask = K.features.find_homography(src, dst, method=0)
    assert H_est.shape == (3, 3)
    assert H_est.dtype == np.float64
    assert mask.shape == (50,)
    assert mask.dtype == np.uint8
    # DLT on clean inliers should recover H_gt to high precision.
    np.testing.assert_allclose(_normalize_h(H_est), _normalize_h(H_gt), atol=1e-8)
    # method=0 returns an all-ones mask (no outlier rejection).
    np.testing.assert_array_equal(mask, np.ones(50, dtype=np.uint8))


def test_find_homography_ransac_rejects_outliers():
    """Build a set with 40 inliers + 10 pure-outlier noise points; RANSAC
    should recover H and mark the outliers."""
    src, dst, H_gt = _make_known_homography_pair(n=40, seed=1)
    rng = np.random.default_rng(2)
    outlier_src = rng.uniform(100, 500, size=(10, 2))
    outlier_dst = rng.uniform(100, 500, size=(10, 2))  # unrelated to src.
    src_all = np.concatenate([src, outlier_src])
    dst_all = np.concatenate([dst, outlier_dst])
    H_est, mask = K.features.find_homography(
        src_all, dst_all, method=8, ransac_threshold=3.0, seed=0
    )
    # At least the 40 true inliers should be marked.
    assert mask.sum() >= 38  # small slack for edge cases in sample draws.
    # The recovered H should be close to ground truth (refit on inliers makes
    # this tight — within 1 pixel of reprojection error on clean inliers).
    homo = np.concatenate([src, np.ones((40, 1))], axis=1)
    proj_est = homo @ H_est.T
    pred = proj_est[:, :2] / proj_est[:, 2:3]
    err = np.linalg.norm(pred - dst, axis=1).mean()
    assert err < 1.0, f"mean reproj err {err:.3f} px exceeds 1.0"


def test_find_homography_rejects_insufficient_points():
    src = np.zeros((3, 2), dtype=np.float64)
    dst = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        K.features.find_homography(src, dst, method=0)


def test_find_homography_rejects_mismatched_length():
    src = np.zeros((5, 2), dtype=np.float64)
    dst = np.zeros((6, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        K.features.find_homography(src, dst)


def test_find_homography_rejects_bad_method():
    src, dst, _ = _make_known_homography_pair(n=10)
    with pytest.raises(ValueError):
        K.features.find_homography(src, dst, method=42)


def test_find_homography_ransac_deterministic_with_seed():
    """Same seed → same H and same inlier mask (property of deterministic RANSAC)."""
    src, dst, _ = _make_known_homography_pair(n=40, seed=3, noise_px=0.5)
    H1, m1 = K.features.find_homography(src, dst, method=8, seed=123)
    H2, m2 = K.features.find_homography(src, dst, method=8, seed=123)
    np.testing.assert_array_equal(H1, H2)
    np.testing.assert_array_equal(m1, m2)


# ---------- ransac_homography (backward-compat surface) ----------


def test_ransac_homography_still_works():
    """Legacy entry point must keep its (H, mask, count) return shape."""
    src, dst, _ = _make_known_homography_pair(n=40, seed=4)
    H, mask, count = K.features.ransac_homography(
        src, dst, threshold=3.0, min_inliers=10, seed=0
    )
    assert H.shape == (3, 3)
    assert mask.shape == (40,)
    assert mask.dtype == np.uint8
    assert count == int(mask.sum())
    assert count >= 30
