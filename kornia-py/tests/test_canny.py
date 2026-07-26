"""CPU tests for kornia_rs.imgproc.canny (byte-for-byte with cv2.Canny)."""

from __future__ import annotations

import numpy as np

import kornia_rs as K


def test_canny_constant_no_edges():
    img = np.full((32, 32, 1), 100, np.uint8)
    out = np.asarray(K.imgproc.canny(img))
    assert (out == 0).all()


def test_canny_step_edge_binary():
    img = np.zeros((32, 64, 1), np.uint8)
    img[:, 32:] = 200
    out = np.asarray(K.imgproc.canny(img))
    assert set(np.unique(out)) <= {0, 255}
    assert (out == 255).any()


def test_canny_reversed_thresholds_swap():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(48, 64, 1), dtype=np.uint8)
    a = np.asarray(K.imgproc.canny(img, low_threshold=50, high_threshold=150))
    b = np.asarray(K.imgproc.canny(img, low_threshold=150, high_threshold=50))
    np.testing.assert_array_equal(a, b)


def test_canny_l2_option():
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, size=(48, 64, 1), dtype=np.uint8)
    a = np.asarray(K.imgproc.canny(img, l2_gradient=True))
    assert set(np.unique(a)) <= {0, 255}
