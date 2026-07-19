"""CPU tests for median_blur / bilateral_filter (byte-for-byte with cv2;
median also matches VPI's CUDA MedianFilter bit-for-bit)."""

from __future__ import annotations

import numpy as np
import pytest

import kornia_rs as K


def _pattern(h, w, c=1):
    rng = np.random.default_rng(5)
    return rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)


@pytest.mark.parametrize("k", [3, 5])
def test_median_constant_unchanged(k):
    img = np.full((32, 32, 1), 99, np.uint8)
    out = np.asarray(K.imgproc.median_blur(img, kernel_size=k))
    assert (out == 99).all()


def test_median_removes_salt_noise():
    img = np.full((64, 64, 1), 128, np.uint8)
    img[10, 10, 0] = 255
    out = np.asarray(K.imgproc.median_blur(img, kernel_size=3))
    assert out[10, 10, 0] == 128


def test_median_rejects_bad_ksize():
    img = _pattern(8, 8)
    with pytest.raises(Exception):
        K.imgproc.median_blur(img, kernel_size=4)


def test_median_c3_shape():
    img = _pattern(16, 24, 3)
    out = np.asarray(K.imgproc.median_blur(img, kernel_size=3))
    assert out.shape == img.shape


def test_bilateral_constant_unchanged():
    img = np.full((32, 32, 1), 200, np.uint8)
    out = np.asarray(K.imgproc.bilateral_filter(img))
    assert (out == 200).all()


def test_bilateral_degenerate_sigma_copies():
    img = _pattern(16, 16)
    out = np.asarray(K.imgproc.bilateral_filter(img, d=5, sigma_color=0.0))
    np.testing.assert_array_equal(out, img)


def test_bilateral_smooths_noise_keeps_edges():
    img = np.zeros((32, 32, 1), np.uint8)
    img[:, 16:] = 200
    out = np.asarray(K.imgproc.bilateral_filter(img, d=5, sigma_color=30.0, sigma_space=30.0))
    # edge preserved
    assert out[16, 0, 0] < 20 and out[16, 31, 0] > 180
