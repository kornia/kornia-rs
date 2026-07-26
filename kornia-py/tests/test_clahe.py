"""CPU tests for kornia_rs.imgproc.clahe (byte-for-byte with cv2.createCLAHE)."""

from __future__ import annotations

import numpy as np

import kornia_rs as K


def _pattern(h, w):
    rng = np.random.default_rng(5)
    return rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)


def test_clahe_shape_and_dtype():
    img = _pattern(48, 64)
    out = np.asarray(K.imgproc.clahe(img))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_clahe_constant_image_stays_constant():
    img = np.full((32, 32, 1), 100, np.uint8)
    out = np.asarray(K.imgproc.clahe(img))
    assert (out == out.flat[0]).all()


def test_clahe_defaults_match_explicit_args():
    img = _pattern(64, 64)
    a = np.asarray(K.imgproc.clahe(img))
    b = np.asarray(K.imgproc.clahe(img, clip_limit=40.0, grid_size=(8, 8)))
    np.testing.assert_array_equal(a, b)


def test_clahe_improves_local_contrast():
    # Low-contrast gradient: CLAHE must widen the value range.
    gx = np.tile((np.arange(256) // 8 + 96).astype(np.uint8), (64, 1))
    out = np.asarray(K.imgproc.clahe(gx[..., None], clip_limit=3.0))
    assert int(out.max()) - int(out.min()) > int(gx.max()) - int(gx.min())
