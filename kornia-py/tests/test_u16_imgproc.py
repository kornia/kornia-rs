"""u16 imgproc coverage — partial Phase 6.

dtype-trivial ops work natively on u16: ``flip_horizontal``,
``flip_vertical``, ``crop``. Math-heavy ops (``resize`` / blur / color /
normalize / rotate) still raise NotImplementedError on u16 with a clearer
remediation hint pointing at ``img.convert(...)``.
"""

import numpy as np
import pytest

from kornia_rs.image import Image


def _rand_u16(h, w, c=3):
    return np.random.randint(0, 65536, (h, w, c), dtype=np.uint16)


# ----------------------------------------------------------- flip on u16


def test_flip_horizontal_u16_lossless():
    arr = _rand_u16(8, 12, 1)
    img = Image(arr)
    flipped = img.flip_horizontal()
    assert flipped.dtype == np.uint16
    np.testing.assert_array_equal(flipped.data, arr[:, ::-1, :])


def test_flip_horizontal_u16_rgb_lossless():
    arr = _rand_u16(8, 12, 3)
    flipped = Image(arr).flip_horizontal()
    np.testing.assert_array_equal(flipped.data, arr[:, ::-1, :])


def test_flip_vertical_u16_lossless():
    arr = _rand_u16(10, 5, 3)
    flipped = Image(arr).flip_vertical()
    np.testing.assert_array_equal(flipped.data, arr[::-1, :, :])


def test_flip_u16_double_flip_identity():
    arr = _rand_u16(8, 8, 1)
    img = Image(arr)
    np.testing.assert_array_equal(img.flip_horizontal().flip_horizontal().data, arr)
    np.testing.assert_array_equal(img.flip_vertical().flip_vertical().data, arr)


# ----------------------------------------------------------- crop on u16


def test_crop_u16_kornia_signature():
    arr = _rand_u16(12, 16, 3)
    cropped = Image(arr).crop(2, 3, 6, 5)  # x, y, w, h
    assert cropped.dtype == np.uint16
    assert cropped.shape == (5, 6, 3)
    np.testing.assert_array_equal(cropped.data, arr[3:8, 2:8, :])


def test_crop_u16_pil_4tuple():
    arr = _rand_u16(12, 16, 1)
    cropped = Image(arr).crop((2, 3, 8, 8))  # left, upper, right, lower
    assert cropped.shape == (5, 6, 1)
    np.testing.assert_array_equal(cropped.data, arr[3:8, 2:8, :])


def test_crop_u16_out_of_bounds_rejects():
    img = Image(_rand_u16(10, 10, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        img.crop(5, 5, 10, 10)


# ----------------------------------------------------------- math-heavy still gated


def test_resize_u16_still_unsupported():
    img = Image(_rand_u16(16, 16, 3))
    with pytest.raises(NotImplementedError, match="convert"):
        img.resize(8, 8)


def test_gaussian_blur_u16_still_unsupported():
    img = Image(_rand_u16(16, 16, 3))
    with pytest.raises(NotImplementedError, match="convert"):
        img.gaussian_blur()


def test_normalize_u16_still_unsupported():
    img = Image(_rand_u16(16, 16, 3))
    with pytest.raises(NotImplementedError, match="convert"):
        img.normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
