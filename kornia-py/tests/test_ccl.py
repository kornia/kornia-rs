"""CPU tests for connected_components (label-exact with cv2 SAUF)."""

from __future__ import annotations

import numpy as np

import kornia_rs as K


def test_ccl_empty():
    n, lab = K.imgproc.connected_components(np.zeros((8, 8, 1), np.uint8))
    assert n == 1
    assert (np.asarray(lab) == 0).all()


def test_ccl_diagonal_connectivity():
    img = np.zeros((2, 2, 1), np.uint8)
    img[0, 0, 0] = 255
    img[1, 1, 0] = 255
    n8, _ = K.imgproc.connected_components(img, connectivity=8)
    n4, _ = K.imgproc.connected_components(img, connectivity=4)
    assert n8 == 2 and n4 == 3


def test_ccl_raster_numbering():
    img = np.zeros((3, 4, 1), np.uint8)
    img[0, 1, 0] = 255
    img[1, 3, 0] = 255
    img[2, 3, 0] = 255
    n, lab = K.imgproc.connected_components(img)
    lab = np.asarray(lab)[..., 0]
    assert n == 3 and lab[0, 1] == 1 and lab[1, 3] == 2


def test_ccl_rejects_bad_connectivity():
    import pytest
    with pytest.raises(Exception):
        K.imgproc.connected_components(np.zeros((4, 4, 1), np.uint8), connectivity=6)
