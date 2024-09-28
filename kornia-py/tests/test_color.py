import kornia_rs as K

import numpy as np


def test_rgb_from_gray():
    img: np.ndarray = np.array([[[1]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.rgb_from_gray(img)
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[1, 1, 1]]]))


def test_bgr_from_rgb():
    img: np.ndarray = np.array([[[1, 2, 3]]], dtype=np.uint8)
    img_bgr: np.ndarray = K.bgr_from_rgb(img)
    assert img_bgr.shape == (1, 1, 3)
    assert np.allclose(img_bgr, np.array([[[3, 2, 1]]]))

def test_gray_from_rgb():
    img: np.ndarray = np.array([[[1, 1, 1]]], dtype=np.uint8)
    img_gray: np.ndarray = K.gray_from_rgb(img)
    assert img_gray.shape == (1, 1, 1)
    assert np.allclose(img_gray, np.array([[[1]]]))
