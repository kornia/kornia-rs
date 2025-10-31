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
    img: np.ndarray = np.array([[[0, 128, 255]]], dtype=np.uint8)
    img_gray: np.ndarray = K.gray_from_rgb(img)
    assert img_gray.shape == (1, 1, 1)
    assert np.allclose(img_gray, np.array([[[104]]]))

def test_rgb_from_rgba():
    img: np.ndarray = np.array([[[0, 1, 2, 255]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.rgb_from_rgba(img)
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[0, 1, 2]]]))


def test_rgb_from_rgba_with_background():
    img: np.ndarray = np.array([[[255, 0, 0, 128]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.rgb_from_rgba(img, background=[100, 100, 100])
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[178, 50, 50]]]))