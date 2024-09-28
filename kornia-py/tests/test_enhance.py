import kornia_rs as K

import numpy as np


def test_add_weighted():
    img1: np.ndarray = np.array([[[1, 2, 3]]], dtype=np.uint8)
    img2: np.ndarray = np.array([[[4, 5, 6]]], dtype=np.uint8)
    img_weighted: np.ndarray = K.add_weighted(img1, 0.5, img2, 0.5, 0.0)
    assert img_weighted.shape == (1, 1, 3)
    assert np.allclose(img_weighted, np.array([[[2, 3, 4]]]))
