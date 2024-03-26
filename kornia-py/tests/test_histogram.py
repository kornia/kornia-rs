from __future__ import annotations
import kornia_rs as K

import numpy as np


def test_histogram():
    # load an image with libjpeg-turbo
    img = np.array([0, 2, 4, 128, 130, 132, 254, 255, 255], dtype=np.uint8).reshape(
        3, 3, 1
    )

    img_histogram: list[int] = K.compute_histogram(img, num_bins=3)

    assert len(img_histogram) == 3
    assert img_histogram[0] == 3
    assert img_histogram[1] == 3
    assert img_histogram[2] == 3
