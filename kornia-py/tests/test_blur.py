import pytest
import numpy as np
import kornia_rs as K
from kornia_rs.image import Image

@pytest.mark.parametrize("channels", [1, 3, 4])
def test_gaussian_blur_host_u8(channels):
    arr = np.random.randint(0, 255, (16, 16, channels), dtype=np.uint8)

    # 1. Test numpy input
    out_arr = K.imgproc.gaussian_blur(arr, (3, 3), (1.0, 1.0))
    assert out_arr.dtype == np.uint8
    assert out_arr.shape == (16, 16, channels)

    # 2. Test Image input
    img = Image(arr)
    out_img = K.imgproc.gaussian_blur(img, (3, 3), (1.0, 1.0))
    assert isinstance(out_img, Image)
    assert out_img.shape == (16, 16, channels)

@pytest.mark.parametrize("channels", [1, 3, 4])
def test_box_blur_host_u8(channels):
    arr = np.random.randint(0, 255, (16, 16, channels), dtype=np.uint8)

    # 1. Test numpy input
    out_arr = K.imgproc.box_blur(arr, (3, 3))
    assert out_arr.dtype == np.uint8
    assert out_arr.shape == (16, 16, channels)

    # 2. Test Image input
    img = Image(arr)
    out_img = K.imgproc.box_blur(img, (3, 3))
    assert isinstance(out_img, Image)
    assert out_img.shape == (16, 16, channels)
