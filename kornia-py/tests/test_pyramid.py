import pytest
import numpy as np
import kornia_rs
from kornia_rs.image import Image

@pytest.fixture
def dummy_image():
    # 8x8 random u8 image with 3 channels
    arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image(arr)

@pytest.fixture
def dummy_image_f32():
    # 8x8 random f32 image with 3 channels
    arr = np.random.rand(8, 8, 3).astype(np.float32)
    return Image(arr)

def test_pyrdown_u8(dummy_image):
    out = kornia_rs.imgproc.pyrdown(dummy_image)
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)

def test_pyrup_u8(dummy_image):
    out = kornia_rs.imgproc.pyrup(dummy_image)
    assert out.dtype == np.uint8
    assert out.shape == (16, 16, 3)

def test_pyrdown_f32(dummy_image_f32):
    out = kornia_rs.imgproc.pyrdown(dummy_image_f32)
    assert out.dtype == np.float32
    assert out.shape == (4, 4, 3)

def test_pyrup_f32(dummy_image_f32):
    out = kornia_rs.imgproc.pyrup(dummy_image_f32)
    assert out.dtype == np.float32
    assert out.shape == (16, 16, 3)

def test_build_pyramid(dummy_image):
    pyramid = kornia_rs.imgproc.build_pyramid(dummy_image, max_level=2)
    assert len(pyramid) == 3
    assert pyramid[0].shape == (8, 8, 3)
    assert pyramid[1].shape == (4, 4, 3)
    assert pyramid[2].shape == (2, 2, 3)
