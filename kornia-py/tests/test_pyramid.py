import pytest
import numpy as np
import kornia_rs
from kornia_rs.image import Image

def _get_cv2():
    return pytest.importorskip("cv2")

@pytest.fixture
def dummy_image():
    # 8x8 random u8 image with 3 channels
    np.random.seed(42)
    arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image(arr)

@pytest.fixture
def dummy_image_f32():
    # 8x8 random f32 image with 3 channels
    np.random.seed(42)
    arr = np.random.rand(8, 8, 3).astype(np.float32)
    return Image(arr)

def test_pyrdown_u8(dummy_image):
    cv2 = _get_cv2()
    out = kornia_rs.imgproc.pyrdown(dummy_image)
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)

    expected = cv2.pyrDown(dummy_image.numpy())
    assert np.allclose(out, expected)

def test_pyrup_u8(dummy_image):
    cv2 = _get_cv2()
    out = kornia_rs.imgproc.pyrup(dummy_image)
    assert out.dtype == np.uint8
    assert out.shape == (16, 16, 3)

    expected = cv2.pyrUp(dummy_image.numpy())
    assert np.allclose(out, expected)

def test_pyrdown_f32(dummy_image_f32):
    cv2 = _get_cv2()
    out = kornia_rs.imgproc.pyrdown(dummy_image_f32)
    assert out.dtype == np.float32
    assert out.shape == (4, 4, 3)

    expected = cv2.pyrDown(dummy_image_f32.numpy())
    assert np.allclose(out, expected, atol=1e-5)

def test_pyrup_f32(dummy_image_f32):
    cv2 = _get_cv2()
    out = kornia_rs.imgproc.pyrup(dummy_image_f32)
    assert out.dtype == np.float32
    assert out.shape == (16, 16, 3)

    expected = cv2.pyrUp(dummy_image_f32.numpy())
    assert np.allclose(out, expected, atol=1e-5)

def test_build_pyramid(dummy_image):
    pyramid = kornia_rs.imgproc.build_pyramid(dummy_image, max_level=4)
    # 8x8 -> 4x4 -> 2x2 -> 1x1, stops at 1x1. Output length = 4.
    assert len(pyramid) == 4
    assert pyramid[0].shape == (8, 8, 3)
    assert pyramid[1].shape == (4, 4, 3)
    assert pyramid[2].shape == (2, 2, 3)
    assert pyramid[3].shape == (1, 1, 3)

    # check no aliasing
    assert pyramid[0] is not dummy_image

def test_pyramid_device(dummy_image):
    if not hasattr(dummy_image, "device"):
        pytest.skip("CUDA not enabled in this build")
    try:
        dev_img = dummy_image.to("cuda")
    except Exception:
        pytest.skip("CUDA device not available")

    out_down = kornia_rs.imgproc.pyrdown(dev_img)
    assert out_down.is_device()
    assert out_down.shape == (4, 4, 3)

    out_up = kornia_rs.imgproc.pyrup(dev_img)
    assert out_up.is_device()
    assert out_up.shape == (16, 16, 3)
