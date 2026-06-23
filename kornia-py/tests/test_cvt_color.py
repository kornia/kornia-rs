"""Regression tests for Image.cvt_color, color_space field, and dtype helpers."""
import numpy as np
import pytest
import kornia_rs
from kornia_rs.image import Image, ColorSpace


def _rgb_u8(h=8, w=8):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_color_space_defaults_and_getter():
    assert Image(np.zeros((4, 4, 3), np.uint8)).color_space == ColorSpace.Rgb
    assert Image(np.zeros((4, 4, 1), np.uint8)).color_space == ColorSpace.Gray
    assert Image(np.zeros((4, 4, 4), np.uint8)).color_space == ColorSpace.Rgba


def test_cvt_color_tag_propagates():
    img = Image(_rgb_u8())
    g = img.cvt_color(ColorSpace.Gray)
    assert g.color_space == ColorSpace.Gray
    assert g.numpy().shape[2] == 1


def test_cvt_color_strict_dtype_error_then_to_float():
    img = Image(_rgb_u8())
    with pytest.raises(ValueError, match="float32"):
        img.cvt_color(ColorSpace.Hsv)
    hsv = img.to_float().cvt_color(ColorSpace.Hsv)
    assert hsv.color_space == ColorSpace.Hsv


def test_cvt_color_unsupported_pair():
    img = Image(_rgb_u8()).to_float().cvt_color(ColorSpace.Hsv)
    with pytest.raises(ValueError, match="no direct"):
        img.cvt_color(ColorSpace.Lab)


def test_cvt_color_parity_with_free_function():
    arr = _rgb_u8().astype(np.float32) / 255.0
    img = Image(arr)  # f32 RGB
    via_method = img.cvt_color(ColorSpace.Hsv).numpy()
    via_free = kornia_rs.imgproc.hsv_from_rgb(arr)
    np.testing.assert_allclose(via_method, via_free, rtol=0, atol=0)


def test_to_uint8_round_trip():
    arr = _rgb_u8()
    img = Image(arr)
    back = img.to_float().to_uint8().numpy()
    np.testing.assert_array_equal(back, arr)


def test_sugar_methods():
    img = Image(_rgb_u8())
    assert img.to_gray().color_space == ColorSpace.Gray
    assert img.to_bgr().color_space == ColorSpace.Bgr
