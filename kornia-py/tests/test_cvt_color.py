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


@pytest.mark.parametrize("src_cs,dst_cs,src_ch,use_f32", [
    (ColorSpace.Rgb, ColorSpace.Gray, 3, False),
    (ColorSpace.Gray, ColorSpace.Rgb, 1, False),
    (ColorSpace.Rgb, ColorSpace.Bgr, 3, False),
    (ColorSpace.Bgr, ColorSpace.Rgb, 3, False),
    (ColorSpace.Rgb, ColorSpace.Rgba, 3, False),
    (ColorSpace.Rgba, ColorSpace.Rgb, 4, False),
    (ColorSpace.Rgb, ColorSpace.Bgra, 3, False),
    (ColorSpace.Bgra, ColorSpace.Rgb, 4, False),
    (ColorSpace.Rgb, ColorSpace.YCbCr, 3, False),
    (ColorSpace.YCbCr, ColorSpace.Rgb, 3, False),
    (ColorSpace.Rgb, ColorSpace.Yuv, 3, False),
    (ColorSpace.Yuv, ColorSpace.Rgb, 3, False),
    (ColorSpace.Rgb, ColorSpace.Hsv, 3, True),
    (ColorSpace.Hsv, ColorSpace.Rgb, 3, True),
    (ColorSpace.Rgb, ColorSpace.Hls, 3, True),
    (ColorSpace.Hls, ColorSpace.Rgb, 3, True),
    (ColorSpace.Rgb, ColorSpace.Lab, 3, True),
    (ColorSpace.Lab, ColorSpace.Rgb, 3, True),
    (ColorSpace.Rgb, ColorSpace.Luv, 3, True),
    (ColorSpace.Luv, ColorSpace.Rgb, 3, True),
    (ColorSpace.Rgb, ColorSpace.Xyz, 3, True),
    (ColorSpace.Xyz, ColorSpace.Rgb, 3, True),
    (ColorSpace.Rgb, ColorSpace.LinearRgb, 3, True),
    (ColorSpace.LinearRgb, ColorSpace.Rgb, 3, True),
])
def test_legal_pairs_parity(src_cs, dst_cs, src_ch, use_f32):
    rng = np.random.default_rng(42)
    if use_f32:
        arr = rng.random((8, 8, src_ch), dtype=np.float32)
    else:
        arr = rng.integers(0, 256, (8, 8, src_ch), dtype=np.uint8)
    img = Image(arr, color_space=src_cs)
    result = img.cvt_color(dst_cs)
    assert result.color_space == dst_cs
