"""Regression tests for Image.cvt_color, color_space field, and dtype helpers."""
import numpy as np
import pytest
import kornia_rs
import kornia_rs.imgproc as imgproc
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


# ---------------------------------------------------------------------------
# Legal-pairs parity matrix
# Columns: src_cs, dst_cs, src_ch, use_f32, expected_dst_ch, free_fn_name
#   free_fn_name=None → shape+dtype check only (no free-fn parity)
# ---------------------------------------------------------------------------
_LEGAL_PARAMS = [
    # u8 pairs with matching free functions
    (ColorSpace.Rgb,       ColorSpace.Gray,      3, False, 1,  "gray_from_rgb"),
    (ColorSpace.Gray,      ColorSpace.Rgb,        1, False, 3,  "rgb_from_gray"),
    (ColorSpace.Rgb,       ColorSpace.Bgr,        3, False, 3,  "bgr_from_rgb"),
    (ColorSpace.Bgr,       ColorSpace.Rgb,        3, False, 3,  None),           # no rgb_from_bgr
    (ColorSpace.Rgb,       ColorSpace.Rgba,       3, False, 4,  None),           # no rgba_from_rgb
    (ColorSpace.Rgba,      ColorSpace.Rgb,        4, False, 3,  "rgb_from_rgba"),
    (ColorSpace.Rgb,       ColorSpace.Bgra,       3, False, 4,  None),           # no bgra_from_rgb
    (ColorSpace.Bgra,      ColorSpace.Rgb,        4, False, 3,  "rgb_from_bgra"),
    # u8 pairs where free functions are f32-only → shape+dtype only
    (ColorSpace.Rgb,       ColorSpace.YCbCr,      3, False, 3,  None),
    (ColorSpace.YCbCr,     ColorSpace.Rgb,        3, False, 3,  None),
    (ColorSpace.Rgb,       ColorSpace.Yuv,        3, False, 3,  None),
    (ColorSpace.Yuv,       ColorSpace.Rgb,        3, False, 3,  None),
    # f32 pairs with matching free functions
    (ColorSpace.Rgb,       ColorSpace.Hsv,        3, True,  3,  "hsv_from_rgb"),
    (ColorSpace.Hsv,       ColorSpace.Rgb,        3, True,  3,  "rgb_from_hsv"),
    (ColorSpace.Rgb,       ColorSpace.Hls,        3, True,  3,  "hls_from_rgb"),
    (ColorSpace.Hls,       ColorSpace.Rgb,        3, True,  3,  "rgb_from_hls"),
    (ColorSpace.Rgb,       ColorSpace.Lab,        3, True,  3,  "lab_from_rgb"),
    (ColorSpace.Lab,       ColorSpace.Rgb,        3, True,  3,  "rgb_from_lab"),
    (ColorSpace.Rgb,       ColorSpace.Luv,        3, True,  3,  "luv_from_rgb"),
    (ColorSpace.Luv,       ColorSpace.Rgb,        3, True,  3,  "rgb_from_luv"),
    (ColorSpace.Rgb,       ColorSpace.Xyz,        3, True,  3,  "xyz_from_rgb"),
    (ColorSpace.Xyz,       ColorSpace.Rgb,        3, True,  3,  "rgb_from_xyz"),
    (ColorSpace.Rgb,       ColorSpace.LinearRgb,  3, True,  3,  "linear_rgb_from_rgb"),
    (ColorSpace.LinearRgb, ColorSpace.Rgb,        3, True,  3,  "rgb_from_linear_rgb"),
]


@pytest.mark.parametrize(
    "src_cs,dst_cs,src_ch,use_f32,expected_dst_ch,free_fn_name",
    _LEGAL_PARAMS,
)
def test_legal_pairs_parity(src_cs, dst_cs, src_ch, use_f32, expected_dst_ch, free_fn_name):
    rng = np.random.default_rng(42)
    if use_f32:
        arr = np.array(rng.random((8, 8, src_ch)), dtype=np.float32)
    else:
        arr = rng.integers(0, 256, (8, 8, src_ch), dtype=np.uint8)

    img = Image(arr, color_space=src_cs)
    result = img.cvt_color(dst_cs)

    # 1. color_space tag
    assert result.color_space == dst_cs

    # 2. shape and dtype
    result_arr = result.numpy()
    assert result_arr.shape == (8, 8, expected_dst_ch), (
        f"{src_cs}->{dst_cs}: expected shape (8, 8, {expected_dst_ch}), got {result_arr.shape}"
    )
    assert result_arr.dtype == arr.dtype, (
        f"{src_cs}->{dst_cs}: dtype changed from {arr.dtype} to {result_arr.dtype}"
    )

    # 3. free-function pixel parity (where available)
    if free_fn_name is not None:
        free_fn = getattr(imgproc, free_fn_name)
        expected = free_fn(arr)
        if use_f32:
            np.testing.assert_allclose(
                result_arr, expected, rtol=0, atol=1e-5,
                err_msg=f"{src_cs}->{dst_cs} parity with {free_fn_name} failed",
            )
        else:
            np.testing.assert_array_equal(
                result_arr, expected,
                err_msg=f"{src_cs}->{dst_cs} parity with {free_fn_name} failed",
            )


# ---------------------------------------------------------------------------
# Illegal-pair rejection matrix
# ---------------------------------------------------------------------------
# ColorSpace is not hashable, so we represent pairs as repr strings.
def _cs_pair(a: ColorSpace, b: ColorSpace) -> tuple:
    return (repr(a), repr(b))


LEGAL_PAIRS = {
    _cs_pair(ColorSpace.Rgb, ColorSpace.Gray),
    _cs_pair(ColorSpace.Gray, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Bgr),
    _cs_pair(ColorSpace.Bgr, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Rgba),
    _cs_pair(ColorSpace.Rgba, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Bgra),
    _cs_pair(ColorSpace.Bgra, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.YCbCr),
    _cs_pair(ColorSpace.YCbCr, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Yuv),
    _cs_pair(ColorSpace.Yuv, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Hsv),
    _cs_pair(ColorSpace.Hsv, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Hls),
    _cs_pair(ColorSpace.Hls, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Lab),
    _cs_pair(ColorSpace.Lab, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Luv),
    _cs_pair(ColorSpace.Luv, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.Xyz),
    _cs_pair(ColorSpace.Xyz, ColorSpace.Rgb),
    _cs_pair(ColorSpace.Rgb, ColorSpace.LinearRgb),
    _cs_pair(ColorSpace.LinearRgb, ColorSpace.Rgb),
}

ALL_CS = [
    ColorSpace.Rgb, ColorSpace.Bgr, ColorSpace.Gray, ColorSpace.Rgba,
    ColorSpace.Bgra, ColorSpace.Hsv, ColorSpace.Hls, ColorSpace.Lab,
    ColorSpace.Luv, ColorSpace.Xyz, ColorSpace.LinearRgb,
    ColorSpace.YCbCr, ColorSpace.Yuv,
]

# Channels and dtype to use for each source color space (minimally valid images).
# ColorSpace is not hashable, so we key by repr string.
_SRC_SPEC = {
    repr(ColorSpace.Gray):       (1, np.uint8),
    repr(ColorSpace.Rgba):       (4, np.uint8),
    repr(ColorSpace.Bgra):       (4, np.uint8),
    repr(ColorSpace.Hsv):        (3, np.float32),
    repr(ColorSpace.Hls):        (3, np.float32),
    repr(ColorSpace.Lab):        (3, np.float32),
    repr(ColorSpace.Luv):        (3, np.float32),
    repr(ColorSpace.Xyz):        (3, np.float32),
    repr(ColorSpace.LinearRgb):  (3, np.float32),
    repr(ColorSpace.Rgb):        (3, np.uint8),
    repr(ColorSpace.Bgr):        (3, np.uint8),
    repr(ColorSpace.YCbCr):      (3, np.uint8),
    repr(ColorSpace.Yuv):        (3, np.uint8),
}


def test_illegal_pairs_all_rejected():
    """Every (src, dst) pair NOT in LEGAL_PAIRS (and src != dst) must raise ValueError."""
    for src in ALL_CS:
        ch, dt = _SRC_SPEC[repr(src)]
        arr = np.zeros((4, 4, ch), dtype=dt)
        img = Image(arr, color_space=src)
        for dst in ALL_CS:
            if src == dst:
                continue  # identity — allowed
            if _cs_pair(src, dst) in LEGAL_PAIRS:
                continue  # legal — skip
            with pytest.raises(ValueError):
                img.cvt_color(dst)
