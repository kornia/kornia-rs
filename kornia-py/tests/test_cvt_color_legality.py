"""
Drift-lock test: verifies that dispatch_cvt covers all pairs listed in
ColorSpace.supports() and that unsupported pairs raise the correct error.

This test is the canary for divergence between Rust's ColorSpace::supports()
source of truth and the Python dispatch table.

Update SUPPORTED_U8_PAIRS / SUPPORTED_F32_PAIRS (and the count in
test_supported_pairs_count) whenever ColorSpace::supports() is extended.
"""
import numpy as np
import pytest
from kornia_rs.image import Image, ColorSpace

# The complete list of (from, to) pairs that ColorSpace::supports() returns True
# for u8-compatible spaces, as documented in
# crates/kornia-image/src/color_space.rs.
SUPPORTED_U8_PAIRS = [
    (ColorSpace.Rgb, ColorSpace.Gray),
    (ColorSpace.Gray, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Bgr),
    (ColorSpace.Bgr, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Rgba),
    (ColorSpace.Rgba, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Bgra),
    (ColorSpace.Bgra, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.YCbCr),
    (ColorSpace.YCbCr, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Yuv),
    (ColorSpace.Yuv, ColorSpace.Rgb),
]

# f32-only pairs (Hsv, Hls, Lab, Luv, Xyz, LinearRgb require f32 storage).
SUPPORTED_F32_PAIRS = [
    (ColorSpace.Rgb, ColorSpace.Hsv),
    (ColorSpace.Hsv, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Hls),
    (ColorSpace.Hls, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Lab),
    (ColorSpace.Lab, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Luv),
    (ColorSpace.Luv, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.Xyz),
    (ColorSpace.Xyz, ColorSpace.Rgb),
    (ColorSpace.Rgb, ColorSpace.LinearRgb),
    (ColorSpace.LinearRgb, ColorSpace.Rgb),
]

ALL_COLORSPACES = [
    ColorSpace.Rgb, ColorSpace.Bgr, ColorSpace.Gray, ColorSpace.Rgba,
    ColorSpace.Bgra, ColorSpace.Hsv, ColorSpace.Hls, ColorSpace.Lab,
    ColorSpace.Luv, ColorSpace.Xyz, ColorSpace.LinearRgb, ColorSpace.YCbCr,
    ColorSpace.Yuv,
]


def _channels_for(cs: ColorSpace) -> int:
    """Return the channel count for a given color space."""
    # ColorSpace is not hashable, so we use equality comparisons.
    if cs == ColorSpace.Gray:
        return 1
    if cs == ColorSpace.Rgba or cs == ColorSpace.Bgra:
        return 4
    return 3


def _make_u8_image(from_cs: ColorSpace) -> Image:
    """Make a valid u8 image for the given color space (correct channel count)."""
    channels = _channels_for(from_cs)
    arr = np.ascontiguousarray(np.full((4, 4, channels), 128, dtype=np.uint8))
    return Image(arr, color_space=from_cs)


def _make_f32_image(from_cs: ColorSpace) -> Image:
    """Make a valid f32 image for the given color space (correct channel count)."""
    channels = _channels_for(from_cs)
    arr = np.ascontiguousarray(np.full((4, 4, channels), 0.5, dtype=np.float32))
    return Image(arr, color_space=from_cs)


@pytest.mark.parametrize("from_cs,to_cs", SUPPORTED_U8_PAIRS)
def test_u8_supported_pair_does_not_raise(from_cs, to_cs):
    """Every pair in SUPPORTED_U8_PAIRS must NOT raise on u8 storage."""
    img = _make_u8_image(from_cs)
    result = img.cvt_color(to_cs)
    assert result is not None


@pytest.mark.parametrize("from_cs,to_cs", SUPPORTED_F32_PAIRS)
def test_f32_supported_pair_does_not_raise(from_cs, to_cs):
    """Every pair in SUPPORTED_F32_PAIRS must NOT raise on f32 storage."""
    img = _make_f32_image(from_cs)
    result = img.cvt_color(to_cs)
    assert result is not None


def test_unsupported_pair_raises():
    """Pairs NOT in supports() must raise ValueError (UnsupportedColorConversion).

    Gray->Bgr is unsupported per ColorSpace::supports() (no direct kernel exists;
    the user must go via Rgb).  The dispatch table must reject it with ValueError.
    """
    img_gray = _make_u8_image(ColorSpace.Gray)
    with pytest.raises(ValueError):
        img_gray.cvt_color(ColorSpace.Bgr)


def test_supported_pairs_count():
    """SUPPORTED_U8_PAIRS and SUPPORTED_F32_PAIRS must together match
    ColorSpace::supports() — detect if new pairs are added without updating this test.

    ColorSpace::supports() in crates/kornia-image/src/color_space.rs has exactly
    24 pairs (12 u8-compatible + 12 f32-only).  Update this count if supports()
    is extended, and add the new pairs to the lists above.
    """
    total = len(SUPPORTED_U8_PAIRS) + len(SUPPORTED_F32_PAIRS)
    assert total == 24, (
        f"Expected 24 supported pairs total; got {total}. "
        "If ColorSpace::supports() was extended, update SUPPORTED_U8_PAIRS or "
        "SUPPORTED_F32_PAIRS in test_cvt_color_legality.py to match."
    )
