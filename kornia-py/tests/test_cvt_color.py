"""Minimal smoke tests for Image.cvt_color, color_space field, and dtype helpers."""
import numpy as np
import kornia_rs
from kornia_rs.image import Image, ColorSpace


def test_color_space_defaults_and_getter():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image(arr)
    assert img.color_space == ColorSpace.Rgb
