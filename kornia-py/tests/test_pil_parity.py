"""PIL-parity surface tests for kornia_rs.image.Image.

Covers the Phase 4 additions:
  - ``Image.open(fp)`` — accepts path / bytes / file-like
  - ``Image.new(mode, size, color)`` — blank canvas
  - ``img.tobytes()`` — raw HWC pixel buffer to ``bytes``
  - ``img.convert(mode)`` — between L / RGB / RGBA / I;16 / RGB;16 / RGBA;16
  - ``img.format`` — set by load/decode/open, ``None`` for in-memory
  - ``img.crop((left, upper, right, lower))`` — PIL 4-tuple signature

Each test asserts a memory or correctness property — perf is covered by
``test_zero_copy_io.py``.
"""

import io
import numpy as np
import pytest

from kornia_rs.image import Image


def _rand_u8(h, w, c=3):
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def _rand_u16(h, w, c=3):
    return np.random.randint(0, 65536, (h, w, c), dtype=np.uint16)


# --------------------------------------------------------------------- format


def test_format_in_memory_is_none():
    img = Image(_rand_u8(32, 32, 3))
    assert img.format is None


def test_format_load_png(tmp_path):
    p = tmp_path / "out.png"
    Image(_rand_u8(32, 32, 3)).save(str(p))
    img = Image.load(str(p))
    assert img.format == "PNG"


def test_format_load_jpeg(tmp_path):
    p = tmp_path / "out.jpg"
    Image(_rand_u8(32, 32, 3)).save(str(p))
    img = Image.load(str(p))
    assert img.format == "JPEG"


def test_format_decode_png_set():
    blob = Image(_rand_u8(32, 32, 3)).encode("png")
    img = Image.decode(bytes(blob))
    assert img.format == "PNG"


def test_format_decode_jpeg_set():
    blob = Image(_rand_u8(32, 32, 3)).encode("jpeg")
    img = Image.decode(bytes(blob))
    assert img.format == "JPEG"


# ---------------------------------------------------------- tobytes / Image.new


def test_tobytes_u8_roundtrip():
    arr = _rand_u8(16, 24, 3)
    img = Image(arr)
    raw = img.tobytes()
    assert isinstance(raw, bytes)
    assert len(raw) == 16 * 24 * 3
    assert raw == arr.tobytes()


def test_tobytes_u16_roundtrip():
    arr = _rand_u16(16, 24, 1)
    img = Image(arr)
    raw = img.tobytes()
    assert isinstance(raw, bytes)
    assert len(raw) == 16 * 24 * 1 * 2
    assert raw == arr.tobytes()


def test_image_new_default_zero():
    img = Image.new("RGB", (8, 4))
    assert img.shape == (4, 8, 3)
    assert img.dtype == np.uint8
    np.testing.assert_array_equal(img.data, np.zeros((4, 8, 3), dtype=np.uint8))


def test_image_new_scalar_color():
    img = Image.new("RGBA", (5, 3), 42)
    assert img.shape == (3, 5, 4)
    assert (img.data == 42).all()


def test_image_new_tuple_color():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    np.testing.assert_array_equal(img.data[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(img.data[3, 3], [10, 20, 30])


def test_image_new_u16_tuple_color():
    img = Image.new("RGB;16", (3, 3), (1000, 2000, 3000))
    assert img.dtype == np.uint16
    np.testing.assert_array_equal(img.data[0, 0], [1000, 2000, 3000])


def test_image_new_rejects_bad_color_arity():
    with pytest.raises(ValueError, match="color tuple has"):
        Image.new("RGB", (4, 4), (1, 2))


def test_image_new_rejects_bad_mode():
    with pytest.raises(ValueError, match="unsupported mode"):
        Image.new("CMYK", (4, 4))


# ---------------------------------------------------------------- crop 4-tuple


def test_crop_pil_4tuple_signature():
    arr = _rand_u8(10, 10, 3)
    img = Image(arr)
    cropped = img.crop((2, 3, 7, 8))  # left, upper, right, lower
    assert cropped.shape == (5, 5, 3)
    np.testing.assert_array_equal(cropped.data, arr[3:8, 2:7, :])


def test_crop_kornia_xywh_signature():
    arr = _rand_u8(10, 10, 3)
    img = Image(arr)
    cropped = img.crop(2, 3, 5, 5)  # x, y, width, height
    assert cropped.shape == (5, 5, 3)
    np.testing.assert_array_equal(cropped.data, arr[3:8, 2:7, :])


def test_crop_pil_and_kornia_match():
    arr = _rand_u8(10, 10, 3)
    img = Image(arr)
    pil = img.crop((2, 3, 7, 8))
    kornia = img.crop(2, 3, 5, 5)
    np.testing.assert_array_equal(pil.data, kornia.data)


def test_crop_rejects_mixed_signatures():
    img = Image(_rand_u8(10, 10, 3))
    with pytest.raises(ValueError, match="not both"):
        img.crop((1, 2, 3, 4), 5, 6, 7)


def test_crop_rejects_bad_pil_box():
    img = Image(_rand_u8(10, 10, 3))
    with pytest.raises(ValueError, match="right >= left"):
        img.crop((5, 5, 2, 8))


# -------------------------------------------------------------- Image.open(fp)


def test_open_path(tmp_path):
    arr = _rand_u8(16, 16, 3)
    p = tmp_path / "in.png"
    Image(arr).save(str(p))

    img = Image.open(str(p))
    np.testing.assert_array_equal(img.data, arr)
    assert img.format == "PNG"


def test_open_bytes():
    arr = _rand_u8(16, 16, 3)
    blob = Image(arr).encode("png")
    img = Image.open(bytes(blob))
    np.testing.assert_array_equal(img.data, arr)
    assert img.format == "PNG"


def test_open_filelike():
    arr = _rand_u8(16, 16, 3)
    blob = Image(arr).encode("png")
    img = Image.open(io.BytesIO(bytes(blob)))
    np.testing.assert_array_equal(img.data, arr)


def test_open_rejects_bad_input():
    with pytest.raises(ValueError, match="path string, bytes"):
        Image.open(42)


# -------------------------------------------------------- convert(mode)


def test_convert_same_mode_copy():
    arr = _rand_u8(8, 8, 3)
    img = Image(arr)
    converted = img.convert("RGB")
    np.testing.assert_array_equal(converted.data, arr)
    # must be an independent buffer
    assert not np.shares_memory(converted.data, img.data)


def test_convert_rgb_to_l():
    img = Image.new("RGB", (4, 4), (255, 255, 255))
    gray = img.convert("L")
    assert gray.mode == "L"
    assert gray.shape == (4, 4, 1)
    assert (gray.data == 255).all()


def test_convert_rgb_to_rgba_alpha_255():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    rgba = img.convert("RGBA")
    assert rgba.mode == "RGBA"
    assert rgba.shape == (4, 4, 4)
    np.testing.assert_array_equal(rgba.data[..., :3], img.data)
    assert (rgba.data[..., 3] == 255).all()


def test_convert_l_to_rgb_replicates():
    img = Image.new("L", (4, 4), 128)
    rgb = img.convert("RGB")
    assert rgb.shape == (4, 4, 3)
    assert (rgb.data == 128).all()


def test_convert_l_to_i16_x257():
    img = Image.new("L", (4, 4), 128)
    i16 = img.convert("I;16")
    assert i16.dtype == np.uint16
    assert i16.shape == (4, 4, 1)
    assert (i16.data == 128 * 257).all()


def test_convert_i16_to_l_high_byte():
    arr = np.full((4, 4, 1), 0xABCD, dtype=np.uint16)
    img = Image(arr)
    assert img.mode == "I;16"
    l = img.convert("L")
    assert l.dtype == np.uint8
    assert (l.data == 0xAB).all()


def test_convert_unsupported_pair():
    img = Image.new("L", (4, 4))
    with pytest.raises(ValueError, match="not supported"):
        img.convert("CMYK")
