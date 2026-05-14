"""float32 (PIL ``"F"`` mode) storage variant tests.

Phase 7 — adds ``ImageData::F32`` alongside U8 / U16. Covers:
  - Constructor: ``Image(arr_f32)`` / ``Image.frombuffer`` / ``Image.fromarray``
  - ``Image.new("F" / "RGBf" / "RGBAf")`` blank canvas + color fill
  - dtype / mode / nbytes / itemsize getters
  - flip + crop on f32 (the dtype-trivial imgproc surface)
  - tobytes round-trip vs numpy.tobytes()
  - TIFF f32 encode produces valid bytes
  - Round-trip via __reduce__ preserves f32 variant
  - JPEG / PNG / WebP reject f32 with a clear error
  - Math-heavy imgproc raises NotImplementedError mentioning "convert"
"""

import numpy as np
import pytest

from kornia_rs.image import Image


def _f32(h, w, c=3):
    return np.random.rand(h, w, c).astype(np.float32)


def test_image_ctor_detects_f32():
    arr = _f32(8, 8, 3)
    img = Image(arr)
    assert img.dtype == np.float32
    assert img.mode == "RGBf"


def test_image_frombuffer_f32():
    arr = _f32(8, 8, 1)
    img = Image.frombuffer(arr)
    assert img.dtype == np.float32
    assert img.mode == "F"


def test_image_fromarray_f32():
    arr = _f32(8, 8, 4)
    img = Image.fromarray(arr)
    assert img.dtype == np.float32
    assert img.mode == "RGBAf"


def test_image_new_f_zero():
    img = Image.new("F", (4, 8))
    assert img.dtype == np.float32
    assert img.shape == (8, 4, 1)
    assert (img.data == 0.0).all()


def test_image_new_rgbf_with_color():
    img = Image.new("RGBf", (3, 3), (0.1, 0.2, 0.3))
    np.testing.assert_allclose(img.data[0, 0], [0.1, 0.2, 0.3])


def test_image_new_f32_scalar_color():
    img = Image.new("F", (5, 5), 1.5)
    assert (img.data == 1.5).all()


def test_image_new_f32_rejects_bad_arity():
    with pytest.raises(ValueError, match="color tuple has"):
        Image.new("RGBf", (4, 4), (0.1, 0.2))


def test_f32_dtype_and_itemsize_and_nbytes():
    arr = _f32(10, 12, 3)
    img = Image(arr)
    assert img.dtype == np.float32
    assert img.nbytes == 10 * 12 * 3 * 4
    assert img.shape == (10, 12, 3)
    assert img.size == (12, 10)


def test_f32_repr():
    img = Image(_f32(4, 4, 3))
    s = repr(img)
    assert "float32" in s
    assert "RGBf" in s


def test_flip_horizontal_f32_lossless():
    arr = _f32(8, 12, 1)
    flipped = Image(arr).flip_horizontal()
    assert flipped.dtype == np.float32
    np.testing.assert_array_equal(flipped.data, arr[:, ::-1, :])


def test_flip_vertical_f32_lossless():
    arr = _f32(10, 5, 3)
    flipped = Image(arr).flip_vertical()
    np.testing.assert_array_equal(flipped.data, arr[::-1, :, :])


def test_crop_f32():
    arr = _f32(12, 16, 3)
    cropped = Image(arr).crop(2, 3, 6, 5)
    assert cropped.dtype == np.float32
    assert cropped.shape == (5, 6, 3)
    np.testing.assert_array_equal(cropped.data, arr[3:8, 2:8, :])


def test_f32_tobytes_roundtrip():
    arr = _f32(8, 12, 3)
    raw = Image(arr).tobytes()
    assert isinstance(raw, bytes)
    assert len(raw) == 8 * 12 * 3 * 4
    assert raw == arr.tobytes()


def test_f32_to_numpy_returns_copy():
    arr = _f32(4, 4, 3)
    img = Image(arr)
    out = img.to_numpy()
    np.testing.assert_array_equal(out, arr)
    assert not np.shares_memory(out, arr)


def test_f32_data_is_zero_copy_view():
    arr = _f32(4, 4, 3)
    img = Image(arr)
    assert np.shares_memory(img.data, arr)


def test_encode_tiff_f32_rgb_writes_valid():
    arr = _f32(8, 12, 3)
    blob = Image(arr).encode("tiff")
    assert blob[0:4] == b"II*\x00"


def test_encode_tiff_f32_mono_writes_valid():
    arr = _f32(8, 8, 1)
    blob = Image(arr).encode("tiff")
    assert blob[0:4] == b"II*\x00"


def test_jpeg_rejects_f32():
    img = Image(_f32(8, 8, 3))
    with pytest.raises(ValueError, match="JPEG cannot encode float32"):
        img.encode("jpeg")


def test_png_rejects_f32():
    img = Image(_f32(8, 8, 3))
    with pytest.raises(ValueError):
        img.encode("png")


def test_webp_rejects_f32():
    img = Image(_f32(8, 8, 3))
    with pytest.raises(ValueError):
        img.encode("webp")


def test_resize_f32_unsupported():
    img = Image(_f32(8, 8, 3))
    with pytest.raises(NotImplementedError, match="convert"):
        img.resize(4, 4)


def test_gaussian_blur_f32_unsupported():
    img = Image(_f32(8, 8, 3))
    with pytest.raises(NotImplementedError, match="convert"):
        img.gaussian_blur()


def test_f32_state_roundtrip_preserves_variant():
    """__reduce__ + reconstruction restores the F32 variant (mode lost-but-default)."""
    arr = _f32(4, 4, 3)
    img = Image(arr)
    cls, args = img.__reduce__()
    restored = cls(*args)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored.data, arr)


def test_f32_eq_self():
    arr = _f32(4, 4, 3)
    img1 = Image(arr.copy())
    img2 = Image(arr.copy())
    assert img1 == img2


def test_f32_eq_different_dtypes_not_equal():
    arr_f = _f32(4, 4, 3)
    arr_u8 = (arr_f * 255).astype(np.uint8)
    assert Image(arr_f) != Image(arr_u8)
