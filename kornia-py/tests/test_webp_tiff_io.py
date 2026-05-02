"""WebP + TIFF coverage tests for Image.encode/save/decode/load.

WebP supports u8 only (1/3/4 channels — gray/RGB/RGBA, lossy at default
quality). TIFF supports u8 and u16 with 1/3 channels (gray + RGB) plus
f32 via the dedicated kornia_rs.io.read_image_tiff_f32 path.
"""

import io
import numpy as np
import pytest

from kornia_rs.image import Image


def _rand_u8(h, w, c=3):
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def _rand_u16(h, w, c=3):
    return np.random.randint(0, 65536, (h, w, c), dtype=np.uint16)


# ----------------------------------------------------------------------- WEBP


def test_encode_webp_u8_rgb():
    arr = _rand_u8(32, 48, 3)
    blob = Image(arr).encode("webp")
    assert isinstance(blob, bytes)
    # WebP RIFF / VP8 signature
    assert blob[0:4] == b"RIFF"
    assert blob[8:12] == b"WEBP"


def test_decode_webp_roundtrip_lossy_close():
    arr = _rand_u8(32, 48, 3)
    blob = Image(arr).encode("webp")
    decoded = Image.decode(bytes(blob))
    assert decoded.format == "WEBP"
    assert decoded.shape == arr.shape
    # WebP is lossy at default quality — assert visually-close, not equal.
    diff = np.abs(decoded.data.astype(np.int32) - arr.astype(np.int32))
    assert diff.mean() < 30.0


def test_save_load_webp_roundtrip(tmp_path):
    arr = _rand_u8(32, 48, 3)
    p = tmp_path / "out.webp"
    Image(arr).save(str(p))
    loaded = Image.load(str(p))
    assert loaded.format == "WEBP"
    assert loaded.shape == arr.shape


def test_webp_rejects_u16():
    arr = _rand_u16(16, 16, 3)
    with pytest.raises(ValueError, match="WebP encode requires uint8"):
        Image(arr).encode("webp")


def test_webp_rgba_roundtrip():
    arr = _rand_u8(16, 16, 4)
    img = Image(arr)
    blob = img.encode("webp")
    decoded = Image.decode(bytes(blob), mode="RGBA")
    assert decoded.shape == (16, 16, 4)
    assert decoded.format == "WEBP"


def test_webp_open_via_bytes():
    arr = _rand_u8(16, 16, 3)
    blob = Image(arr).encode("webp")
    decoded = Image.open(io.BytesIO(bytes(blob)))
    assert decoded.format == "WEBP"


# ----------------------------------------------------------------------- TIFF


def test_encode_tiff_u8_rgb():
    arr = _rand_u8(16, 24, 3)
    blob = Image(arr).encode("tiff")
    assert isinstance(blob, bytes)
    # Little-endian TIFF magic
    assert blob[0:4] == b"II*\x00"


def test_tiff_u8_lossless_roundtrip():
    arr = _rand_u8(16, 24, 3)
    img = Image(arr)
    blob = img.encode("tiff")
    decoded = Image.decode(bytes(blob))
    assert decoded.format == "TIFF"
    np.testing.assert_array_equal(decoded.data, arr)


def test_tiff_u16_lossless_roundtrip():
    arr = _rand_u16(16, 24, 3)
    img = Image(arr)
    blob = img.encode("tiff")
    decoded = Image.decode(bytes(blob))
    assert decoded.format == "TIFF"
    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded.data, arr)


def test_tiff_u16_gray_lossless_roundtrip():
    arr = _rand_u16(16, 16, 1)
    img = Image(arr)
    blob = img.encode("tiff")
    decoded = Image.decode(bytes(blob), mode="L")
    np.testing.assert_array_equal(decoded.data, arr)


def test_save_load_tiff_roundtrip(tmp_path):
    arr = _rand_u8(20, 20, 3)
    p = tmp_path / "out.tiff"
    Image(arr).save(str(p))
    loaded = Image.load(str(p))
    assert loaded.format == "TIFF"
    np.testing.assert_array_equal(loaded.data, arr)


def test_save_load_tif_extension(tmp_path):
    """Common alternate extension (.tif) should also work."""
    arr = _rand_u8(20, 20, 3)
    p = tmp_path / "out.tif"
    Image(arr).save(str(p))
    loaded = Image.load(str(p))
    assert loaded.format == "TIFF"


def test_tiff_open_via_bytes():
    arr = _rand_u8(16, 16, 3)
    blob = Image(arr).encode("tiff")
    decoded = Image.open(bytes(blob))
    assert decoded.format == "TIFF"


def test_encode_unsupported_format_rejects():
    arr = _rand_u8(8, 8, 3)
    with pytest.raises(ValueError, match="Unsupported format"):
        Image(arr).encode("bmp")


# --------------------------------------------------------------- zero-copy invariants


def test_webp_encode_zero_copy_input():
    arr = _rand_u8(16, 16, 3)
    img = Image(arr)
    arr_before = arr.copy()
    _ = img.encode("webp")
    np.testing.assert_array_equal(arr, arr_before)
    assert np.shares_memory(img.data, arr)


def test_tiff_encode_zero_copy_input():
    arr = _rand_u16(16, 16, 3)
    img = Image(arr)
    arr_before = arr.copy()
    _ = img.encode("tiff")
    np.testing.assert_array_equal(arr, arr_before)
    assert np.shares_memory(img.data, arr)
