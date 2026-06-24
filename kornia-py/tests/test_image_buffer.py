"""Tests for Image buffer-protocol, pickle, dlpack, equality, and overflow."""
import pickle
import struct
import sys

import numpy as np
import pytest

from kornia_rs.image import Image, ColorSpace


# ---------------------------------------------------------------------------
# Read-only ingest
# ---------------------------------------------------------------------------

def test_from_buffer_bytes_readonly():
    """bytes is read-only — from_buffer succeeds; memoryview exposes readonly=True."""
    data = bytes(range(4 * 4 * 3))
    img = Image.from_buffer(data, width=4, height=4, channels=3)
    assert img.shape == (4, 4, 3)
    # The buffer is read-only: memoryview must reflect that
    mv = memoryview(img)
    assert mv.readonly is True
    # Writing through the view must raise
    with pytest.raises((TypeError, ValueError)):
        mv[0] = 1


def test_from_buffer_bytearray_writable():
    """bytearray is writable — from_buffer succeeds and memoryview is not readonly."""
    data = bytearray(4 * 4 * 3)
    img = Image.from_buffer(data, width=4, height=4, channels=3)
    assert img.shape == (4, 4, 3)
    mv = memoryview(img)
    # bytearray-backed buffer should be writable
    assert mv.readonly is False


# ---------------------------------------------------------------------------
# from_buffer dtypes / edges
# ---------------------------------------------------------------------------

def test_from_buffer_float32():
    raw = bytearray(struct.pack('<' + 'f' * (4 * 4 * 3), *([0.5] * (4 * 4 * 3))))
    img = Image.from_buffer(raw, width=4, height=4, channels=3, dtype="float32")
    assert img.dtype == np.float32
    assert img.shape == (4, 4, 3)


def test_from_buffer_uint16():
    raw = bytearray(struct.pack('<' + 'H' * (4 * 4 * 1), *([1000] * (4 * 4 * 1))))
    img = Image.from_buffer(raw, width=4, height=4, channels=1, dtype="uint16")
    assert img.dtype == np.uint16
    assert img.shape == (4, 4, 1)


def test_from_buffer_memoryview():
    ba = bytearray(4 * 4 * 3)
    mv = memoryview(ba)
    img = Image.from_buffer(mv, width=4, height=4, channels=3)
    assert img.shape == (4, 4, 3)


def test_from_buffer_undersized_raises():
    """Buffer too small for requested dimensions raises ValueError or OverflowError."""
    data = bytearray(10)  # way too small for 4x4x3=48 bytes
    with pytest.raises((ValueError, OverflowError)):
        Image.from_buffer(data, width=4, height=4, channels=3)


# ---------------------------------------------------------------------------
# Pickle with non-default color_space
# ---------------------------------------------------------------------------

def test_pickle_bgr_image():
    rng = np.random.default_rng(13)
    arr = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    img = Image(arr, color_space=ColorSpace.Bgr)
    loaded = pickle.loads(pickle.dumps(img))
    assert loaded.color_space == ColorSpace.Bgr
    np.testing.assert_array_equal(loaded.numpy(), arr)


def test_pickle_after_cvt_color():
    rng = np.random.default_rng(14)
    arr = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    img = Image(arr).to_bgr()
    loaded = pickle.loads(pickle.dumps(img))
    assert loaded.color_space == ColorSpace.Bgr


def test_pickle_colorspace_enum():
    for cs in [ColorSpace.Rgb, ColorSpace.Bgr, ColorSpace.Gray, ColorSpace.Hsv]:
        assert pickle.loads(pickle.dumps(cs)) == cs


# ---------------------------------------------------------------------------
# Owned-output independence
# ---------------------------------------------------------------------------

def test_cvt_color_output_independent_of_source():
    """cvt_color output is an owned buffer; mutating source does not affect it."""
    rng = np.random.default_rng(15)
    arr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8).copy()
    img = Image.from_numpy(arr, copy=False)
    gray = img.cvt_color(ColorSpace.Gray)
    pixels_before = gray.numpy().copy()
    arr[:] = 0
    np.testing.assert_array_equal(gray.numpy(), pixels_before)


def test_to_float_output_independent():
    """to_float output is an owned buffer; mutating source does not affect it."""
    rng = np.random.default_rng(16)
    arr = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8).copy()
    img = Image.from_numpy(arr, copy=False)
    out = img.to_float()
    pixels_before = out.numpy().copy()
    arr[:] = 255
    np.testing.assert_array_equal(out.numpy(), pixels_before)


# ---------------------------------------------------------------------------
# to_numpy copy semantics
# ---------------------------------------------------------------------------

def test_to_numpy_copy_false_shares_memory():
    """to_numpy(False) is a zero-copy view (positional arg, not keyword-only)."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image(arr)
    view = img.to_numpy(False)  # positional, not keyword-only
    assert np.shares_memory(view, img.numpy())


def test_to_numpy_copy_true_is_independent():
    """to_numpy(True) produces an independent copy."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image(arr)
    copy = img.to_numpy(True)
    copy[:] = 99
    assert not np.any(img.numpy() == 99)


def test_to_numpy_default_is_copy():
    """to_numpy() default (copy=True) does not share memory."""
    arr = np.ones((4, 4, 3), dtype=np.uint8) * 7
    img = Image(arr)
    out = img.to_numpy()  # default copy=True
    assert not np.shares_memory(out, img.numpy())


# ---------------------------------------------------------------------------
# __dlpack__ kwargs
# ---------------------------------------------------------------------------

def test_dlpack_non_cpu_device_raises_buffer_error():
    """Non-CPU dl_device raises BufferError."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image(arr)
    with pytest.raises(BufferError):
        img.__dlpack__(dl_device=(2, 0))  # kDLCUDA=2


def test_dlpack_copy_true_raises_not_implemented():
    """copy=True raises NotImplementedError."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image(arr)
    with pytest.raises(NotImplementedError):
        img.__dlpack__(copy=True)


# ---------------------------------------------------------------------------
# __eq__ / __hash__
# ---------------------------------------------------------------------------

def test_eq_different_color_space_not_equal():
    """Same pixels but different color_space → not equal."""
    rng = np.random.default_rng(20)
    arr = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    img_rgb = Image(arr, color_space=ColorSpace.Rgb)
    img_bgr = Image(arr, color_space=ColorSpace.Bgr)
    assert img_rgb != img_bgr


def test_image_hashable():
    """Image must be usable as a dict key / set member."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image(arr)
    s = {img}  # must not raise
    assert img in s


def test_hash_consistent_with_eq():
    """Equal images must have equal hashes."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img1 = Image(arr)
    img2 = Image(arr.copy())
    assert img1 == img2
    assert hash(img1) == hash(img2)


# ---------------------------------------------------------------------------
# Size overflow
# ---------------------------------------------------------------------------

def test_from_buffer_size_overflow_raises():
    """Requesting width*height*channels that overflows usize raises OverflowError."""
    data = bytearray(3)
    big = sys.maxsize  # 2^63-1 on 64-bit
    with pytest.raises((OverflowError, ValueError)):
        Image.from_buffer(data, width=big, height=big, channels=3)
