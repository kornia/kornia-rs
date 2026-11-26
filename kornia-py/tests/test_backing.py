import numpy as np
import pytest
from kornia_rs.image import Image


def test_from_numpy_zero_copy_shares_memory():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = Image(arr)                      # borrow by default
    arr[1, 2, 0] = 200
    assert img.numpy()[1, 2, 0] == 200    # shares memory
    img.numpy()[3, 3, 1] = 50
    assert arr[3, 3, 1] == 50


def test_from_numpy_copy_is_independent():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image.from_numpy(arr, copy=True)
    arr[0, 0, 0] = 99
    assert img.numpy()[0, 0, 0] == 0


def test_owned_output_survives_source_free():
    arr = np.random.randint(0, 255, (16, 16, 3), np.uint8)
    img = Image(arr)
    out = img.resize(8, 8)                # owned output
    del arr, img
    assert out.numpy().shape == (8, 8, 3)  # still valid


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.int32])
def test_from_numpy_strided_slice_all_dtypes(dtype):
    """Test that strided slices gather correctly and match numpy across all supported dtypes."""
    if issubclass(dtype, np.floating):
        raw = np.linspace(0.0, 100.0, 32 * 48 * 3, dtype=dtype).reshape(32, 48, 3)
    else:
        raw = np.arange(32 * 48 * 3, dtype=dtype).reshape(32, 48, 3)

    # Subsampled view: step 2 vertically and step 3 horizontally
    strided = raw[::2, ::3, :]
    assert not strided.flags.c_contiguous

    img = Image(strided)
    out = img.numpy()

    assert out.shape == strided.shape
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, strided)

    # Verify that the ingested buffer is owned and independent from the source
    orig_val = raw[0, 0, 0]
    raw[0, 0, 0] = (orig_val + 50) if not issubclass(dtype, np.floating) else (orig_val + 50.0)
    assert out[0, 0, 0] == orig_val


def test_from_numpy_transposed():
    """Test non-contiguous transposed views (e.g., column-major / Fortran order)."""
    raw = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    transposed = raw.transpose(1, 0, 2)  # shape (30, 20, 3)
    assert not transposed.flags.c_contiguous

    img = Image(transposed)
    out = img.numpy()

    assert out.shape == (30, 20, 3)
    np.testing.assert_array_equal(out, transposed)


def test_from_numpy_negative_strides():
    """Test flipped views with negative strides."""
    raw = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)

    # Vertical flip
    v_flipped = raw[::-1, :, :]
    img_v = Image(v_flipped)
    np.testing.assert_array_equal(img_v.numpy(), v_flipped)

    # Horizontal flip
    h_flipped = raw[:, ::-1, :]
    img_h = Image(h_flipped)
    np.testing.assert_array_equal(img_h.numpy(), h_flipped)

    # Both flipped
    both_flipped = raw[::-1, ::-1, :]
    img_both = Image(both_flipped)
    np.testing.assert_array_equal(img_both.numpy(), both_flipped)


def test_from_numpy_broadcast_array():
    """Test broadcast array with stride 0."""
    raw = np.arange(10 * 3, dtype=np.float32).reshape(10, 1, 3)
    broadcast = np.broadcast_to(raw, (10, 20, 3))
    assert not broadcast.flags.c_contiguous

    img = Image(broadcast)
    out = img.numpy()

    assert out.shape == (10, 20, 3)
    np.testing.assert_array_equal(out, broadcast)


def test_from_numpy_negative_and_broadcast_channel_strides():
    """Test negative, zero-stride broadcast, and channel-outer strides across gather paths.

    - ``[:, :, ::-1]`` (`strides[2] == -1`, `C == 3`) hits the reversed 3-channel loop.
    - Broadcast (`strides[2] == 0`) and ``CHW -> HWC`` transpose hit the general aligned loop.
    """
    raw = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    bgr = raw[:, :, ::-1]
    assert bgr.strides[2] == -1
    np.testing.assert_array_equal(Image(bgr).numpy(), bgr)

    gray_broadcast = np.broadcast_to(raw[:, :, :1], (12, 16, 3))
    assert gray_broadcast.strides[2] == 0
    np.testing.assert_array_equal(Image(gray_broadcast).numpy(), gray_broadcast)

    chw_to_hwc = np.arange(3 * 12 * 16, dtype=np.uint8).reshape(3, 12, 16).transpose(1, 2, 0)
    assert chw_to_hwc.strides[2] == 12 * 16
    np.testing.assert_array_equal(Image(chw_to_hwc).numpy(), chw_to_hwc)


def test_from_numpy_empty_strided_slice():
    """Zero-width strided slice with non-zero height must not trigger pointer offset UB."""
    raw = np.empty((6, 4, 3), dtype=np.uint8)
    empty_w = raw[::2, :0, :]
    img = Image(empty_w)
    out = img.numpy()
    assert out.shape == (3, 0, 3)
    assert out.dtype == np.uint8


def test_from_numpy_strided_misaligned_borrow():
    """Strided view with misaligned element strides still copies safely when copy=False."""
    rec = np.zeros((3, 4, 3), dtype=np.dtype([("val", "<f4"), ("pad", "u1")], align=False))
    expected = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
    rec["val"] = expected
    view = rec["val"]
    assert not view.flags.aligned
    img = Image.from_numpy(view, copy=False)
    out = img.numpy()
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float32
    rec["val"][0, 0, 0] = -1.0
    assert out[0, 0, 0] == 0.0
