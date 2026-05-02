"""Image ↔ torch zero-copy contract tests.

The Image class doesn't import torch — the bridge is via numpy. Both
``torch.from_numpy(img.data)`` and ``torch.from_dlpack(img.data)`` must
share storage with the underlying numpy buffer (which itself is a
zero-copy view of the user's input array, per test_zero_copy_io.py).

These tests pin the contract: a mutation through the torch tensor must
be visible on the original numpy array, and the buffer pointers must
match. Skipped gracefully if torch isn't installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kornia_rs.image import Image


# --------------------------------------------------------------- u8


def test_u8_torch_from_numpy_shares_buffer():
    arr = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
    img = Image(arr)
    t = torch.from_numpy(img.data)
    assert tuple(t.shape) == arr.shape
    assert t.dtype == torch.uint8
    assert t.data_ptr() == arr.ctypes.data, "torch tensor and numpy array do not share storage"


def test_u8_torch_mutation_visible_on_numpy():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = Image(arr)
    t = torch.from_numpy(img.data)

    t[2, 3, 1] = 200
    assert int(arr[2, 3, 1]) == 200, "mutation through torch did not propagate — extra copy somewhere"


def test_u8_torch_dlpack_shares_buffer():
    arr = np.random.randint(0, 256, (16, 24, 3), dtype=np.uint8)
    img = Image(arr)
    t = torch.from_dlpack(img.data)
    assert t.data_ptr() == arr.ctypes.data
    t[0, 0, 0] = 99
    assert int(arr[0, 0, 0]) == 99


# --------------------------------------------------------------- u16


def test_u16_torch_from_numpy_shares_buffer():
    arr = np.zeros((10, 12, 1), dtype=np.uint16)
    img = Image(arr)
    t = torch.from_numpy(img.data)
    assert t.dtype == torch.int16 or t.dtype == torch.uint16  # torch's u16 dtype name varies by version
    assert t.data_ptr() == arr.ctypes.data


def test_u16_torch_mutation_visible_on_numpy():
    arr = np.zeros((4, 4, 1), dtype=np.uint16)
    img = Image(arr)
    t = torch.from_numpy(img.data)
    t[1, 1, 0] = 50000
    assert int(arr[1, 1, 0]) == 50000


# --------------------------------------------------------------- f32


def test_f32_torch_from_numpy_shares_buffer():
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    img = Image(arr)
    t = torch.from_numpy(img.data)
    assert t.dtype == torch.float32
    assert t.data_ptr() == arr.ctypes.data


def test_f32_torch_mutation_visible_on_numpy():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    img = Image(arr)
    t = torch.from_numpy(img.data)
    t[2, 2, 1] = 0.75
    assert float(arr[2, 2, 1]) == 0.75


def test_f32_torch_dlpack_shares_buffer():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    img = Image(arr)
    t = torch.from_dlpack(img.data)
    assert t.data_ptr() == arr.ctypes.data


# --------------------------------------------------------------- decode → torch


def test_decoded_image_to_torch_independent_per_decode():
    """A freshly-decoded Image owns a fresh PyArray. Two torch tensors from
    two decodes of the same blob must NOT share storage."""
    arr = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    blob = Image(arr).encode("png")

    a = Image.decode(bytes(blob))
    b = Image.decode(bytes(blob))
    ta = torch.from_numpy(a.data)
    tb = torch.from_numpy(b.data)
    assert ta.data_ptr() != tb.data_ptr()


def test_image_data_torch_survives_image_drop():
    """The torch tensor must keep the numpy buffer alive after the Image goes
    out of scope — a pure refcount-bump, no fresh copy required."""
    arr = np.full((8, 8, 3), 7, dtype=np.uint8)
    img = Image(arr)
    np_view = img.data  # zero-copy view
    t = torch.from_numpy(np_view)
    del img  # Image gone; numpy view + tensor still hold refs to arr's storage
    assert int(t[0, 0, 0]) == 7
    np_view[1, 1, 1] = 11
    assert int(t[1, 1, 1]) == 11


# --------------------------------------------------------------- HWC vs CHW reminder


def test_torch_permute_is_a_copy():
    """Documentation/contract test: HWC -> CHW via .permute is a *view*, but
    .contiguous() after that *is* a copy. Useful for users wiring Image into
    a CNN pipeline."""
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    img = Image(arr)
    t_hwc = torch.from_numpy(img.data)
    assert t_hwc.is_contiguous()
    t_chw_view = t_hwc.permute(2, 0, 1)
    assert not t_chw_view.is_contiguous()
    # permute alone is still zero-copy (just a stride trick)
    assert t_chw_view.data_ptr() == arr.ctypes.data
    # But contiguous-CHW for a CNN forward pass IS a copy
    t_chw_contig = t_chw_view.contiguous()
    assert t_chw_contig.data_ptr() != arr.ctypes.data
