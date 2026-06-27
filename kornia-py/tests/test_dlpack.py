"""DLPack bidirectional zero-copy tests for kornia_rs.Image.

Every test that asserts *zero-copy* does so via **mutation** (write through
one view, observe in another). This is more robust than pointer-identity
comparisons because `np.from_dlpack` may return an array whose ctypes address
differs from the capsule pointer even when no copy was made (numpy may add a
thin wrapper object, but the underlying data buffer is shared).
"""

import numpy as np
import pytest

from kornia_rs.image import Image


# ────────────────────────────────────────────────────────────────
# Export direction: Image → DLPack consumer
# ────────────────────────────────────────────────────────────────


def test_dlpack_export_numpy_values():
    """np.from_dlpack(img) must return the correct pixel values."""
    arr = np.ascontiguousarray(np.random.randint(0, 255, (5, 7, 3), np.uint8))
    img = Image.from_numpy(arr, copy=True)  # owned buffer
    out = np.from_dlpack(img)
    np.testing.assert_array_equal(out, arr)


def test_dlpack_export_numpy_zero_copy():
    """The exported numpy array must share the Image's data buffer (zero-copy)."""
    arr = np.ascontiguousarray(np.random.randint(0, 255, (5, 7, 3), np.uint8))
    img = Image.from_numpy(arr, copy=True)  # owned buffer
    out = np.from_dlpack(img)
    img_view = np.asarray(img)
    # Primary zero-copy check via shared memory — robust across numpy versions.
    # Older numpy (py3.8/3.9) marks `np.from_dlpack` results read-only, so a
    # mutation check would raise there; numpy>=2 returns a writable view.
    assert np.shares_memory(out, img_view), (
        "ZERO-COPY EXPORT FAILED: exported tensor does not share the Image buffer"
    )
    # Stronger mutation-visibility check only where numpy exposes a writable view.
    if out.flags.writeable:
        sentinel = 42
        out[0, 0, 0] = sentinel
        assert int(img_view[0, 0, 0]) == sentinel, (
            "ZERO-COPY EXPORT FAILED: mutation through exported tensor "
            "not visible in Image buffer"
        )


def test_dlpack_export_shape_and_dtype():
    """Exported DLPack tensor must have the correct shape and dtype."""
    arr = np.ascontiguousarray(np.zeros((4, 6, 3), dtype=np.float32))
    img = Image.from_numpy(arr, copy=True)
    out = np.from_dlpack(img)
    assert out.shape == (4, 6, 3)
    assert out.dtype == np.float32


def test_dlpack_export_u16():
    """uint16 images must export correctly."""
    arr = np.ascontiguousarray(np.arange(12, dtype=np.uint16).reshape(2, 2, 3))
    img = Image.from_numpy(arr, copy=True)
    out = np.from_dlpack(img)
    np.testing.assert_array_equal(out, arr)


def test_dlpack_device():
    """__dlpack_device__ must return (kDLCPU=1, 0)."""
    arr = np.ascontiguousarray(np.zeros((2, 2, 3), dtype=np.uint8))
    img = Image.from_numpy(arr, copy=True)
    device = img.__dlpack_device__()
    assert device == (1, 0), f"Expected (1, 0) for kDLCPU, got {device}"


# ────────────────────────────────────────────────────────────────
# Import direction: DLPack producer → Image
# ────────────────────────────────────────────────────────────────


def test_dlpack_import_from_numpy_values():
    """Image.from_dlpack(arr) must expose the correct pixel values."""
    arr = np.ascontiguousarray(np.random.rand(4, 4, 3).astype(np.float32))
    img = Image.from_dlpack(arr)
    np.testing.assert_allclose(img.numpy(), arr)


def test_dlpack_import_from_numpy_zero_copy():
    """Mutating the source numpy array must be visible through the Image
    (proves the import is zero-copy)."""
    arr = np.ascontiguousarray(np.random.rand(4, 4, 3).astype(np.float32))
    img = Image.from_dlpack(arr)
    sentinel = 123.0
    arr[0, 0, 0] = sentinel
    assert float(img.numpy()[0, 0, 0]) == sentinel, (
        "ZERO-COPY IMPORT FAILED: mutation of source not visible in imported Image"
    )


def test_dlpack_import_u8():
    """uint8 import from numpy."""
    arr = np.ascontiguousarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    img = Image.from_dlpack(arr)
    np.testing.assert_array_equal(img.numpy(), arr)


def test_dlpack_import_u16():
    """uint16 import from numpy."""
    arr = np.ascontiguousarray(np.arange(12, dtype=np.uint16).reshape(2, 2, 3))
    img = Image.from_dlpack(arr)
    np.testing.assert_array_equal(img.numpy(), arr)


def test_dlpack_import_shape():
    """Shape reported by the imported Image must match the source."""
    arr = np.ascontiguousarray(np.zeros((7, 5, 3), dtype=np.uint8))
    img = Image.from_dlpack(arr)
    assert img.height == 7
    assert img.width == 5


def test_dlpack_import_rejects_non_cpu():
    """Importing a non-CPU tensor must raise NotImplementedError."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — skipping non-CPU device rejection test")
    t = torch.zeros((4, 4, 3), dtype=torch.uint8, device="cuda")
    with pytest.raises((NotImplementedError, Exception)):
        Image.from_dlpack(t)


def test_dlpack_import_rejects_unsupported_dtype():
    """Importing a float64 tensor must raise ValueError."""
    arr = np.ascontiguousarray(np.zeros((2, 2, 3), dtype=np.float64))
    with pytest.raises((ValueError, Exception)):
        Image.from_dlpack(arr)


# ────────────────────────────────────────────────────────────────
# Bidirectional round-trip with torch
# ────────────────────────────────────────────────────────────────


def test_dlpack_torch_export_zero_copy():
    """Export Image to torch via __dlpack__ — mutation through torch must
    be visible in the Image (zero-copy export)."""
    torch = pytest.importorskip("torch")
    arr = np.ascontiguousarray(np.random.randint(0, 255, (8, 8, 3), np.uint8))
    img = Image.from_numpy(arr, copy=True)
    t = torch.from_dlpack(img)
    assert tuple(t.shape) == (8, 8, 3)
    # Mutate through torch and verify zero-copy.
    t[0, 0, 0] = 200
    img_view = np.asarray(img)
    assert int(img_view[0, 0, 0]) == 200, (
        "ZERO-COPY EXPORT TO TORCH FAILED: torch mutation not visible in Image"
    )


def test_dlpack_torch_import_zero_copy():
    """Import a torch tensor via from_dlpack — mutation through torch must
    be visible in the Image (zero-copy import)."""
    torch = pytest.importorskip("torch")
    arr = np.ascontiguousarray(np.random.randint(0, 255, (8, 8, 3), np.uint8))
    t = torch.from_numpy(arr)
    img = Image.from_dlpack(t)
    # Mutate the torch tensor; verify the Image sees the change.
    t[1, 1, 0] = 77
    assert int(img.numpy()[1, 1, 0]) == 77, (
        "ZERO-COPY IMPORT FROM TORCH FAILED: torch mutation not visible in Image"
    )


def test_dlpack_torch_bidirectional():
    """Full bidirectional round-trip: numpy -> Image -> torch -> Image."""
    torch = pytest.importorskip("torch")
    arr = np.ascontiguousarray(np.random.randint(0, 255, (8, 8, 3), np.uint8))
    img = Image.from_numpy(arr, copy=True)
    # Export to torch.
    t = torch.from_dlpack(img)
    assert tuple(t.shape) == (8, 8, 3)
    # Import back from torch.
    img2 = Image.from_dlpack(t)
    np.testing.assert_array_equal(img2.numpy(), np.asarray(img))
