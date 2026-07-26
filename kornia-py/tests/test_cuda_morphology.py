"""Tests for the device (CUDA) path of dilate / erode.

u8 device images run the CUDA morphology kernel, bit-identical to the numpy
CPU path (same tap multiset, same border index mapping). Skipped wholesale
without a GPU or a cuda wheel.
"""

import numpy as np
import pytest

import kornia_rs
from kornia_rs import imgproc

from _cuda_helpers import dev as _dev

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available(),
    reason="CUDA not available (no GPU or CPU-only wheel)",
)


def _pattern_u8(h, w, c=3):
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)


@pytest.mark.parametrize("op", ["dilate", "erode"])
@pytest.mark.parametrize(
    "border", ["constant", "replicate", "reflect101", "reflect", "wrap"]
)
def test_device_matches_numpy_cpu_bit_exact(op, border):
    a = _pattern_u8(41, 63)
    fn = getattr(imgproc, op)
    cpu = fn(a, kernel="box", size=(3, 3), border=border, constant_value=9)
    out = fn(_dev(a), kernel="box", size=(3, 3), border=border, constant_value=9)
    assert "cuda" in str(out.device)
    np.testing.assert_array_equal(out.cpu().numpy(), cpu)


@pytest.mark.parametrize("kernel,size", [("cross", (5, 5)), ("ellipse", (3, 5))])
def test_device_kernel_shapes_bit_exact(kernel, size):
    a = _pattern_u8(41, 63)
    cpu = imgproc.dilate(a, kernel=kernel, size=size)
    out = imgproc.dilate(_dev(a), kernel=kernel, size=size)
    np.testing.assert_array_equal(out.cpu().numpy(), cpu)


def test_numpy_path_unchanged_and_shapes():
    a = _pattern_u8(16, 16, 1)
    out = imgproc.erode(a, kernel="box", size=(3, 3))
    assert isinstance(out, np.ndarray)
    assert out.shape == a.shape


def test_bad_kernel_and_border_raise():
    a = _pattern_u8(16, 16)
    with pytest.raises(ValueError, match="kernel shape"):
        imgproc.dilate(a, kernel="hexagon")
    with pytest.raises(ValueError, match="border mode"):
        imgproc.dilate(a, border="void")
