"""Tests for the device (CUDA) path of compute_histogram / equalize_hist.

u8 single-channel device images run the CUDA kernels; counts are exactly
equal to the CPU's and equalize output is byte-identical to the CPU path
(which itself is byte-for-byte with cv2.equalizeHist). Skipped wholesale
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


def _pattern_u8(h, w):
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)


@pytest.mark.parametrize("shape", [(48, 64), (43, 67), (1, 1)])
@pytest.mark.parametrize("num_bins", [256, 64, 10])
def test_histogram_device_matches_cpu(shape, num_bins):
    img = _pattern_u8(*shape)
    cpu = imgproc.compute_histogram(img, num_bins=num_bins)
    gpu = imgproc.compute_histogram(_dev(img), num_bins=num_bins)
    assert cpu == gpu
    assert sum(gpu) == shape[0] * shape[1]


@pytest.mark.parametrize("shape", [(48, 64), (43, 67), (1, 1)])
def test_equalize_device_matches_cpu(shape):
    img = _pattern_u8(*shape)
    cpu = imgproc.equalize_hist(img)
    out = imgproc.equalize_hist(_dev(img))
    assert "cuda" in str(out.device)
    np.testing.assert_array_equal(np.asarray(out.to_numpy()), np.asarray(cpu))


def test_equalize_device_constant_image_identity():
    img = np.full((16, 16, 1), 77, np.uint8)
    out = np.asarray(imgproc.equalize_hist(_dev(img)).to_numpy())
    assert (out == 77).all()


def test_equalize_device_rejects_multichannel():
    rgb = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(Exception, match="single-channel"):
        imgproc.equalize_hist(_dev(rgb))
