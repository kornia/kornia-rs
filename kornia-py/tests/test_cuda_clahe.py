"""Tests for the device (CUDA) path of clahe.

u8 single-channel device images run the CUDA LUT-build + blend kernels,
byte-identical to the CPU path (which is byte-for-byte with
cv2.createCLAHE). Skipped wholesale without a GPU or a cuda wheel.
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


@pytest.mark.parametrize("shape", [(48, 64), (43, 67), (9, 33)])
@pytest.mark.parametrize("grid", [(8, 8), (4, 3), (1, 1)])
@pytest.mark.parametrize("clip", [40.0, 2.5, 0.0])
def test_clahe_device_matches_cpu(shape, grid, clip):
    img = _pattern_u8(*shape)
    cpu = np.asarray(imgproc.clahe(img, clip_limit=clip, grid_size=grid))
    out = imgproc.clahe(_dev(img), clip_limit=clip, grid_size=grid)
    assert "cuda" in str(out.device)
    np.testing.assert_array_equal(np.asarray(out.to_numpy()), cpu)


def test_clahe_device_rejects_multichannel():
    rgb = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(Exception, match="single-channel"):
        imgproc.clahe(_dev(rgb))
