"""Device (CUDA) path of canny: byte-identical to the CPU path (which is
byte-for-byte with cv2.Canny). Skipped without a GPU."""

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


@pytest.mark.parametrize("shape", [(48, 64), (43, 67), (17, 9)])
@pytest.mark.parametrize("thr", [(50.0, 150.0), (20.0, 60.0)])
@pytest.mark.parametrize("l2", [False, True])
def test_canny_device_matches_cpu(shape, thr, l2):
    img = _pattern_u8(*shape)
    lo, hi = thr
    cpu = np.asarray(imgproc.canny(img, low_threshold=lo, high_threshold=hi, l2_gradient=l2))
    out = imgproc.canny(_dev(img), low_threshold=lo, high_threshold=hi, l2_gradient=l2)
    np.testing.assert_array_equal(np.asarray(out.to_numpy()), cpu)


def test_canny_device_rejects_multichannel():
    rgb = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(Exception, match="single-channel"):
        imgproc.canny(_dev(rgb))
