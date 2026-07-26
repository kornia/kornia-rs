"""Device (CUDA) paths of median_blur / bilateral_filter: byte-identical to
the CPU paths (which are byte-for-byte with cv2). Skipped without a GPU."""

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


def _pattern_u8(h, w, c=1):
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)


@pytest.mark.parametrize("shape", [(48, 64), (43, 67), (5, 9)])
@pytest.mark.parametrize("k", [3, 5])
def test_median_device_matches_cpu(shape, k):
    img = _pattern_u8(*shape)
    cpu = np.asarray(imgproc.median_blur(img, kernel_size=k))
    out = imgproc.median_blur(_dev(img), kernel_size=k)
    np.testing.assert_array_equal(np.asarray(out.to_numpy()), cpu)


def test_median_device_c3_matches_cpu():
    img = _pattern_u8(33, 21, 3)
    cpu = np.asarray(imgproc.median_blur(img, kernel_size=5))
    out = imgproc.median_blur(_dev(img), kernel_size=5)
    np.testing.assert_array_equal(np.asarray(out.to_numpy()), cpu)


@pytest.mark.parametrize("params", [(5, 50.0, 50.0), (3, 25.0, 10.0), (9, 75.0, 75.0)])
def test_bilateral_device_matches_cpu(params):
    d, sc, ss = params
    img = _pattern_u8(48, 64)
    cpu = np.asarray(imgproc.bilateral_filter(img, d=d, sigma_color=sc, sigma_space=ss))
    out = imgproc.bilateral_filter(_dev(img), d=d, sigma_color=sc, sigma_space=ss)
    np.testing.assert_array_equal(np.asarray(out.to_numpy()), cpu)


def test_bilateral_device_rejects_multichannel():
    rgb = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(Exception, match="single-channel"):
        imgproc.bilateral_filter(_dev(rgb))
