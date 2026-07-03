import numpy as np
import pytest

from kornia_rs import IMAGENET_MEAN as MEAN
from kornia_rs import IMAGENET_STD as STD
from kornia_rs.image import Image


def _f64_reference(src_u8, dst_w, dst_h, mean, std):
    """f32 bilinear (OpenCV half-pixel) + normalize + HWC->CHW, in float64."""
    sh, sw, _ = src_u8.shape
    sx = sw / dst_w
    sy = sh / dst_h
    s = src_u8.astype(np.float64)
    out = np.empty((3, dst_h, dst_w), np.float64)
    for dy in range(dst_h):
        fy = max((dy + 0.5) * sy - 0.5, 0.0)
        y0 = min(int(fy), sh - 1)
        y1 = min(y0 + 1, sh - 1)
        wy = fy - y0
        for dx in range(dst_w):
            fx = max((dx + 0.5) * sx - 0.5, 0.0)
            x0 = min(int(fx), sw - 1)
            x1 = min(x0 + 1, sw - 1)
            wx = fx - x0
            for c in range(3):
                top = s[y0, x0, c] + wx * (s[y0, x1, c] - s[y0, x0, c])
                bot = s[y1, x0, c] + wx * (s[y1, x1, c] - s[y1, x0, c])
                val = top + wy * (bot - top)
                out[c, dy, dx] = (val / 255.0 - mean[c]) / std[c]
    return out


@pytest.mark.parametrize(
    "sh,sw,dw,dh",
    [
        (1080, 1920, 640, 640),  # downscale, non-integer ratio
        (480, 640, 320, 240),    # exact 2x (box dispatch path)
        (333, 517, 224, 224),    # odd dims
        (200, 200, 416, 416),    # upscale
    ],
)
def test_resize_normalize_matches_f64_reference(sh, sw, dw, dh):
    src = np.ascontiguousarray((np.random.default_rng(0).random((sh, sw, 3)) * 255).astype(np.uint8))
    out = Image(src).resize_normalize_to_tensor(dw, dh, MEAN, STD)
    k = out.numpy()
    assert k.shape == (3, dh, dw)
    assert k.dtype == np.float32
    ref = _f64_reference(src, dw, dh, MEAN, STD)
    assert np.max(np.abs(k.astype(np.float64) - ref)) < 1e-3


def test_resize_normalize_zero_copy():
    src = np.ascontiguousarray((np.random.default_rng(1).random((100, 120, 3)) * 255).astype(np.uint8))
    out = Image(src).resize_normalize_to_tensor(60, 50, MEAN, STD)
    v = out.numpy()
    # data_ptr is the host address of the same buffer numpy() views.
    assert out.data_ptr == v.ctypes.data
    assert out.nbytes == 3 * 50 * 60 * 4
    # numpy() is a live view, not a copy.
    v[0, 0, 0] = 1234.0
    assert out.numpy()[0, 0, 0] == np.float32(1234.0)


def test_resize_normalize_close_to_cv2():
    cv2 = pytest.importorskip("cv2")
    src = np.ascontiguousarray((np.random.default_rng(2).random((720, 1280, 3)) * 255).astype(np.uint8))
    out = Image(src).resize_normalize_to_tensor(224, 224, MEAN, STD).numpy()
    r = cv2.resize(src, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    ref = ((r - np.array(MEAN, np.float32)) / np.array(STD, np.float32)).transpose(2, 0, 1)
    # We interpolate in f32 (no u8 requantization), so we differ from cv2's
    # u8-rounded resize by at most ~1 LSB / (255*std) ≈ 0.02 in normalized units.
    assert np.max(np.abs(out - ref)) < 0.05


def test_resize_normalize_rejects_non_rgb():
    gray = np.ascontiguousarray((np.random.default_rng(3).random((32, 32, 1)) * 255).astype(np.uint8))
    with pytest.raises(ValueError):
        Image(gray).resize_normalize_to_tensor(16, 16, MEAN, STD)
