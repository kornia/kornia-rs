import numpy as np
import pytest

import kornia_rs as K

imgproc = K.imgproc


def test_yuyv_encode_shape_and_layout():
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)  # 1x2
    buf = imgproc.yuyv_from_rgb(rgb)
    assert buf.shape == (2 * 1 * 2,)  # W*H*2
    assert buf.dtype == np.uint8
    # Layout Y0 U Y1 V; Y differs per pixel, U/V shared.
    y0, u, y1, v = buf.tolist()
    assert y0 != y1
    assert 0 <= u <= 255 and 0 <= v <= 255


def test_nv12_encode_shape():
    rgb = np.ascontiguousarray((np.random.default_rng(0).random((6, 8, 3)) * 255).astype(np.uint8))
    buf = imgproc.nv12_from_rgb(rgb)
    assert buf.shape == (8 * 6 * 3 // 2,)
    assert buf.dtype == np.uint8


def _smooth_rgb(w, h):
    # Gentle gradient (~1 level/pixel) so chroma subsampling error stays small —
    # the realistic case. Steep per-pixel chroma is fundamentally lossy under 4:2:x.
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([64 + xx % 64, 64 + yy % 64, np.full_like(xx, 128)], axis=-1)
    return np.ascontiguousarray(rgb.astype(np.uint8))


@pytest.mark.parametrize("w,h", [(8, 6), (64, 48), (32, 16)])
def test_yuyv_round_trip(w, h):
    rgb = _smooth_rgb(w, h)
    back = imgproc.rgb_from_yuyv(imgproc.yuyv_from_rgb(rgb), w, h)
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 5


@pytest.mark.parametrize("w,h", [(8, 6), (64, 48), (32, 16)])
def test_nv12_round_trip(w, h):
    rgb = _smooth_rgb(w, h)
    back = imgproc.rgb_from_nv12(imgproc.nv12_from_rgb(rgb), w, h)
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 6


def test_constant_color_round_trip_exact():
    # Constant color → no subsampling error.
    rgb = np.full((8, 8, 3), [180, 60, 30], dtype=np.uint8)
    rgb = np.ascontiguousarray(rgb)
    for enc, dec, args in [
        (imgproc.yuyv_from_rgb, imgproc.rgb_from_yuyv, (8, 8)),
        (imgproc.nv12_from_rgb, imgproc.rgb_from_nv12, (8, 8)),
    ]:
        back = dec(enc(rgb), *args)
        assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 2
