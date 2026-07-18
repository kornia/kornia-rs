import kornia_rs as K

import numpy as np
import pytest

imgproc = K.imgproc


# ----- f32 round-trip correctness (no external deps) -----

def _rand_rgb01(h=17, w=13, seed=0):
    return np.ascontiguousarray(
        np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)
    )


@pytest.mark.parametrize(
    "fwd,inv,scale",
    [
        ("hsv_from_rgb", "rgb_from_hsv", 255.0),  # our HSV/HLS convention is [0,255]
        ("hls_from_rgb", "rgb_from_hls", 255.0),
        ("xyz_from_rgb", "rgb_from_xyz", 1.0),
        ("lab_from_rgb", "rgb_from_lab", 1.0),
        ("luv_from_rgb", "rgb_from_luv", 1.0),
        ("ycbcr_from_rgb", "rgb_from_ycbcr", 1.0),
        ("yuv_from_rgb", "rgb_from_yuv", 1.0),
        ("linear_rgb_from_rgb", "rgb_from_linear_rgb", 1.0),
    ],
)
def test_f32_round_trip(fwd, inv, scale):
    rgb = _rand_rgb01() * np.float32(scale)
    mid = getattr(imgproc, fwd)(rgb)
    back = getattr(imgproc, inv)(mid)
    assert back.shape == rgb.shape
    assert np.max(np.abs(back - rgb)) < (scale * 1e-3)


def test_sepia_shape_and_range():
    rgb = _rand_rgb01()
    out = imgproc.sepia_from_rgb(rgb)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0


# ----- byte-exactness / equivalence vs OpenCV (skipped if cv2 unavailable) -----

def test_xyz_ycbcr_match_cv2_f32_precision():
    cv2 = pytest.importorskip("cv2")
    a = _rand_rgb01(64, 64)
    # Pure-matrix conversions match cv2 to f32 precision (≤1 ULP; op-order rounding).
    assert np.max(np.abs(imgproc.xyz_from_rgb(a) - cv2.cvtColor(a, cv2.COLOR_RGB2XYZ))) < 1e-6
    assert np.max(np.abs(imgproc.ycbcr_from_rgb(a) - cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb))) < 1e-6


def test_lab_luv_close_cv2():
    cv2 = pytest.importorskip("cv2")
    a = _rand_rgb01(64, 64)
    # cv2's own f32 Lab/Luv are ~0.5 off true; we match within cv2's error band.
    assert np.max(np.abs(imgproc.lab_from_rgb(a) - cv2.cvtColor(a, cv2.COLOR_RGB2Lab))) < 1.0
    assert np.max(np.abs(imgproc.luv_from_rgb(a) - cv2.cvtColor(a, cv2.COLOR_RGB2Luv))) < 1.0


def test_bayer_byte_exact_cv2():
    cv2 = pytest.importorskip("cv2")
    mosaic = np.ascontiguousarray(
        (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    )
    k = imgproc.rgb_from_bayer(mosaic[..., None], "rggb")  # rggb == cv2 BayerBG2RGB
    c = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2RGB)
    assert np.array_equal(k[2:-2, 2:-2], c[2:-2, 2:-2])  # interior byte-exact


@pytest.mark.parametrize(
    "fn,code,shape",
    [
        ("rgb_from_yuyv", "COLOR_YUV2RGB_YUYV", "422"),
        ("rgb_from_uyvy", "COLOR_YUV2RGB_UYVY", "422"),
        ("rgb_from_yvyu", "COLOR_YUV2RGB_YVYU", "422"),
        ("rgb_from_nv12", "COLOR_YUV2RGB_NV12", "420"),
        ("rgb_from_nv21", "COLOR_YUV2RGB_NV21", "420"),
        ("rgb_from_i420", "COLOR_YUV2RGB_I420", "420"),
        ("rgb_from_yv12", "COLOR_YUV2RGB_YV12", "420"),
    ],
)
def test_video_decoders_byte_exact_cv2(fn, code, shape):
    cv2 = pytest.importorskip("cv2")
    w, h = 64, 48
    rng = np.random.default_rng(1)
    if shape == "422":
        buf = np.ascontiguousarray((rng.random(w * h * 2) * 255).astype(np.uint8))
        cv_in = buf.reshape(h, w, 2)
    else:
        buf = np.ascontiguousarray((rng.random(w * h * 3 // 2) * 255).astype(np.uint8))
        cv_in = buf.reshape(h * 3 // 2, w)
    k = getattr(imgproc, fn)(buf, w, h)
    c = cv2.cvtColor(cv_in, getattr(cv2, code))
    assert np.array_equal(k, c)


def test_rgb_from_gray():
    img: np.ndarray = np.array([[[1]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.imgproc.rgb_from_gray(img)
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[1, 1, 1]]]))


def test_bgr_from_rgb():
    img: np.ndarray = np.array([[[1, 2, 3]]], dtype=np.uint8)
    img_bgr: np.ndarray = K.imgproc.bgr_from_rgb(img)
    assert img_bgr.shape == (1, 1, 3)
    assert np.allclose(img_bgr, np.array([[[3, 2, 1]]]))

def test_gray_from_rgb():
    # Formula: (4899*R + 9617*G + 1868*B + 8192) >> 14 — OpenCV's exact Q14
    # BT.601 (byte-parity with cv2.cvtColor RGB2GRAY).
    # R=0, G=128, B=255 → (0 + 1230976 + 476340 + 8192) >> 14 = 104
    img: np.ndarray = np.array([[[0, 128, 255]]], dtype=np.uint8)
    img_gray: np.ndarray = K.imgproc.gray_from_rgb(img)
    assert img_gray.shape == (1, 1, 1)
    assert np.allclose(img_gray, np.array([[[104]]]))

def test_rgb_from_rgba():
    img: np.ndarray = np.array([[[0, 1, 2, 255]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.imgproc.rgb_from_rgba(img)
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[0, 1, 2]]]))


def test_rgb_from_rgba_with_background():
    img: np.ndarray = np.array([[[255, 0, 0, 128]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.imgproc.rgb_from_rgba(img, background=[100, 100, 100])
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[178, 50, 50]]]))

def test_rgb_from_bgra():
    img: np.ndarray = np.array([[[0, 0, 255, 128]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.imgproc.rgb_from_bgra(img)
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[255, 0, 0]]]))

def test_rgb_from_bgra_with_background():
    img: np.ndarray = np.array([[[0, 0, 255, 128]]], dtype=np.uint8)
    img_rgb: np.ndarray = K.imgproc.rgb_from_bgra(img, background=[100, 100, 100])
    assert img_rgb.shape == (1, 1, 3)
    assert np.allclose(img_rgb, np.array([[[178, 50, 50]]]))
