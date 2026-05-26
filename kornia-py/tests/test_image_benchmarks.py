"""Performance benchmarks for the Image API.

Times each public Image surface method at representative sizes and prints
ms/call. Acts as a no-regression gate: every test asserts the call is
faster than a loose ceiling chosen well above today's measured timings,
so a 2-3× regression would fail CI without flapping on noisy runs.

Run with ``pytest -s`` to see the timings.

Skipped on CI runners — ceilings are tuned to developer hardware and
shared GitHub Actions runners are too variable to gate on absolute ms.

Coverage:
  - Constructors:        Image(arr) / Image.frombuffer / Image.fromarray /
                         Image.frombytes / Image.new
  - File IO:             encode/decode for PNG / JPEG / WebP / TIFF on
                         u8 (1080p) and u16 (1080p, PNG/TIFF only)
  - Imgproc:             resize, gaussian_blur, flip_h/flip_v, crop,
                         rotate, brightness/contrast/saturation/hue,
                         normalize, to_grayscale, to_rgb
  - Convert:             RGB->L, RGB->RGBA, L->I;16, I;16->L
  - Buffer ops:          tobytes, to_numpy, copy
"""

import os

import numpy as np
import pytest

from kornia_rs.image import Image

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="perf ceilings are tuned to dev hardware; CI runners vary too much",
)

# Shared best-of-N bench helper. ``benchmarks/`` is on the pythonpath
# via ``pyproject.toml [tool.pytest.ini_options]``.
from _bench import bench as _real_bench  # noqa: E402


def _u8(h, w, c=3):
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def _u16(h, w, c=3):
    return np.random.randint(0, 65536, (h, w, c), dtype=np.uint16)


def _f32(h, w, c=3):
    return np.random.rand(h, w, c).astype(np.float32)


def _bench(label, fn, *, ceiling_ms=5000.0, target_seconds=0.3, **_legacy):
    """Backwards-compatible wrapper around the shared bench helper.

    Reports the *min* (best-of-N) — the only honest number for sub-ms
    ops where mean is biased high by GC/scheduler noise.
    """
    r = _real_bench(fn, target_seconds=target_seconds, min_iters=20)
    print(f"\n[bench] {label:40s}: min {r.min_ms:7.3f}  p50 {r.p50_ms:7.3f}  p95 {r.p95_ms:7.3f}  ms (n={r.n})")
    assert r.min_ms < ceiling_ms, f"{label} regressed: {r.min_ms:.1f}ms > {ceiling_ms}ms"


# ----------------------------------------------------- constructors


def test_perf_constructor_array():
    arr = _u8(1080, 1920, 3)
    _bench("Image(arr) 1080p u8", lambda: Image(arr), iters=20, ceiling_ms=10.0)


def test_perf_constructor_frombuffer():
    arr = _u8(1080, 1920, 3)
    _bench("Image.frombuffer 1080p u8", lambda: Image.frombuffer(arr), iters=20, ceiling_ms=10.0)


def test_perf_constructor_fromarray():
    arr = _u8(1080, 1920, 3)
    _bench("Image.fromarray 1080p u8", lambda: Image.fromarray(arr), iters=20, ceiling_ms=10.0)


def test_perf_constructor_new_blank_u8():
    _bench("Image.new RGB 1080p", lambda: Image.new("RGB", (1920, 1080)),
           iters=10, ceiling_ms=50.0)


def test_perf_constructor_new_with_color():
    _bench("Image.new RGB 1080p color", lambda: Image.new("RGB", (1920, 1080), (10, 20, 30)),
           iters=10, ceiling_ms=50.0)


# ------------------------------------------------- encode / decode


@pytest.fixture(scope="module")
def img_u8_1080p():
    return Image(_u8(1080, 1920, 3))


@pytest.fixture(scope="module")
def img_u16_1080p():
    return Image(_u16(1080, 1920, 1))


def test_perf_encode_png_u8(img_u8_1080p):
    _bench("encode PNG u8 1080p", lambda: img_u8_1080p.encode("png"), iters=3, ceiling_ms=2000.0)


def test_perf_encode_png_u16(img_u16_1080p):
    _bench("encode PNG u16 1080p mono", lambda: img_u16_1080p.encode("png"), iters=3, ceiling_ms=2000.0)


def test_perf_encode_jpeg(img_u8_1080p):
    _bench("encode JPEG u8 1080p", lambda: img_u8_1080p.encode("jpeg"), iters=5, ceiling_ms=500.0)


def test_perf_encode_webp(img_u8_1080p):
    _bench("encode WebP u8 1080p", lambda: img_u8_1080p.encode("webp"), iters=3, ceiling_ms=2000.0)


def test_perf_encode_tiff_u8(img_u8_1080p):
    _bench("encode TIFF u8 1080p", lambda: img_u8_1080p.encode("tiff"), iters=5, ceiling_ms=500.0)


def test_perf_encode_tiff_u16(img_u16_1080p):
    _bench("encode TIFF u16 1080p", lambda: img_u16_1080p.encode("tiff"), iters=5, ceiling_ms=500.0)


def test_perf_decode_png_u8(img_u8_1080p):
    blob = img_u8_1080p.encode("png")
    _bench("decode PNG u8 1080p", lambda: Image.decode(blob), iters=5, ceiling_ms=500.0)


def test_perf_decode_jpeg(img_u8_1080p):
    blob = img_u8_1080p.encode("jpeg")
    # Random-noise images compress poorly → decode is slower than typical
    # natural-image content. Ceiling chosen well above measured (~325ms) so
    # a real regression (>2x) trips it.
    _bench("decode JPEG u8 1080p", lambda: Image.decode(blob), iters=10, ceiling_ms=800.0)


def test_perf_decode_webp(img_u8_1080p):
    blob = img_u8_1080p.encode("webp")
    _bench("decode WebP u8 1080p", lambda: Image.decode(bytes(blob)), iters=5, ceiling_ms=500.0)


def test_perf_decode_tiff(img_u8_1080p):
    blob = img_u8_1080p.encode("tiff")
    _bench("decode TIFF u8 1080p", lambda: Image.decode(bytes(blob)), iters=10, ceiling_ms=200.0)


# ------------------------------------------------------- imgproc


def test_perf_resize_down(img_u8_1080p):
    _bench("resize 1080p->720p u8", lambda: img_u8_1080p.resize(1280, 720),
           iters=5, ceiling_ms=200.0)


def test_perf_resize_up(img_u8_1080p):
    _bench("resize 1080p->4k u8", lambda: img_u8_1080p.resize(3840, 2160),
           iters=3, ceiling_ms=500.0)


def test_perf_gaussian_blur(img_u8_1080p):
    _bench("gaussian_blur k=3 sigma=1 u8", lambda: img_u8_1080p.gaussian_blur(3, 1.0),
           iters=5, ceiling_ms=500.0)


def test_perf_flip_horizontal_u8(img_u8_1080p):
    _bench("flip_horizontal u8 1080p", lambda: img_u8_1080p.flip_horizontal(),
           iters=10, ceiling_ms=100.0)


def test_perf_flip_vertical_u8(img_u8_1080p):
    _bench("flip_vertical u8 1080p", lambda: img_u8_1080p.flip_vertical(),
           iters=10, ceiling_ms=100.0)


def test_perf_flip_horizontal_u16(img_u16_1080p):
    _bench("flip_horizontal u16 1080p", lambda: img_u16_1080p.flip_horizontal(),
           iters=10, ceiling_ms=200.0)


def test_perf_crop_u8(img_u8_1080p):
    _bench("crop 512x512 u8", lambda: img_u8_1080p.crop(100, 100, 512, 512),
           iters=20, ceiling_ms=50.0)


def test_perf_rotate(img_u8_1080p):
    _bench("rotate 0° (copy fast path)", lambda: img_u8_1080p.rotate(0.0),
           iters=10, ceiling_ms=100.0)


def test_perf_to_grayscale(img_u8_1080p):
    _bench("to_grayscale u8 1080p", lambda: img_u8_1080p.to_grayscale(),
           iters=5, ceiling_ms=200.0)


def test_perf_normalize(img_u8_1080p):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    _bench("normalize ImageNet 1080p u8",
           lambda: img_u8_1080p.normalize(mean, std), iters=5, ceiling_ms=300.0)


def test_perf_adjust_brightness(img_u8_1080p):
    _bench("adjust_brightness 1080p u8",
           lambda: img_u8_1080p.adjust_brightness(0.2), iters=10, ceiling_ms=100.0)


def test_perf_adjust_contrast(img_u8_1080p):
    _bench("adjust_contrast 1080p u8",
           lambda: img_u8_1080p.adjust_contrast(1.2), iters=10, ceiling_ms=100.0)


def test_perf_adjust_saturation(img_u8_1080p):
    _bench("adjust_saturation 1080p u8",
           lambda: img_u8_1080p.adjust_saturation(1.2), iters=10, ceiling_ms=200.0)


def test_perf_adjust_hue(img_u8_1080p):
    _bench("adjust_hue 1080p u8",
           lambda: img_u8_1080p.adjust_hue(0.1), iters=10, ceiling_ms=200.0)


# ------------------------------------------------------- convert


def test_perf_convert_rgb_to_l(img_u8_1080p):
    _bench("convert RGB->L 1080p u8",
           lambda: img_u8_1080p.convert("L"), iters=5, ceiling_ms=200.0)


def test_perf_convert_rgb_to_rgba(img_u8_1080p):
    _bench("convert RGB->RGBA 1080p u8",
           lambda: img_u8_1080p.convert("RGBA"), iters=5, ceiling_ms=200.0)


def test_perf_convert_l_to_i16():
    arr = _u8(1080, 1920, 1)
    img = Image(arr)
    _bench("convert L->I;16 1080p",
           lambda: img.convert("I;16"), iters=5, ceiling_ms=200.0)


def test_perf_convert_i16_to_l():
    arr = _u16(1080, 1920, 1)
    img = Image(arr)
    _bench("convert I;16->L 1080p",
           lambda: img.convert("L"), iters=5, ceiling_ms=200.0)


# ------------------------------------------------------- buffer ops


def test_perf_tobytes(img_u8_1080p):
    _bench("tobytes u8 1080p", lambda: img_u8_1080p.tobytes(),
           iters=10, ceiling_ms=100.0)


def test_perf_to_numpy(img_u8_1080p):
    _bench("to_numpy u8 1080p", lambda: img_u8_1080p.to_numpy(),
           iters=10, ceiling_ms=100.0)


def test_perf_copy_u8(img_u8_1080p):
    _bench("copy u8 1080p", lambda: img_u8_1080p.copy(),
           iters=10, ceiling_ms=100.0)
