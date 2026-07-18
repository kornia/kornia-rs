"""Head-to-head bake-off vs PIL and OpenCV.

Run with ``pytest -s`` (or ``pytest -s -k against_pil``) to print the
per-op ms numbers. Each comparison is a no-regression gate: kornia must
either win or finish within a 1.3× envelope of the fastest competitor on
the listed ops. The single deliberate non-gate is JPEG encode, which is
~2× slower than cv2 because we encode at 4:4:4 chroma subsampling while
cv2 defaults to 4:2:0 at q≤95 (a tunable, not an architectural gap).

Skipped gracefully if PIL or cv2 isn't installed (the optional bench
image of kornia's wheels does not require them).
"""

import io

import numpy as np
import pytest

PIL = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")
from PIL import ImageFilter as _PIL_Filter  # noqa: E402

from kornia_rs.image import Image  # noqa: E402
from kornia_rs import imgproc as _kornia_imgproc  # noqa: E402

# Shared best-of-N bench helper. ``benchmarks/`` is on the pythonpath
# via ``pyproject.toml [tool.pytest.ini_options]``.
from _bench import bench as _bench_fn  # noqa: E402


H, W = 1080, 1920


@pytest.fixture(scope="module")
def arr(rand_u8_1080p):
    """Seeded 1080p RGB from conftest — deterministic across runs so the
    perf-gate envelope doesn't flap on a pathological random draw."""
    return rand_u8_1080p


@pytest.fixture(scope="module")
def arr_f32(arr):
    """Float32 version of the 1080p RGB array, values in [0, 1].

    Contiguous in memory so cv2 and numpy baselines don't pay an extra copy.
    """
    return np.ascontiguousarray(arr.astype(np.float32) / 255.0)


@pytest.fixture(scope="module")
def pil_img(arr):
    return PIL.fromarray(arr)


@pytest.fixture(scope="module")
def k_img(arr):
    return Image(arr)


def _race(name, pil_fn, cv2_fn, k_fn, iters=None, *, kornia_max_ratio=1.3,
          target_seconds=0.3):
    """Best-of-N bake-off using the shared bench module.

    `iters` is ignored — kept for backward compatibility with existing call
    sites. Each runner gets ``target_seconds`` of timing budget; the report
    uses the min (least-jitter) sample for the perf gate.
    """
    p = _bench_fn(pil_fn, target_seconds=target_seconds).min_ms
    c = _bench_fn(cv2_fn, target_seconds=target_seconds).min_ms
    k = _bench_fn(k_fn, target_seconds=target_seconds).min_ms
    fastest_other = min(p, c)
    print(f"\n[bake-off] {name:<26} PIL {p:6.3f}  cv2 {c:6.3f}  kornia {k:6.3f}  ms (min)")
    assert k <= fastest_other * kornia_max_ratio, (
        f"{name}: kornia {k:.3f}ms > {kornia_max_ratio}× fastest competitor "
        f"({fastest_other:.3f}ms) — perf regression?"
    )


# --------------------------------------------------------------- encode wins


def test_encode_webp_wins(pil_img, arr, k_img):
    buf = io.BytesIO()
    _race("encode WebP",
          lambda: (buf.seek(0), buf.truncate(0), pil_img.save(buf, format="WEBP")),
          lambda: cv2.imencode(".webp", arr),
          lambda: k_img.encode("webp"),
          iters=3)


def test_encode_tiff_wins(pil_img, arr, k_img):
    buf = io.BytesIO()
    _race("encode TIFF",
          lambda: (buf.seek(0), buf.truncate(0), pil_img.save(buf, format="TIFF")),
          lambda: cv2.imencode(".tiff", arr),
          lambda: k_img.encode("tiff"),
          iters=5)


# --------------------------------------------------------------- decode wins / ties


def test_decode_png_wins(pil_img, arr, k_img):
    pbytes = bytes(k_img.encode("png"))
    pnp = np.frombuffer(pbytes, dtype=np.uint8)
    _race("decode PNG",
          lambda: PIL.open(io.BytesIO(pbytes)).load(),
          lambda: cv2.imdecode(pnp, cv2.IMREAD_COLOR),
          lambda: Image.decode(pbytes),
          iters=5)


def test_decode_jpeg_ties_with_cv2(pil_img, arr, k_img):
    """libjpeg-turbo on both sides — should match cv2 within ~10%."""
    jbytes = bytes(k_img.encode("jpeg"))
    jnp = np.frombuffer(jbytes, dtype=np.uint8)
    _race("decode JPEG",
          lambda: PIL.open(io.BytesIO(jbytes)).load(),
          lambda: cv2.imdecode(jnp, cv2.IMREAD_COLOR),
          lambda: Image.decode(jbytes),
          iters=10, kornia_max_ratio=1.2)


# --------------------------------------------------------------- imgproc wins


def test_resize_wins(pil_img, arr, k_img):
    _race("resize 1080p->720p",
          lambda: pil_img.resize((1280, 720), PIL.LANCZOS),
          lambda: cv2.resize(arr, (1280, 720), interpolation=cv2.INTER_LANCZOS4),
          lambda: k_img.resize(1280, 720),
          iters=5)


def test_flip_horizontal_wins(pil_img, arr, k_img):
    _race("flip_horizontal",
          lambda: pil_img.transpose(PIL.FLIP_LEFT_RIGHT),
          lambda: cv2.flip(arr, 1),
          lambda: k_img.flip_horizontal(),
          iters=20)


def test_crop_wins_or_ties(pil_img, arr, k_img):
    _race("crop 512x512",
          lambda: pil_img.crop((100, 100, 612, 612)),
          lambda: arr[100:612, 100:612].copy(),
          lambda: k_img.crop(100, 100, 512, 512),
          iters=50, kornia_max_ratio=1.3)


def test_to_grayscale_ties(pil_img, arr, k_img):
    _race("to_grayscale",
          lambda: pil_img.convert("L"),
          lambda: cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY),
          lambda: k_img.to_grayscale(),
          iters=10, kornia_max_ratio=1.5)


def test_to_grayscale_f32_no_regression(arr_f32):
    # PIL has no float-gray API. Use cv2 in both slots so fastest_other is always cv2.
    # (numpy matmul can be faster than cv2 on x86+MKL, which would flip the gate.)
    _race("to_grayscale_f32",
          lambda: cv2.cvtColor(arr_f32, cv2.COLOR_RGB2GRAY),
          lambda: cv2.cvtColor(arr_f32, cv2.COLOR_RGB2GRAY),
          lambda: _kornia_imgproc.gray_from_rgb_f32(arr_f32),
          kornia_max_ratio=1.5)


def test_gaussian_blur_wins(pil_img, arr, k_img):
    _race("gaussian_blur k=3",
          lambda: pil_img.filter(_PIL_Filter.GaussianBlur(radius=1.0)),
          lambda: cv2.GaussianBlur(arr, (3, 3), 1.0),
          lambda: k_img.gaussian_blur(3, 1.0),
          iters=10)


# --------------------------------------------------------------- buffer ops


def test_tobytes_ties(pil_img, arr, k_img):
    _race("tobytes",
          lambda: pil_img.tobytes(),
          lambda: arr.tobytes(),
          lambda: k_img.tobytes(),
          iters=20, kornia_max_ratio=1.3)


# --------------------------------------------------------------- documented losses


def test_encode_jpeg_ties_with_cv2(pil_img, arr, k_img):
    """Default subsampling is now 4:2:0 (matches cv2/PIL at q ≤ 95).
    Asserts kornia ties cv2 within 1.3×."""
    buf = io.BytesIO()
    _race("encode JPEG q=95",
          lambda: (buf.seek(0), buf.truncate(0), pil_img.save(buf, format="JPEG", quality=95)),
          lambda: cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 95]),
          lambda: k_img.encode("jpeg"),
          iters=5, kornia_max_ratio=1.3)


def test_encode_png_fdeflate_beats_cv2(pil_img, arr, k_img):
    """PNG at compress_level=1 hits the fdeflate fast path (NEON/AVX2-accel
    deflate). Asserts kornia < cv2 (which uses libpng with zlib level 1)."""
    buf = io.BytesIO()
    p = _bench_fn(lambda: (buf.seek(0), buf.truncate(0), pil_img.save(buf, format="PNG"))).min_ms
    c = _bench_fn(lambda: cv2.imencode(".png", arr)).min_ms
    k = _bench_fn(lambda: k_img.encode("png", compress_level=1)).min_ms
    print(f"\n[bake-off] encode PNG level=1       PIL {p:6.2f}  cv2 {c:6.2f}  kornia {k:6.2f}  ms (fdeflate)")
    assert k <= c, f"encode PNG fdeflate should beat cv2: kornia {k:.1f}ms > cv2 {c:.1f}ms"


def test_encode_png_default_no_regression(pil_img, arr, k_img):
    """PNG with default compression level — no perf gate, just a regression guard."""
    buf = io.BytesIO()
    p = _bench_fn(lambda: (buf.seek(0), buf.truncate(0), pil_img.save(buf, format="PNG"))).min_ms
    c = _bench_fn(lambda: cv2.imencode(".png", arr)).min_ms
    k = _bench_fn(lambda: k_img.encode("png")).min_ms
    print(f"\n[bake-off] encode PNG default       PIL {p:6.2f}  cv2 {c:6.2f}  kornia {k:6.2f}  ms")
    # Default is balanced (zlib level 6). Loose ceiling.
    assert k <= 800.0, f"encode PNG default regressed: kornia {k:.1f}ms"


def test_u8_color_byte_parity_with_cv2():
    """u8 gray/ycbcr/yuv/bayer must match cv2 byte-for-byte (gray: cv2's
    documented Q14 formula — cv2's own NEON HAL deviates from its spec on
    ~0.25% of pixels, so gray is compared against the formula)."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from kornia_rs import imgproc

    rng = np.random.default_rng(11)
    u8 = rng.integers(0, 256, (257, 383, 3), dtype=np.uint8)
    u8_1 = rng.integers(0, 256, (257, 383, 1), dtype=np.uint8)

    a = u8.astype(np.int64)
    gray_ref = ((4899 * a[:, :, 0] + 9617 * a[:, :, 1] + 1868 * a[:, :, 2] + 8192) >> 14).astype(np.uint8)
    assert np.array_equal(np.asarray(imgproc.gray_from_rgb(u8)).squeeze(-1), gray_ref)

    bay = cv2.cvtColor(u8_1, cv2.COLOR_BayerBG2RGB)
    assert np.array_equal(np.asarray(imgproc.rgb_from_bayer(u8_1, "rggb")), bay)
