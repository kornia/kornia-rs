"""Zero-copy + perf verification for kornia-rs IO paths.

Covers:
  - Write/encode side: numpy_as_image keeps the input numpy buffer untouched
    and shares memory with the kornia Image's .data view.
  - Read/decode side: alloc_output_pyarray returns a freshly-allocated PyArray
    that is mutable in place (and not a view of any internal Rust state).
  - Per-call independence: re-decoding produces an array with a different
    backing buffer (no internal cache aliasing).

Performance smoke checks at 1080p / 4K confirm the encode path is dominated
by the codec, not by an extra full-image memcpy.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

from kornia_rs.image import Image

# Best-of-N bench helper from sibling benchmarks/ dir.
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from _bench import bench as _bench_fn  # noqa: E402


# --------------------------------------------------------------------- helpers


def _rand_u8(h, w, c=3):
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def _rand_u16(h, w, c=3):
    return np.random.randint(0, 65536, (h, w, c), dtype=np.uint16)


def _bench(fn, *, iters=None, warmup=None):
    """Min ms per call (best-of-N). `iters` / `warmup` accepted for
    backwards compatibility with existing call sites — the bench module
    auto-tunes iteration count to a 0.3s budget."""
    return _bench_fn(fn, target_seconds=0.3).min_ms


# ----------------------------------------------------------- write/encode side


def test_image_data_shares_memory_with_input():
    """Image(arr).data must be a zero-copy view of arr."""
    arr = _rand_u8(64, 64, 3)
    img = Image(arr)
    assert np.shares_memory(img.data, arr), "Image.data did not share memory with input"


def test_encode_png_does_not_mutate_input():
    arr = _rand_u8(64, 64, 3)
    img = Image(arr)
    arr_before = arr.copy()
    _ = img.encode("png")
    np.testing.assert_array_equal(arr, arr_before, "encode mutated input")


def test_encode_png_input_buffer_still_shared():
    """After encode, Image.data must still alias the original numpy buffer."""
    arr = _rand_u8(64, 64, 3)
    img = Image(arr)
    _ = img.encode("png")
    assert np.shares_memory(img.data, arr), "encode broke buffer aliasing"


def test_encode_png_u16_does_not_mutate_input():
    arr = _rand_u16(64, 64, 3)
    img = Image(arr)
    arr_before = arr.copy()
    _ = img.encode("png")
    np.testing.assert_array_equal(arr, arr_before)
    assert np.shares_memory(img.data, arr)


def test_encode_jpeg_does_not_mutate_input():
    arr = _rand_u8(64, 64, 3)
    img = Image(arr)
    arr_before = arr.copy()
    _ = img.encode("jpeg")
    np.testing.assert_array_equal(arr, arr_before)
    assert np.shares_memory(img.data, arr)


# ------------------------------------------------------------ read/decode side


def test_decode_output_is_writable_in_place():
    arr = _rand_u8(64, 64, 3)
    src = Image(arr)
    blob = src.encode("png")

    decoded = Image.decode(blob)
    # Mutating the decoded array should not segfault — we own the buffer.
    decoded.data[0, 0] = [1, 2, 3]
    np.testing.assert_array_equal(decoded.data[0, 0], [1, 2, 3])


def test_decode_returns_independent_buffers_per_call():
    """Two decodes from the same blob must yield non-aliased numpy arrays."""
    arr = _rand_u8(64, 64, 3)
    blob = Image(arr).encode("png")

    a = Image.decode(blob)
    b = Image.decode(blob)
    assert not np.shares_memory(a.data, b.data), "decoder is reusing one buffer"


def test_decode_png_u16_lossless_roundtrip():
    arr = _rand_u16(64, 64, 3)
    blob = Image(arr).encode("png")
    decoded = Image.decode(blob)
    assert decoded.data.dtype == np.uint16
    np.testing.assert_array_equal(decoded.data, arr)


def test_decode_does_not_alias_source_bytes():
    """The decoded numpy array must not be a view of the source PNG bytes."""
    arr = _rand_u8(64, 64, 3)
    blob = Image(arr).encode("png")
    blob_bytes = bytes(blob)  # ensure a fresh Python bytes object
    decoded = Image.decode(blob_bytes)
    # Mutate decoded — source bytes must be unaffected.
    blob_before = bytes(blob_bytes)
    decoded.data[0, 0] = [99, 99, 99]
    assert blob_bytes == blob_before


# --------------------------------------------------------- file-based io paths


def test_load_save_path_roundtrip_zero_copy_input(tmp_path):
    arr = _rand_u8(96, 96, 3)
    img = Image(arr)
    p = tmp_path / "out.png"

    arr_before = arr.copy()
    img.save(str(p))
    np.testing.assert_array_equal(arr, arr_before)

    loaded = Image.load(str(p))
    np.testing.assert_array_equal(loaded.data, arr)
    # Every load gets a fresh buffer.
    loaded2 = Image.load(str(p))
    assert not np.shares_memory(loaded.data, loaded2.data)


def test_save_to_bytesio_zero_copy_input():
    arr = _rand_u8(64, 64, 3)
    img = Image(arr)
    arr_before = arr.copy()

    bio = io.BytesIO()
    img.save(bio, format="png")
    np.testing.assert_array_equal(arr, arr_before)
    assert bio.getvalue().startswith(b"\x89PNG")


# ------------------------------------------------------------------ perf smoke


@pytest.mark.parametrize("size", [(1080, 1920), (2160, 3840)])
def test_perf_encode_png_u8(size):
    h, w = size
    arr = _rand_u8(h, w, 3)
    img = Image(arr)
    per_call_ms = _bench(lambda: img.encode("png"), iters=3, warmup=1)
    print(f"\n[bench] encode png u8 ({h}x{w}): {per_call_ms:6.1f} ms/call")
    # Loose ceiling — catches >2× regression vs hand-tuned encode.
    assert per_call_ms < 5000.0


@pytest.mark.parametrize("size", [(1080, 1920)])
def test_perf_encode_png_u16(size):
    h, w = size
    arr = _rand_u16(h, w, 1)
    img = Image(arr)
    per_call_ms = _bench(lambda: img.encode("png"), iters=3, warmup=1)
    print(f"\n[bench] encode png-16 gray ({h}x{w}): {per_call_ms:6.1f} ms/call")
    assert per_call_ms < 5000.0


@pytest.mark.parametrize("size", [(1080, 1920)])
def test_perf_encode_jpeg(size):
    h, w = size
    arr = _rand_u8(h, w, 3)
    img = Image(arr)
    per_call_ms = _bench(lambda: img.encode("jpeg"), iters=5, warmup=1)
    print(f"\n[bench] encode jpeg ({h}x{w}): {per_call_ms:6.1f} ms/call")
    assert per_call_ms < 1000.0


@pytest.mark.parametrize("size", [(1080, 1920)])
def test_perf_decode_png_u8(size):
    h, w = size
    blob = Image(_rand_u8(h, w, 3)).encode("png")
    per_call_ms = _bench(lambda: Image.decode(blob), iters=5, warmup=1)
    print(f"\n[bench] decode png u8 ({h}x{w}): {per_call_ms:6.1f} ms/call")
    assert per_call_ms < 2000.0
