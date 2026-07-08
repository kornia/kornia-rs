"""Integration tests asserting the CUDA paths don't leak device memory.

Each test brackets a hot loop with ``cuda.mem_get_info()`` (free device bytes)
and asserts the free count returns to its post-warmup baseline — a genuine
per-iteration leak (a device buffer that is never freed, a DLPack keepalive that
never releases) would make the free count fall monotonically and blow the
tolerance.

The warmup iterations let the driver's stream-ordered memory pool reach its
high-water mark first, so steady-state reuse — not a leak — is what we measure.
``test_mem_probe_is_sensitive`` is the positive control: it proves the probe
actually observes allocations, so the leak assertions can't pass vacuously.

Skipped wholesale without the ``cuda`` feature / a CUDA device (or an older
build lacking ``mem_get_info``).
"""

import gc

import numpy as np
import pytest

import kornia_rs
from kornia_rs.image import Image
from kornia_rs.cuda import Stream

from _cuda_helpers import dev as _dev, dzeros as _dzeros

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available() or not hasattr(cuda, "mem_get_info"),
    reason="kornia_rs.cuda not built, no CUDA device, or build lacks mem_get_info",
)

RNG = np.random.default_rng(0)

# Slack for driver / memory-pool bookkeeping that isn't a real leak. Buffers
# below are sized (and iterations chosen) so any true per-iteration leak is
# tens of MB — an order of magnitude past this tolerance.
TOL = 8 * 1024 * 1024  # 8 MiB


def _rgb(h=256, w=256):
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rgbf(h=256, w=256):
    return RNG.random((h, w, 3), dtype=np.float32)


def _free() -> int:
    """Free device bytes, after collecting Python garbage so dropped device
    buffers are actually released before the reading."""
    gc.collect()
    return cuda.mem_get_info()[0]


def assert_no_leak(body, iters=100, warmup=15, tol=TOL):
    """Run ``body`` ``warmup`` times to reach steady state, then ``iters`` more
    and assert free device memory did not fall by more than ``tol``."""
    for _ in range(warmup):
        body()
    base = _free()
    for _ in range(iters):
        body()
    end = _free()
    leaked = base - end
    assert leaked <= tol, (
        f"leaked ~{leaked / 1e6:.1f} MB over {iters} iters "
        f"(baseline free {base / 1e6:.1f} MB, end {end / 1e6:.1f} MB, "
        f"tol {tol / 1e6:.1f} MB)"
    )


# ── positive control ─────────────────────────────────────────────────────────


def test_mem_probe_is_sensitive():
    """Guard against the leak tests being vacuous: the probe must actually see
    allocations grow and shrink."""
    base = _free()
    # 8 x 1024*1024*3 f32 ≈ 8 x 12.6 MB ≈ 100 MB held live.
    hold = [_dzeros(1024, 1024, 3, dtype="float32") for _ in range(8)]
    held_free = _free()
    assert base - held_free > 32 * 1024 * 1024, (
        "mem_get_info did not observe ~100 MB of live device allocations — "
        "the probe is not sensitive, leak tests would be meaningless"
    )
    del hold
    recovered = _free()
    assert recovered >= base - TOL, "freed device buffers were not released"


# ── allocation / transfer round-trips ────────────────────────────────────────


def test_no_leak_from_numpy_download_roundtrip():
    a = _rgb()

    def body():
        img = _dev(a)  # H2D alloc
        _ = img.numpy()  # D2H copy
        del img

    assert_no_leak(body)


def test_no_leak_to_cuda_cpu_roundtrip():
    host = Image.from_numpy(_rgb())

    def body():
        dev = host.to_cuda()  # H2D alloc
        back = dev.cpu()  # D2H
        del dev, back

    assert_no_leak(body)


def test_no_leak_cuda_zeros():
    def body():
        img = _dzeros(256, 256, 3, dtype="float32")
        del img

    assert_no_leak(body)


# ── color-op kernels (each allocates a fresh device destination) ──────────────


def test_no_leak_color_op_chain():
    dev = _dev(_rgb())

    def body():
        ycc = kornia_rs.imgproc.ycbcr_from_rgb(dev)  # alloc dst
        rgb = kornia_rs.imgproc.rgb_from_ycbcr(ycc)  # alloc dst
        del ycc, rgb

    assert_no_leak(body)


def test_no_leak_f32_color_op():
    dev = _dev(_rgbf())

    def body():
        lab = kornia_rs.imgproc.lab_from_rgb(dev)
        back = kornia_rs.imgproc.rgb_from_lab(lab)
        del lab, back

    assert_no_leak(body)


def test_no_leak_channel_expanding_ops():
    gray = _dev(RNG.integers(0, 256, (256, 256, 1), dtype=np.uint8))

    def body():
        cmap = kornia_rs.imgproc.apply_colormap(gray, "jet")  # 1 -> 3ch
        rgba = kornia_rs.imgproc.rgba_from_rgb(cmap)  # 3 -> 4ch
        del cmap, rgba

    assert_no_leak(body)


# ── DLPack export / import keepalive paths ───────────────────────────────────


def test_no_leak_dlpack_export_capsule_unconsumed():
    """An exported capsule that no consumer ever claims must run its deleter on
    GC (release the keepalive) rather than leak it."""
    dev = _dev(_rgb())

    def body():
        cap = dev.__dlpack__()  # never handed to a consumer
        del cap

    assert_no_leak(body)


def test_no_leak_dlpack_import_zerocopy():
    """Zero-copy import: the imported alias holds a keepalive on the producer;
    when both drop, the single underlying buffer must be freed exactly once."""

    def body():
        src = _dev(_rgb())  # alloc
        alias = Image.from_dlpack(src)  # keepalive alias
        del alias
        del src

    assert_no_leak(body)


def test_no_leak_foreign_stream_fence():
    """A foreign stream fences each op with a freshly recorded CUDA event; those
    events must be destroyed, not leaked, across the loop."""
    a = _rgb()
    fs = Stream.from_handle(Stream.default().cuda_stream_ptr)

    def body():
        dev = _dev(a, stream=fs)
        del dev

    assert_no_leak(body)


# ── fused preprocessor (persistent staging must not grow unbounded) ───────────


def test_no_leak_preprocessor_run():
    pre = kornia_rs.Preprocessor(format="rgb")
    frame = _rgb(192, 256).reshape(-1).copy()

    def body():
        t = pre.run(frame, 256, 192, 128, 128)  # fresh output tensor each call
        del t

    assert_no_leak(body)


def test_no_leak_preprocessor_run_into():
    """run(..., out=out) reuses a preallocated output — no per-call device
    allocation, and the non-owning destination wrapper must not free/leak the
    buffer."""
    pre = kornia_rs.Preprocessor(format="rgb")
    frame = _rgb(192, 256).reshape(-1).copy()
    out = pre.alloc_output(128, 128)

    def body():
        pre.run(frame, 256, 192, 128, 128, out=out)

    assert_no_leak(body)


def test_no_leak_preprocessor_run_batch():
    pre = kornia_rs.Preprocessor(mode="letterbox", format="rgb")
    frames = [_rgb(192, 256).reshape(-1).copy() for _ in range(4)]

    def body():
        t = pre.run(frames, 256, 192, 128, 128)
        _ = t.numpy()
        del t

    assert_no_leak(body, iters=60)


# ── end-to-end pipeline ──────────────────────────────────────────────────────


def test_no_leak_full_pipeline():
    a = _rgb()

    def body():
        dev = _dev(a)
        gray = kornia_rs.imgproc.gray_from_rgb(dev)
        rgb = kornia_rs.imgproc.rgb_from_gray(gray)
        cap = rgb.__dlpack__()
        _ = rgb.numpy()
        del dev, gray, rgb, cap

    assert_no_leak(body)
