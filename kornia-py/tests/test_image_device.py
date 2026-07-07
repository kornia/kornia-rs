"""Tests for the unified device-aware ``Image`` API (``Image.cuda.*``, ``.cpu()``,
``.to_cuda()``, ``.device``, auto-download ``.numpy()``, DLPack + CUDA Array
Interface interop).

Skipped wholesale when the wheel was built without the ``cuda`` feature or no
CUDA device is present.
"""

import numpy as np
import pytest

import kornia_rs
from kornia_rs.image import Image
from kornia_rs.cuda import Stream

from _cuda_helpers import dev as _dev, dzeros as _dzeros

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available(),
    reason="kornia_rs.cuda not built or no CUDA device",
)

RNG = np.random.default_rng(0)


def _rgb(h=48, w=64):
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rgbf(h=32, w=40):
    return RNG.random((h, w, 3), dtype=np.float32)


def test_from_numpy_roundtrip_uint8():
    a = _rgb()
    img = _dev(a)
    assert img.device == "cuda:0"
    # .numpy() auto-downloads (D2H) a device image.
    np.testing.assert_array_equal(img.numpy(), a)


def test_from_numpy_roundtrip_float32():
    a = _rgbf()
    img = _dev(a)
    assert img.device == "cuda:0"
    np.testing.assert_array_equal(img.numpy(), a)


def test_host_image_device_is_cpu():
    a = _rgb()
    img = Image.from_numpy(a)
    assert img.device == "cpu"


def test_to_cuda_then_cpu_roundtrip():
    a = _rgb()
    host = Image.from_numpy(a)
    dev = host.to_cuda()
    assert dev.device == "cuda:0"
    back = dev.cpu()
    assert back.device == "cpu"
    np.testing.assert_array_equal(back.numpy(), a)


def test_cuda_zeros():
    img = _dzeros(width=8, height=6, channels=3, dtype="uint8")
    assert img.device == "cuda:0"
    assert (img.height, img.width, img.channels) == (6, 8, 3)
    np.testing.assert_array_equal(img.numpy(), np.zeros((6, 8, 3), np.uint8))


def test_unsupported_dtype_on_device_raises():
    a = RNG.integers(0, 1000, (8, 8, 3), dtype=np.uint16)
    with pytest.raises(ValueError):
        _dev(a)


def test_cuda_array_interface_present_on_device_only():
    a = _rgb()
    dev = _dev(a)
    cai = dev.__cuda_array_interface__
    assert cai["version"] == 3
    assert cai["shape"] == (a.shape[0], a.shape[1], a.shape[2])
    assert cai["typestr"] == "|u1"
    ptr, readonly = cai["data"]
    assert isinstance(ptr, int) and ptr != 0
    assert readonly is False
    # Host images must NOT advertise the CUDA array interface.
    host = Image.from_numpy(a)
    with pytest.raises(AttributeError):
        _ = host.__cuda_array_interface__


def test_stream_default_and_protocol():
    s = Stream.default()
    assert isinstance(s.cuda_stream_ptr, int)
    ver, handle = s.__cuda_stream__()
    assert ver == 0
    assert handle == s.cuda_stream_ptr
    s.synchronize()
    # A stream can be threaded through a transfer.
    img = _dev(_rgb(), stream=s)
    assert img.device == "cuda:0"


def test_stream_from_handle_roundtrips():
    s = Stream.from_handle(0)  # 0 == the (valid) legacy default stream
    assert s.cuda_stream_ptr == 0
    ver, handle = s.__cuda_stream__()
    assert ver == 0 and handle == 0
    s.synchronize()  # syncing the null stream is valid


def test_stream_from_cuda_stream_protocol_object():
    # Any object implementing the cuda-python / cuda.core __cuda_stream__
    # protocol is accepted — kornia's own Stream is one such object.
    base = Stream.default()
    adopted = Stream.from_cuda_stream(base)
    assert adopted.cuda_stream_ptr == base.cuda_stream_ptr


def test_stream_from_cuda_stream_int_and_attrs():
    assert Stream.from_cuda_stream(0).cuda_stream_ptr == 0  # bare int handle

    class WithPtr:  # cupy-style .ptr
        ptr = 0

    class WithHandle:  # cuda.core-style .handle
        handle = 0

    assert Stream.from_cuda_stream(WithPtr()).cuda_stream_ptr == 0
    assert Stream.from_cuda_stream(WithHandle()).cuda_stream_ptr == 0


def test_stream_from_cuda_stream_rejects_non_stream():
    with pytest.raises(ValueError):
        Stream.from_cuda_stream("not a stream")


def test_foreign_stream_threaded_through_transfers():
    """A foreign (adopted) stream is accepted by every device constructor; the
    result is correct (kornia runs on its default stream, fenced into the
    foreign one)."""
    a = _rgb()
    fs = Stream.from_handle(Stream.default().cuda_stream_ptr)

    dev = _dev(a, stream=fs)
    assert dev.device == "cuda:0"
    np.testing.assert_array_equal(dev.numpy(), a)

    dev2 = Image.from_numpy(a).to_cuda(stream=fs)
    np.testing.assert_array_equal(dev2.numpy(), a)

    z = _dzeros(8, 6, 3, dtype="uint8", stream=fs)
    np.testing.assert_array_equal(z.numpy(), np.zeros((6, 8, 3), np.uint8))


def test_device_numpy_is_readonly_copy():
    """`.numpy()` / `.data` / `np.asarray` / `to_numpy(copy=False)` on a device
    image return a read-only host copy — writes must not silently vanish into the
    throwaway D2H buffer."""
    a = _rgb()
    dev = _dev(a)
    for arr in (dev.numpy(), dev.data, np.asarray(dev), dev.to_numpy(copy=False)):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[:] = 0
        np.testing.assert_array_equal(arr, a)
    # A host image stays a writable zero-copy view.
    host = Image.from_numpy(a.copy())
    assert host.numpy().flags.writeable


def test_host_only_transforms_raise_on_device():
    """Every host transform is host-only and raises on a device image (the stubs
    document this); `.cpu()` first is the fix."""
    dev = _dev(_rgb())
    host_only = [
        ("resize", lambda i: i.resize(16, 16)),
        ("copy", lambda i: i.copy()),
        ("crop", lambda i: i.crop(0, 0, 8, 8)),
        ("flip_horizontal", lambda i: i.flip_horizontal()),
        ("flip_vertical", lambda i: i.flip_vertical()),
        ("rotate", lambda i: i.rotate(90.0)),
        (
            "resize_normalize_to_tensor",
            lambda i: i.resize_normalize_to_tensor(
                16, 16, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            ),
        ),
        ("memoryview", lambda i: memoryview(i)),
    ]
    for name, call in host_only:
        # memoryview surfaces the host guard as BufferError; the rest as ValueError.
        with pytest.raises((ValueError, BufferError)):
            call(dev)
    # After moving to host the same call works.
    assert dev.cpu().resize(16, 16).device == "cpu"


def test_dlpack_stream_argument_paths():
    """The device `__dlpack__` validates/handles the DLPack `stream` argument:
    -1 (no sync), 0/1/2 default sentinels + a real handle (non-blocking fence),
    None (host sync) each return a PyCapsule; invalid negatives are rejected.
    Exercised without a torch consumer."""
    dev = _dev(_rgb())
    for stream in (-1, 0, 1, 2, Stream.default().cuda_stream_ptr, None):
        cap = dev.__dlpack__(stream=stream)
        assert type(cap).__name__ == "PyCapsule"
    for bad in (-2, -100):
        with pytest.raises(ValueError):
            dev.__dlpack__(stream=bad)


def test_dlpack_roundtrip_with_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    dev = _dev(a)
    # Export device Image -> torch (zero-copy), values match.
    t = torch.from_dlpack(dev)
    assert t.is_cuda
    np.testing.assert_array_equal(t.cpu().numpy(), a)
    # Import a device torch tensor back into a unified device Image.
    t2 = torch.as_tensor(a, device="cuda")
    img2 = Image.from_dlpack(t2)
    assert img2.device == "cuda:0"
    np.testing.assert_array_equal(img2.numpy(), a)


def test_from_dlpack_copy_isolates_and_zerocopy_keepalive():
    """DLPack import copy semantics + keepalive (regression coverage):

    - `copy=True` (default) produces an OWNED device buffer; mutating the
      producer afterwards must NOT change the imported image.
    - `copy=False` aliases the producer AND keeps it alive: dropping the source
      tensor while the image lives must not free the device memory (no UAF).
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()

    # copy=True: independent buffer.
    t = torch.as_tensor(a, device="cuda")
    owned = Image.from_dlpack(t, copy=True)
    t.zero_()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(owned.numpy(), a)  # unchanged by producer write

    # copy=False: zero-copy alias that survives the producer being dropped.
    t2 = torch.as_tensor(a, device="cuda")
    aliased = Image.from_dlpack(t2, copy=False)
    del t2
    import gc

    gc.collect()
    # Force torch's caching allocator to actually recycle the freed block: churn
    # a batch of same-shaped tensors so a broken keepalive (premature free) would
    # have its bytes overwritten here rather than lingering intact by luck.
    churn = [torch.full_like(torch.as_tensor(a, device="cuda"), 0) for _ in range(8)]
    torch.cuda.synchronize()
    del churn
    gc.collect()
    np.testing.assert_array_equal(aliased.numpy(), a)  # keepalive prevented free


def test_from_dlpack_infers_device():
    """The universal Image.from_dlpack infers host vs CUDA from the source."""
    torch = pytest.importorskip("torch")
    a = _rgb()
    # Host tensor -> host Image.
    host = Image.from_dlpack(torch.from_numpy(a))
    assert host.device == "cpu"
    np.testing.assert_array_equal(host.numpy(), a)
    if torch.cuda.is_available():
        # CUDA tensor -> device Image (auto-inferred).
        dev = Image.from_dlpack(torch.as_tensor(a, device="cuda"))
        assert dev.device == "cuda:0"
        np.testing.assert_array_equal(dev.numpy(), a)


def test_cuda_color_op_on_unified_image():
    a = _rgb()
    gray = kornia_rs.imgproc.gray_from_rgb(_dev(a))
    assert gray.device == "cuda:0" and gray.channels == 1
    cpu_gray = np.asarray(kornia_rs.imgproc.gray_from_rgb(a)).squeeze()
    np.testing.assert_array_equal(gray.numpy().squeeze(-1), cpu_gray)
