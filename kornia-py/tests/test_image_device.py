"""Tests for the unified device-aware ``Image`` API (``Image.cuda.*``, ``.cpu()``,
``.to_cuda()``, ``.device``, auto-download ``.numpy()``, DLPack + CUDA Array
Interface interop).

Skipped wholesale when the wheel was built without the ``cuda`` feature or no
CUDA device is present.
"""

import numpy as np
import pytest

import kornia_rs
from kornia_rs.image import Image, Stream

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
    img = Image.cuda.from_numpy(a)
    assert img.device == "cuda:0"
    # .numpy() auto-downloads (D2H) a device image.
    np.testing.assert_array_equal(img.numpy(), a)


def test_from_numpy_roundtrip_float32():
    a = _rgbf()
    img = Image.cuda.from_numpy(a)
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
    img = Image.cuda.zeros(width=8, height=6, channels=3, dtype="uint8")
    assert img.device == "cuda:0"
    assert (img.height, img.width, img.channels) == (6, 8, 3)
    np.testing.assert_array_equal(img.numpy(), np.zeros((6, 8, 3), np.uint8))


def test_unsupported_dtype_on_device_raises():
    a = RNG.integers(0, 1000, (8, 8, 3), dtype=np.uint16)
    with pytest.raises(ValueError):
        Image.cuda.from_numpy(a)


def test_cuda_array_interface_present_on_device_only():
    a = _rgb()
    dev = Image.cuda.from_numpy(a)
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
    img = Image.cuda.from_numpy(_rgb(), stream=s)
    assert img.device == "cuda:0"


def test_dlpack_roundtrip_with_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    dev = Image.cuda.from_numpy(a)
    # Export device Image -> torch (zero-copy), values match.
    t = torch.from_dlpack(dev)
    assert t.is_cuda
    np.testing.assert_array_equal(t.cpu().numpy(), a)
    # Import a device torch tensor back into a unified device Image.
    t2 = torch.as_tensor(a, device="cuda")
    img2 = Image.cuda.from_dlpack(t2)
    assert img2.device == "cuda:0"
    np.testing.assert_array_equal(img2.numpy(), a)


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
    gray = cuda.gray_from_rgb(Image.cuda.from_numpy(a))
    assert gray.device == "cuda:0" and gray.channels == 1
    cpu_gray = np.asarray(kornia_rs.imgproc.gray_from_rgb(a)).squeeze()
    np.testing.assert_array_equal(gray.numpy().squeeze(-1), cpu_gray)
