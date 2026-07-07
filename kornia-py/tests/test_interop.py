"""Zero-copy device interop: DLPack / CUDA Array Interface / streams / TensorRT.

Verifies that the device-resident surfaces genuinely SHARE the device buffer
(pointer identity + write-through), not merely produce equal values. Gated on a
torch-CUDA build (and TensorRT for the last test), so it runs only where those
consumers are present — the kornia device suite itself has no such dependency.
"""

import numpy as np
import pytest

import kornia_rs
from kornia_rs.image import Image
from kornia_rs.cuda import Stream

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available(),
    reason="kornia_rs.cuda not built or no CUDA device",
)

RNG = np.random.default_rng(0)


def _rgb(h=48, w=64):
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _dptr(img: Image) -> int:
    """Raw device pointer of a device Image (via the CUDA Array Interface)."""
    return img.__cuda_array_interface__["data"][0]


def test_dlpack_export_is_zero_copy():
    """device Image -> torch shares the buffer: same pointer, and a write through
    torch is visible in kornia (proves aliasing, not a copy)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    dev = Image.from_numpy(a).to_cuda()
    t = torch.from_dlpack(dev)
    assert t.is_cuda and tuple(t.shape) == a.shape
    assert t.data_ptr() == _dptr(dev)  # same device pointer — no copy
    np.testing.assert_array_equal(t.cpu().numpy(), a)
    # Write through torch, observe in kornia.
    t[0, 0, 0] = 7
    torch.cuda.synchronize()
    assert dev.numpy()[0, 0, 0] == 7


def test_dlpack_import_zero_copy_alias():
    """torch CUDA tensor -> Image.from_dlpack(copy=False) aliases the producer."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    t = torch.as_tensor(a, device="cuda")
    img = Image.from_dlpack(t, copy=False)
    assert img.device == "cuda:0"
    assert _dptr(img) == t.data_ptr()
    t[1, 1, 1] = 9
    torch.cuda.synchronize()
    assert img.numpy()[1, 1, 1] == 9


def test_dlpack_import_copy_isolates():
    """copy=True produces an owned buffer that the producer can't mutate."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    t = torch.as_tensor(a, device="cuda")
    owned = Image.from_dlpack(t, copy=True)
    assert _dptr(owned) != t.data_ptr()
    t.zero_()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(owned.numpy(), a)


def test_cai_and_dlpack_agree():
    """The CUDA Array Interface pointer matches the DLPack export pointer."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    dev = Image.from_numpy(_rgb()).to_cuda()
    assert _dptr(dev) == torch.from_dlpack(dev).data_ptr()


def test_cudatensor_zero_copy_to_torch():
    """CudaPreprocessor output -> torch is zero-copy; CAI ptr == data_ptr."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    pre = cuda.CudaPreprocessor(format="rgb")
    ct = pre.run(a.reshape(-1).copy(), a.shape[1], a.shape[0], 32, 32)
    t = torch.from_dlpack(ct)
    assert t.data_ptr() == ct.data_ptr == ct.__cuda_array_interface__["data"][0]
    assert tuple(t.shape) == (1, 3, 32, 32) and t.dtype == torch.float32


def test_adopt_torch_stream():
    """A torch CUDA stream is adopted by kornia (by handle and via the .ptr
    protocol) and correctly threads device work; the preprocessor fences its
    output into that stream for a zero-copy hand-off."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")
    a = _rgb()
    ts = torch.cuda.Stream()
    assert Stream.from_handle(ts.cuda_stream).cuda_stream_ptr == ts.cuda_stream

    class WithPtr:
        ptr = ts.cuda_stream

    assert Stream.from_cuda_stream(WithPtr()).cuda_stream_ptr == ts.cuda_stream

    ks = Stream.from_handle(ts.cuda_stream)
    dev = Image.from_numpy(a).to_cuda(ks)
    np.testing.assert_array_equal(dev.numpy(), a)

    pre = cuda.CudaPreprocessor(format="rgb")
    out = pre.run(a.reshape(-1).copy(), a.shape[1], a.shape[0], 32, 32, stream=ks)
    with torch.cuda.stream(ts):
        t = torch.from_dlpack(out)
        s = t.sum()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(s))


def test_tensorrt_binds_kornia_data_ptr_zero_copy():
    """Bind a kornia CudaTensor's device pointer straight into a TensorRT engine
    input (set_tensor_address) — no copy — and verify inference."""
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA")

    N, C, H, W = 1, 3, 64, 64
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    inp = net.add_input("images", trt.float32, (N, C, H, W))
    relu = net.add_activation(inp, trt.ActivationType.RELU)
    relu.get_output(0).name = "out"
    net.mark_output(relu.get_output(0))
    plan = builder.build_serialized_network(net, builder.create_builder_config())
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()

    a = RNG.integers(0, 256, (H * W * 3,), dtype=np.uint8)
    pre = cuda.CudaPreprocessor(format="rgb")
    tin = pre.run(a, W, H, H, W)  # device CudaTensor, zero-copy handoff
    tout = torch.empty((N, C, H, W), dtype=torch.float32, device="cuda")

    stream = torch.cuda.Stream()
    ctx.set_tensor_address("images", tin.data_ptr)  # kornia device ptr, no copy
    ctx.set_tensor_address("out", tout.data_ptr())
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    ref = torch.relu(torch.from_dlpack(tin))
    assert torch.allclose(tout, ref, atol=1e-5)
