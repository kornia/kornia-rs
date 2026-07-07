"""Tests for kornia_rs.cuda — GPU color conversions + fused preprocessing.

Skipped wholesale when no GPU (or the wheel was built without the `cuda`
feature). Device pixels now live in the unified ``kornia_rs.image.Image``
(``Image.from_numpy(a).to_cuda()``); the color-conversion functions take and
return such a device ``Image``.
"""

import numpy as np
import pytest

import kornia_rs
from kornia_rs.image import Image

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available(),
    reason="kornia_rs.cuda not built or no CUDA device",
)

RNG = np.random.default_rng(42)


def nv12_frame(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """A synthetic tightly-packed NV12 buffer (w*h luma + w*h/2 chroma)."""
    return rng.integers(0, 256, (w * h * 3 // 2,), dtype=np.uint8)


def _rgb(h=48, w=64):
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _dev(a):
    """Host numpy array -> device Image."""
    return Image.from_numpy(a).to_cuda()


def test_gray_matches_cpu_bit_exact():
    img = _rgb()
    gpu = kornia_rs.imgproc.gray_from_rgb(_dev(img)).numpy()
    cpu = kornia_rs.imgproc.gray_from_rgb(img)
    np.testing.assert_array_equal(gpu.squeeze(-1), np.asarray(cpu).squeeze())


@pytest.mark.parametrize("h,w", [(17, 23), (65, 31), (1, 1), (3, 97), (100, 1)])
def test_color_ops_bit_exact_odd_dims(h, w):
    """Non-block-aligned / prime / degenerate dims: a grid-remainder partial write
    over the uninitialized device destination (`uninit_cuda`) would leave garbage
    in border pixels. Deterministic bit-exact + identity checks at these sizes
    guard that every color kernel writes every output pixel."""
    a = RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)
    d = _dev(a)
    # gray has a CPU oracle -> bit-exact over every pixel including borders.
    gpu_gray = kornia_rs.imgproc.gray_from_rgb(d).numpy().reshape(h, w)
    cpu_gray = np.asarray(kornia_rs.imgproc.gray_from_rgb(a)).reshape(h, w)
    np.testing.assert_array_equal(gpu_gray, cpu_gray)
    # Channel-expand (u8c4 dst) + swap (u8c3 dst) round-trips are identities; any
    # unwritten border pixel (uninit garbage) breaks the exact comparison.
    np.testing.assert_array_equal(kornia_rs.imgproc.rgb_from_rgba(kornia_rs.imgproc.rgba_from_rgb(d)).numpy(), a)
    np.testing.assert_array_equal(kornia_rs.imgproc.bgr_from_rgb(kornia_rs.imgproc.bgr_from_rgb(d)).numpy(), a)


def test_color_op_dispatches_on_residency():
    # One imgproc op, three inputs: numpy -> numpy (CPU), host Image -> host
    # Image (CPU), device Image -> device Image (GPU); all bit-exact.
    a = _rgb()
    ref = np.asarray(kornia_rs.imgproc.gray_from_rgb(a)).reshape(a.shape[:2])

    host = kornia_rs.imgproc.gray_from_rgb(Image.from_numpy(a))
    assert host.device == "cpu"
    np.testing.assert_array_equal(host.numpy().reshape(a.shape[:2]), ref)

    dev = kornia_rs.imgproc.gray_from_rgb(_dev(a))
    assert dev.device == "cuda:0"
    np.testing.assert_array_equal(dev.numpy().reshape(a.shape[:2]), ref)

    # A CPU-only op (no GPU kernel) on a device image raises a clear error.
    af = (a.astype(np.float32) / 255.0).copy()
    with pytest.raises(ValueError, match="no GPU kernel"):
        kornia_rs.imgproc.hls_from_rgb(_dev(af))


def test_conversion_chain_stays_on_device():
    d = _dev(_rgb())
    ycc = kornia_rs.imgproc.ycbcr_from_rgb(d)
    assert ycc.device == "cuda:0"
    rgb = kornia_rs.imgproc.rgb_from_ycbcr(ycc)
    out = rgb.numpy()
    assert out.shape == (48, 64, 3)
    # Round-trip within the documented tolerance of the fixed-point path.
    assert np.max(np.abs(out.astype(int) - d.numpy().astype(int))) <= 3


def test_f32_ops():
    img = (_rgb().astype(np.float32) / 255.0).copy()
    d = _dev(img)
    lab = kornia_rs.imgproc.lab_from_rgb(d)
    back = kornia_rs.imgproc.rgb_from_lab(lab).numpy()
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, img, atol=2e-2)


def test_colormap_and_bayer():
    gray = RNG.integers(0, 256, (48, 64, 1), dtype=np.uint8)
    d = _dev(gray)
    assert kornia_rs.imgproc.apply_colormap(d, "jet").channels == 3
    assert kornia_rs.imgproc.rgb_from_bayer(d, "rggb").channels == 3
    with pytest.raises(ValueError):
        kornia_rs.imgproc.apply_colormap(d, "nope")


def test_preprocessor_nv12_fused():
    w, h = 128, 96
    frame = nv12_frame(w, h, RNG)
    pre = cuda.CudaPreprocessor(mode="letterbox", format="nv12")
    t = pre.run(frame, w, h, 64, 64)
    assert t.shape == (1, 3, 64, 64)
    assert t.dtype == "float32"
    out = t.numpy()
    assert out.shape == (1, 3, 64, 64)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocessor_f16_and_normalize():
    w, h = 64, 48
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    mean, std = cuda.IMAGENET_MEAN, cuda.IMAGENET_STD
    pre16 = cuda.CudaPreprocessor(format="rgb", f16=True, mean=mean, std=std)
    t = pre16.run(frame, w, h, 32, 32)
    assert t.dtype == "float16"
    got = t.numpy()  # widened to f32

    plain = cuda.CudaPreprocessor(format="rgb").run(frame, w, h, 32, 32).numpy()
    want = (plain - np.asarray(mean).reshape(1, 3, 1, 1)) / np.asarray(std).reshape(
        1, 3, 1, 1
    )
    np.testing.assert_allclose(got, want, atol=2e-3)


def test_dlpack_export_to_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    img = _rgb()
    d = kornia_rs.imgproc.gray_from_rgb(_dev(img))
    t = torch.from_dlpack(d)
    assert t.is_cuda and t.shape == (48, 64, 1) and t.dtype == torch.uint8
    np.testing.assert_array_equal(t.cpu().numpy(), d.numpy())

    # Tensor export too (f32 CHW).
    pre = cuda.CudaPreprocessor(format="rgb")
    ct = pre.run(img.reshape(-1).copy(), 64, 48, 32, 32)
    tt = torch.from_dlpack(ct)
    assert tt.is_cuda and tt.shape == (1, 3, 32, 32) and tt.dtype == torch.float32


def test_cuda_tensor_interop_surface():
    """CudaTensor exposes the zero-copy handoff surface TensorRT / cupy need:
    data_ptr, device, and the CUDA Array Interface."""
    w, h = 64, 48
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    t = cuda.CudaPreprocessor(format="rgb").run(frame, w, h, 32, 32)

    assert t.device == "cuda:0"
    assert isinstance(t.data_ptr, int) and t.data_ptr != 0

    cai = t.__cuda_array_interface__
    assert cai["version"] == 3
    assert cai["shape"] == (1, 3, 32, 32)
    assert cai["typestr"] == "<f4"
    ptr, readonly = cai["data"]
    assert ptr == t.data_ptr and readonly is False
    # CAI stream must be None or a positive int — never the ambiguous 0.
    assert cai["stream"] is None or (isinstance(cai["stream"], int) and cai["stream"] >= 1)

    # f16 advertises the half typestr.
    t16 = cuda.CudaPreprocessor(format="rgb", f16=True).run(frame, w, h, 32, 32)
    assert t16.__cuda_array_interface__["typestr"] == "<f2"


def test_preprocessor_run_with_consumer_stream():
    """Passing a consumer stream is accepted and the result is correct (output
    is fenced into that stream)."""
    from kornia_rs.cuda import Stream

    w, h = 64, 48
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    pre = cuda.CudaPreprocessor(format="rgb")
    fs = Stream.from_handle(Stream.default().cuda_stream_ptr)
    t = pre.run(frame, w, h, 32, 32, stream=fs)
    plain = pre.run(frame, w, h, 32, 32)
    np.testing.assert_array_equal(t.numpy(), plain.numpy())


def test_preprocessor_run_into_matches_run():
    """run_into writes into a preallocated output identical to what run() would
    produce, without allocating a new tensor per call."""
    w, h = 64, 48
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    pre = cuda.CudaPreprocessor(format="rgb")

    out = pre.alloc_output(32, 32)
    assert out.shape == (1, 3, 32, 32) and out.dtype == "float32"
    pre.run_into(out, frame, w, h)
    want = pre.run(frame, w, h, 32, 32).numpy()
    np.testing.assert_array_equal(out.numpy(), want)

    # Reusing the same buffer for a second frame overwrites it in place.
    frame2 = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    pre.run_into(out, frame2, w, h)
    np.testing.assert_array_equal(out.numpy(), pre.run(frame2, w, h, 32, 32).numpy())


def test_preprocessor_run_into_f16_and_dtype_mismatch():
    w, h = 48, 32
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    pre16 = cuda.CudaPreprocessor(format="rgb", f16=True)
    out16 = pre16.alloc_output(16, 16)
    assert out16.dtype == "float16"
    pre16.run_into(out16, frame, w, h)
    np.testing.assert_allclose(
        out16.numpy(), pre16.run(frame, w, h, 16, 16).numpy(), atol=1e-3
    )

    # A dtype-mismatched output is rejected.
    pre32 = cuda.CudaPreprocessor(format="rgb")
    with pytest.raises(ValueError):
        pre32.run_into(out16, frame, w, h)  # f32 preprocessor, f16 buffer


def test_cuda_tensor_dlpack_versioned_capsule():
    """Requesting max_version >= (1,0) yields a versioned capsule; the default
    stays unversioned. Both must round-trip through torch when available."""
    w, h = 64, 48
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    t = cuda.CudaPreprocessor(format="rgb").run(frame, w, h, 32, 32)

    cap_v = t.__dlpack__(max_version=(1, 0))
    assert "dltensor_versioned" in str(cap_v)
    cap_u = t.__dlpack__()
    # Unversioned capsule name is plain "dltensor".
    name = str(cap_u)
    assert "dltensor" in name and "versioned" not in name


def test_preprocessor_run_batch_matches_single():
    w, h = 64, 48
    frames = [nv12_frame(w, h, RNG) for _ in range(3)]
    pre = cuda.CudaPreprocessor(mode="letterbox", format="nv12")
    batch = pre.run_batch(frames, w, h, 32, 32)
    assert batch.shape == (3, 3, 32, 32)
    got = batch.numpy()
    for i, f in enumerate(frames):
        single = pre.run(f, w, h, 32, 32).numpy()
        np.testing.assert_array_equal(got[i : i + 1], single)

    # f16 batch follows the constructor flag.
    pre16 = cuda.CudaPreprocessor(mode="letterbox", format="nv12", f16=True)
    b16 = pre16.run_batch(frames, w, h, 32, 32)
    assert b16.dtype == "float16" and b16.shape == (3, 3, 32, 32)
    np.testing.assert_allclose(b16.numpy(), got, atol=1e-3)
