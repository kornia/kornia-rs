"""Tests for kornia_rs.cuda — skipped wholesale when no GPU (or the wheel was
built without the `cuda` feature)."""

import numpy as np
import pytest

import kornia_rs

cuda = getattr(kornia_rs, "cuda", None)
pytestmark = pytest.mark.skipif(
    cuda is None or not cuda.is_available(),
    reason="kornia_rs.cuda not built or no CUDA device",
)

RNG = np.random.default_rng(42)


def _rgb(h=48, w=64):
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_upload_download_roundtrip():
    img = _rgb()
    d = cuda.upload(img)
    assert (d.height, d.width, d.channels, d.dtype) == (48, 64, 3, "uint8")
    back = d.download()
    np.testing.assert_array_equal(back, img)


def test_gray_matches_cpu_bit_exact():
    img = _rgb()
    gpu = cuda.gray_from_rgb(cuda.upload(img)).download()
    cpu = kornia_rs.imgproc.gray_from_rgb(img)
    np.testing.assert_array_equal(gpu.squeeze(-1), np.asarray(cpu).squeeze())


def test_conversion_chain_stays_on_device():
    d = cuda.upload(_rgb())
    ycc = cuda.ycbcr_from_rgb(d)
    rgb = cuda.rgb_from_ycbcr(ycc)
    out = rgb.download()
    assert out.shape == (48, 64, 3)
    # Round-trip within the documented tolerance of the fixed-point path.
    assert np.max(np.abs(out.astype(int) - d.download().astype(int))) <= 3


def test_f32_ops():
    img = (_rgb().astype(np.float32) / 255.0).copy()
    d = cuda.upload(img)
    lab = cuda.lab_from_rgb(d)
    back = cuda.rgb_from_lab(lab).download()
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, img, atol=2e-2)


def test_colormap_and_bayer():
    gray = RNG.integers(0, 256, (48, 64, 1), dtype=np.uint8)
    d = cuda.upload(gray)
    assert cuda.apply_colormap(d, "jet").channels == 3
    assert cuda.rgb_from_bayer(d, "rggb").channels == 3
    with pytest.raises(ValueError):
        cuda.apply_colormap(d, "nope")


def test_preprocessor_nv12_fused():
    w, h = 128, 96
    frame = RNG.integers(0, 256, (w * h * 3 // 2,), dtype=np.uint8)
    pre = cuda.CudaPreprocessor(mode="letterbox", format="nv12")
    t = pre.run(frame, w, h, 64, 64)
    assert t.shape == (1, 3, 64, 64)
    assert t.dtype == "float32"
    out = t.download()
    assert out.shape == (1, 3, 64, 64)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocessor_f16_and_normalize():
    w, h = 64, 48
    frame = RNG.integers(0, 256, (w * h * 3,), dtype=np.uint8)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    pre16 = cuda.CudaPreprocessor(format="rgb", f16=True, mean=mean, std=std)
    t = pre16.run(frame, w, h, 32, 32)
    assert t.dtype == "float16"
    got = t.download()  # widened to f32

    plain = cuda.CudaPreprocessor(format="rgb").run(frame, w, h, 32, 32).download()
    want = (plain - np.asarray(mean).reshape(1, 3, 1, 1)) / np.asarray(std).reshape(
        1, 3, 1, 1
    )
    np.testing.assert_allclose(got, want, atol=2e-3)


def test_dlpack_export_to_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    img = _rgb()
    d = cuda.gray_from_rgb(cuda.upload(img))
    t = torch.from_dlpack(d)
    assert t.is_cuda and t.shape == (48, 64, 1) and t.dtype == torch.uint8
    np.testing.assert_array_equal(t.cpu().numpy(), d.download())

    # Tensor export too (f32 CHW).
    pre = cuda.CudaPreprocessor(format="rgb")
    ct = pre.run(img.reshape(-1).copy(), 64, 48, 32, 32)
    tt = torch.from_dlpack(ct)
    assert tt.is_cuda and tt.shape == (1, 3, 32, 32) and tt.dtype == torch.float32


def test_from_dlpack_torch_roundtrip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    src = RNG.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    t = torch.from_numpy(src).cuda()
    img = cuda.from_dlpack(t)
    assert (img.height, img.width, img.channels) == (48, 64, 3)
    assert img.dtype == "uint8"
    np.testing.assert_array_equal(img.download(), src)

    # Owned copy: mutating the producer afterwards must not affect the image.
    t.zero_()
    np.testing.assert_array_equal(img.download(), src)

    # f32 single-channel too.
    fsrc = RNG.random((16, 20, 1), dtype=np.float32)
    fimg = cuda.from_dlpack(torch.from_numpy(fsrc).cuda())
    assert fimg.dtype == "float32"
    np.testing.assert_array_equal(fimg.download(), fsrc)

    # Host tensors are rejected with a pointer to upload().
    with pytest.raises(ValueError, match="upload"):
        cuda.from_dlpack(torch.from_numpy(src))


def test_preprocessor_run_batch_matches_single():
    w, h = 64, 48
    frames = [
        RNG.integers(0, 256, (w * h * 3 // 2,), dtype=np.uint8) for _ in range(3)
    ]
    pre = cuda.CudaPreprocessor(mode="letterbox", format="nv12")
    batch = pre.run_batch(frames, w, h, 32, 32)
    assert batch.shape == (3, 3, 32, 32)
    got = batch.download()
    for i, f in enumerate(frames):
        single = pre.run(f, w, h, 32, 32).download()
        np.testing.assert_array_equal(got[i : i + 1], single)
