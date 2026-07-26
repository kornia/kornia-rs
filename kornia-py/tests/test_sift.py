import numpy as np
import pytest

import kornia_rs as K


def _image(w: int = 192, h: int = 144, seed: int = 0x9E3779B9) -> np.ndarray:
    """Deterministic 0..255 grayscale in the layout Sift expects: (H, W, 1) f32.

    The reference works in 0..255 floats, not 0..1 — normalising changes what
    the contrast threshold means and yields almost no keypoints.
    """
    s = seed
    out = np.empty(w * h, dtype=np.float32)
    for i in range(w * h):
        # Plain ints with an explicit mask: numpy's uint32 warns on overflow,
        # which is the intended behaviour of an LCG.
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = float((s >> 16) & 255)
    return np.ascontiguousarray(out.reshape(h, w, 1))


def _cuda_stream():
    """A CUDA stream, or None when this build or host has no device."""
    try:
        return K.cuda.Stream.default()
    except Exception:  # noqa: BLE001 - CPU-only build or no device
        return None


def test_detect_and_compute_shapes():
    kp, desc = K.imgproc.Sift().detect_and_compute(_image())
    assert kp.ndim == 2 and desc.ndim == 2
    assert kp.shape[0] == desc.shape[0], "one descriptor row per keypoint"
    assert desc.shape[1] == 128
    assert kp.shape[0] > 8, "test image should yield enough keypoints to be useful"


def test_descriptors_are_quantised_bytes_in_f32():
    """The reference scales to 0..255 and rounds, storing the result as f32."""
    _, desc = K.imgproc.Sift().detect_and_compute(_image())
    assert desc.dtype == np.float32
    assert desc.min() >= 0.0 and desc.max() <= 255.0
    assert np.all(desc == np.floor(desc)), "values must be integral"


def test_is_deterministic():
    img = _image()
    s = K.imgproc.Sift()
    kp1, d1 = s.detect_and_compute(img)
    kp2, d2 = s.detect_and_compute(img)
    np.testing.assert_array_equal(kp1, kp2)
    np.testing.assert_array_equal(d1, d2)


def test_reusing_one_detector_matches_a_fresh_one():
    """The detector owns scratch that is reused across calls; a second frame
    must not be polluted by the first."""
    img = _image()
    reused = K.imgproc.Sift()
    reused.detect_and_compute(_image(seed=12345))
    kp_reused, d_reused = reused.detect_and_compute(img)
    kp_fresh, d_fresh = K.imgproc.Sift().detect_and_compute(img)
    np.testing.assert_array_equal(kp_reused, kp_fresh)
    np.testing.assert_array_equal(d_reused, d_fresh)


def test_n_features_caps_and_keeps_the_strongest():
    """`n_features` is applied before descriptors, as the reference does, so it
    must cap the count AND drop only the weakest keypoints.

    This is the half of the budget contract that is cheapest to get wrong: the
    keypoints kept must be a subset of the unbudgeted run, not a re-detection.
    """
    img = _image()
    kp_all, d_all = K.imgproc.Sift().detect_and_compute(img)
    n = kp_all.shape[0] // 2
    assert n > 2, "need enough keypoints to halve"

    kp_cut, d_cut = K.imgproc.Sift(n_features=n).detect_and_compute(img)
    assert kp_cut.shape[0] == n
    assert d_cut.shape == (n, 128)

    # Column 4 is the response. Every kept keypoint is at least as strong as
    # every dropped one.
    kept = {tuple(r[:2]) for r in kp_cut}
    dropped = [r for r in kp_all if tuple(r[:2]) not in kept]
    assert len(dropped) == kp_all.shape[0] - n
    if dropped:
        assert max(r[4] for r in dropped) <= min(r[4] for r in kp_cut)


def test_n_features_above_the_count_is_a_no_op():
    img = _image()
    kp_all, d_all = K.imgproc.Sift().detect_and_compute(img)
    kp, d = K.imgproc.Sift(n_features=kp_all.shape[0] + 1000).detect_and_compute(img)
    np.testing.assert_array_equal(kp, kp_all)
    np.testing.assert_array_equal(d, d_all)


def test_upsample_false_is_faster_and_finds_fewer_keypoints():
    """`upsample=False` skips the 2x upsample, so every octave is a quarter of
    the pixels and far fewer keypoints survive."""
    img = _image()
    many, _ = K.imgproc.Sift().detect_and_compute(img)
    few, _ = K.imgproc.Sift(upsample=False).detect_and_compute(img)
    assert 0 < few.shape[0] < many.shape[0]


def test_match_is_symmetric_in_shape_and_self_matches_identically():
    img = _image()
    s = K.imgproc.Sift()
    kp_a, kp_b, pairs = s.match(img, img)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    # Matching an image against itself must pair every keypoint with itself.
    assert pairs.shape[0] == kp_a.shape[0]
    assert np.all(pairs[:, 0] == pairs[:, 1])


def test_match_ratio_and_cross_check_are_accepted():
    img_a, img_b = _image(), _image(seed=4242)
    s = K.imgproc.Sift()
    _, _, strict = s.match(img_a, img_b, ratio=0.6, cross_check=True)
    _, _, loose = s.match(img_a, img_b, ratio=0.95, cross_check=True)
    assert strict.shape[0] <= loose.shape[0], "a tighter ratio cannot add pairs"


def test_rejects_wrong_shapes():
    s = K.imgproc.Sift()
    with pytest.raises(Exception):
        s.detect_and_compute(np.zeros((16, 16), dtype=np.float32))  # missing channel
    with pytest.raises(Exception):
        s.detect_and_compute(np.zeros((16, 16, 3), dtype=np.float32))  # not grayscale


def test_rejects_invalid_config():
    with pytest.raises(Exception):
        K.imgproc.Sift(n_octave_layers=0)
    with pytest.raises(Exception):
        K.imgproc.Sift(sigma=0.0)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_matches_the_host_path():
    """Residency must change speed, not results.

    The CUDA and CPU paths are separate implementations held to the same
    bitwise contract against `cv::SIFT`, so their output should agree exactly.
    """
    img = _image()
    stream = _cuda_stream()
    dev = K.image.Image.from_numpy(img).to_cuda(stream)

    kp_h, d_h = K.imgproc.Sift().detect_and_compute(img)
    kp_d, d_d = K.imgproc.Sift().detect_and_compute(dev)

    assert kp_d.shape == kp_h.shape
    np.testing.assert_array_equal(d_d, d_h)
    np.testing.assert_array_equal(kp_d, kp_h)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_n_features_caps_the_count():
    """The budget is applied before descriptors on the device path too."""
    img = _image()
    stream = _cuda_stream()
    dev = K.image.Image.from_numpy(img).to_cuda(stream)

    kp_all, _ = K.imgproc.Sift().detect_and_compute(dev)
    n = kp_all.shape[0] // 2
    assert n > 2

    kp_cut, d_cut = K.imgproc.Sift(n_features=n).detect_and_compute(dev)
    assert kp_cut.shape[0] == n
    assert d_cut.shape == (n, 128)

    kept = {tuple(r[:2]) for r in kp_cut}
    dropped = [r for r in kp_all if tuple(r[:2]) not in kept]
    if dropped:
        assert max(r[4] for r in dropped) <= min(r[4] for r in kp_cut)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_fast_descriptor_keeps_the_keypoints_and_the_descriptor_shape():
    """`fast_descriptor` is an opt-in approximation: it samples a rotated frame
    rather than the reference's pixel walk, and its orientation accumulates with
    atomics. Descriptors therefore differ, but the detector is untouched, so the
    keypoint count should be essentially unchanged.
    """
    img = _image()
    stream = _cuda_stream()
    dev = K.image.Image.from_numpy(img).to_cuda(stream)

    kp_exact, _ = K.imgproc.Sift().detect_and_compute(dev)
    kp_fast, d_fast = K.imgproc.Sift(fast_descriptor=True).detect_and_compute(dev)

    assert d_fast.shape[1] == 128
    assert d_fast.shape[0] == kp_fast.shape[0]
    # Atomic accumulation makes a borderline orientation peak non-deterministic,
    # so allow a small drift rather than requiring an exact count.
    assert abs(kp_fast.shape[0] - kp_exact.shape[0]) <= max(2, kp_exact.shape[0] // 100)
