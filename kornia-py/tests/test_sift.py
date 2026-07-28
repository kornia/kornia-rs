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


def _device(img: np.ndarray):
    return K.image.Image.from_numpy(img).to_cuda(_cuda_stream())


def _col(kp, name: str) -> np.ndarray:
    """One field of a keypoint list as an array.

    The list is one Python object per keypoint, so bulk assertions build the
    column once here rather than re-walking it per comparison.
    """
    return np.fromiter((getattr(k, name) for k in kp), float, len(kp))


def _kp_block(kp) -> list:
    """A keypoint list flattened to comparable tuples, for equality asserts."""
    return [
        (k.x, k.y, k.size, k.angle, k.response, k.packed_octave) for k in kp
    ]


def _desc_numpy(desc) -> np.ndarray:
    """Descriptors as a host (N, 128) array, whichever container they came in.

    The device path returns a ``(1, 1, N, 128)`` Tensor — rank 4 because that is
    what ``Tensor`` currently models — so the leading singleton axes come off
    here rather than at every call site.
    """
    if isinstance(desc, np.ndarray):
        return desc
    return desc.numpy()[0, 0]


# ── Keypoints ────────────────────────────────────────────────────────────────


def test_detect_and_compute_shapes():
    kp, desc = K.imgproc.Sift().detect_and_compute(_image())
    assert isinstance(kp, list), "shaped like cv2.SIFT.detectAndCompute"
    assert len(kp) > 8, "test image should yield enough keypoints to be useful"
    assert desc.shape == (len(kp), 128), "one descriptor row per keypoint"
    k = kp[0]
    for name in ("x", "y", "size", "angle", "response", "xi"):
        assert isinstance(getattr(k, name), float)
    for name in ("octave", "layer", "packed_octave"):
        assert isinstance(getattr(k, name), int)


def test_keypoints_are_in_bounds():
    img = _image()
    kp, _ = K.imgproc.Sift().detect_and_compute(img)
    h, w = img.shape[:2]
    x, y = _col(kp, "x"), _col(kp, "y")
    assert x.min() >= 0 and x.max() <= w
    assert y.min() >= 0 and y.max() <= h
    assert np.all(_col(kp, "size") > 0)
    angle = _col(kp, "angle")
    assert angle.min() >= 0 and angle.max() < 360


def test_octave_layer_xi_decode_the_packed_field():
    """``packed_octave`` is OpenCV's own three-values-in-an-int32; the decoded
    fields must be exactly ``unpackOctave``, including the sign extension that
    turns the low byte 255 into ``firstOctave = -1``."""
    kp, _ = K.imgproc.Sift().detect_and_compute(_image())
    for k in kp:
        lo = k.packed_octave & 255
        assert k.octave == (lo if lo < 128 else lo | -128)
        assert k.layer == (k.packed_octave >> 8) & 255
        # In float32, as the decode computes it: Python would otherwise use
        # float64 and disagree in the last bit on ~20% of the values.
        expect_xi = np.float32((k.packed_octave >> 16) & 255) / np.float32(
            255.0
        ) - np.float32(0.5)
        assert np.float32(k.xi) == expect_xi
        assert -0.5 <= k.xi <= 0.5

    # upsample=True is firstOctave=-1, so the finest octave must be present.
    assert min(k.octave for k in kp) == -1


def test_keypoint_repr_is_readable():
    kp, _ = K.imgproc.Sift().detect_and_compute(_image())
    r = repr(kp[0])
    assert r.startswith("SiftKeypoint(")
    for field in ("x=", "y=", "size=", "angle=", "response=", "octave=", "layer="):
        assert field in r


# ── Detection ────────────────────────────────────────────────────────────────


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
    assert _kp_block(kp1) == _kp_block(kp2)
    np.testing.assert_array_equal(d1, d2)


def test_reusing_one_detector_matches_a_fresh_one():
    """The detector owns scratch that is reused across calls; a second frame
    must not be polluted by the first."""
    img = _image()
    reused = K.imgproc.Sift()
    reused.detect_and_compute(_image(seed=12345))
    kp_reused, d_reused = reused.detect_and_compute(img)
    kp_fresh, d_fresh = K.imgproc.Sift().detect_and_compute(img)
    assert _kp_block(kp_reused) == _kp_block(kp_fresh)
    np.testing.assert_array_equal(d_reused, d_fresh)


def test_n_features_caps_and_keeps_the_strongest():
    """`n_features` is applied before descriptors, as the reference does, so it
    must cap the count AND drop only the weakest keypoints.

    This is the half of the budget contract that is cheapest to get wrong: the
    keypoints kept must be a subset of the unbudgeted run, not a re-detection.
    """
    img = _image()
    kp_all, _ = K.imgproc.Sift().detect_and_compute(img)
    n = len(kp_all) // 2
    assert n > 2, "need enough keypoints to halve"

    kp_cut, d_cut = K.imgproc.Sift(n_features=n).detect_and_compute(img)
    assert len(kp_cut) == n
    assert d_cut.shape == (n, 128)

    kept = {(k.x, k.y) for k in kp_cut}
    dropped = [k.response for k in kp_all if (k.x, k.y) not in kept]
    assert len(dropped) == len(kp_all) - n
    if dropped:
        assert max(dropped) <= min(k.response for k in kp_cut)


def test_n_features_above_the_count_is_a_no_op():
    img = _image()
    kp_all, d_all = K.imgproc.Sift().detect_and_compute(img)
    kp, d = K.imgproc.Sift(n_features=len(kp_all) + 1000).detect_and_compute(img)
    assert _kp_block(kp) == _kp_block(kp_all)
    np.testing.assert_array_equal(d, d_all)


def test_upsample_false_is_faster_and_finds_fewer_keypoints():
    """`upsample=False` skips the 2x upsample, so every octave is a quarter of
    the pixels and far fewer keypoints survive."""
    img = _image()
    many, _ = K.imgproc.Sift().detect_and_compute(img)
    few, _ = K.imgproc.Sift(upsample=False).detect_and_compute(img)
    assert 0 < len(few) < len(many)
    # Without the doubling there is no octave below zero.
    assert min(k.octave for k in few) == 0


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


# ── Matching ─────────────────────────────────────────────────────────────────


def test_match_takes_descriptors_and_self_matches_identically():
    s = K.imgproc.Sift()
    _, desc = s.detect_and_compute(_image())
    pairs = s.match(desc, desc)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert pairs.dtype == np.int32
    # Matching a block against itself must pair every row with itself.
    assert pairs.shape[0] == desc.shape[0]
    assert np.all(pairs[:, 0] == pairs[:, 1])


def test_match_detects_once_and_matches_many():
    """The point of splitting detect from match: one detection feeds several
    matches, which the old image-taking signature could not express."""
    s = K.imgproc.Sift()
    _, d_ref = s.detect_and_compute(_image())
    _, d_a = s.detect_and_compute(_image(seed=4242))
    _, d_b = s.detect_and_compute(_image(seed=777))
    assert s.match(d_ref, d_a).shape[1] == 2
    assert s.match(d_ref, d_b).shape[1] == 2
    # d_ref survived both matches unmodified.
    _, d_ref2 = s.detect_and_compute(_image())
    np.testing.assert_array_equal(d_ref, d_ref2)


def test_match_ratio_and_cross_check_are_accepted():
    s = K.imgproc.Sift()
    _, da = s.detect_and_compute(_image())
    _, db = s.detect_and_compute(_image(seed=4242))
    strict = s.match(da, db, ratio=0.6, cross_check=True)
    loose = s.match(da, db, ratio=0.95, cross_check=True)
    assert strict.shape[0] <= loose.shape[0], "a tighter ratio cannot add pairs"


def test_match_rejects_a_wrong_descriptor_width():
    """The matcher derives its row count from the buffer length, so a block with
    the wrong width would be silently read as a different number of rows."""
    s = K.imgproc.Sift()
    _, d = s.detect_and_compute(_image())
    with pytest.raises(Exception):
        s.match(d[:, :64], d)


# ── CUDA ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_matches_the_host_path():
    """Residency must change speed and containers, not results.

    The CUDA and CPU paths are separate implementations held to the same
    bitwise contract against `cv::SIFT`, so their output should agree exactly.
    """
    img = _image()
    kp_h, d_h = K.imgproc.Sift().detect_and_compute(img)
    kp_d, d_d = K.imgproc.Sift().detect_and_compute(_device(img))

    assert len(kp_d) == len(kp_h)
    assert _kp_block(kp_d) == _kp_block(kp_h)
    np.testing.assert_array_equal(_desc_numpy(d_d), d_h)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_descriptors_stay_on_device():
    """The whole reason the device path returns a Tensor: a 2515x128 block costs
    more to move to the host than the detection that produced it."""
    kp, desc = K.imgproc.Sift().detect_and_compute(_device(_image()))
    assert not isinstance(desc, np.ndarray)
    assert desc.device.startswith("cuda:")
    assert desc.shape == (1, 1, len(kp), 128)
    assert desc.dtype == "float32"


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_descriptors_survive_the_next_detect():
    """They are a fresh allocation, not a view into the plan's single output
    buffer — a view would change under a caller holding two frames, which is
    exactly what frame-to-frame matching does."""
    s = K.imgproc.Sift()
    _, first = s.detect_and_compute(_device(_image()))
    before = _desc_numpy(first).copy()
    s.detect_and_compute(_device(_image(seed=4242)))
    np.testing.assert_array_equal(_desc_numpy(first), before)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_match_agrees_with_the_host_matcher():
    s = K.imgproc.Sift()
    img_a, img_b = _image(), _image(seed=4242)
    _, ha = s.detect_and_compute(img_a)
    _, hb = s.detect_and_compute(img_b)
    _, da = s.detect_and_compute(_device(img_a))
    _, db = s.detect_and_compute(_device(img_b))

    host = s.match(ha, hb)
    dev = s.match(da, db)
    np.testing.assert_array_equal(dev, host)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_match_refuses_mixed_residency():
    """Refused rather than silently transferred — the transfer is the expensive
    part, and hiding it is how a frame budget disappears."""
    s = K.imgproc.Sift()
    img = _image()
    _, host = s.detect_and_compute(img)
    _, dev = s.detect_and_compute(_device(img))
    with pytest.raises(Exception):
        s.match(host, dev)
    with pytest.raises(Exception):
        s.match(dev, host)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_cuda_n_features_caps_the_count():
    """The budget is applied before descriptors on the device path too."""
    dev = _device(_image())
    kp_all, _ = K.imgproc.Sift().detect_and_compute(dev)
    n = len(kp_all) // 2
    assert n > 2

    kp_cut, d_cut = K.imgproc.Sift(n_features=n).detect_and_compute(dev)
    assert len(kp_cut) == n
    assert d_cut.shape == (1, 1, n, 128)

    kept = {(k.x, k.y) for k in kp_cut}
    dropped = [k.response for k in kp_all if (k.x, k.y) not in kept]
    if dropped:
        assert max(dropped) <= min(k.response for k in kp_cut)


@pytest.mark.skipif(_cuda_stream() is None, reason="no CUDA device")
def test_fast_descriptor_keeps_the_keypoints_and_the_descriptor_shape():
    """`fast_descriptor` is an opt-in approximation: it samples a rotated frame
    rather than the reference's pixel walk, and its orientation accumulates with
    atomics. Descriptors therefore differ, but the detector is untouched, so the
    keypoint count should be essentially unchanged.
    """
    dev = _device(_image())
    kp_exact, _ = K.imgproc.Sift().detect_and_compute(dev)
    kp_fast, d_fast = K.imgproc.Sift(fast_descriptor=True).detect_and_compute(dev)

    assert d_fast.shape == (1, 1, len(kp_fast), 128)
    # Atomic accumulation makes a borderline orientation peak non-deterministic,
    # so allow a small drift rather than requiring an exact count.
    assert abs(len(kp_fast) - len(kp_exact)) <= max(2, len(kp_exact) // 100)
