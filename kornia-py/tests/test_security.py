"""Memory-safety regression tests for the PyO3 bindings.

Every test here targets a concrete bug from a security audit of ``kornia-py``:
before the fix each input below crashed the interpreter (out-of-bounds read or
write, heap overflow), leaked stale heap memory to Python, wrote through a
read-only buffer, or leaked memory on every call. After the fix each raises a
clean Python exception (or behaves correctly).
"""

import ctypes
import gc
import sys
import threading
import weakref

import numpy as np
import pytest

import kornia_rs as K
from kornia_rs.image import Image

# ---------------------------------------------------------------------------
# H1: adjust_brightness read h*w*c bytes from arr.data() with no contiguity check
# ---------------------------------------------------------------------------


def test_adjust_brightness_rejects_broadcast_view():
    # A zero-stride broadcast has 3 bytes of real memory but reports 48M elements.
    a = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (4000, 4000, 3))
    with pytest.raises(ValueError, match="C-contiguous"):
        K.imgproc.adjust_brightness(a, 0.0)


def test_adjust_brightness_rejects_negative_stride_view():
    base = np.zeros((64, 64, 3), np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        K.imgproc.adjust_brightness(base[::-1], 0.0)


def test_adjust_brightness_contiguous_still_works():
    a = np.full((4, 5, 3), 10, np.uint8)
    out = K.imgproc.adjust_brightness(a, 10 / 255)
    assert out.shape == a.shape
    assert (out == 20).all()


# ---------------------------------------------------------------------------
# H2: bundle_adjust / pose_graph_optimize read arrays with no contiguity check
# ---------------------------------------------------------------------------


def test_bundle_adjust_rejects_broadcast_inputs():
    p = 2_000_000
    r = np.broadcast_to(np.eye(3), (p, 3, 3))
    t = np.broadcast_to(np.zeros(3), (p, 3))
    with pytest.raises(ValueError, match="C-contiguous"):
        K.k3d.bundle_adjust(
            r, t, np.zeros((1, 3)), np.zeros((1, 4)), np.eye(3), max_iterations=0
        )


def test_bundle_adjust_rejects_strided_k():
    k = np.eye(6)[::2, ::2]  # (3, 3) view with non-unit strides
    assert not k.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        K.k3d.bundle_adjust(
            np.eye(3)[None].copy(),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((0, 4)),
            k,
            max_iterations=0,
        )


def test_pose_graph_optimize_rejects_broadcast_inputs():
    p = 2_000_000
    r = np.broadcast_to(np.eye(3), (p, 3, 3))
    t = np.broadcast_to(np.zeros(3), (p, 3))
    with pytest.raises(ValueError, match="C-contiguous"):
        K.k3d.pose_graph_optimize(r, t, np.zeros((0, 15)), [0], 0)


def test_pose_graph_optimize_contiguous_still_works():
    r = np.stack([np.eye(3)] * 2)
    t = np.zeros((2, 3))
    edge = np.concatenate([[0.0, 1.0], np.eye(3).ravel(), np.zeros(3), [1.0]])[None]
    r_out, t_out, _, _ = K.k3d.pose_graph_optimize(r, t, edge, [0], 5)
    np.testing.assert_allclose(r_out, r)
    np.testing.assert_allclose(t_out, t)


def test_ransac_homography_rejects_misaligned_points():
    buf = np.zeros(20 * 2 * 8 + 1, np.uint8)
    pts = np.frombuffer(buf.data, dtype=np.float64, count=40, offset=1).reshape(20, 2)
    assert not pts.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        K.k3d.ransac_homography(pts, pts)


def test_match_descriptors_rejects_strided_input():
    d = np.zeros((10, 64), np.uint8)[:, ::2]
    assert d.shape == (10, 32) and not d.flags.c_contiguous
    with pytest.raises(ValueError):
        K.features.match_descriptors(d, d)


# ---------------------------------------------------------------------------
# H3: unchecked size products from Python ints (heap overflow on wrap-around)
# ---------------------------------------------------------------------------


def test_resize_normalize_to_tensor_overflowing_size():
    img = Image(np.full((8, 8, 3), 7, np.uint8))
    # 3 * 1 * 2**62 * 4 wraps to 0 in unchecked usize math -> 1-byte buffer.
    with pytest.raises((OverflowError, ValueError, MemoryError)):
        img.resize_normalize_to_tensor(1, 1 << 62, [0.0] * 3, [1.0] * 3)


def test_resize_normalize_to_tensor_still_works():
    img = Image(np.full((8, 8, 3), 255, np.uint8))
    out = img.resize_normalize_to_tensor(4, 4, [0.0] * 3, [1.0] * 3)
    arr = out.numpy()
    assert arr.shape == (3, 4, 4)
    np.testing.assert_allclose(arr, 1.0, atol=1e-5)


def test_frombytes_overflowing_dims():
    # 2**62 * 4 * 1 * 1 wraps to 0 == len(b"")
    with pytest.raises(OverflowError):
        Image.frombytes(b"", 1 << 62, 4, channels=1)


def test_resize_nearest_overflowing_dims():
    img = Image(np.zeros((4, 4, 1), np.uint8))
    with pytest.raises(OverflowError):
        img.resize(1 << 62, 1 << 62)


def test_pipeline_overflowing_dims():
    a = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(OverflowError):
        K.pipeline.resize_normalize_to_tensor(a, (1 << 62, 4), [0.0] * 3, [1.0] * 3)


# ---------------------------------------------------------------------------
# H4: YUV decoders left the last column/row of an uninitialised output unwritten
# ---------------------------------------------------------------------------


YUV_422 = ["rgb_from_yuyv", "rgb_from_uyvy", "rgb_from_yvyu"]
YUV_420 = ["rgb_from_nv12", "rgb_from_nv21", "rgb_from_i420", "rgb_from_yv12"]


@pytest.mark.parametrize(
    "fn,factor,w,h",
    # 4:2:2 decodes 2-pixel groups per row: only the width must be even.
    [(fn, 2.0, w, h) for fn in YUV_422 for (w, h) in [(101, 50), (5, 5)]]
    # 4:2:0 decodes 2x2 blocks: both dimensions must be even.
    + [(fn, 1.5, w, h) for fn in YUV_420 for (w, h) in [(101, 50), (100, 51), (5, 5)]],
)
def test_yuv_decode_rejects_odd_dims(fn, factor, w, h):
    src = np.full(int(w * h * factor), 128, np.uint8)
    with pytest.raises(ValueError, match="even"):
        getattr(K.imgproc, fn)(src, w, h)


@pytest.mark.parametrize("fn", YUV_422)
def test_yuv422_decode_odd_height_fully_written(fn):
    # Packed 4:2:2 has no vertical subsampling, so an odd height is valid and
    # every row (including the last) must be written.
    w, h = 100, 51
    out = getattr(K.imgproc, fn)(np.full(w * h * 2, 128, np.uint8), w, h)
    assert out.shape == (h, w, 3)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 1


def test_yuv_decode_even_dims_fully_written():
    # Dirty the heap so an uninitialised output would show 0x41 bytes.
    junk = [np.full((50, 100, 3), 0x41, np.uint8) for _ in range(8)]
    del junk
    out = K.imgproc.rgb_from_yuyv(np.full(100 * 50 * 2, 128, np.uint8), 100, 50)
    assert out.shape == (50, 100, 3)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 1


# ---------------------------------------------------------------------------
# M1: out= accepted read-only arrays and arrays aliasing the source
# ---------------------------------------------------------------------------


def _readonly_u8(shape):
    return np.frombuffer(bytes(int(np.prod(shape))), np.uint8).reshape(shape)


@pytest.mark.parametrize(
    "call",
    [
        lambda src, out: K.imgproc.median_blur(src, 3, out=out),
        lambda src, out: K.imgproc.bilateral_filter(src, 5, 50.0, 50.0, out=out),
        lambda src, out: K.imgproc.clahe(src, 2.0, (2, 2), out=out),
        lambda src, out: K.imgproc.warp_affine(
            src, [1, 0, 0, 0, 1, 0], (30, 30), "bilinear", out=out
        ),
        lambda src, out: K.imgproc.warp_perspective(
            src, [1, 0, 0, 0, 1, 0, 0, 0, 1], (30, 30), "bilinear", out=out
        ),
    ],
)
def test_out_param_rejects_readonly(call):
    src = np.full((30, 30, 1), 200, np.uint8)
    out = _readonly_u8((30, 30, 1))
    assert not out.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        call(src, out)
    assert out.max() == 0  # the immutable bytes buffer was not touched


@pytest.mark.parametrize(
    "call",
    [
        lambda src, out: K.imgproc.median_blur(src, 3, out=out),
        lambda src, out: K.imgproc.bilateral_filter(src, 5, 50.0, 50.0, out=out),
        lambda src, out: K.imgproc.clahe(src, 2.0, (2, 2), out=out),
        lambda src, out: K.imgproc.warp_affine(
            src, [1, 0, 0, 0, 1, 0], (30, 30), "bilinear", out=out
        ),
    ],
)
def test_out_param_rejects_aliasing_source(call):
    src = np.full((30, 30, 1), 200, np.uint8)
    with pytest.raises(ValueError, match="share memory"):
        call(src, src)
    # A different view over the same memory is aliasing too.
    with pytest.raises(ValueError, match="share memory"):
        call(src, src.view())


def test_out_param_valid_still_works():
    src = np.full((30, 30, 1), 200, np.uint8)
    out = np.zeros_like(src)
    res = K.imgproc.median_blur(src, 3, out=out)
    assert res is out
    assert (out == 200).all()


# ---------------------------------------------------------------------------
# DLPack helpers: a ctypes-built producer so capsule ownership can be observed
# ---------------------------------------------------------------------------


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER),
]


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", _DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DELETER),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class _CtypesProducer:
    """Minimal DLPack producer whose deleter counts its invocations."""

    def __init__(self, versioned, device_type=1, h=4, w=4, c=3):
        self.buf = (ctypes.c_uint8 * (h * w * c))(*([7] * (h * w * c)))
        self.shape = (ctypes.c_int64 * 3)(h, w, c)
        self.deleted = 0
        self.versioned = versioned
        self.device_type = device_type

        def _deleter(_ptr):
            self.deleted += 1

        self._deleter = _DELETER(_deleter)  # keep the thunk alive
        tensor = _DLTensor(
            data=ctypes.cast(self.buf, ctypes.c_void_p),
            device=_DLDevice(device_type, 0),
            ndim=3,
            dtype=_DLDataType(1, 8, 1),  # kDLUInt, 8 bits
            shape=self.shape,
            strides=None,
            byte_offset=0,
        )
        if versioned:
            self.mt = _DLManagedTensorVersioned(
                version=_DLPackVersion(1, 0),
                manager_ctx=None,
                deleter=self._deleter,
                flags=0,
                dl_tensor=tensor,
            )
        else:
            self.mt = _DLManagedTensor(dl_tensor=tensor, manager_ctx=None, deleter=self._deleter)

    def __dlpack__(self, *, max_version=None, stream=None, dl_device=None, copy=None):
        name = b"dltensor_versioned" if self.versioned else b"dltensor"
        return _PyCapsule_New(ctypes.addressof(self.mt), name, None)

    def __dlpack_device__(self):
        return (self.device_type, 0)


# ---------------------------------------------------------------------------
# M2: from_dlpack consumed the capsule but never called the deleter (leak)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("versioned", [False, True])
def test_from_dlpack_calls_deleter_exactly_once(versioned):
    prod = _CtypesProducer(versioned)
    img = Image.from_dlpack(prod)
    assert prod.deleted == 0  # still in use
    np.testing.assert_array_equal(img.to_numpy(), np.full((4, 4, 3), 7, np.uint8))
    del img
    gc.collect()
    assert prod.deleted == 1  # released exactly once (no leak, no double free)
    gc.collect()
    assert prod.deleted == 1


def test_from_dlpack_releases_numpy_source():
    arr = np.ones((64, 64, 3), np.uint8)
    ref = weakref.ref(arr)
    img = Image.from_dlpack(arr)
    del arr
    gc.collect()
    assert ref() is not None  # the Image keeps the source alive
    del img
    gc.collect()
    assert ref() is None  # ...and releases it afterwards (used to leak forever)


def test_from_dlpack_no_refcount_growth():
    arr = np.ones((16, 16, 3), np.uint8)
    before = sys.getrefcount(arr)
    for _ in range(50):
        img = Image.from_dlpack(arr)
        del img
    gc.collect()
    assert sys.getrefcount(arr) == before


# ---------------------------------------------------------------------------
# M3 (host side): a non-host backing must never be dereferenced on the host
# ---------------------------------------------------------------------------


def test_non_host_dlpack_image_refuses_host_access():
    # kDLOneAPI (14) is not CPU: the imported backing must never be read on
    # the host (the audit found host kernels reading such device pointers).
    prod = _CtypesProducer(versioned=True, device_type=14)
    try:
        img = Image.from_dlpack(prod)
    except (ValueError, RuntimeError, TypeError):
        return  # refusing the import outright is also safe
    for op in (lambda: img.tobytes(), lambda: img.copy(), lambda: img.to_float()):
        with pytest.raises(ValueError):
            op()
    del img
    gc.collect()
    assert prod.deleted == 1


# ---------------------------------------------------------------------------
# M4: sample_depth read masks through raw pointers after dropping references
# ---------------------------------------------------------------------------


def test_sample_depth_concurrent_list_mutation():
    depth = np.full((256, 256), 1000, np.uint16)
    masks = [np.ones((256, 256), np.uint8) for _ in range(32)]
    stop = threading.Event()

    def mutate():
        while not stop.is_set():
            masks.clear()
            masks.extend(np.ones((256, 256), np.uint8) for _ in range(32))

    t = threading.Thread(target=mutate)
    t.start()
    try:
        for _ in range(200):
            res = K.depth.sample_depth(depth, list(masks) or [np.ones((1, 1), np.uint8)])
            assert all(v == (1000, True) for v in res)
            K.depth.sample_depth(depth, masks)
    finally:
        stop.set()
        t.join()


def test_sample_depth_rejects_empty_mask():
    depth = np.full((4, 4), 1, np.uint16)
    with pytest.raises(ValueError):
        K.depth.sample_depth(depth, [np.zeros((0, 4), np.uint8)])


def test_sample_depth_rejects_misaligned_depth():
    buf = np.zeros(4 * 4 * 2 + 1, np.uint8)
    depth = np.frombuffer(buf.data, dtype=np.uint16, count=16, offset=1).reshape(4, 4)
    assert not depth.flags.aligned
    with pytest.raises(ValueError):
        K.depth.sample_depth(depth, [np.ones((4, 4), np.uint8)])


# ---------------------------------------------------------------------------
# L3: legacy __dlpack__ exported read-only memory as a writable tensor
# ---------------------------------------------------------------------------


def test_legacy_dlpack_refuses_readonly_image():
    img = Image.from_buffer(bytes(4 * 4 * 3), 4, 4, 3)
    with pytest.raises(BufferError):
        img.__dlpack__()
    # The versioned protocol carries the read-only flag and is allowed.
    cap = img.__dlpack__(max_version=(1, 0))
    assert cap is not None


def test_legacy_dlpack_writable_image_still_exports():
    img = Image(np.zeros((4, 4, 3), np.uint8))
    assert img.__dlpack__() is not None


# ---------------------------------------------------------------------------
# L4: rle_to_mask allocated h*w (unchecked) before validating the counts
# ---------------------------------------------------------------------------


def test_rle_to_mask_overflowing_shape():
    with pytest.raises(OverflowError):
        K.segmentation.rle_to_mask([0], (1 << 62, 8))


def test_rle_to_mask_counts_exceed_shape():
    with pytest.raises(ValueError):
        K.segmentation.rle_to_mask([10, 10], (2, 2))


def test_rle_to_mask_huge_alloc_is_catchable():
    # 2**30 x 2**30 = 1 EiB: must raise MemoryError, not abort the process.
    with pytest.raises((MemoryError, ValueError)):
        K.segmentation.rle_to_mask([], (1 << 30, 1 << 30))


def test_rle_to_mask_still_works():
    m = K.segmentation.rle_to_mask([1, 2, 1], (2, 2))
    np.testing.assert_array_equal(m, np.array([[0, 1], [1, 0]], np.uint8))


# ---------------------------------------------------------------------------
# L5: Image.crop bounds check `x + width > src_w` could wrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_crop_overflowing_offset(dtype):
    img = Image(np.zeros((8, 8, 3), dtype))
    big = (1 << 64) - 1
    with pytest.raises(ValueError, match="out of bounds"):
        img.crop(big, 0, 2, 2)
    with pytest.raises(ValueError, match="out of bounds"):
        img.crop(0, big - 1, 2, 3)


# ---------------------------------------------------------------------------
# L1: misaligned typed buffers
# ---------------------------------------------------------------------------


def test_from_buffer_rejects_misaligned_float32():
    buf = bytearray(4 * 4 * 1 * 4 + 1)
    with pytest.raises(ValueError, match="aligned"):
        Image.from_buffer(memoryview(buf)[1:], 4, 4, 1, dtype="float32")


def test_from_numpy_copies_misaligned_float32():
    buf = np.zeros(4 * 4 * 4 + 1, np.uint8)
    arr = np.frombuffer(buf.data, dtype=np.float32, count=16, offset=1).reshape(4, 4, 1)
    assert not arr.flags.aligned
    img = Image(arr)
    out = img.to_numpy()
    assert out.dtype == np.float32 and out.shape == (4, 4, 1)
    assert out.ctypes.data % 4 == 0


def test_sobel_rejects_misaligned_float32():
    buf = np.zeros(8 * 8 * 4 + 1, np.uint8)
    arr = np.frombuffer(buf.data, dtype=np.float32, count=64, offset=1).reshape(8, 8, 1)
    with pytest.raises(ValueError, match="aligned"):
        K.imgproc.sobel(arr)


# ---------------------------------------------------------------------------
# L2 (concrete variant): Preprocessor wrote into a buffer Python could shrink
# ---------------------------------------------------------------------------


def test_preprocessor_survives_resized_output_buffer():
    pp = K.pipeline.Preprocessor((64, 64), (32, 32), [0.0] * 3, [1.0] * 3)
    img = np.full((64, 64, 3), 255, np.uint8)
    out = pp(img)
    assert out.shape == (3, 32, 32)
    # Shrink the shared internal buffer in place (realloc to 1 element).
    out.resize((1,), refcheck=False)
    out2 = pp(img)  # used to write 3*32*32 floats into the 1-element buffer
    assert out2.shape == (3, 32, 32)
    np.testing.assert_allclose(out2, 1.0, atol=1e-5)


def test_preprocessor_survives_readonly_output_buffer():
    pp = K.pipeline.Preprocessor((64, 64), (32, 32), [0.0] * 3, [1.0] * 3)
    img = np.zeros((64, 64, 3), np.uint8)
    out = pp(img)
    out.setflags(write=False)
    out2 = pp(np.full((64, 64, 3), 255, np.uint8))
    np.testing.assert_allclose(out, 0.0)  # frozen buffer untouched
    np.testing.assert_allclose(out2, 1.0, atol=1e-5)


def test_from_numpy_strided_misaligned_elements_are_copied_safely():
    """A field view of a packed record array has an aligned base pointer but a
    5-byte stride, so its f32 elements are misaligned. Copying it must not read
    them through a typed view (UB); the values must still round-trip."""
    rec = np.zeros((2, 2, 3), dtype=np.dtype([("b", "<f4"), ("a", "u1")], align=False))
    rec["b"] = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    view = rec["b"]
    assert not view.flags.aligned
    img = Image.from_numpy(view, copy=True)
    np.testing.assert_array_equal(np.asarray(img), np.arange(12, dtype=np.float32).reshape(2, 2, 3))
