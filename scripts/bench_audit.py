#!/usr/bin/env python3
"""Whole-library per-op audit: kornia CPU + GPU vs OpenCV CPU and VPI-CUDA.

The acceptance lens is the per-op 10x goal (every GPU op >= 10x vs both
baselines) with the documented physics exceptions: bandwidth-bound ops
cannot beat a competent GPU kernel (VPI) 10x on the same DRAM, and
sub-millisecond cv2 CPU baselines put 10x inside fixed launch+sync
overhead. Each row reports the honest ratio; the audit write-up
classifies PASS / envelope-capped / overhead-capped.

Method: locked clocks required (asserts), owned CUDA stream,
stream-synchronized rounds (min of rounds, median within round) — the
same methodology as scripts/bench_per_op.py. VPI completion is forced
via rlock_cpu. cv2 CPU baselines drift +/-80% run-to-run; kornia GPU is
stable +/-3% — compare within one sitting only.

Run: python3 scripts/bench_audit.py
"""
import time
import statistics as st

import numpy as np
from kornia_rs import imgproc
from kornia_rs.cuda import Stream
from kornia_rs.image import Image

H, W = 1080, 1920
DH, DW = 720, 1280
AFF = [0.87, -0.5, 400.0, 0.5, 0.87, -100.0]
PSP = [0.9, 0.12, 40.0, -0.08, 1.05, -20.0, 6.0e-5, -4.5e-5, 1.0]

with open("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq") as f:
    cur = f.read().strip()
with open("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq") as f:
    assert cur == f.read().strip(), "GPU clocks not locked — run sudo jetson_clocks"

rng = np.random.default_rng(0)
img3 = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
img1 = rng.integers(0, 256, size=(H, W, 1), dtype=np.uint8)
# Structured gray for edge/label ops (noise input is pathological there).
try:
    import cv2

    smooth = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 4)
    gray = (smooth * 255).astype(np.uint8)[..., None]
    binary = ((smooth > float(np.median(smooth))) * 255).astype(np.uint8)[..., None]
except ImportError:
    cv2 = None
    gray = img1
    binary = ((img1 > 127) * 255).astype(np.uint8)

stream = Stream.new()
d3 = Image.from_numpy(img3).to_cuda(stream)
d1 = Image.from_numpy(img1).to_cuda(stream)
dgray = Image.from_numpy(gray).to_cuda(stream)
dbin = Image.from_numpy(binary).to_cuda(stream)
h3 = Image.from_numpy(img3)
h1 = Image.from_numpy(img1)
img3f = rng.random((H, W, 3), dtype=np.float32)
img1f = rng.random((H, W, 1), dtype=np.float32)
h3f = Image.from_numpy(img3f)
h1f = Image.from_numpy(img1f)
d3f = Image.from_numpy(img3f).to_cuda(stream)
d1f = Image.from_numpy(img1f).to_cuda(stream)
hgray = Image.from_numpy(gray)
hbin = Image.from_numpy(binary)


def gpu_bench(fn, warm=50, iters=100, rounds=3):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm):
            fn()
        stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        stream.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3 / iters)
        warm = 5
    return best


def cpu_bench(fn, warm=10, iters=50, rounds=3):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        best = min(best, st.median(ts))
        warm = 2
    return best


ROWS = []  # (name, kcpu_fn, kgpu_fn, cv2_fn, vpi_key)


def row(name, kcpu=None, kgpu=None, cv=None, vpi_key=None):
    ROWS.append((name, kcpu, kgpu, cv, vpi_key))


# --- geometry ---------------------------------------------------------------
for mode in ["nearest", "bilinear", "bicubic", "lanczos"]:
    cvflag = None
    if cv2:
        cvflag = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[mode]
    row(
        f"resize-{mode}",
        kcpu=lambda m=mode: imgproc.resize(h3, (DH, DW), m),
        kgpu=lambda m=mode: imgproc.resize(d3, (DH, DW), m),
        cv=(lambda f=cvflag: cv2.resize(img3, (DW, DH), interpolation=f)) if cv2 else None,
        vpi_key=f"resize-{mode}",
    )

m_a = np.array(AFF, dtype=np.float64).reshape(2, 3)
m_p = np.array(PSP, dtype=np.float64).reshape(3, 3)
row(
    "warp-affine",
    kcpu=lambda: imgproc.warp_affine(h3, AFF, (H, W), "bilinear"),
    kgpu=lambda: imgproc.warp_affine(d3, AFF, (H, W), "bilinear"),
    cv=(lambda: cv2.warpAffine(img3, m_a, (W, H))) if cv2 else None,
    vpi_key="warp-affine",
)
row(
    "warp-perspective",
    kcpu=lambda: imgproc.warp_perspective(h3, PSP, (H, W), "bilinear"),
    kgpu=lambda: imgproc.warp_perspective(d3, PSP, (H, W), "bilinear"),
    cv=(lambda: cv2.warpPerspective(img3, m_p, (W, H))) if cv2 else None,
    vpi_key="warp-perspective",
)

# --- morphology + filters ---------------------------------------------------
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) if cv2 else None
row(
    "dilate-3x3-c1",
    kcpu=lambda: imgproc.dilate(h1, kernel="box", size=(3, 3)),
    kgpu=lambda: imgproc.dilate(d1, kernel="box", size=(3, 3)),
    cv=(lambda: cv2.dilate(img1, se)) if cv2 else None,
    vpi_key="dilate",
)
row(
    "erode-3x3-c1",
    kcpu=lambda: imgproc.erode(h1, kernel="box", size=(3, 3)),
    kgpu=lambda: imgproc.erode(d1, kernel="box", size=(3, 3)),
    cv=(lambda: cv2.erode(img1, se)) if cv2 else None,
    vpi_key="erode",
)
row(
    "gaussian-5x5-c3",
    kcpu=lambda: imgproc.gaussian_blur(h3, (5, 5), (1.5, 1.5)),
    kgpu=lambda: imgproc.gaussian_blur(d3, (5, 5), (1.5, 1.5)),
    cv=(lambda: cv2.GaussianBlur(img3, (5, 5), 1.5)) if cv2 else None,
)
row(
    "gaussian-5x5-c1",
    kgpu=lambda: imgproc.gaussian_blur(d1, (5, 5), (1.5, 1.5)),
    cv=(lambda: cv2.GaussianBlur(img1, (5, 5), 1.5)) if cv2 else None,
    vpi_key="gaussian",
)
row(
    "box-5x5-c3",
    kcpu=lambda: imgproc.box_blur(h3, (5, 5)),
    kgpu=lambda: imgproc.box_blur(d3, (5, 5)),
    cv=(lambda: cv2.blur(img3, (5, 5))) if cv2 else None,
)
row(
    "box-5x5-c1",
    kgpu=lambda: imgproc.box_blur(d1, (5, 5)),
    cv=(lambda: cv2.blur(img1, (5, 5))) if cv2 else None,
    vpi_key="box",
)
row(
    "sobel-3x3-f32",
    kcpu=lambda: imgproc.sobel(h1f, 3),
    kgpu=lambda: imgproc.sobel(d1f, 3),
    cv=(lambda: cv2.Sobel(img1f, cv2.CV_32F, 1, 0, ksize=3)) if cv2 else None,
)
row(
    "median-3x3-c1",
    kcpu=lambda: imgproc.median_blur(h1, 3),
    kgpu=lambda: imgproc.median_blur(d1, 3),
    cv=(lambda: cv2.medianBlur(img1, 3)) if cv2 else None,
    vpi_key="median3",
)
row(
    "median-5x5-c1",
    kcpu=lambda: imgproc.median_blur(h1, 5),
    kgpu=lambda: imgproc.median_blur(d1, 5),
    cv=(lambda: cv2.medianBlur(img1, 5)) if cv2 else None,
    vpi_key="median5",
)
row(
    "bilateral-d5-c1",
    kcpu=lambda: imgproc.bilateral_filter(h1, 5, 50.0, 50.0),
    kgpu=lambda: imgproc.bilateral_filter(d1, 5, 50.0, 50.0),
    cv=(lambda: cv2.bilateralFilter(img1, 5, 50.0, 50.0)) if cv2 else None,
    vpi_key="bilateral",
)

# --- histogram / enhancement ------------------------------------------------
row(
    "histogram-c1",
    kcpu=lambda: imgproc.compute_histogram(img1, 256),
    kgpu=lambda: imgproc.compute_histogram(d1, 256),
    cv=(lambda: cv2.calcHist([img1], [0], None, [256], [0, 256])) if cv2 else None,
    vpi_key="histogram",
)
row(
    "equalize-hist-c1",
    kcpu=lambda: imgproc.equalize_hist(h1),
    kgpu=lambda: imgproc.equalize_hist(d1),
    cv=(lambda: cv2.equalizeHist(img1)) if cv2 else None,
    vpi_key="eqhist",
)
clahe_cv = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8)) if cv2 else None
row(
    "clahe-c1",
    kcpu=lambda: imgproc.clahe(h1),
    kgpu=lambda: imgproc.clahe(d1),
    cv=(lambda: clahe_cv.apply(img1)) if cv2 else None,
)

# --- edges / labeling -------------------------------------------------------
row(
    "canny",
    kcpu=lambda: imgproc.canny(hgray, 50.0, 150.0),
    kgpu=lambda: imgproc.canny(dgray, 50.0, 150.0),
    cv=(lambda: cv2.Canny(gray, 50.0, 150.0)) if cv2 else None,
    vpi_key="canny",
)
row(
    "connected-components",
    kcpu=lambda: imgproc.connected_components(hbin, connectivity=8),
    kgpu=lambda: imgproc.connected_components(dbin, connectivity=8),
    cv=(lambda: cv2.connectedComponents(binary[:, :, 0], connectivity=8)) if cv2 else None,
)

# --- color ------------------------------------------------------------------
row(
    "gray_from_rgb-u8",
    kcpu=lambda: imgproc.gray_from_rgb(h3),
    kgpu=lambda: imgproc.gray_from_rgb(d3),
    cv=(lambda: cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)) if cv2 else None,
    vpi_key="convert-gray",
)
for kop, cvcode in [
    ("hsv_from_rgb", "COLOR_RGB2HSV"),
    ("lab_from_rgb", "COLOR_RGB2Lab"),
    ("ycbcr_from_rgb", "COLOR_RGB2YCrCb"),
]:
    fn = getattr(imgproc, kop)
    code = getattr(cv2, cvcode) if cv2 else None
    row(
        f"{kop}-f32",
        kcpu=lambda f=fn: f(h3f),
        kgpu=lambda f=fn: f(d3f),
        cv=(lambda c=code: cv2.cvtColor(img3f, c)) if cv2 else None,
    )

# --- VPI baselines ----------------------------------------------------------
VPI = {}
try:
    import vpi

    v3 = vpi.asimage(img3, vpi.Format.RGB8)
    v1 = vpi.asimage(img1[:, :, 0])
    vgray = vpi.asimage(gray[:, :, 0])

    def vrun(name, make):
        def f():
            with vpi.Backend.CUDA:
                o = make()
            with o.rlock_cpu():
                pass

        try:
            VPI[name] = cpu_bench(f, warm=15, rounds=2)
        except Exception as e:  # noqa: BLE001 - per-op isolation
            print(f"vpi {name}: {e}")

    for mode, interp in [
        ("resize-nearest", vpi.Interp.NEAREST),
        ("resize-bilinear", vpi.Interp.LINEAR),
        ("resize-bicubic", vpi.Interp.CATMULL_ROM),
    ]:
        vrun(mode, lambda i=interp: v3.rescale((DW, DH), interp=i))
    m33 = np.array(
        [[AFF[0], AFF[1], AFF[2]], [AFF[3], AFF[4], AFF[5]], [0, 0, 1]], dtype=np.float32
    )
    vrun("warp-affine", lambda: v3.perspwarp(m33))
    vrun("warp-perspective", lambda: v3.perspwarp(np.array(PSP, np.float32).reshape(3, 3)))
    se_v = np.ones((3, 3), dtype=np.uint8)
    vrun("dilate", lambda: v1.dilate(se_v))
    vrun("erode", lambda: v1.erode(se_v))
    vrun("gaussian", lambda: v1.gaussian_filter(5, 1.5))
    vrun("box", lambda: v1.box_filter(5))
    vrun("median3", lambda: v1.median_filter(np.ones((3, 3), np.uint8)))
    vrun("median5", lambda: v1.median_filter(np.ones((5, 5), np.uint8)))
    vrun("bilateral", lambda: v1.bilateral_filter(5, 50.0, 50.0))
    vrun("histogram", lambda: v1.histogram(bins=256))
    vrun("eqhist", lambda: v1.eqhist())
    vrun("canny", lambda: vgray.canny(thresh_strong=150.0, thresh_weak=50.0))
    vrun("convert-gray", lambda: v3.convert(vpi.Format.Y8_ER))
except Exception as e:  # noqa: BLE001
    print(f"vpi unavailable/partial: {e}")

# --- run + print ------------------------------------------------------------
print(f"\nPer-op audit, 1080p u8 (ms; x = baseline / kornia-gpu)")
print(
    f"{'op':22s} {'k-cpu':>8s} {'k-gpu':>8s} {'cv2':>8s} {'vpi':>8s}"
    f" {'xcv-gpu':>8s} {'xvpi':>8s} {'xcv-cpu':>8s}"
)
for name, kcpu, kgpu, cv, vpi_key in ROWS:
    def cell(fn, bench):
        if fn is None:
            return float("nan")
        try:
            return bench(fn)
        except Exception as e:  # noqa: BLE001 - per-op isolation
            print(f"{name}: {e}")
            return float("nan")

    tc = cell(kcpu, cpu_bench)
    tg = cell(kgpu, gpu_bench)
    tv = VPI.get(vpi_key, float("nan"))
    tcv = cell(cv, cpu_bench)
    xg = tcv / tg
    xv = tv / tg
    xc = tcv / tc
    print(
        f"{name:22s} {tc:8.3f} {tg:8.3f} {tcv:8.3f} {tv:8.3f}"
        f" {xg:8.2f} {xv:8.2f} {xc:8.2f}"
    )
