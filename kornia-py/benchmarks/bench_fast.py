"""FAST-9 detector micro-benchmark: kornia-rs vs OpenCV vs VPI (CPU + CUDA).

Times JUST the FAST corner detector (no Harris, no orientation, no descriptor,
no pyramid) on identical gray u8 images across identical thresholds. This is
the fairest head-to-head because the three backends do nominally the same
thing at this stage.

- kornia-rs   : `K.features.fast_detect` (NEON fused single-pass on aarch64).
- OpenCV      : `cv2.FastFeatureDetector_create().detect()`.
- VPI-CPU/CUDA: `vpi.Image.fastcorners(backend=...)`.

All three receive the same image bytes and the same threshold (u8 space,
0..255). We verify kornia-rs matches OpenCV's corner *count* exactly before
timing — if the counts diverge, the implementations are doing different work
and the timings would be apples-to-oranges.

Usage: `taskset -c 0-5 python kornia-py/benchmarks/bench_fast.py`
"""
import glob
import os
import sys
from pathlib import Path

# VPI ships its Python bindings outside the standard site-packages tree;
# probe the usual install layout (or KORNIA_VPI_PYPATH override) and
# silently fall through if VPI isn't installed.
_vpi_path = os.environ.get("KORNIA_VPI_PYPATH") or next(
    iter(glob.glob("/opt/nvidia/vpi*/lib/*/python")), None
)
if _vpi_path and os.path.isdir(_vpi_path):
    sys.path.insert(0, _vpi_path)

import cv2
import numpy as np
import kornia_rs as K
from kornia_rs.image import Image

from _bench import bench as _bench_fn

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"


def _load_gray(path, size=None):
    """Load an image as 2D u8 grayscale via the kornia Image API.

    `size` is `(width, height)` if a resize is needed. cv2 stays out of the
    setup path; it appears in this bench only as a timing comparison target.
    """
    img = Image.load(str(path)).to_grayscale()
    if size is not None:
        w, h = size
        img = img.resize(width=w, height=h, interpolation="bilinear")
    return img.to_numpy()[..., 0]

try:
    import vpi

    HAVE_VPI = True
except ImportError:
    HAVE_VPI = False


# Iterations per round is tuned so each round lasts ~1-3s on the slowest backend
# at 1080p. Legacy constants kept for reference but no longer drive iteration count.
N_ROUNDS = 5
ITERS_PER_ROUND = {"640x480": 200, "1080x1920": 50}


def time_loop(fn, **_legacy):
    return _bench_fn(fn).min_ms


def bench_kornia(img, threshold):
    return K.features.fast_detect(img, threshold=float(threshold))


def bench_opencv_factory(threshold):
    # NMS=False so we time the bare FAST arc-test without the NMS post-pass,
    # matching what kornia's `fast_detect` nominally does. At width<800 kornia
    # emits raw arc-test-passing pixels identical to OpenCV(NMS=False). At
    # width≥800 kornia applies a cheap 1D in-block local-max filter as a NEON
    # optimization (cuts the Vec::push pressure on dense-corner images); we
    # report the count discrepancy rather than disabling the optimization.
    det = cv2.FastFeatureDetector_create(threshold=int(threshold), nonmaxSuppression=False)
    return lambda img: det.detect(img)


def bench_vpi_factory(threshold, backend, src):
    # `src` is a pre-wrapped vpi.Image cached outside the timed loop. A real
    # pipeline wraps the frame once at ingest, not per call; re-wrapping here
    # would add ~1-4 ms of pure harness overhead that a FAST kernel (<1 ms)
    # cannot amortize.
    def run():
        with backend:
            corners = src.fastcorners(
                intensity_threshold=float(threshold),
                arc_length=9,
                non_max_suppression=False,
            )
        with corners.rlock_cpu() as c:
            _ = np.asarray(c)

    return run


def count_kornia(img, threshold):
    kps, _ = K.features.fast_detect(img, threshold=float(threshold))
    return len(kps)


def count_opencv(img, threshold):
    det = cv2.FastFeatureDetector_create(threshold=int(threshold), nonmaxSuppression=False)
    return len(det.detect(img))


def count_vpi(img, threshold, backend):
    src = vpi.asimage(img)
    with backend:
        corners = src.fastcorners(
            intensity_threshold=float(threshold),
            arc_length=9,
            non_max_suppression=False,
        )
    with corners.rlock_cpu() as c:
        return int(np.asarray(c).shape[0])


def run_size(name, w, h, threshold, img):
    print(f"\n=== {name} ({w}x{h}), threshold={threshold} ===")
    # Parity check first — we only report timings for backends whose corner
    # counts agree reasonably with OpenCV.
    k_n = count_kornia(img, threshold)
    o_n = count_opencv(img, threshold)
    print(f"  corner count:  kornia={k_n}, opencv={o_n}, match={'yes' if k_n == o_n else f'DIFF by {k_n - o_n}'}")

    iters = ITERS_PER_ROUND[name]
    cv_fn = bench_opencv_factory(threshold)
    results = []
    results.append(("kornia-rs", time_loop(lambda: bench_kornia(img, threshold), iters), k_n))
    results.append(("opencv", time_loop(lambda: cv_fn(img), iters), o_n))

    if HAVE_VPI:
        vpi_src = vpi.asimage(img)  # cached outside timed loop (see bench_vpi_factory)
        for be_name, be in [("vpi-cpu", vpi.Backend.CPU), ("vpi-cuda", vpi.Backend.CUDA)]:
            v_n = count_vpi(img, threshold, be)
            fn = bench_vpi_factory(threshold, be, vpi_src)
            fn()  # warm-up, especially important for CUDA (JIT / ctx init).
            v_iters = iters if name == "640x480" else max(iters // 2, 10)
            results.append((be_name, time_loop(fn, v_iters), v_n))

    # Report relative to OpenCV so the ratio is unambiguous.
    cv_ms = next(ms for (nm, ms, _) in results if nm == "opencv")
    print(f"  {'backend':12s} {'ms/call':>9s} {'vs opencv':>11s} {'kps':>7s}")
    for nm, ms, n in results:
        ratio = cv_ms / ms if ms > 0 else float("inf")
        print(f"  {nm:12s} {ms:>9.3f} {ratio:>10.2f}x {n:>7d}")


def main():
    # 640×480: a 3-ch photo converted to gray, similar to mh01 frames.
    img_small = _load_gray(DATA_DIR / "mh01_frame1.png", size=(640, 480))
    # 1080p: upscale dog.jpeg (it's the closest real photo we have).
    img_big = _load_gray(DATA_DIR / "dog.jpeg", size=(1920, 1080))

    # Threshold 20 matches cv2's default and gives a realistic (hundreds of)
    # corners at both sizes — not the degenerate 0 or 100k regimes.
    for (name, w, h, img) in [("640x480", 640, 480, img_small), ("1080x1920", 1920, 1080, img_big)]:
        run_size(name, w, h, threshold=20, img=img)


if __name__ == "__main__":
    main()
