"""Benchmark kornia-rs ORB vs OpenCV ORB vs NVIDIA VPI ORB on natural imagery.

Timing target: 640x480 and 1920x1080 (set via ORB_BENCH_SIZES). Primary
image is a natural SLAM frame (set via ORB_BENCH_IMG, defaults to dog.jpeg
which has rich outdoor-like texture and matches the Rust-side quality test
`test_orb_compare_with_opencv`).

Reports:
  * kornia-rs vs OpenCV vs VPI(CPU) vs VPI(CUDA) wall-time (ms per call).
  * Keypoint overlap @ 1/3/5 px — quality vs OpenCV. The Rust test floor is
    15% at 3 px; natural images typically land in the 40–75% band.

Tip: drop a KITTI odometry frame into tests/data/ and point ORB_BENCH_IMG
at it for true outdoor imagery.
"""
import json
import os
import sys
import time
import numpy as np
import kornia_rs as K
import cv2

# VPI ships its Python bindings outside the standard site-packages tree.
sys.path.insert(0, "/opt/nvidia/vpi3/lib/aarch64-linux-gnu/python")
try:
    import vpi
    HAVE_VPI = True
except ImportError:
    HAVE_VPI = False

N_ITERS = int(os.environ.get("ORB_BENCH_N", "200"))
N_WARMUP = int(os.environ.get("ORB_BENCH_WARMUP", "10"))
TEST_IMG = os.environ.get(
    "ORB_BENCH_IMG",
    "/home/nvidia/kornia-rs/tests/data/dog.jpeg",
)
QUALITY_IMGS = [
    "/home/nvidia/kornia-rs/tests/data/dog.jpeg",
    "/home/nvidia/kornia-rs/tests/data/mh01_frame1.png",
    "/home/nvidia/kornia-rs/tests/data/mh01_frame2.png",
]

# VPI ORB parameters — matches the sample defaults scaled up for 500 features.
VPI_PARAMS = dict(intensity_threshold=20, max_features_per_level=500, max_pyr_levels=3)


def bench(name, fn, n=N_ITERS, warmup=N_WARMUP):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:40s} {elapsed:8.3f} ms")
    return elapsed


def _xy_from_vpi(corners_array):
    """VPI returns (x, y, level, reserved); scale xy back to base-image coords."""
    if corners_array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    scale = np.power(2.0, corners_array[:, 2]).astype(np.float32)
    return np.stack([corners_array[:, 0] * scale, corners_array[:, 1] * scale], axis=1)


def vpi_orb_run(gray, backend, src=None):
    """Call VPI ORB end-to-end (pyramid + orb) and return base-image xy coords.

    If `src` is None (default) the numpy array is wrapped on every call, which
    adds ~1 ms/call of pure overhead that a real pipeline would not pay (the
    image is wrapped once at ingest, not per frame). Pass a cached
    `vpi.asimage(gray)` to time the kernel path fairly — the bench uses the
    cached version for timing and the uncached path only for one-shot quality
    checks."""
    if src is None:
        src = vpi.asimage(gray)
    with backend:
        pyr = src.gaussian_pyramid(VPI_PARAMS["max_pyr_levels"])
        corners, descriptors = pyr.orb(**VPI_PARAMS)
    with corners.rlock_cpu() as c:
        return _xy_from_vpi(np.asarray(c, dtype=np.float32))


def overlap_stats(kps_a, kps_b):
    """Overlap of kps_a against reference kps_b at 1/3/5 px."""
    a_xy = np.asarray(kps_a, dtype=np.float32).reshape(-1, 2)
    b_xy = np.asarray(kps_b, dtype=np.float32).reshape(-1, 2)
    if len(a_xy) == 0 or len(b_xy) == 0:
        return {1: 0.0, 3: 0.0, 5: 0.0}
    d = np.linalg.norm(a_xy[:, None, :] - b_xy[None, :, :], axis=2)
    min_d = d.min(axis=1)
    return {t: float((min_d <= t).mean()) for t in (1, 3, 5)}


def cv_xy(kps_cv):
    return np.asarray([kp.pt for kp in kps_cv], dtype=np.float32) if kps_cv else np.zeros((0, 2), dtype=np.float32)


def quality_report():
    """Cross-image overlap vs OpenCV — same config as the timing bench."""
    print(f"\n{'='*72}\nQuality check across natural images (overlap vs OpenCV)\n{'='*72}")
    orb_cv = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=31, firstLevel=0, WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE,
                             patchSize=31, fastThreshold=20)
    hdr = f"  {'image':28s} {'n_k':>4s} {'n_cv':>4s} {'n_vpi':>5s}"
    hdr += f"  {'k@3':>5s} {'vpi@3':>6s}"
    print(hdr)
    for path in QUALITY_IMGS:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feat_k = K.features.orb_detect_and_compute(img)
        kps_cv, _ = orb_cv.detectAndCompute(img, None)
        k_xy = np.asarray(feat_k.keypoints_xy, dtype=np.float32).reshape(-1, 2)
        cv_pts = cv_xy(kps_cv)
        ok = overlap_stats(k_xy, cv_pts)

        if HAVE_VPI:
            vpi_xy = vpi_orb_run(img, vpi.Backend.CPU)
            ov = overlap_stats(vpi_xy, cv_pts)
            n_vpi = len(vpi_xy)
            vpi_3 = f"{ov[3]*100:5.1f}%"
        else:
            n_vpi = 0
            vpi_3 = "  n/a"

        name = os.path.basename(path)
        print(f"  {name:28s} {len(k_xy):>4d} {len(kps_cv):>4d} {n_vpi:>5d}"
              f"  {ok[3]*100:4.1f}% {vpi_3:>6s}")


def run_benchmarks():
    results = {}

    for label, (h, w) in [("640x480", (480, 640)), ("1920x1080", (1080, 1920))]:
        print(f"\n{'='*72}")
        print(f"Image size: {label} (HxW={h}x{w}, gray uint8)")
        print(f"{'='*72}")

        src = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
        assert src is not None, f"missing test image: {TEST_IMG}"
        gray = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

        orb_cv = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8,
                                 edgeThreshold=31, firstLevel=0, WTA_K=2,
                                 scoreType=cv2.ORB_HARRIS_SCORE,
                                 patchSize=31, fastThreshold=20)

        print(f"\n--- ORB detect + compute  (input: {os.path.basename(TEST_IMG)}) ---")
        feat_k = K.features.orb_detect_and_compute(gray)
        kps_cv, _ = orb_cv.detectAndCompute(gray, None)
        k_xy = np.asarray(feat_k.keypoints_xy, dtype=np.float32).reshape(-1, 2)
        cv_pts = cv_xy(kps_cv)
        o_k = overlap_stats(k_xy, cv_pts)
        print(f"  kornia={len(k_xy)} kps, opencv={len(kps_cv)} kps")
        print(f"  kornia overlap vs OpenCV: 1px={o_k[1]*100:4.1f}%  "
              f"3px={o_k[3]*100:4.1f}%  5px={o_k[5]*100:4.1f}%")

        if HAVE_VPI:
            vpi_cpu_xy = vpi_orb_run(gray, vpi.Backend.CPU)
            vpi_cuda_xy = vpi_orb_run(gray, vpi.Backend.CUDA)
            o_vpi_cpu = overlap_stats(vpi_cpu_xy, cv_pts)
            o_vpi_cuda = overlap_stats(vpi_cuda_xy, cv_pts)
            print(f"  vpi-cpu={len(vpi_cpu_xy)} kps  overlap vs OpenCV: "
                  f"1px={o_vpi_cpu[1]*100:4.1f}%  3px={o_vpi_cpu[3]*100:4.1f}%  "
                  f"5px={o_vpi_cpu[5]*100:4.1f}%")
            print(f"  vpi-cuda={len(vpi_cuda_xy)} kps  overlap vs OpenCV: "
                  f"1px={o_vpi_cuda[1]*100:4.1f}%  3px={o_vpi_cuda[3]*100:4.1f}%  "
                  f"5px={o_vpi_cuda[5]*100:4.1f}%")

        row = {
            "kornia":  bench("kornia-rs",  lambda: K.features.orb_detect_and_compute(gray)),
            "opencv":  bench("opencv",     lambda: orb_cv.detectAndCompute(gray, None)),
        }
        if HAVE_VPI:
            # Cache the VPI image wrapper outside the hot loop — a real app wraps once
            # and reuses; the per-call `vpi.asimage(gray)` is pure bench-harness overhead.
            vpi_src = vpi.asimage(gray)
            row["vpi_cpu"] = bench(
                "vpi (CPU, cached src)",
                lambda: vpi_orb_run(gray, vpi.Backend.CPU, src=vpi_src),
            )
            row["vpi_cuda"] = bench(
                "vpi (CUDA, cached src)",
                lambda: vpi_orb_run(gray, vpi.Backend.CUDA, src=vpi_src),
            )

        results[label] = row

    return results


if __name__ == "__main__":
    if not HAVE_VPI:
        print("[warn] VPI not importable — VPI comparison will be skipped.")
    quality_report()
    results = run_benchmarks()
    print("\n" + json.dumps(results, indent=2))
