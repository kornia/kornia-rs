"""OpenCV-only two-view pose benchmark: sweep every RANSAC/USAC method.

Loads the EuRoC MH_01 frame pair (via the kornia Image API for consistent
decode bytes across cv2 versions), runs OpenCV ORB → BFMatcher(Lowe+
cross-check) → `findEssentialMat(method=...)` → `recoverPose`, and scores
each backend on wall-clock time + pose error against the Vicon-derived
ground truth.

Used to compare OpenCV 4.13 (latest stable on PyPI) against OpenCV 5.0.0-pre
(via `opencv-python-rolling==5.0.0.20221015`, built from the opencv 5.x
branch). Loading via kornia (rather than `cv2.imread`) keeps the input
pixels identical across cv2 versions, so the comparison only reflects
the algorithm differences.

Usage:
    python kornia-py/benchmarks/bench_opencv_two_view.py
    N_ITERS=200 python kornia-py/benchmarks/bench_opencv_two_view.py

Switch OpenCV versions by running under different venvs (e.g. one with
`opencv-python==4.13` and one with `opencv-python-rolling==5.0.0.*`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from kornia_rs.image import Image

from _bench import bench as _bench_fn

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data"


def _load_gray(path):
    """Load a 2D u8 grayscale image via kornia. cv2 stays purely as the
    thing being compared (across versions, across RANSAC methods)."""
    return Image.load(str(path)).to_grayscale().to_numpy()[..., 0]

# EuRoC MH_01_easy cam0.
K_MH01 = np.array(
    [[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
GT_ROT_DEG = 2.7021
GT_T_DIR = np.array([0.2422, -0.2330, 0.9418], dtype=np.float64)
GT_T_DIR /= np.linalg.norm(GT_T_DIR)

N_ITERS = int(os.environ.get("N_ITERS", "50"))
N_WARMUP = int(os.environ.get("N_WARMUP", "5"))

METHODS = [
    ("RANSAC (legacy 5-pt)",     cv2.RANSAC),
    ("LMEDS",                    cv2.LMEDS),
    ("USAC_DEFAULT",             cv2.USAC_DEFAULT),
    ("USAC_FAST",                cv2.USAC_FAST),
    ("USAC_ACCURATE",            cv2.USAC_ACCURATE),
    ("USAC_MAGSAC",              cv2.USAC_MAGSAC),
    ("USAC_PROSAC",              cv2.USAC_PROSAC),
    ("USAC_PARALLEL",            cv2.USAC_PARALLEL),
]


def rotation_angle_deg(r: np.ndarray) -> float:
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def t_dir_err_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = t_est / np.linalg.norm(t_est)
    return float(np.degrees(np.arccos(np.clip(abs(float(t_est @ t_gt)), 0.0, 1.0))))


def median_ms(fn, n: int = N_ITERS, warmup: int = N_WARMUP) -> float:
    """Backwards-compat shim around benchmarks/_bench.py — reports min ms.

    The shared helper auto-tunes iteration count to a 1s budget; the legacy
    n / warmup args are accepted but ignored. min_ms is the right number for
    sub-millisecond ops; mean is biased high by GC/scheduler noise.
    """
    r = _bench_fn(fn, target_seconds=1.0, min_iters=100)
    return r.min_ms


@dataclass
class Result:
    name: str
    pose_ms: float
    rot_err_deg: float
    t_err_deg: float
    n_inliers: int


def orb_match(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared ORB+Lowe+cross-check match so the RANSAC sweep sees identical input."""
    orb = cv2.ORB_create(
        nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
        firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31, fastThreshold=20,
    )
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [a for pair in knn if len(pair) == 2 for a, b in [pair] if a.distance < 0.75 * b.distance]
    rev = bf.knnMatch(d2, d1, k=1)
    rmap = {pair[0].queryIdx: pair[0].trainIdx for pair in rev if pair}
    good = [m for m in good if rmap.get(m.trainIdx) == m.queryIdx]

    pts1 = np.float64([k1[m.queryIdx].pt for m in good])
    pts2 = np.float64([k2[m.trainIdx].pt for m in good])
    return pts1, pts2


def bench_method(pts1: np.ndarray, pts2: np.ndarray, method: int, name: str) -> Result:
    # Single quality run.
    E, mask = cv2.findEssentialMat(pts1, pts2, K_MH01, method=method,
                                   prob=0.9999, threshold=1.0)
    n_in, R, t, _ = cv2.recoverPose(E, pts1, pts2, K_MH01, mask=mask)

    def pose():
        E_, m = cv2.findEssentialMat(pts1, pts2, K_MH01, method=method,
                                     prob=0.9999, threshold=1.0)
        cv2.recoverPose(E_, pts1, pts2, K_MH01, mask=m)

    return Result(
        name=name,
        pose_ms=median_ms(pose),
        rot_err_deg=abs(rotation_angle_deg(R) - GT_ROT_DEG),
        t_err_deg=t_dir_err_deg(t.flatten(), GT_T_DIR),
        n_inliers=int(n_in),
    )


def print_table(results: list[Result], version: str) -> None:
    hdr = f"{'method':<24} {'pose(ms)':>10} {'rot_err°':>10} {'t_err°':>9} {'inliers':>8}"
    print(f"\nOpenCV {version}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<24} {r.pose_ms:>10.3f} {r.rot_err_deg:>10.3f} "
              f"{r.t_err_deg:>9.3f} {r.n_inliers:>8d}")


def main() -> None:
    img1 = _load_gray(DATA_DIR / "mh01_frame1.png")
    img2 = _load_gray(DATA_DIR / "mh01_frame2.png")

    pts1, pts2 = orb_match(img1, img2)
    print(f"OpenCV {cv2.__version__}  ·  {len(pts1)} matches  ·  "
          f"median over {N_ITERS} iter ({N_WARMUP} warmup)")
    print(f"GT: rot={GT_ROT_DEG}°  t_dir={GT_T_DIR.tolist()}")

    results = [bench_method(pts1, pts2, method, name) for name, method in METHODS]
    print_table(results, cv2.__version__)


if __name__ == "__main__":
    main()
