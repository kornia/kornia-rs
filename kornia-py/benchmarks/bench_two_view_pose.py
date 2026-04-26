"""SLAM bootstrap baseline: kornia-rs vs OpenCV on EuRoC MH_01 frame pair.

One canonical image pair with ground-truth relative pose. Each backend runs
the full monocular-SLAM initialization pipeline:

    ORB detect+describe → Hamming match → F/E RANSAC → decompose → (R, t)

and is scored on BOTH wall-clock time AND pose accuracy against ground truth.
This is the "is kornia-rs the fastest *and* most accurate library" test.

Pair:
    tests/data/mh01_frame1.png   (1403636633263555584)
    tests/data/mh01_frame2.png   (1403636634263555584)   — 20 frames apart

Ground truth (derived from EuRoC MH_01_easy state_groundtruth_estimate0
via kornia-py/scripts/derive_mh01_gt.py — verified against the 200 Hz Vicon
table, Δ=0 ns at both frames):
    Rotation:     2.7021°
    Translation:  direction [0.2422, -0.2330, 0.9418]  (scale 0.6585 m,
                  unobservable from monocular geometry)

Intrinsics: EuRoC MH_01_easy cam0
    fx=458.654, fy=457.296, cx=367.215, cy=248.375

Usage:
    python kornia-py/benchmarks/bench_two_view_pose.py           # 50 timing iters, seed 42
    N_ITERS=200 python kornia-py/benchmarks/bench_two_view_pose.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import kornia_rs as K
from kornia_rs.image import Image

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"


def _load_gray(path):
    """Load a 2D u8 grayscale image via the kornia Image API. cv2 stays as
    a comparison-only target (ORB, BFMatcher, findEssentialMat)."""
    return Image.load(str(path)).to_grayscale().to_numpy()[..., 0]

K_MH01 = np.array(
    [[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
GT_ROT_DEG = 2.7021
GT_T_DIR = np.array([0.2422, -0.2330, 0.9418], dtype=np.float64)
GT_T_DIR /= np.linalg.norm(GT_T_DIR)

N_ITERS = int(os.environ.get("N_ITERS", "50"))
N_WARMUP = int(os.environ.get("N_WARMUP", "5"))
SEED = 42


def rotation_angle_deg(r: np.ndarray) -> float:
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def t_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = t_est / np.linalg.norm(t_est)
    return float(np.degrees(np.arccos(np.clip(abs(float(t_est @ t_gt)), 0.0, 1.0))))


@dataclass
class Result:
    name: str
    t_detect_ms: float
    t_match_ms: float
    t_pose_ms: float
    rot_err_deg: float
    t_err_deg: float
    n_matches: int
    n_inliers: int

    @property
    def t_total_ms(self) -> float:
        return self.t_detect_ms + self.t_match_ms + self.t_pose_ms


def median_ms(fn, n=N_ITERS, warmup=N_WARMUP) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def bench_kornia(
    img1: np.ndarray,
    img2: np.ndarray,
    use_5pt: bool = False,
    label: str = "kornia-rs",
) -> Result:
    # Single run for quality + inlier counts.
    f1 = K.features.orb_detect_and_compute(img1)
    f2 = K.features.orb_detect_and_compute(img2)

    m = K.features.match_descriptors(
        f1.descriptors, f2.descriptors, max_ratio=0.75, cross_check=True
    )

    pts1 = np.ascontiguousarray(f1.keypoints_xy[m[:, 0]], dtype=np.float64)
    pts2 = np.ascontiguousarray(f2.keypoints_xy[m[:, 1]], dtype=np.float64)

    pose = K.k3d.two_view_estimate(
        pts1, pts2, K_MH01,
        seed=SEED, min_parallax_deg=0.5,
        use_5pt_essential=use_5pt,
    )

    # Timing medians for stable numbers.
    t_det = median_ms(lambda: (
        K.features.orb_detect_and_compute(img1),
        K.features.orb_detect_and_compute(img2),
    ))
    t_mat = median_ms(lambda: K.features.match_descriptors(
        f1.descriptors, f2.descriptors, max_ratio=0.75, cross_check=True
    ))
    t_pose = median_ms(lambda: K.k3d.two_view_estimate(
        pts1, pts2, K_MH01,
        seed=SEED, min_parallax_deg=0.5,
        use_5pt_essential=use_5pt,
    ))

    return Result(
        name=label,
        t_detect_ms=t_det,
        t_match_ms=t_mat,
        t_pose_ms=t_pose,
        rot_err_deg=abs(rotation_angle_deg(pose.rotation) - GT_ROT_DEG),
        t_err_deg=t_direction_error_deg(pose.translation, GT_T_DIR),
        n_matches=len(pts1),
        n_inliers=int(pose.inlier_count),
    )


def bench_opencv(
    img1: np.ndarray,
    img2: np.ndarray,
    method: int = cv2.RANSAC,
    label: str = "opencv",
) -> Result:
    """Run the ORB→BF→E→(R,t) pipeline using a chosen RANSAC backend.

    `method` can be cv2.RANSAC (legacy 5-point), cv2.USAC_ACCURATE (MSAC + LO),
    or cv2.USAC_MAGSAC (σ-marginalized MAGSAC++). All three ship in OpenCV 4.5+
    via the shared USAC implementation — same algorithms OpenCV 5 ships.
    """
    orb = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )

    # Single run for quality.
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    # Lowe ratio + symmetric cross-check, same semantics as the kornia side.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(d1, d2, k=2)
    good = []
    for pair in matches_knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)
    # Symmetric check: the match must also be 1→2 best from 2→1 side.
    matches_knn_rev = bf.knnMatch(d2, d1, k=1)
    rev = {pair[0].queryIdx: pair[0].trainIdx for pair in matches_knn_rev if pair}
    good = [m for m in good if rev.get(m.trainIdx) == m.queryIdx]

    pts1 = np.float64([k1[m.queryIdx].pt for m in good])
    pts2 = np.float64([k2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K_MH01, method=method, prob=0.9999, threshold=1.0
    )
    n_inliers_e, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K_MH01, mask=mask)

    # Timing medians.
    def det():
        orb.detectAndCompute(img1, None)
        orb.detectAndCompute(img2, None)

    def match():
        mk = bf.knnMatch(d1, d2, k=2)
        rv = bf.knnMatch(d2, d1, k=1)
        _ = [pair[0] for pair in mk if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        _ = rv

    def pose():
        E_, mk = cv2.findEssentialMat(
            pts1, pts2, K_MH01, method=method, prob=0.9999, threshold=1.0
        )
        cv2.recoverPose(E_, pts1, pts2, K_MH01, mask=mk)

    return Result(
        name=label,
        t_detect_ms=median_ms(det),
        t_match_ms=median_ms(match),
        t_pose_ms=median_ms(pose),
        rot_err_deg=abs(rotation_angle_deg(R) - GT_ROT_DEG),
        t_err_deg=t_direction_error_deg(t.flatten(), GT_T_DIR),
        n_matches=len(pts1),
        n_inliers=int(n_inliers_e),
    )


def print_table(results: list[Result]) -> None:
    hdr = f"{'backend':<18} {'detect(ms)':>11} {'match(ms)':>10} {'pose(ms)':>10} {'total(ms)':>11} {'rot_err°':>9} {'t_err°':>8} {'matches':>8} {'inliers':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.name:<18} {r.t_detect_ms:>11.3f} {r.t_match_ms:>10.3f} "
            f"{r.t_pose_ms:>10.3f} {r.t_total_ms:>11.3f} "
            f"{r.rot_err_deg:>9.3f} {r.t_err_deg:>8.3f} "
            f"{r.n_matches:>8d} {r.n_inliers:>8d}"
        )
    # For multi-backend runs (kornia + one or more opencv variants), report
    # kornia's total speedup against each opencv row and the winner on accuracy.
    if len(results) >= 2 and results[0].name == "kornia-rs":
        k = results[0]
        print("-" * len(hdr))
        for o in results[1:]:
            rot_winner = "kornia" if k.rot_err_deg < o.rot_err_deg else o.name
            t_winner = "kornia" if k.t_err_deg < o.t_err_deg else o.name
            print(
                f"{'vs ' + o.name:<18} "
                f"{o.t_detect_ms/k.t_detect_ms:>10.2f}x "
                f"{o.t_match_ms/k.t_match_ms:>9.2f}x "
                f"{o.t_pose_ms/k.t_pose_ms:>9.2f}x "
                f"{o.t_total_ms/k.t_total_ms:>10.2f}x "
                f"{rot_winner:>9} "
                f"{t_winner:>8}"
            )


def main() -> None:
    img1 = _load_gray(DATA_DIR / "mh01_frame1.png")
    img2 = _load_gray(DATA_DIR / "mh01_frame2.png")

    print(f"SLAM bootstrap baseline — EuRoC MH_01 pair ({img1.shape[1]}x{img1.shape[0]})")
    print(f"Ground truth:  rot={GT_ROT_DEG}°  t_dir={GT_T_DIR.tolist()}")
    print(f"Timing: median over {N_ITERS} iterations ({N_WARMUP} warmup)")
    print()

    # kornia-rs vs every RANSAC/USAC method OpenCV ships. All OpenCV rows
    # share the same ORB→BF→ratio+cross-check match set; only the essential-
    # matrix estimator differs. kornia-rs runs its own full pipeline.
    opencv_methods = [
        ("opencv-ransac",    cv2.RANSAC),        # legacy 5-point + vanilla RANSAC
        ("opencv-lmeds",     cv2.LMEDS),         # least median of squares (no threshold)
        ("opencv-usac-def",  cv2.USAC_DEFAULT),  # RHO / PROSAC-equivalent default
        ("opencv-usac-fast", cv2.USAC_FAST),     # fewer iter, early termination
        ("opencv-usac-acc",  cv2.USAC_ACCURATE), # LO-MSAC + degeneracy check
        ("opencv-usac-mag",  cv2.USAC_MAGSAC),   # σ-marginalized (MAGSAC++)
        ("opencv-usac-pro",  cv2.USAC_PROSAC),   # quality-ordered sampling
        ("opencv-usac-par",  cv2.USAC_PARALLEL), # multithreaded
    ]
    # Default config uses the 8-point fundamental solver — kornia-slam's
    # bootstrap consumes F downstream. The 5-point row stays for direct
    # comparison vs the on-manifold essential path (callers opt in with
    # `use_5pt_essential=True` when translation-direction priority matters
    # more than F-availability).
    results = [
        bench_kornia(img1, img2, use_5pt=False, label="kornia-rs"),
        bench_kornia(img1, img2, use_5pt=True,  label="kornia-rs-5pt"),
    ]
    results.extend(
        bench_opencv(img1, img2, method=m, label=label)
        for label, m in opencv_methods
    )
    print_table(results)


if __name__ == "__main__":
    main()
