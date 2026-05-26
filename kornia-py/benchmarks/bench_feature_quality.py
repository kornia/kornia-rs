"""Feature-pipeline quality + speed benchmark harness.

What this is for
----------------
When we want to swap a piece of the ORB pipeline (detector, descriptor,
matcher, or RANSAC) for a candidate improvement (e.g. BEBLID descriptor
instead of BRIEF, PROSAC instead of LO-RANSAC), we need quantitative
evidence that the new setup is actually better than the current one.
This script is that evidence.

What it measures
----------------
For each (pipeline × scenario × trial) it records:
  * quality  — inlier count, corner reprojection error in pixels
  * speed    — ms for detect+describe, match, RANSAC, and end-to-end
  * coverage — number of keypoints detected, number of good matches

Then per-pipeline aggregates (median / mean / p95) are printed both
globally and per-scenario family (rotation / scale / illumination / …).
A final "winner count" shows, for each scenario, which pipeline
produced the lowest reprojection error — so a proposed upgrade has
to actually win most scenarios to be called an improvement.

How to add a new pipeline
-------------------------
A pipeline is a ``Pipeline`` dataclass: a ``detect`` callable, a
``match`` callable, and a ``solve`` callable. See ``KORNIA_PIPELINE``
and ``OPENCV_ORB_BRIEF_PIPELINE`` below for templates. To test
e.g. ORB+BEBLID, create a new pipeline that reuses the ORB detector
but swaps in BEBLID as the descriptor step.

How to add a new scenario
-------------------------
Scenarios are declared in ``build_scenarios()``. Each one is a
``Scenario`` with a name, a family, an ``H_gt`` generator and an
optional image-perturbation callable. They're deliberately the same
set used by ``kornia-py/tests/test_orb_slam_robustness.py`` — if a
pipeline passes the robustness tests, the harness will show by how
much margin, and if it fails, by how much it's behind.

Running
-------
  .venv/bin/python benchmarks/bench_feature_quality.py
  .venv/bin/python benchmarks/bench_feature_quality.py --trials 3 --only-fast
  .venv/bin/python benchmarks/bench_feature_quality.py --save results.json
  .venv/bin/python benchmarks/bench_feature_quality.py --baseline results.json

Missing backends (e.g. opencv-contrib not installed) are skipped with
a warning, not a hard failure — so the harness works in any env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import cv2  # required
import kornia_rs as K  # required
from kornia_rs.image import Image


def _load_gray(path):
    """Load a 2D u8 grayscale image via the kornia Image API. cv2 stays as
    a comparison-only target (ORB / BEBLID / AKAZE pipelines)."""
    return Image.load(str(path)).to_grayscale().to_numpy()[..., 0]


def _warp_perspective_gray(img, H, w, h):
    """Apply a 3x3 homography to a 2D grayscale image via kornia's
    `warp_perspective`. Used to synthesize warped test data."""
    arr3 = img[..., None] if img.ndim == 2 else img
    out3 = K.imgproc.warp_perspective(
        arr3, tuple(H.astype(np.float32).flatten()), (h, w), "bilinear"
    )
    return out3[..., 0] if img.ndim == 2 else out3


def _perspective_transform(pts, H):
    """Numpy port of cv2.perspectiveTransform: apply 3x3 H to (N, 1, 2) pts."""
    p = pts.reshape(-1, 2)
    h_pts = np.concatenate([p, np.ones((len(p), 1), dtype=p.dtype)], axis=1)
    out = h_pts @ H.T
    return (out[:, :2] / out[:, 2:3]).reshape(pts.shape)


# -------------------------------------------------------------------
# Data classes: a Pipeline is (detect, match, solve). A Scenario is a
# controlled perturbation of a reference image with a known H_gt.
# -------------------------------------------------------------------


@dataclass
class Pipeline:
    """One end-to-end feature pipeline under test."""

    name: str
    detect: Callable[[np.ndarray], tuple]  # (kps_xy, descriptors)
    match: Callable[[np.ndarray, np.ndarray], np.ndarray]  # desc_a, desc_b -> (M, 2) int
    solve: Callable[[np.ndarray, np.ndarray], tuple]  # src_pts, dst_pts -> (H, mask)
    available: bool = True
    skip_reason: str = ""


@dataclass
class Scenario:
    """A synthetic transformation + optional image perturbation."""

    name: str
    family: str  # one of: rotation, scale, affine, perspective, illum, noise, blur, compound
    make_h: Callable[[int, int], np.ndarray]  # (w, h) -> 3x3 H_gt
    perturb: Optional[Callable[[np.ndarray], np.ndarray]] = None  # applied to warped img


@dataclass
class TrialResult:
    pipeline: str
    scenario: str
    family: str
    trial: int
    n_kp_a: int
    n_kp_b: int
    n_matches: int
    n_inliers: int
    reproj_err_px: float
    ms_detect: float
    ms_match: float
    ms_solve: float
    ms_total: float
    failed: bool = False


@dataclass
class Aggregate:
    pipeline: str
    n: int
    inlier_mean: float
    reproj_median_px: float
    reproj_mean_px: float
    reproj_p95_px: float
    success_rate: float  # fraction of trials with finite reproj err
    ms_detect_median: float
    ms_match_median: float
    ms_solve_median: float
    ms_total_median: float


# -------------------------------------------------------------------
# Scenario helpers — identical math to test_orb_slam_robustness.py so
# that passing the benchmark implies passing the robustness suite.
# -------------------------------------------------------------------


def _h_identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _h_rotation(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a),  np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return Tb @ R @ T


def _h_scale(cx: float, cy: float, s: float) -> np.ndarray:
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    S = np.diag([s, s, 1.0]).astype(np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return Tb @ S @ T


def _h_affine(cx, cy, angle_deg, sx, sy, tx, ty):
    a = np.deg2rad(angle_deg)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    A = np.array([[sx * np.cos(a), -sy * np.sin(a), 0],
                  [sx * np.sin(a),  sy * np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)
    return Tb @ A @ T


def _h_perspective(cx, cy, angle_deg, skew):
    a = np.deg2rad(angle_deg)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a),  np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    P = np.array([[1, 0, 0], [0, 1, 0], [skew, skew, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return Tb @ P @ R @ T


def _apply_illumination(img: np.ndarray, gain: float, bias: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def _apply_motion_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    # Horizontal averaging filter (matches the cv2.filter2D row-kernel
    # version we used to ship). Synthetic perturbation only — never timed
    # — so a plain numpy sliding window is correct + cv2-free.
    pad = kernel_size // 2
    padded = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size, axis=1)
    return windows.mean(axis=-1).astype(np.uint8)


def _apply_gaussian_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def build_scenarios() -> List[Scenario]:
    """The canonical set of SLAM-style perturbations."""
    out = []

    def w(s):  # closure-capture helper for lambdas below
        return lambda W, H: s(W / 2, H / 2)

    # rotation
    for deg in [-30, -15, -5, 5, 15, 30, 45]:
        out.append(Scenario(
            name=f"rot_{deg:+d}deg", family="rotation",
            make_h=(lambda d: (lambda W, H: _h_rotation(W / 2, H / 2, d)))(deg),
        ))
    # scale
    for s in [0.7, 0.85, 1.15, 1.3, 1.5]:
        out.append(Scenario(
            name=f"scale_{s:.2f}x", family="scale",
            make_h=(lambda s_: (lambda W, H: _h_scale(W / 2, H / 2, s_)))(s),
        ))
    # affine
    for (deg, sx, sy, tx, ty) in [(5, 1.1, 0.95, 5, -5), (-10, 0.9, 1.05, -10, 10)]:
        out.append(Scenario(
            name=f"affine_{deg:+d}deg_{sx:.2f}x{sy:.2f}", family="affine",
            make_h=(lambda p: (lambda W, H: _h_affine(W / 2, H / 2, *p)))((deg, sx, sy, tx, ty)),
        ))
    # perspective
    for skew in [5e-5, 1e-4, 2e-4]:
        out.append(Scenario(
            name=f"persp_{skew:.0e}", family="perspective",
            make_h=(lambda sk: (lambda W, H: _h_perspective(W / 2, H / 2, 10, sk)))(skew),
        ))
    # illumination — identity H, perturb warped image
    for (g, b) in [(0.5, 0), (1.5, 0), (1.0, -40), (1.0, 40)]:
        gg, bb = g, b
        out.append(Scenario(
            name=f"illum_gain{gg:.1f}_bias{bb:+d}", family="illum",
            make_h=(lambda: (lambda W, H: _h_rotation(W / 2, H / 2, 3)))(),
            perturb=(lambda g_, b_: lambda img: _apply_illumination(img, g_, b_))(gg, bb),
        ))
    # noise
    for sigma in [2, 5, 10]:
        out.append(Scenario(
            name=f"noise_sigma{sigma}", family="noise",
            make_h=(lambda: (lambda W, H: _h_rotation(W / 2, H / 2, 3)))(),
            perturb=(lambda s_: lambda img: _apply_gaussian_noise(img, s_))(sigma),
        ))
    # motion blur
    for k in [3, 5, 7]:
        out.append(Scenario(
            name=f"blur_k{k}", family="blur",
            make_h=(lambda: (lambda W, H: _h_rotation(W / 2, H / 2, 3)))(),
            perturb=(lambda k_: lambda img: _apply_motion_blur(img, k_))(k),
        ))
    # compound — the worst-case SLAM stress
    out.append(Scenario(
        name="compound_slam", family="compound",
        make_h=lambda W, H: _h_affine(W / 2, H / 2, 8, 1.12, 0.95, 5, -5),
        perturb=lambda img: _apply_gaussian_noise(_apply_illumination(img, 1.2, -15), 3, seed=2),
    ))
    return out


# -------------------------------------------------------------------
# Pipeline factories. Each factory returns a Pipeline; if the backend
# is not available, it returns a Pipeline with available=False so the
# runner can skip it cleanly.
# -------------------------------------------------------------------


def kornia_pipeline() -> Pipeline:
    """Full native kornia-rs pipeline: FAST+Harris+BRIEF, kornia matcher, kornia RANSAC."""

    def detect(img):
        feat = K.features.orb_detect_and_compute(img)
        xy = np.asarray(feat.keypoints_xy, dtype=np.float32).reshape(-1, 2)
        desc = np.asarray(feat.descriptors, dtype=np.uint8).reshape(-1, 32)
        return xy, desc

    def match(da, db):
        if len(da) == 0 or len(db) == 0:
            return np.zeros((0, 2), np.int64)
        return K.features.match_descriptors(da, db, cross_check=False, max_ratio=0.8)

    def solve(src, dst):
        if len(src) < 4:
            return None, np.zeros(len(src), np.uint8)
        try:
            H, mask = K.k3d.find_homography(
                src.astype(np.float64), dst.astype(np.float64),
                method=8, ransac_threshold=3.0, min_inliers=4, seed=0,
            )
        except ValueError:
            return None, np.zeros(len(src), np.uint8)
        return H, mask

    return Pipeline("kornia-rs", detect, match, solve)


def opencv_orb_pipeline() -> Pipeline:
    """OpenCV baseline: cv2.ORB_create with all defaults equivalent to ours."""
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                         firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                         patchSize=31, fastThreshold=20)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(img):
        kps, desc = orb.detectAndCompute(img, None)
        xy = np.asarray([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
        desc = desc if desc is not None else np.zeros((0, 32), np.uint8)
        return xy, desc

    def match(da, db):
        if len(da) < 2 or len(db) < 2:
            return np.zeros((0, 2), np.int64)
        pairs = bf.knnMatch(da, db, k=2)
        good = [(p[0].queryIdx, p[0].trainIdx) for p in pairs
                if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        return np.asarray(good, dtype=np.int64) if good else np.zeros((0, 2), np.int64)

    def solve(src, dst):
        if len(src) < 4:
            return None, np.zeros(len(src), np.uint8)
        H, mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
        if mask is None:
            mask = np.zeros(len(src), np.uint8)
        else:
            mask = mask.ravel().astype(np.uint8)
        return H, mask

    return Pipeline("opencv-orb", detect, match, solve)


def opencv_orb_beblid_pipeline() -> Pipeline:
    """Drop-in test: OpenCV ORB detector + BEBLID descriptor. This is the
    experiment we care about — does replacing BRIEF with BEBLID improve
    matching on our scenarios, while keeping the FAST+Harris detector?"""
    try:
        beblid = cv2.xfeatures2d.BEBLID_create(6.25, 101)  # 101 = 256-bit
    except (AttributeError, cv2.error) as e:
        return Pipeline("opencv-orb+beblid", lambda i: (None, None),
                        lambda a, b: None, lambda s, d: (None, None),
                        available=False, skip_reason=f"BEBLID unavailable: {e}")

    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                         firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                         patchSize=31, fastThreshold=20)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(img):
        kps = orb.detect(img, None)
        kps, desc = beblid.compute(img, kps)
        xy = np.asarray([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
        desc = desc if desc is not None else np.zeros((0, 32), np.uint8)
        return xy, desc

    def match(da, db):
        if len(da) < 2 or len(db) < 2:
            return np.zeros((0, 2), np.int64)
        pairs = bf.knnMatch(da, db, k=2)
        good = [(p[0].queryIdx, p[0].trainIdx) for p in pairs
                if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        return np.asarray(good, dtype=np.int64) if good else np.zeros((0, 2), np.int64)

    def solve(src, dst):
        if len(src) < 4:
            return None, np.zeros(len(src), np.uint8)
        H, mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
        if mask is None:
            mask = np.zeros(len(src), np.uint8)
        else:
            mask = mask.ravel().astype(np.uint8)
        return H, mask

    return Pipeline("opencv-orb+beblid", detect, match, solve)


def opencv_akaze_pipeline() -> Pipeline:
    """Full AKAZE — nonlinear-scale-space detector + M-LDB binary descriptor.
    Included as a 'quality ceiling' reference, not a real-time candidate."""
    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(img):
        kps, desc = akaze.detectAndCompute(img, None)
        xy = np.asarray([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
        if desc is None:
            desc = np.zeros((0, 61), np.uint8)
        return xy, desc

    def match(da, db):
        if len(da) < 2 or len(db) < 2:
            return np.zeros((0, 2), np.int64)
        pairs = bf.knnMatch(da, db, k=2)
        good = [(p[0].queryIdx, p[0].trainIdx) for p in pairs
                if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        return np.asarray(good, dtype=np.int64) if good else np.zeros((0, 2), np.int64)

    def solve(src, dst):
        if len(src) < 4:
            return None, np.zeros(len(src), np.uint8)
        H, mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
        if mask is None:
            mask = np.zeros(len(src), np.uint8)
        else:
            mask = mask.ravel().astype(np.uint8)
        return H, mask

    return Pipeline("opencv-akaze", detect, match, solve)


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------


def corner_reproj_error(H_est, H_gt, w, h) -> float:
    if H_est is None:
        return float("inf")
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    try:
        a = _perspective_transform(corners, H_est).reshape(-1, 2)
        b = _perspective_transform(corners, H_gt).reshape(-1, 2)
        return float(np.linalg.norm(a - b, axis=1).mean())
    except (ValueError, ZeroDivisionError):
        return float("inf")


def run_one(pipeline: Pipeline, img: np.ndarray, scenario: Scenario, trial: int) -> TrialResult:
    h, w = img.shape
    H_gt = scenario.make_h(w, h)
    warped = _warp_perspective_gray(img, H_gt, w, h)
    if scenario.perturb is not None:
        warped = scenario.perturb(warped)

    t0 = time.perf_counter()
    xy_a, desc_a = pipeline.detect(img)
    xy_b, desc_b = pipeline.detect(warped)
    t1 = time.perf_counter()

    matches = pipeline.match(desc_a, desc_b)
    t2 = time.perf_counter()

    if len(matches) < 4:
        return TrialResult(
            pipeline.name, scenario.name, scenario.family, trial,
            len(xy_a), len(xy_b), len(matches), 0, float("inf"),
            (t1 - t0) * 1000, (t2 - t1) * 1000, 0.0, (t2 - t0) * 1000,
            failed=True,
        )

    src = xy_a[matches[:, 0]].astype(np.float64)
    dst = xy_b[matches[:, 1]].astype(np.float64)
    H_est, mask = pipeline.solve(src, dst)
    t3 = time.perf_counter()

    inliers = int(mask.sum()) if mask is not None else 0
    err = corner_reproj_error(H_est, H_gt, w, h)

    return TrialResult(
        pipeline.name, scenario.name, scenario.family, trial,
        len(xy_a), len(xy_b), len(matches), inliers, err,
        (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t3 - t0) * 1000,
        failed=not np.isfinite(err),
    )


def aggregate(results: List[TrialResult], pipeline: str) -> Aggregate:
    rs = [r for r in results if r.pipeline == pipeline]
    if not rs:
        return Aggregate(pipeline, 0, 0, float("inf"), float("inf"), float("inf"), 0.0, 0, 0, 0, 0)
    # Successes only for reproj; failures contribute to success_rate.
    ok = [r for r in rs if np.isfinite(r.reproj_err_px)]
    errs = np.array([r.reproj_err_px for r in ok]) if ok else np.array([float("inf")])
    inls = np.array([r.n_inliers for r in rs])
    return Aggregate(
        pipeline=pipeline,
        n=len(rs),
        inlier_mean=float(inls.mean()),
        reproj_median_px=float(np.median(errs)) if ok else float("inf"),
        reproj_mean_px=float(errs.mean()) if ok else float("inf"),
        reproj_p95_px=float(np.percentile(errs, 95)) if ok else float("inf"),
        success_rate=len(ok) / len(rs),
        ms_detect_median=float(np.median([r.ms_detect for r in rs])),
        ms_match_median=float(np.median([r.ms_match for r in rs])),
        ms_solve_median=float(np.median([r.ms_solve for r in rs])),
        ms_total_median=float(np.median([r.ms_total for r in rs])),
    )


def print_aggregate_table(pipelines: List[str], aggs: List[Aggregate]):
    print(f"\n{'='*110}")
    print(f"Aggregate across all scenarios × trials (per pipeline)")
    print(f"{'='*110}")
    print(f"{'pipeline':22s} {'n':>4s} {'succ%':>6s} {'inl':>6s} "
          f"{'reproj_med':>11s} {'reproj_mean':>12s} {'reproj_p95':>11s} "
          f"{'ms_det':>8s} {'ms_mat':>8s} {'ms_slv':>8s} {'ms_tot':>8s}")
    print("-" * 110)
    for a in aggs:
        print(f"{a.pipeline:22s} {a.n:>4d} {a.success_rate*100:>5.1f}% "
              f"{a.inlier_mean:>6.1f} {a.reproj_median_px:>10.3f}  "
              f"{a.reproj_mean_px:>11.3f}  {a.reproj_p95_px:>10.3f}  "
              f"{a.ms_detect_median:>7.2f}  {a.ms_match_median:>7.2f}  "
              f"{a.ms_solve_median:>7.2f}  {a.ms_total_median:>7.2f}")


def print_per_family_table(results: List[TrialResult], pipelines: List[str]):
    families = sorted({r.family for r in results})
    print(f"\n{'='*110}")
    print("Median reprojection error (px) per pipeline × scenario family — lower is better")
    print(f"{'='*110}")
    header = f"{'family':14s}" + "".join(f"{p:>22s}" for p in pipelines)
    print(header)
    print("-" * len(header))
    for fam in families:
        row = f"{fam:14s}"
        for p in pipelines:
            errs = [r.reproj_err_px for r in results
                    if r.pipeline == p and r.family == fam and np.isfinite(r.reproj_err_px)]
            if errs:
                row += f"{np.median(errs):>22.3f}"
            else:
                row += f"{'N/A':>22s}"
        print(row)


def print_winner_table(results: List[TrialResult], pipelines: List[str]):
    """Count, per pipeline, on how many (scenario, trial) combos it had the lowest reproj error."""
    # group by (scenario, trial)
    by_key = {}
    for r in results:
        k = (r.scenario, r.trial)
        by_key.setdefault(k, []).append(r)

    wins = {p: 0 for p in pipelines}
    finite_runs = 0
    for k, rs in by_key.items():
        finite = [r for r in rs if np.isfinite(r.reproj_err_px)]
        if not finite:
            continue
        finite_runs += 1
        best = min(finite, key=lambda r: r.reproj_err_px)
        wins[best.pipeline] = wins.get(best.pipeline, 0) + 1

    print(f"\n{'='*60}")
    print(f"Per-scenario winner count (of {finite_runs} finite trials)")
    print(f"{'='*60}")
    for p in pipelines:
        pct = 100 * wins[p] / finite_runs if finite_runs else 0
        print(f"  {p:24s}  {wins[p]:>4d}  ({pct:.1f}%)")


def compare_to_baseline(results: List[TrialResult], pipelines: List[str],
                        baseline_path: Path, probe_pipeline: str):
    """A/B compare current-run 'probe_pipeline' vs saved-baseline 'probe_pipeline'.
    Only the named pipeline is A/B'd — the others are shown purely as context."""
    with open(baseline_path) as f:
        base = json.load(f)
    base_results = [TrialResult(**r) for r in base["results"]]

    a = [r for r in base_results if r.pipeline == probe_pipeline and np.isfinite(r.reproj_err_px)]
    b = [r for r in results if r.pipeline == probe_pipeline and np.isfinite(r.reproj_err_px)]
    if not a or not b:
        print(f"[baseline] no data for pipeline '{probe_pipeline}' in one of the runs")
        return

    def med(xs, attr):
        return float(np.median([getattr(x, attr) for x in xs]))

    print(f"\n{'='*70}")
    print(f"A/B vs baseline for pipeline '{probe_pipeline}'")
    print(f"{'='*70}")
    for attr, label, better in [
        ("reproj_err_px", "reproj_med px", "lower"),
        ("n_inliers", "inliers", "higher"),
        ("ms_total", "ms_total", "lower"),
    ]:
        old, new = med(a, attr), med(b, attr)
        delta = new - old
        pct = 100 * delta / old if old else 0
        arrow = "↓" if delta < 0 else "↑"
        verdict = "improved" if (
            (better == "lower" and delta < 0) or (better == "higher" and delta > 0)
        ) else "regressed"
        print(f"  {label:20s}  {old:>10.3f}  →  {new:>10.3f}   {arrow}{abs(pct):5.1f}%  [{verdict}]")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--trials", type=int, default=3,
                    help="trials per (pipeline, scenario) to average across noise")
    ap.add_argument("--only-fast", action="store_true",
                    help="skip the slow AKAZE/reference pipelines")
    ap.add_argument("--images", nargs="+", default=None,
                    help="override default images; defaults to dog.jpeg + mh01_frame1.png")
    ap.add_argument("--save", type=Path, default=None,
                    help="write raw results to a JSON file")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="compare probe pipeline against saved baseline JSON")
    ap.add_argument("--probe", default="kornia-rs",
                    help="pipeline name to A/B against baseline")
    args = ap.parse_args()

    if args.images is None:
        data_dir = Path(__file__).resolve().parents[2] / "tests" / "data"
        args.images = [
            str(data_dir / "dog.jpeg"),
            str(data_dir / "mh01_frame1.png"),
        ]

    pipelines = [
        kornia_pipeline(),
        opencv_orb_pipeline(),
        opencv_orb_beblid_pipeline(),
    ]
    if not args.only_fast:
        pipelines.append(opencv_akaze_pipeline())

    # drop unavailable backends
    for p in pipelines:
        if not p.available:
            print(f"[skip] {p.name}: {p.skip_reason}")
    pipelines = [p for p in pipelines if p.available]
    scenarios = build_scenarios()

    print(f"Benchmark: {len(pipelines)} pipelines × {len(scenarios)} scenarios × "
          f"{args.trials} trials × {len(args.images)} images")
    print(f"Pipelines: {[p.name for p in pipelines]}")

    results: List[TrialResult] = []
    for img_path in args.images:
        if not Path(img_path).is_file():
            print(f"[skip image] {img_path}")
            continue
        img = _load_gray(img_path)
        print(f"\n--- {Path(img_path).name} ({img.shape[1]}x{img.shape[0]}) ---")
        for p in pipelines:
            # warm-up pass so JIT/caches don't taint trial 0 timings
            try:
                p.detect(img)
            except Exception as e:
                print(f"  [warmup fail] {p.name}: {e}")
                continue
            for sc in scenarios:
                for t in range(args.trials):
                    try:
                        r = run_one(p, img, sc, t)
                        results.append(r)
                    except Exception as e:
                        print(f"  [error] {p.name} / {sc.name} / trial {t}: {e}")

    pipeline_names = [p.name for p in pipelines]
    aggs = [aggregate(results, p) for p in pipeline_names]
    print_aggregate_table(pipeline_names, aggs)
    print_per_family_table(results, pipeline_names)
    print_winner_table(results, pipeline_names)

    if args.save:
        payload = {
            "results": [r.__dict__ for r in results],
            "pipelines": pipeline_names,
            "n_scenarios": len(scenarios),
            "trials": args.trials,
            "images": args.images,
        }
        args.save.write_text(json.dumps(payload, indent=2))
        print(f"\n[saved] {args.save}")

    if args.baseline is not None:
        compare_to_baseline(results, pipeline_names, args.baseline, args.probe)


if __name__ == "__main__":
    sys.exit(main() or 0)
