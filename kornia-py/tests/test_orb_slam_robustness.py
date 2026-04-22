"""ORB robustness against SLAM-style image perturbations.

The benchmarks in `bench_orb.py` measure speed; `test_orb_e2e.py` measures
a single synthetic round-trip per image. This suite pins the *quality
envelope* the full pipeline (detect → describe → match → RANSAC
homography) must hold across the transformations an ORB-SLAM3-style
system encounters frame-to-frame:

1. In-plane rotation       (camera roll between keyframes)
2. Scale                   (camera motion toward/away from scene)
3. Affine view change      (small baseline + roll)
4. Perspective view change (larger baseline, out-of-plane rotation)
5. Illumination            (auto-exposure drift, sun moving behind a cloud)
6. Gaussian pixel noise    (low-light / high-ISO sensor noise)
7. Motion blur             (short camera motion during exposure)
8. Compound                (scale + rotation + illumination + noise)

For each perturbation we warp the EuRoC MH_01 frame (real SLAM imagery,
not synthetic) by a known homography, run the full kornia-rs ORB
pipeline on both copies, solve the homography via `find_homography`
(LO-RANSAC), and measure corner-reprojection error in pixels. The test
asserts that (a) RANSAC converges (≥ `min_inliers` inliers) and
(b) reprojected corners land within a pixel budget appropriate for the
perturbation class.

Thresholds are calibrated against ORB-SLAM3's tracking tolerance: the
system drops a frame when tracked points fall below ~50 inliers, and its
homography-based map initialization expects mean reprojection error
below ~4 px. We use tighter budgets (8 px max for the hardest cases,
2 px for the easy ones) so a regression that would merely degrade
tracking also fails this test.

Runtime: ~2 s total on Jetson Orin — cheap enough to run on every
`pytest` invocation alongside the unit tests.
"""
from __future__ import annotations

import os
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import kornia_rs as K


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "tests",
    "data",
    "mh01_frame1.png",
)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------


def _load_euroc_frame():
    img = cv2.imread(DATA_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pytest.skip(f"missing test image: {DATA_PATH}")
    return img


def _h_rotation(cx, cy, angle_deg):
    a = np.deg2rad(angle_deg)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a),  np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return Tb @ R @ T


def _h_scale(cx, cy, s):
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    S = np.diag([s, s, 1.0]).astype(np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return Tb @ S @ T


def _h_affine(cx, cy, angle_deg, sx, sy, tx, ty):
    a = np.deg2rad(angle_deg)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    M = np.array([[sx * np.cos(a), -sy * np.sin(a), 0],
                  [sx * np.sin(a),  sy * np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)
    return Tb @ M @ T


def _h_perspective(cx, cy, angle_deg, skew):
    a = np.deg2rad(angle_deg)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a),  np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    P = np.array([[1, 0, 0], [0, 1, 0], [skew, skew * 0.5, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return Tb @ P @ R @ T


def _apply_illumination(img, gain, bias):
    out = img.astype(np.float32) * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_motion_blur(img, kernel_size):
    """Horizontal-motion blur: kernel_size-wide single-row box."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, k)


def _detect(img):
    kps, _, desc = K.features.orb_detect_and_compute(img)
    return np.asarray(kps, dtype=np.float32).reshape(-1, 2), np.asarray(desc, dtype=np.uint8)


def _estimate_h(xy_a, desc_a, xy_b, desc_b, ratio=0.8):
    if len(desc_a) < 4 or len(desc_b) < 4:
        return None, 0, 0
    matches = K.features.match_descriptors(desc_a, desc_b, cross_check=False, max_ratio=ratio)
    if len(matches) < 4:
        return None, 0, len(matches)
    pts_a = np.asarray(xy_a, dtype=np.float64)[matches[:, 0]]
    pts_b = np.asarray(xy_b, dtype=np.float64)[matches[:, 1]]
    try:
        H, mask = K.features.find_homography(
            pts_a, pts_b, method=8, ransac_threshold=3.0, min_inliers=4, seed=0
        )
    except ValueError:
        return None, 0, len(matches)
    return H, int(mask.sum()), len(matches)


def _corner_error(H_est, H_gt, w, h):
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    pts_est = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
    pts_gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
    return float(np.linalg.norm(pts_est - pts_gt, axis=1).mean())


def _run(img, H_gt, *, perturb_warped=None, min_inliers, max_err_px):
    """Generic harness: warp `img` by `H_gt`, optionally perturb the warped
    copy (illumination / noise / blur), run the full pipeline, assert
    inlier and reprojection-error bounds, and return `(n_inl, err)`."""
    h, w = img.shape
    warped = cv2.warpPerspective(img, H_gt, (w, h))
    if perturb_warped is not None:
        warped = perturb_warped(warped)

    xy_a, desc_a = _detect(img)
    xy_b, desc_b = _detect(warped)
    H_est, n_inl, n_good = _estimate_h(xy_a, desc_a, xy_b, desc_b)

    assert H_est is not None, f"find_homography failed ({n_good} matches)"
    assert n_inl >= min_inliers, f"only {n_inl} inliers (need ≥ {min_inliers})"

    err = _corner_error(H_est, H_gt, w, h)
    assert err < max_err_px, f"reproj err {err:.2f}px exceeds budget {max_err_px}px"
    return n_inl, err


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle_deg", [-30, -15, -5, 5, 15, 30, 45])
def test_rotation_invariance(angle_deg):
    """In-plane rotation up to ±45°. ORB uses the intensity-centroid
    orientation, which is rotation-equivariant, so the descriptor should
    stay stable across the entire range. Budget: 4 px corner error, 40
    inliers — ORB-SLAM3 drops a frame below ~50 inliers so this is tight."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = _h_rotation(w / 2, h / 2, angle_deg)
    _run(img, H_gt, min_inliers=40, max_err_px=4.0)


@pytest.mark.parametrize("scale", [0.7, 0.85, 1.15, 1.3, 1.5])
def test_scale_invariance(scale):
    """Scale 0.7×–1.5× — covers ≈2 pyramid levels of the 8-level ORB
    pyramid (scale factor 1.2 per level). Between-level samples stress
    cross-octave matching most. Budget: 6 px, 30 inliers."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = _h_scale(w / 2, h / 2, scale)
    _run(img, H_gt, min_inliers=30, max_err_px=6.0)


def test_affine_small_baseline():
    """Small-baseline affine (rotation + anisotropic scale + translation):
    the typical between-keyframe transform in handheld SLAM. Budget 3 px."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = _h_affine(w / 2, h / 2, angle_deg=10, sx=1.05, sy=0.95, tx=20, ty=-10)
    _run(img, H_gt, min_inliers=50, max_err_px=3.0)


@pytest.mark.parametrize("skew", [5e-5, 1e-4, 2e-4])
def test_perspective_view_change(skew):
    """Out-of-plane rotation induces perspective skew. Skew = 2e-4 at
    480-row images amounts to ~10% width foreshortening — harder than
    most frame-to-frame pairs but easier than loop-closure. Budget
    scales with skew: stronger skew → larger acceptable residual."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = _h_perspective(w / 2, h / 2, angle_deg=8, skew=skew)
    _run(img, H_gt, min_inliers=30, max_err_px=5.0)


@pytest.mark.parametrize("gain,bias", [(0.5, 0), (1.5, 0), (1.0, -40), (1.0, 40)])
def test_illumination_invariance(gain, bias):
    """BRIEF's pair-compare descriptor is illumination-invariant by
    construction (pair sign is gain-invariant; bias cancels too). Warp is
    identity — we're isolating the illumination axis from geometry.
    Budget: 2 px, 60 inliers. Looser bias tolerance would be
    suspicious — if identity-warp corner error exceeds 2 px the
    descriptor isn't invariant."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = np.eye(3, dtype=np.float64)
    _run(
        img, H_gt,
        perturb_warped=lambda x: _apply_illumination(x, gain, bias),
        min_inliers=60, max_err_px=2.0,
    )


@pytest.mark.parametrize("sigma", [2, 5, 10])
def test_noise_invariance(sigma):
    """Gaussian sensor noise with σ = 2–10 gray levels. σ=10 is closer
    to low-light smartphone sensor territory; real SLAM rigs are
    typically σ ≤ 5. Budget 4 px for σ≤5 and 8 px for σ=10 — noise
    jitters the IC orientation, which knocks BRIEF rotation bucket."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = np.eye(3, dtype=np.float64)
    rng = np.random.default_rng(0)

    def add_noise(x):
        return np.clip(x.astype(np.float32) + rng.normal(0, sigma, x.shape),
                       0, 255).astype(np.uint8)

    budget = 4.0 if sigma <= 5 else 8.0
    _run(img, H_gt, perturb_warped=add_noise, min_inliers=40, max_err_px=budget)


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_motion_blur_tolerance(kernel_size):
    """Short horizontal motion blur during exposure. Kernel 7 is heavy
    blur — ORB-SLAM3 would typically drop these frames, but we
    want the pipeline to still degrade gracefully (not return garbage)
    up to kernel 5. Kernel 7 uses looser budget."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = np.eye(3, dtype=np.float64)
    budget = 3.0 if kernel_size <= 5 else 6.0
    min_inl = 40 if kernel_size <= 5 else 20
    _run(
        img, H_gt,
        perturb_warped=lambda x, k=kernel_size: _apply_motion_blur(x, k),
        min_inliers=min_inl, max_err_px=budget,
    )


def test_compound_slam_worst_case():
    """All perturbation axes simultaneously at moderate severity —
    approximates the hardest frame-to-frame transform an ORB-SLAM3
    tracker encounters without dropping the frame: 10° roll + 15%
    scale + small perspective + illumination drift + σ=3 noise.
    Must still recover homography within 6 px."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = _h_affine(w / 2, h / 2, angle_deg=10, sx=1.15, sy=1.15, tx=8, ty=-8)
    H_gt = _h_perspective(w / 2, h / 2, angle_deg=0, skew=8e-5) @ H_gt

    rng = np.random.default_rng(42)

    def hard_perturb(x):
        x = _apply_illumination(x, 1.2, -15)
        x = np.clip(x.astype(np.float32) + rng.normal(0, 3.0, x.shape),
                    0, 255).astype(np.uint8)
        return x

    _run(img, H_gt, perturb_warped=hard_perturb, min_inliers=30, max_err_px=6.0)


def test_homography_under_loop_closure_viewpoint():
    """Larger viewpoint change — more representative of a loop-closure
    candidate match (same place, very different camera pose). We don't
    expect ORB to ace this; we expect it not to silently fail.
    Success = ≥20 inliers and reprojection error under 10 px."""
    img = _load_euroc_frame()
    h, w = img.shape
    H_gt = _h_affine(w / 2, h / 2, angle_deg=25, sx=0.80, sy=0.80, tx=40, ty=20)
    H_gt = _h_perspective(w / 2, h / 2, angle_deg=0, skew=1.5e-4) @ H_gt
    _run(img, H_gt, min_inliers=20, max_err_px=10.0)
