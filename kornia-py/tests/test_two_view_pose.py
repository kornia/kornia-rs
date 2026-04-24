"""End-to-end SLAM bootstrap test: EuRoC MH_01 frame1 → frame2.

Verifies the full pipeline that ORB-SLAM-style monocular SLAM uses for cold-start
initialization:

    ORB detect+describe → Hamming match → fundamental RANSAC → model selection
    → essential decomposition → cheirality → (R, t)

and checks the recovered relative pose against the EuRoC ground truth:

    Rotation:      2.7021°  (Vicon-derived, allow ≤ 5°)
    Translation:   direction [0.2422, -0.2330, 0.9418]   (allow ≤ 15°)

The same test exists Rust-side at
`crates/kornia-3d/src/pose/twoview.rs::test_two_view_euroc_mh01`.
"""
import numpy as np
import os
import pytest

import kornia_rs as K


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")

# EuRoC MH_01_easy cam0 intrinsics.
K_MH01 = np.array(
    [
        [458.654, 0.0, 367.215],
        [0.0, 457.296, 248.375],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Ground-truth relative pose (frame1 → frame2, camera frame).
GT_ROT_DEG = 2.7021
GT_T_DIR = np.array([0.2422, -0.2330, 0.9418], dtype=np.float64)
GT_T_DIR /= np.linalg.norm(GT_T_DIR)


def _rotation_angle_deg(r: np.ndarray) -> float:
    """Angle of a 3×3 rotation matrix in degrees (axis-angle magnitude)."""
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _translation_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle between unit directions, folded to [0, 90°] (sign ambiguous from F)."""
    t_est = t_est / np.linalg.norm(t_est)
    t_gt = t_gt / np.linalg.norm(t_gt)
    cos_angle = float(np.clip(abs(np.dot(t_est, t_gt)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


@pytest.fixture(scope="module")
def mh01_matches():
    """Load the two MH_01 frames, run ORB on both, return matched pixel pairs."""
    img1 = K.io.read_image_png_u8(os.path.join(DATA_DIR, "mh01_frame1.png"), "mono")
    img2 = K.io.read_image_png_u8(os.path.join(DATA_DIR, "mh01_frame2.png"), "mono")
    img1 = np.asarray(img1).squeeze()
    img2 = np.asarray(img2).squeeze()

    feat1 = K.features.orb_detect_and_compute(img1)
    feat2 = K.features.orb_detect_and_compute(img2)

    # Lowe ratio test + mutual best; cross_check keeps only symmetric NN pairs.
    matches = K.features.match_descriptors(
        feat1.descriptors,
        feat2.descriptors,
        max_ratio=0.75,
        cross_check=True,
    )
    if len(matches) < 15:
        pytest.skip(f"too few ORB matches on MH01: {len(matches)} (need ≥ 15)")

    pts1 = np.ascontiguousarray(feat1.keypoints_xy[matches[:, 0]], dtype=np.float64)
    pts2 = np.ascontiguousarray(feat2.keypoints_xy[matches[:, 1]], dtype=np.float64)
    return pts1, pts2


def test_two_view_pose_recovers_mh01_ground_truth(mh01_matches):
    """Full F → E → (R, t) decomposition on real SLAM imagery."""
    pts1, pts2 = mh01_matches

    pose = K.features.two_view_estimate(
        pts1,
        pts2,
        K_MH01,
        ransac_threshold=1.0,
        max_iterations=2000,
        min_inliers_f=15,
        min_inliers_h=8,
        homography_inlier_ratio=0.8,
        min_parallax_deg=0.5,
        seed=42,
    )

    # Monocular frame-to-frame motion on the MH01 warehouse: general 3D
    # motion, not a planar scene — expect the fundamental model to win.
    assert pose.model_type == "fundamental", f"expected fundamental, got {pose.model_type}"

    # Rotation angle (axis-angle magnitude) — GT 2.7021°, allow 5°.
    angle_deg = _rotation_angle_deg(pose.rotation)
    assert abs(angle_deg - GT_ROT_DEG) < 5.0, (
        f"rotation error too large: estimated {angle_deg:.2f}°, GT {GT_ROT_DEG}°"
    )

    # Translation direction — GT [0.242, -0.233, 0.942], allow 15°.
    t_err_deg = _translation_direction_error_deg(pose.translation, GT_T_DIR)
    assert t_err_deg < 15.0, f"translation direction error too large: {t_err_deg:.2f}°"

    # Cheirality + triangulation should produce at least a few 3D points.
    assert pose.points3d.shape[1] == 3
    assert pose.points3d.shape[0] >= 10, (
        f"expected ≥ 10 triangulated points, got {pose.points3d.shape[0]}"
    )

    # All triangulated points should be in front of the view-1 camera (Z > 0).
    assert (pose.points3d[:, 2] > 0.0).all(), "triangulated points behind view-1 camera"


def test_two_view_pose_return_shapes(mh01_matches):
    """Pin the public output contract: dtypes and shapes."""
    pts1, pts2 = mh01_matches
    pose = K.features.two_view_estimate(pts1, pts2, K_MH01, seed=42)

    assert pose.rotation.shape == (3, 3) and pose.rotation.dtype == np.float64
    assert pose.translation.shape == (3,) and pose.translation.dtype == np.float64
    # Unit-length direction, not metric scale.
    np.testing.assert_allclose(np.linalg.norm(pose.translation), 1.0, atol=1e-9)
    assert pose.model.shape == (3, 3) and pose.model.dtype == np.float64
    assert pose.inliers.shape == (len(pts1),) and pose.inliers.dtype == np.uint8
    assert pose.inlier_count > 0
    assert pose.inlier_indices.dtype == np.int64
    assert pose.points3d.ndim == 2 and pose.points3d.shape[1] == 3
    # Indices must fit within the input correspondence arrays.
    assert (pose.inlier_indices >= 0).all()
    assert (pose.inlier_indices < len(pts1)).all()


def test_two_view_pose_rejects_mismatched_shapes():
    """Shape-validation errors should raise ValueError, not silently produce garbage."""
    pts1 = np.zeros((10, 2), dtype=np.float64)
    pts2 = np.zeros((11, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="matching length"):
        K.features.two_view_estimate(pts1, pts2, K_MH01)

    # (N, 3) instead of (N, 2) — also rejected.
    bad = np.zeros((10, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        K.features.two_view_estimate(bad, bad, K_MH01)


def test_two_view_pose_rejects_bad_intrinsics():
    """A non-3×3 intrinsics matrix should raise ValueError."""
    pts1 = np.zeros((20, 2), dtype=np.float64)
    pts2 = np.zeros((20, 2), dtype=np.float64)
    bad_k = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        K.features.two_view_estimate(pts1, pts2, bad_k)
