"""End-to-end ORB quality test: homography round-trip reprojection error.

Takes a real image, warps it with a known homography, runs the full ORB
pipeline (detect → describe → match → RANSAC homography) on both copies,
and measures how well the estimated homography matches the ground truth
via corner reprojection error in pixels.

This is the real-world quality metric — it reflects how usable the
descriptors are for SLAM/SfM/image registration, which is what ORB is
actually used for. Byte-level parity vs OpenCV is a proxy; this measures
the thing we actually care about.

Tested backends: kornia-rs, OpenCV, VPI (CPU + CUDA if available). We use
OpenCV's BFMatcher on all descriptor sets so the matcher is held constant
and we're comparing only detector+descriptor quality.
"""
import sys
sys.path.insert(0, "/opt/nvidia/vpi3/lib/aarch64-linux-gnu/python")
import cv2
import numpy as np
import kornia_rs as K

try:
    import vpi
    HAVE_VPI = True
except ImportError:
    HAVE_VPI = False


def make_test_homography(w, h, seed=0):
    """Synthetic but realistic homography: rotate + translate + perspective."""
    rng = np.random.default_rng(seed)
    cx, cy = w / 2.0, h / 2.0
    angle_deg = rng.uniform(-15, 15)
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-w * 0.05, w * 0.05)
    ty = rng.uniform(-h * 0.05, h * 0.05)
    # Small perspective skew.
    p0 = rng.uniform(-1e-4, 1e-4)
    p1 = rng.uniform(-1e-4, 1e-4)

    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    a = np.deg2rad(angle_deg)
    R = np.array([[scale * np.cos(a), -scale * np.sin(a), 0],
                  [scale * np.sin(a),  scale * np.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    Tb = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)
    P = np.array([[1, 0, 0], [0, 1, 0], [p0, p1, 1]], dtype=np.float64)
    return Tb @ P @ R @ T


def corner_reproj_error(H_est, H_gt, w, h):
    """Reproject the 4 image corners through H_est and H_gt; return mean L2 px error."""
    if H_est is None:
        return float("inf")
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    pts_est = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
    pts_gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
    return float(np.linalg.norm(pts_est - pts_gt, axis=1).mean())


def kornia_detect(img):
    feat = K.features.orb_detect_and_compute(img)
    xy = np.asarray(feat.keypoints_xy, dtype=np.float32).reshape(-1, 2)
    desc = np.asarray(feat.descriptors, dtype=np.uint8)
    return xy, desc


def opencv_detect(img):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                         firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                         patchSize=31, fastThreshold=20)
    kps, desc = orb.detectAndCompute(img, None)
    xy = np.asarray([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)
    return xy, desc if desc is not None else np.zeros((0, 32), np.uint8)


def vpi_detect(img, backend):
    src = vpi.asimage(img)
    with backend:
        pyr = src.gaussian_pyramid(3)
        corners, descriptors = pyr.orb(intensity_threshold=20,
                                       max_features_per_level=500,
                                       max_pyr_levels=3)
    with corners.rlock_cpu() as c, descriptors.rlock_cpu() as d:
        c_arr = np.asarray(c, dtype=np.float32)
        raw = np.asarray(d)
        desc = raw.view(np.uint8).reshape(len(raw), -1)[:, :32] if raw.dtype.itemsize > 1 else raw.reshape(len(c_arr), 32)
    if c_arr.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
    scale = np.power(2.0, c_arr[:, 2]).astype(np.float32)
    xy = np.stack([c_arr[:, 0] * scale, c_arr[:, 1] * scale], axis=1)
    return xy, desc


def estimate_homography_cv(xy_a, desc_a, xy_b, desc_b, ratio=0.8):
    """OpenCV BFMatcher + Lowe's ratio + cv2.findHomography. Held constant across
    detector backends so we compare detector/descriptor quality, not matcher/solver."""
    if len(desc_a) < 4 or len(desc_b) < 4:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 4:
        return None, 0, len(good)
    src = np.float32([xy_a[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([xy_b[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if mask is None:
        return H, 0, len(good)
    return H, int(mask.sum()), len(good)


def estimate_homography_kornia(xy_a, desc_a, xy_b, desc_b, ratio=0.8):
    """All-kornia-rs path: kornia matcher + kornia find_homography (RANSAC + DLT refit).
    Demonstrates the native pipeline without any OpenCV in the matching stage.
    `method=8` mirrors `cv2.RANSAC` — runs RANSAC then refits H via DLT on the
    full inlier set (LO-RANSAC), which is what cv2.findHomography does."""
    if len(desc_a) < 4 or len(desc_b) < 4:
        return None, 0, 0
    matches = K.features.match_descriptors(
        desc_a, desc_b, cross_check=False, max_ratio=ratio
    )
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


def run_test(img, H_gt, name, detect_fn, estimator=estimate_homography_cv):
    warped = cv2.warpPerspective(img, H_gt, (img.shape[1], img.shape[0]))
    xy_a, desc_a = detect_fn(img)
    xy_b, desc_b = detect_fn(warped)
    H_est, n_inl, n_good = estimator(xy_a, desc_a, xy_b, desc_b)
    err = corner_reproj_error(H_est, H_gt, img.shape[1], img.shape[0])
    print(f"  {name:14s} kps={len(xy_a):4d}/{len(xy_b):4d}  matches={n_good:4d}  "
          f"inliers={n_inl:4d}  reproj_err={err:7.3f} px")
    return err, n_inl


def main():
    img_paths = [
        "/home/nvidia/kornia-rs/tests/data/dog.jpeg",
        "/home/nvidia/kornia-rs/tests/data/mh01_frame1.png",
    ]
    # Each backend = (name, detect_fn, estimator). cv2.BFMatcher + cv2.findHomography
    # is held constant across all detector backends; "kornia-full" swaps both the
    # matcher and the RANSAC solver for the native kornia-rs ones — it's a
    # demo of the full native pipeline, not an apples-to-apples detector comparison.
    backends = [
        ("kornia-rs", kornia_detect, estimate_homography_cv),
        ("kornia-full", kornia_detect, estimate_homography_kornia),
        ("opencv", opencv_detect, estimate_homography_cv),
    ]
    if HAVE_VPI:
        backends.append(("vpi-cpu", lambda img: vpi_detect(img, vpi.Backend.CPU), estimate_homography_cv))
        backends.append(("vpi-cuda", lambda img: vpi_detect(img, vpi.Backend.CUDA), estimate_homography_cv))

    # Multiple random homographies per image to smooth out the RANSAC noise.
    N_TRIALS = 5
    print(f"{'='*80}\nEnd-to-end ORB: homography round-trip reprojection error\n{'='*80}")
    print(f"{N_TRIALS} trials per image × backend; lower err = better.\n")

    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [skip] {path}")
            continue
        h, w = img.shape
        print(f"--- {path.split('/')[-1]} ({w}x{h}) ---")
        results = {name: [] for name, _, _ in backends}
        for seed in range(N_TRIALS):
            H_gt = make_test_homography(w, h, seed=seed)
            print(f"  trial {seed}:")
            for name, fn, est in backends:
                err, inl = run_test(img, H_gt, name, fn, est)
                results[name].append((err, inl))

        print(f"  === aggregate ===")
        print(f"  {'backend':14s} {'mean_err':>10s} {'median_err':>12s} {'max_err':>10s} {'mean_inl':>10s}")
        for name, _, _ in backends:
            errs = np.array([r[0] for r in results[name]])
            inls = np.array([r[1] for r in results[name]])
            print(f"  {name:14s} {errs.mean():>10.3f} {np.median(errs):>12.3f} "
                  f"{errs.max():>10.3f} {inls.mean():>10.1f}")
        print()


if __name__ == "__main__":
    main()
