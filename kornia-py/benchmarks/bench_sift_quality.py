"""SIFT speed AND matching quality: kornia-rs (CUDA + NEON) vs OpenCV.

Run:  python3 kornia-py/benchmarks/bench_sift_quality.py

Why two geometric tests
-----------------------
Keypoint counts and wall time say nothing about whether the descriptors are
any good, and a detector can be made arbitrarily fast by being wrong. These
two probe different failure modes:

  * **Homography** — warp a planar image by a known H. Every correct match is
    by construction consistent with that H, so this measures descriptor
    invariance under rotation and scale. It says nothing about 3D parallax.
  * **Epipolar** — a real stereo-motion pair (EuRoC MH01, consecutive frames).
    No homography exists; correspondences satisfy only the epipolar
    constraint. Scored by symmetric epipolar distance under a RANSAC
    fundamental matrix, which is the geometry SfM and VO actually rely on.

A backend can look fine on one and fail the other. An early version of this
port scored a healthy median epipolar error while returning a fifth of
OpenCV's homography matches — the descriptors for every octave but the first
were zero, and a zero descriptor is equidistant from everything, so the ratio
test *rejected* rather than mismatched. Reporting both is what made that
visible.

Threading
---------
Both engines are reported at one thread and at all cores, because the two
answer different questions and quoting only one is how a comparison goes
wrong. The single-thread column is the *kernel* diagnostic: it isolates the
arithmetic from the scheduler. The all-core column is what a caller actually
gets. Comparing one engine's all-core figure against another's single-thread
figure — which this harness originally did, via a `setNumThreads(1)` left in
from the first measurement — flatters the parallel one by whatever its
scaling happens to be.

Both engines parallelise, and by similar amounts: on this host OpenCV scales
2.15x across six cores and this one 3.7x. Because rayon's pool size is fixed
when the process starts, only the row matching `RAYON_NUM_THREADS` is a
like-for-like comparison; the others are printed to show OpenCV's scaling
curve, and their ratios are deliberately left blank rather than reported as
if they meant something.

    for n in 1 2 4 6; do RAYON_NUM_THREADS=$n python3 <this file> ; done

Columns
-------
  kp        keypoints on frame 1
  ms        median detect+describe wall time
  H match   mutual-NN matches surviving the ratio test, summed over 4 warps
  H ok      of those, reprojecting within 3 px under the known H
  F match   matches on the stereo pair
  F inl     RANSAC fundamental-matrix inliers
  inl%      F inl / F match
  sed       median symmetric epipolar distance of the inliers, px

Matching is cv2's BFMatcher for every engine, so the column compares
descriptors rather than matchers. `bench_sift_matchers` below compares the
matcher implementations against each other on identical descriptors.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

import kornia_rs as K

RATIO, PX_OK = 0.8, 3.0
ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "tests" / "data" / name
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"missing test image: {p}")
    return img


def _match(d1, d2):
    """Mutual nearest neighbour with Lowe's ratio test."""
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.empty((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < RATIO * n.distance]
    rev = bf.knnMatch(d2, d1, k=1)
    back = {r[0].queryIdx: r[0].trainIdx for r in rev if r}
    return np.array(
        [[m.queryIdx, m.trainIdx] for m in good if back.get(m.trainIdx, -1) == m.queryIdx],
        dtype=int,
    )


def _timed(fn, reps=5):
    fn()
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2]


def thread_scaling(img):
    """cv2 across thread counts, against this process's fixed rayon pool.

    Only the row whose thread count equals the rayon pool is a real
    comparison — see the module docstring for the sweep loop.
    """
    sift = cv2.SIFT_create()
    arr = np.ascontiguousarray(img.astype(np.float32)[..., None])
    detector = K.imgproc.Sift()

    pool = int(os.environ.get("RAYON_NUM_THREADS") or (os.cpu_count() or 1))
    neon_ms = _timed(lambda: detector.detect_and_compute(arr))
    ncpu = cv2.getNumberOfCPUs()

    print(f"\nthread scaling (mh01_frame1, fo=-1) — rayon pool = {pool}")
    print(f"{'threads':<10}{'opencv ms':>12}{'kornia ms':>12}{'ratio':>9}")
    for nt in sorted({1, 2, 4, ncpu}):
        cv2.setNumThreads(nt)
        cv_ms = _timed(lambda: sift.detectAndCompute(img, None))
        # A ratio is only meaningful where both engines have the same cores.
        if nt == pool:
            print(f"{nt:<10}{cv_ms:>12.1f}{neon_ms:>12.1f}{cv_ms / neon_ms:>8.2f}x")
        else:
            print(f"{nt:<10}{cv_ms:>12.1f}{'-':>12}{'-':>9}")
    cv2.setNumThreads(1)


def homography_audit(img, detect):
    """Correct-match count over four known warps."""
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    hs = {}
    for a in (15, 30):
        hs[f"rot{a}"] = np.vstack([cv2.getRotationMatrix2D(c, a, 1.0), [0, 0, 1]])
    for s in (0.8, 0.6):
        hs[f"scale{s}"] = np.array(
            [[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)], [0, 0, 1]]
        )
    tot_m = tot_ok = 0
    p1, d1 = detect(img)
    for mat in hs.values():
        warp = cv2.warpPerspective(img, mat, (w, h))
        p2, d2 = detect(warp)
        ms = _match(d1, d2)
        if len(ms) == 0:
            continue
        src = np.hstack([p1[ms[:, 0]], np.ones((len(ms), 1), np.float32)])
        proj = (mat @ src.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - p2[ms[:, 1]], axis=1)
        tot_m += len(ms)
        tot_ok += int((err < PX_OK).sum())
    return tot_m, tot_ok


def epipolar_audit(img_a, img_b, detect):
    """Inliers under a RANSAC fundamental matrix, plus symmetric epipolar error."""
    p1, d1 = detect(img_a)
    p2, d2 = detect(img_b)
    ms = _match(d1, d2)
    if len(ms) < 8:
        return len(ms), 0, float("nan")
    a, b = p1[ms[:, 0]], p2[ms[:, 1]]
    f, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0, 0.999)
    if f is None or mask is None:
        return len(ms), 0, float("nan")
    inl = mask.ravel().astype(bool)
    ah = np.hstack([a, np.ones((len(a), 1), np.float32)])
    bh = np.hstack([b, np.ones((len(b), 1), np.float32)])
    fa = (f @ ah.T).T  # epipolar lines in image B
    ftb = (f.T @ bh.T).T  # epipolar lines in image A
    num = np.sum(bh * fa, axis=1) ** 2
    den = fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2
    sed = np.sqrt(num / np.maximum(den, 1e-12))
    med = float(np.median(sed[inl])) if inl.any() else float("nan")
    return len(ms), int(inl.sum()), med


def _engines(stream):
    """(name, detect) pairs. `detect` returns (points Nx2, descriptors Nx128)."""
    cv_sift = cv2.SIFT_create()

    def cv_detect(img):
        k, d = cv_sift.detectAndCompute(img, None)
        return np.array([[p.pt[0], p.pt[1]] for p in k], dtype=np.float32), d

    cache = {}

    def kornia_detect(img, device, **kw):
        # One detector per configuration, reused: it owns the pipeline's
        # scratch and rebuilding it per call would dominate the measurement.
        key = (device, tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = K.imgproc.Sift(**kw)
        arr = np.ascontiguousarray(img.astype(np.float32)[..., None])
        src = K.image.Image.from_numpy(arr).to_cuda(stream) if device else arr
        kp, desc = cache[key].detect_and_compute(src)
        return np.ascontiguousarray(kp[:, :2]), np.ascontiguousarray(desc)

    # OpenCV at both extremes: one thread isolates kernel quality, all cores is
    # what a caller actually gets. Quoting only the first flatters us by cv2's
    # scaling factor, which is how this table originally read.
    def cv_detect_mt(img):
        cv2.setNumThreads(cv2.getNumberOfCPUs())
        try:
            return cv_detect(img)
        finally:
            cv2.setNumThreads(1)

    out = [("opencv (1 thread)", cv_detect), ("opencv (all cores)", cv_detect_mt)]
    if stream is not None:
        out += [
            ("cuda fo=-1", lambda im: kornia_detect(im, True)),
            ("cuda fo=-1 fast", lambda im: kornia_detect(im, True, fast_descriptor=True)),
            ("cuda fo=0 4oct", lambda im: kornia_detect(im, True, upsample=False, max_octaves=4)),
        ]
    out += [
        ("neon fo=-1", lambda im: kornia_detect(im, False)),
        ("neon fo=0 4oct", lambda im: kornia_detect(im, False, upsample=False, max_octaves=4)),
    ]
    return out


def bench_sift_matchers(stream):
    """Matcher implementations on identical descriptors, so this isolates the
    matcher rather than the descriptors feeding it."""
    img = _load("mh01_frame1.png")
    other = _load("mh01_frame2.png")
    sift = K.imgproc.Sift()
    a = np.ascontiguousarray(img.astype(np.float32)[..., None])
    b = np.ascontiguousarray(other.astype(np.float32)[..., None])
    _, da = sift.detect_and_compute(a)
    _, db = sift.detect_and_compute(b)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    cv_pairs = {tuple(x) for x in _match(da, db)}
    neon_pairs = {tuple(x) for x in sift.match(a, b)[2]}
    agree = cv_pairs == neon_pairs
    print(f"\nmatcher pair sets identical (neon vs cv2 BFMatcher): {agree}  "
          f"({len(neon_pairs)} pairs)")
    if stream is not None:
        da_d = K.image.Image.from_numpy(a).to_cuda(stream)
        db_d = K.image.Image.from_numpy(b).to_cuda(stream)
        cuda_pairs = {tuple(x) for x in sift.match(da_d, db_d)[2]}
        print(f"matcher pair sets identical (cuda vs cv2 BFMatcher): "
              f"{cuda_pairs == cv_pairs}  ({len(cuda_pairs)} pairs)")
    del bf


def main():
    try:
        stream = K.cuda.Stream.default()
    except Exception:  # noqa: BLE001 - CPU-only build or no device
        stream = None
        print("no CUDA device; benchmarking the CPU path only")

    img_a = _load("mh01_frame1.png")
    img_b = _load("mh01_frame2.png")
    cv2.setNumThreads(1)

    print(f"quality table: kornia at its rayon pool "
          f"({os.environ.get('RAYON_NUM_THREADS') or os.cpu_count()} threads); "
          f"opencv reported at 1 thread and at all cores")
    hdr = (f"{'engine':<20}{'kp':>7}{'ms':>9}{'H match':>9}{'H ok':>7}"
           f"{'F match':>9}{'F inl':>7}{'inl%':>7}{'sed':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name, detect in _engines(stream):
        kp, _ = detect(img_a)
        ms = _timed(lambda d=detect: d(img_a))
        hm, ho = homography_audit(img_a, detect)
        fm, fi, sed = epipolar_audit(img_a, img_b, detect)
        pct = 100 * fi / fm if fm else 0.0
        print(f"{name:<20}{len(kp):>7}{ms:>8.1f}{hm:>9}{ho:>7}"
              f"{fm:>9}{fi:>7}{pct:>6.1f}%{sed:>7.2f}")

    bench_sift_matchers(stream)
    thread_scaling(img_a)


if __name__ == "__main__":
    main()
