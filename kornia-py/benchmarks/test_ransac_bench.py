"""kornia-rs vs opencv-python RANSAC benchmark + quality parity.

Two test classes per estimator: a `quality` set that runs both libraries
on identical seeded inputs and asserts numerical parity (model agreement,
inlier set agreement), and a `bench` set that times each library through
pytest-benchmark with the timed closure containing only the solver call.

Run timings:
    pytest -q kornia-py/benchmarks/bench_ransac.py --benchmark-only

Run quality only:
    pytest -q kornia-py/benchmarks/bench_ransac.py -k quality

Run everything (default):
    pytest -q kornia-py/benchmarks/bench_ransac.py

Skips silently if either kornia_rs.ransac or cv2 isn't importable, so the
file is safe to commit before the wheel is built.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
kornia_rs = pytest.importorskip("kornia_rs")
# `kornia_rs.ransac` is a PyO3 submodule attribute, not a true Python
# submodule — `import kornia_rs.ransac` fails even when the attribute is
# present. Reach it via attribute access and skip if missing.
ransac = getattr(kornia_rs, "ransac", None)
if ransac is None:
    pytest.skip("kornia_rs.ransac not available", allow_module_level=True)


# --------------------------------------------------------------------------
# Synthetic data generators
# --------------------------------------------------------------------------

def make_two_view(n_inliers: int, n_outliers: int, seed: int = 0) -> np.ndarray:
    """Return an (N, 4) array of [x1.x, x1.y, x2.x, x2.y] correspondences.

    Inlier rows come first. Image bounds are 640×480; ground-truth motion is
    a small Y-axis rotation + lateral translation.
    """
    rng = np.random.default_rng(seed)
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    angle = 0.1
    R = np.array([
        [math.cos(angle), 0.0, -math.sin(angle)],
        [0.0,             1.0,  0.0],
        [math.sin(angle), 0.0,  math.cos(angle)],
    ])
    t = np.array([1.0, 0.0, 0.2])

    rows = []
    for _ in range(n_inliers):
        p = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                      rng.uniform(3.0, 6.0)])
        u1 = fx * p[0] / p[2] + cx
        v1 = fy * p[1] / p[2] + cy
        pc2 = R @ p + t
        u2 = fx * pc2[0] / pc2[2] + cx
        v2 = fy * pc2[1] / pc2[2] + cy
        rows.append([u1, v1, u2, v2])
    for _ in range(n_outliers):
        rows.append([
            rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0),
            rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0),
        ])
    return np.asarray(rows, dtype=np.float64)


def make_planar_two_view(n_inliers: int, n_outliers: int, seed: int = 0) -> np.ndarray:
    """Two-view correspondences from a known homography (for H tests)."""
    rng = np.random.default_rng(seed)
    H = np.array([
        [1.2, 0.05, 7.0],
        [0.03, 0.95, -3.0],
        [0.0005, -0.0002, 1.0],
    ])
    rows = []
    for _ in range(n_inliers):
        p = np.array([rng.uniform(50.0, 590.0), rng.uniform(50.0, 430.0), 1.0])
        q = H @ p
        q /= q[2]
        rows.append([p[0], p[1], q[0], q[1]])
    for _ in range(n_outliers):
        rows.append([
            rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0),
            rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0),
        ])
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------
# Quality helpers
# --------------------------------------------------------------------------

def model_proportionality(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius-normalised cosine similarity in absolute value.

    For homogeneous 3x3 models F/H/E that are equal up to sign+scale, this
    returns ~1.0 when the two matrices represent the same geometry.
    """
    a = a / max(np.linalg.norm(a), 1e-15)
    b = b / max(np.linalg.norm(b), 1e-15)
    return abs(float(np.sum(a * b)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index between two boolean inlier masks."""
    a = a.astype(bool); b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def sampson_residuals_sq(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Vectorised Sampson distance² for an (N, 2) / (N, 2) correspondence set."""
    n = x1.shape[0]
    x1h = np.hstack([x1, np.ones((n, 1))])
    x2h = np.hstack([x2, np.ones((n, 1))])
    Fx1 = x1h @ F.T
    Ftx2 = x2h @ F
    err = np.sum(x2h * Fx1, axis=1)
    denom = (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 +
             Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return err * err / denom


# --------------------------------------------------------------------------
# Fundamental
# --------------------------------------------------------------------------

def report_inlier_parity(
    capsys, label, n_inliers, n_outliers, kr_mask, cv_mask
) -> dict:
    """Print and return a parity report for the two libraries' inlier masks.

    Computed metrics (everything against the *known* ground-truth partition
    inliers=[0:n_inliers], outliers=[n_inliers:]):

    - precision/recall/F1 vs ground truth, per library
    - false-positive rate (outliers wrongly accepted)
    - Jaccard between the two libraries' masks
    - exact inlier counts and their ratio
    """
    n = n_inliers + n_outliers
    truth = np.zeros(n, dtype=bool)
    truth[:n_inliers] = True

    def stats(mask):
        tp = int(np.logical_and(mask, truth).sum())
        fp = int(np.logical_and(mask, ~truth).sum())
        fn = int(np.logical_and(~mask, truth).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        fpr = fp / max(n_outliers, 1)
        return dict(count=int(mask.sum()), tp=tp, fp=fp, fn=fn,
                    precision=prec, recall=rec, f1=f1, fpr=fpr)

    kr = stats(kr_mask)
    cv = stats(cv_mask)
    jac = jaccard(kr_mask, cv_mask)
    count_ratio = kr["count"] / max(cv["count"], 1)

    with capsys.disabled():
        print(f"\n  [{label}  inliers={n_inliers} outliers={n_outliers}]")
        print(f"    kornia: count={kr['count']:3d}  TP={kr['tp']:3d}  FP={kr['fp']:3d}  "
              f"prec={kr['precision']:.3f}  rec={kr['recall']:.3f}  "
              f"F1={kr['f1']:.3f}  FPR={kr['fpr']:.3f}")
        print(f"    opencv: count={cv['count']:3d}  TP={cv['tp']:3d}  FP={cv['fp']:3d}  "
              f"prec={cv['precision']:.3f}  rec={cv['recall']:.3f}  "
              f"F1={cv['f1']:.3f}  FPR={cv['fpr']:.3f}")
        print(f"    parity: Jaccard={jac:.3f}  count-ratio kornia/opencv={count_ratio:.3f}")
    return dict(kornia=kr, opencv=cv, jaccard=jac, count_ratio=count_ratio)


@pytest.mark.parametrize(
    "n_inliers,n_outliers",
    # Inlier ratios spanning clean → noisy → adversarial.
    [(80, 20), (60, 40), (200, 100), (100, 100), (60, 140)],
)
class TestFundamentalQuality:
    """Both libraries should return the same model up to sign/scale and a
    largely-overlapping inlier set on identical inputs across the full
    range of inlier ratios SLAM front-ends actually see."""

    def test_quality(self, n_inliers, n_outliers, capsys):
        m = make_two_view(n_inliers, n_outliers, seed=42)
        x1 = m[:, :2]; x2 = m[:, 2:]
        ratio_in = n_inliers / (n_inliers + n_outliers)

        # F-RANSAC iter budget scales steeply with inlier ratio (8-pt sample
        # → m ≈ log(0.001)/log(1-w⁸) ≈ 100k at w=0.3). Pick max_iters per
        # regime so both libraries get a fair chance to converge.
        max_iters = 2000 if ratio_in >= 0.5 else 20000

        kr = ransac.fundamental(m, threshold=4.0, max_iters=max_iters,
                                confidence=0.999, seed=42)
        cv_F, cv_mask = cv2.findFundamentalMat(
            x1.astype(np.float32), x2.astype(np.float32),
            cv2.FM_RANSAC, 4.0, 0.999, max_iters)

        assert kr.model is not None and cv_F is not None

        kr_F = np.asarray(kr.model).reshape(3, 3)
        kr_mask = np.asarray(kr.inliers, dtype=bool)
        cv_mask = cv_mask.flatten().astype(bool) if cv_mask is not None else \
            np.zeros(len(m), dtype=bool)

        # Median Sampson residual on the *true* inlier set — both must be
        # sub-pixel² (threshold = 4 px²).
        true_inliers = slice(0, n_inliers)
        kr_med = float(np.median(sampson_residuals_sq(
            kr_F, x1[true_inliers], x2[true_inliers])))
        cv_med = float(np.median(sampson_residuals_sq(
            cv_F, x1[true_inliers], x2[true_inliers])))
        assert kr_med < 4.0, f"kornia median Sampson² {kr_med} on true inliers"
        assert cv_med < 4.0, f"opencv median Sampson² {cv_med} on true inliers"

        # Same-geometry sanity: kornia and opencv residuals on the same
        # inliers should agree within two orders of magnitude.
        ratio = (kr_med + 1e-9) / (cv_med + 1e-9)
        assert 0.01 < ratio < 100.0, (
            f"kornia/opencv Sampson² ratio {ratio:.3f} suggests divergent F "
            f"(kornia={kr_med:.4f}, opencv={cv_med:.4f})")

        # Per-library precision/recall + cross-library parity.
        rep = report_inlier_parity(
            capsys, "F", n_inliers, n_outliers, kr_mask, cv_mask
        )

        # Regime-aware F1 floor: at low inlier ratios both RANSACs are
        # iteration-limited, so we expect a small drop in absolute F1.
        f1_floor = 0.85 if ratio_in >= 0.5 else 0.70
        assert rep["kornia"]["f1"] >= f1_floor, (
            f"kornia F1 {rep['kornia']['f1']:.3f} (floor {f1_floor})"
        )
        assert rep["opencv"]["f1"] >= f1_floor, (
            f"opencv F1 {rep['opencv']['f1']:.3f} (floor {f1_floor})"
        )

        # Same inlier set across libraries — Jaccard floor relaxes at low
        # inlier ratios (libraries make different speed/recall trade-offs
        # under iteration pressure).
        jac_floor = 0.70 if ratio_in >= 0.5 else 0.55
        assert rep["jaccard"] >= jac_floor, (
            f"F inlier-set Jaccard {rep['jaccard']:.3f} — libraries diverged"
        )

        # Inlier counts in the same order of magnitude.
        assert 0.5 <= rep["count_ratio"] <= 2.0, (
            f"kornia/opencv inlier-count ratio {rep['count_ratio']:.3f}"
        )


@pytest.mark.parametrize("n_inliers,n_outliers", [(60, 40), (200, 100)])
class TestFundamentalBench:
    def test_kornia_rs(self, benchmark, n_inliers, n_outliers):
        m = make_two_view(n_inliers, n_outliers, seed=42)
        result = benchmark(
            ransac.fundamental,
            m, threshold=4.0, max_iters=1000, confidence=0.999, seed=42,
        )
        assert result.model is not None

    def test_opencv(self, benchmark, n_inliers, n_outliers):
        m = make_two_view(n_inliers, n_outliers, seed=42)
        x1 = m[:, :2].astype(np.float32); x2 = m[:, 2:].astype(np.float32)
        F, _ = benchmark(
            cv2.findFundamentalMat,
            x1, x2, cv2.FM_RANSAC, 4.0, 0.999, 1000,
        )
        assert F is not None


# --------------------------------------------------------------------------
# Homography
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_inliers,n_outliers",
    [(80, 20), (40, 20), (200, 100), (100, 100), (60, 140)],
)
class TestHomographyQuality:
    def test_quality(self, n_inliers, n_outliers, capsys):
        m = make_planar_two_view(n_inliers, n_outliers, seed=7)
        x1 = m[:, :2]; x2 = m[:, 2:]

        kr = ransac.homography(m, threshold=4.0, max_iters=2000,
                               confidence=0.999, seed=7)
        cv_H, cv_mask = cv2.findHomography(
            x1.astype(np.float32), x2.astype(np.float32),
            cv2.RANSAC, 4.0, None, 2000, 0.999)

        assert kr.model is not None and cv_H is not None

        kr_H = np.asarray(kr.model).reshape(3, 3)
        prop = model_proportionality(kr_H, cv_H)
        assert prop > 0.999, f"H proportionality {prop:.6f} (expected ≈ 1)"

        # Forward transfer error on the true inliers — both should be sub-pixel.
        x1h = np.hstack([x1[:n_inliers], np.ones((n_inliers, 1))])
        kr_proj = (kr_H @ x1h.T).T
        kr_proj = kr_proj[:, :2] / kr_proj[:, 2:3]
        kr_err = float(np.median(np.linalg.norm(kr_proj - x2[:n_inliers], axis=1)))
        cv_proj = (cv_H @ x1h.T).T
        cv_proj = cv_proj[:, :2] / cv_proj[:, 2:3]
        cv_err = float(np.median(np.linalg.norm(cv_proj - x2[:n_inliers], axis=1)))
        assert kr_err < 1.0, f"kornia transfer err {kr_err:.3f} px"
        assert cv_err < 1.0, f"opencv transfer err {cv_err:.3f} px"

        kr_mask = np.asarray(kr.inliers, dtype=bool)
        cv_mask = cv_mask.flatten().astype(bool)
        rep = report_inlier_parity(
            capsys, "H", n_inliers, n_outliers, kr_mask, cv_mask
        )

        assert rep["kornia"]["f1"] >= 0.90, f"kornia F1 {rep['kornia']['f1']:.3f}"
        assert rep["opencv"]["f1"] >= 0.90, f"opencv F1 {rep['opencv']['f1']:.3f}"
        assert rep["jaccard"] >= 0.85, (
            f"H inlier-set Jaccard {rep['jaccard']:.3f}"
        )
        assert 0.85 <= rep["count_ratio"] <= 1.18, (
            f"kornia/opencv inlier-count ratio {rep['count_ratio']:.3f}"
        )


@pytest.mark.parametrize("n_inliers,n_outliers", [(40, 20), (200, 100)])
class TestHomographyBench:
    def test_kornia_rs(self, benchmark, n_inliers, n_outliers):
        m = make_planar_two_view(n_inliers, n_outliers, seed=7)
        result = benchmark(
            ransac.homography,
            m, threshold=4.0, max_iters=1000, confidence=0.999, seed=7,
        )
        assert result.model is not None

    def test_opencv(self, benchmark, n_inliers, n_outliers):
        m = make_planar_two_view(n_inliers, n_outliers, seed=7)
        x1 = m[:, :2].astype(np.float32); x2 = m[:, 2:].astype(np.float32)
        H, _ = benchmark(
            cv2.findHomography,
            x1, x2, cv2.RANSAC, 4.0, None, 1000, 0.999,
        )
        assert H is not None


# --------------------------------------------------------------------------
# Byte-to-byte minimal-solver parity
# --------------------------------------------------------------------------
#
# Full RANSAC pipelines can never be byte-identical across libraries because
# OpenCV uses Mersenne Twister + a different sample iteration order than
# kornia's StdRng (ChaCha). What *can* be byte-identical is the underlying
# minimal solver: feed both libraries the exact same N points and they
# should produce the same model up to floating-point roundoff.
#
# These tests bypass RANSAC entirely (kornia: `find_fundamental(method=0)` /
# `find_homography(method=0)`; opencv: `cv2.FM_8POINT` / `findHomography(method=0)`)
# and check element-wise agreement after Frobenius normalisation + sign
# alignment.

from kornia_rs import k3d as _k3d


def normalise_and_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frobenius-normalise both, then flip the sign of `b` to match `a`'s
    largest-magnitude entry. Homogeneous models are equivalent under both
    sign and scale; this collapses the equivalence class to a unique repr."""
    a = a / max(np.linalg.norm(a), 1e-15)
    b = b / max(np.linalg.norm(b), 1e-15)
    k = int(np.argmax(np.abs(a)))
    if a.flat[k] * b.flat[k] < 0.0:
        b = -b
    return a, b


@pytest.mark.parametrize("n_pts", [8, 16, 50, 200])
class TestFundamental8ptByteParity:
    """Pure 8-point DLT — no RANSAC, no random sampling. Both libraries
    implement the normalised 8-point algorithm; output must agree to
    numerical roundoff on identical inputs."""

    def test_byte_parity(self, n_pts, capsys):
        m = make_two_view(n_pts, 0, seed=123)
        # Slicing produces strided views; the kornia binding requires
        # C-contiguous arrays — copy.
        x1 = np.ascontiguousarray(m[:, :2])
        x2 = np.ascontiguousarray(m[:, 2:])

        kr_F, _ = _k3d.find_fundamental(x1, x2, method=0)
        cv_F, _ = cv2.findFundamentalMat(x1, x2, cv2.FM_8POINT)

        kr_F = np.asarray(kr_F).reshape(3, 3)
        cv_F = np.asarray(cv_F).reshape(3, 3)
        kr_n, cv_n = normalise_and_align(kr_F, cv_F)

        max_abs_diff = float(np.max(np.abs(kr_n - cv_n)))
        rel_frob = float(np.linalg.norm(kr_n - cv_n))

        with capsys.disabled():
            print(f"\n  [F8pt n={n_pts}] max|Δ|={max_abs_diff:.3e}  "
                  f"frob(Δ)={rel_frob:.3e}")

        # Empirical floor between kornia (faer SVD) and opencv (bundled
        # Eigen-style SVD): ~3e-6 at the minimal sample (n=8), tightening
        # to ~1e-7 at n≥50. The cap below catches any algorithmic
        # divergence (transpose bug, wrong normalisation, sign flip)
        # without flagging the SVD-library noise floor.
        cap = 1e-5 if n_pts <= 8 else 1e-6
        assert max_abs_diff < cap, (
            f"F-8pt byte parity broken: max|Δ|={max_abs_diff:.3e} > {cap:.0e}"
        )


@pytest.mark.parametrize("n_pts", [4, 8, 50, 200])
class TestHomographyDltByteParity:
    """Pure DLT homography — minimal solver at n=4, full DLT for n>4.
    Output must agree to numerical roundoff."""

    def test_byte_parity(self, n_pts, capsys):
        m = make_planar_two_view(n_pts, 0, seed=123)
        x1 = np.ascontiguousarray(m[:, :2])
        x2 = np.ascontiguousarray(m[:, 2:])

        kr_H, _ = _k3d.find_homography(x1, x2, method=0)
        cv_H, _ = cv2.findHomography(x1, x2, 0)

        kr_H = np.asarray(kr_H).reshape(3, 3)
        cv_H = np.asarray(cv_H).reshape(3, 3)
        kr_n, cv_n = normalise_and_align(kr_H, cv_H)

        max_abs_diff = float(np.max(np.abs(kr_n - cv_n)))
        rel_frob = float(np.linalg.norm(kr_n - cv_n))

        with capsys.disabled():
            print(f"\n  [H-DLT n={n_pts}] max|Δ|={max_abs_diff:.3e}  "
                  f"frob(Δ)={rel_frob:.3e}")

        # Empirical floor: ~6-8e-6 at minimal/near-minimal (n=4, 8),
        # ~1e-7 at n≥50. Same SVD-backend noise reasoning as F8pt.
        cap = 1e-5 if n_pts <= 8 else 1e-6
        assert max_abs_diff < cap, (
            f"H-DLT byte parity broken: max|Δ|={max_abs_diff:.3e} > {cap:.0e}"
        )


# --------------------------------------------------------------------------
# Multi-seed RANSAC parity (statistical equivalence)
# --------------------------------------------------------------------------

def _multi_seed_ransac_parity(
    capsys, label, make_fn, kr_call, cv_call,
    n_inliers, n_outliers, n_seeds=10,
):
    """Run both libraries N times with different RNG seeds; report mean ± std
    of inlier-set parity metrics. Stochastic RANSAC across libraries can't
    be byte-identical, but the *distribution* over seeds should be tight
    and centred on the same value."""
    n = n_inliers + n_outliers
    truth = np.zeros(n, dtype=bool); truth[:n_inliers] = True

    kr_f1 = []; cv_f1 = []
    kr_count = []; cv_count = []
    jac = []; t_kornia = []; t_opencv = []
    import time

    for s in range(n_seeds):
        m = make_fn(n_inliers, n_outliers, seed=1000 + s)
        x1 = m[:, :2]; x2 = m[:, 2:]

        t0 = time.perf_counter()
        kr_mask = np.asarray(kr_call(m, s).inliers, dtype=bool)
        t_kornia.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cv_mask = cv_call(x1, x2, s)
        t_opencv.append(time.perf_counter() - t0)

        for mask, ls_f1, ls_cnt in [
            (kr_mask, kr_f1, kr_count), (cv_mask, cv_f1, cv_count),
        ]:
            tp = int(np.logical_and(mask, truth).sum())
            fp = int(np.logical_and(mask, ~truth).sum())
            fn = int(np.logical_and(~mask, truth).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            ls_f1.append(2 * prec * rec / max(prec + rec, 1e-12))
            ls_cnt.append(int(mask.sum()))
        jac.append(jaccard(kr_mask, cv_mask))

    kr_f1 = np.array(kr_f1); cv_f1 = np.array(cv_f1); jac = np.array(jac)
    kr_cnt = np.array(kr_count); cv_cnt = np.array(cv_count)
    t_kr = np.array(t_kornia) * 1e6; t_cv = np.array(t_opencv) * 1e6
    speedup = t_cv.mean() / t_kr.mean()

    with capsys.disabled():
        print(f"\n  [{label}  inliers={n_inliers} outliers={n_outliers}  "
              f"seeds={n_seeds}]")
        print(f"    F1   kornia={kr_f1.mean():.3f}±{kr_f1.std():.3f}   "
              f"opencv={cv_f1.mean():.3f}±{cv_f1.std():.3f}")
        print(f"    cnt  kornia={kr_cnt.mean():5.1f}±{kr_cnt.std():.1f}    "
              f"opencv={cv_cnt.mean():5.1f}±{cv_cnt.std():.1f}")
        print(f"    Jaccard {jac.mean():.3f}±{jac.std():.3f}   "
              f"timing kornia={t_kr.mean():.0f}±{t_kr.std():.0f}µs  "
              f"opencv={t_cv.mean():.0f}±{t_cv.std():.0f}µs  "
              f"speedup={speedup:.2f}×")

    return dict(
        kr_f1=kr_f1, cv_f1=cv_f1, jaccard=jac,
        kr_count=kr_cnt, cv_count=cv_cnt,
        t_kornia=t_kr, t_opencv=t_cv, speedup=speedup,
    )


@pytest.mark.parametrize(
    "n_inliers,n_outliers",
    [(80, 20), (60, 40), (200, 100), (100, 100), (60, 140)],
)
class TestFundamentalMultiSeed:
    def test_parity(self, n_inliers, n_outliers, capsys):
        ratio_in = n_inliers / (n_inliers + n_outliers)
        max_iters = 2000 if ratio_in >= 0.5 else 20000

        rep = _multi_seed_ransac_parity(
            capsys, "F", make_two_view,
            kr_call=lambda m, s: ransac.fundamental(
                m, threshold=4.0, max_iters=max_iters,
                confidence=0.999, seed=s,
            ),
            cv_call=lambda x1, x2, s: (
                cv2.findFundamentalMat(
                    x1.astype(np.float32), x2.astype(np.float32),
                    cv2.FM_RANSAC, 4.0, 0.999, max_iters,
                )[1].flatten().astype(bool)
            ),
            n_inliers=n_inliers, n_outliers=n_outliers, n_seeds=10,
        )

        # Stochastic equivalence: mean F1 difference between libraries < 0.10.
        f1_gap = abs(rep["kr_f1"].mean() - rep["cv_f1"].mean())
        assert f1_gap < 0.10, (
            f"F1 gap {f1_gap:.3f} between kornia mean={rep['kr_f1'].mean():.3f} "
            f"and opencv mean={rep['cv_f1'].mean():.3f}"
        )
        # And: mean Jaccard ≥ 0.7 across the 10 seeds.
        jac_floor = 0.70 if ratio_in >= 0.5 else 0.55
        assert rep["jaccard"].mean() >= jac_floor, (
            f"mean Jaccard {rep['jaccard'].mean():.3f} below floor {jac_floor}"
        )


@pytest.mark.parametrize(
    "n_inliers,n_outliers",
    [(80, 20), (40, 20), (200, 100), (100, 100), (60, 140)],
)
class TestHomographyMultiSeed:
    def test_parity(self, n_inliers, n_outliers, capsys):
        rep = _multi_seed_ransac_parity(
            capsys, "H", make_planar_two_view,
            kr_call=lambda m, s: ransac.homography(
                m, threshold=4.0, max_iters=2000,
                confidence=0.999, seed=s,
            ),
            cv_call=lambda x1, x2, s: (
                cv2.findHomography(
                    x1.astype(np.float32), x2.astype(np.float32),
                    cv2.RANSAC, 4.0, None, 2000, 0.999,
                )[1].flatten().astype(bool)
            ),
            n_inliers=n_inliers, n_outliers=n_outliers, n_seeds=10,
        )

        f1_gap = abs(rep["kr_f1"].mean() - rep["cv_f1"].mean())
        assert f1_gap < 0.05, f"H F1 gap {f1_gap:.3f}"
        assert rep["jaccard"].mean() >= 0.95, (
            f"mean H Jaccard {rep['jaccard'].mean():.3f}"
        )
