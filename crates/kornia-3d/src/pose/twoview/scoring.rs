//! SoA residual scoring and architecture-specific kernels.
//!
//! Two-view scoring uses an inclusive squared-error cutoff and compares
//! inlier count first, then residual sum. Kernels preserve lane grouping and
//! accumulation order. The generic consensus API has a different contract.

#![allow(clippy::needless_range_loop)]

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Borrowed structure-of-arrays coordinates for a validated correspondence set.
///
/// All four slices have equal length. The model-specific wrappers construct
/// these from paired inputs after validating the number of correspondences.
/// Scoring owns mask initialization, so callers cannot retain stale inliers.
pub(super) struct ScoringPoints<'a> {
    x1_x: &'a [f64],
    x1_y: &'a [f64],
    x2_x: &'a [f64],
    x2_y: &'a [f64],
}

impl<'a> ScoringPoints<'a> {
    /// Group the four equally sized coordinate slices without allocation.
    pub(super) fn new(x1_x: &'a [f64], x1_y: &'a [f64], x2_x: &'a [f64], x2_y: &'a [f64]) -> Self {
        debug_assert_eq!(x1_x.len(), x1_y.len());
        debug_assert_eq!(x1_x.len(), x2_x.len());
        debug_assert_eq!(x1_x.len(), x2_y.len());
        Self {
            x1_x,
            x1_y,
            x2_x,
            x2_y,
        }
    }

    /// Score a fundamental matrix and completely overwrite the caller's mask.
    #[inline]
    pub(super) fn fundamental(
        &self,
        model: &Mat3F64,
        threshold_sq: f64,
        mask: &mut [bool],
    ) -> (usize, f64) {
        debug_assert_eq!(mask.len(), self.x1_x.len());
        mask.fill(false);
        score_inliers_f(
            model,
            self.x1_x,
            self.x1_y,
            self.x2_x,
            self.x2_y,
            threshold_sq,
            mask,
        )
    }

    /// Score a homography and completely overwrite the caller's mask.
    #[inline]
    pub(super) fn homography(
        &self,
        model: &Mat3F64,
        threshold_sq: f64,
        mask: &mut [bool],
    ) -> (usize, f64) {
        debug_assert_eq!(mask.len(), self.x1_x.len());
        mask.fill(false);
        score_inliers_h(
            model,
            self.x1_x,
            self.x1_y,
            self.x2_x,
            self.x2_y,
            threshold_sq,
            mask,
        )
    }

    /// Return only strict improvements, with a complete replacement mask.
    ///
    /// Bounded scoring is the F7 policy; F8 retains its full-scoring path.
    /// At high incumbent support, compute the mask on the first pass. Otherwise
    /// losing roots avoid mask traffic and only a winner materializes its mask.
    /// A rejection may leave `mask` unchanged or partial; callers ignore it.
    #[inline]
    pub(super) fn fundamental_candidate<const BOUNDED: bool>(
        &self,
        model: &Mat3F64,
        threshold_sq: f64,
        incumbent: (usize, f64),
        mask: &mut [bool],
    ) -> Option<(usize, f64)> {
        let (best_count, best_score) = incumbent;
        let masked = BOUNDED && best_count >= self.x1_x.len() - self.x1_x.len() / 4;
        let (count, score) = if masked {
            mask.fill(false);
            score_inliers_f_bounded_masked(
                model,
                self.x1_x,
                self.x1_y,
                self.x2_x,
                self.x2_y,
                threshold_sq,
                mask,
                best_count,
                best_score,
            )?
        } else if BOUNDED {
            score_inliers_f_bounded_count(
                model,
                self.x1_x,
                self.x1_y,
                self.x2_x,
                self.x2_y,
                threshold_sq,
                best_count,
                best_score,
            )?
        } else {
            self.fundamental(model, threshold_sq, mask)
        };
        if !(count > best_count || (count == best_count && score < best_score)) {
            return None;
        }
        if BOUNDED && !masked {
            let materialized = self.fundamental(model, threshold_sq, mask);
            debug_assert_eq!(materialized.0, count);
            debug_assert_eq!(materialized.1.to_bits(), score.to_bits());
        }
        Some((count, score))
    }
}

/// Twice the signed area of the triangle (a, b, c) — zero iff collinear.
#[inline]
fn triangle_area2(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// True if any 3 of the 4 points are (near-)collinear.
///
/// Uses a scale-aware threshold: points spread over a 100-px patch and a 10-px
/// patch should both survive unless they're genuinely collinear. The threshold
/// scales with the sample's bounding-box area to avoid false positives at small
/// scales and false negatives at large scales.
pub(super) fn sample_is_degenerate(pts: &[[f64; 2]; 4]) -> bool {
    let mut xmin = pts[0][0];
    let mut xmax = pts[0][0];
    let mut ymin = pts[0][1];
    let mut ymax = pts[0][1];
    for p in pts.iter().skip(1) {
        if p[0] < xmin {
            xmin = p[0];
        }
        if p[0] > xmax {
            xmax = p[0];
        }
        if p[1] < ymin {
            ymin = p[1];
        }
        if p[1] > ymax {
            ymax = p[1];
        }
    }
    let span = (xmax - xmin).max(ymax - ymin).max(1.0);
    // Area threshold: 1% of the sample's bounding square. Well below any
    // non-degenerate sample but reliably nonzero for genuine triangles.
    let eps = 0.01 * span * span;
    let triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];
    for (i, j, k) in triples {
        if triangle_area2(&pts[i], &pts[j], &pts[k]).abs() < eps {
            return true;
        }
    }
    false
}

/// Computes the squared reprojection error for mapping `x1` to `x2` via the homography `h`.
pub(super) fn homography_reproj_error(h: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
    let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
    let hx = *h * x1h;
    if hx.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    let u = hx.x / hx.z;
    let v = hx.y / hx.z;
    let dx = u - x2.x;
    let dy = v - x2.y;
    dx * dx + dy * dy
}

/// Batched H reprojection scorer. For N correspondences in SoA layout and one
/// candidate homography, returns (inlier_count, score_sum) and populates
/// `inliers`. Bypasses `glam::DMat3 * DVec3` (scalar 9-mul per-call) with a
/// 2-lane f64 NEON FMA chain on aarch64. One-time SoA conversion at RANSAC
/// entry makes the per-iteration cost dominated by FMA + 2× `vdivq_f64`.
#[inline]
pub(super) fn score_inliers_h(
    h: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
) -> (usize, f64) {
    let n = x1_x.len();
    // Column-major glam::DMat3: row-r, col-c = {x,y,z}_axis[r] at column c.
    let a = h.x_axis.x;
    let b = h.y_axis.x;
    let c = h.z_axis.x;
    let d = h.x_axis.y;
    let e = h.y_axis.y;
    let f = h.z_axis.y;
    let g = h.x_axis.z;
    let hh = h.y_axis.z;
    let ii = h.z_axis.z;

    let mut count = 0usize;
    let mut score = 0.0f64;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut idx = 0usize;

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64. Validated SoA slices and mask
    // have equal lengths; the kernel bounds every vector load and store.
    let mut idx = unsafe {
        score_inliers_h_neon(
            (a, b, c, d, e, f, g, hh, ii),
            x1_x,
            x1_y,
            x2_x,
            x2_y,
            thresh_sq,
            inliers,
            &mut count,
            &mut score,
        )
    };

    #[cfg(target_arch = "x86_64")]
    let mut idx = if kornia_imgproc::simd::cpu_features().has_avx2
        && kornia_imgproc::simd::cpu_features().has_fma
    {
        // SAFETY: AVX2/FMA support is runtime checked. Equally sized SoA
        // slices and mask satisfy the kernel's bounded access requirements.
        unsafe {
            score_inliers_h_avx2(
                (a, b, c, d, e, f, g, hh, ii),
                x1_x,
                x1_y,
                x2_x,
                x2_y,
                thresh_sq,
                inliers,
                &mut count,
                &mut score,
            )
        }
    } else {
        0usize
    };

    // Scalar tail (and full fallback on non-aarch64).
    while idx < n {
        let x = x1_x[idx];
        let y = x1_y[idx];
        let hw = g * x + hh * y + ii;
        if hw.abs() >= 1e-12 {
            let u = (a * x + b * y + c) / hw;
            let v = (d * x + e * y + f) / hw;
            let dx = u - x2_x[idx];
            let dy = v - x2_y[idx];
            let dd = dx * dx + dy * dy;
            if dd <= thresh_sq {
                inliers[idx] = true;
                count += 1;
                score += dd;
            }
        }
        idx += 1;
    }
    (count, score)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_h_neon(
    coeffs: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    // SAFETY: The caller guarantees NEON and equally sized SoA/mask slices.
    // Vector accesses require at least two remaining entries; the loop checks
    // that bound and stores only those lanes. Scalar tails stay with the caller.
    unsafe {
        use std::arch::aarch64::*;
        let (a, b, c, d, e, f, g, hh, ii) = coeffs;
        let a_v = vdupq_n_f64(a);
        let b_v = vdupq_n_f64(b);
        let c_v = vdupq_n_f64(c);
        let d_v = vdupq_n_f64(d);
        let e_v = vdupq_n_f64(e);
        let f_v = vdupq_n_f64(f);
        let g_v = vdupq_n_f64(g);
        let h_v = vdupq_n_f64(hh);
        let i_v = vdupq_n_f64(ii);
        let n = x1_x.len();
        let mut idx = 0usize;
        while idx + 2 <= n {
            let x1 = vld1q_f64(x1_x.as_ptr().add(idx));
            let y1 = vld1q_f64(x1_y.as_ptr().add(idx));
            let x2 = vld1q_f64(x2_x.as_ptr().add(idx));
            let y2 = vld1q_f64(x2_y.as_ptr().add(idx));
            // hx = a*x + b*y + c ; hy = d*x + e*y + f ; hw = g*x + h*y + i
            let hx = vfmaq_f64(vfmaq_f64(c_v, x1, a_v), y1, b_v);
            let hy = vfmaq_f64(vfmaq_f64(f_v, x1, d_v), y1, e_v);
            let hw = vfmaq_f64(vfmaq_f64(i_v, x1, g_v), y1, h_v);
            let u = vdivq_f64(hx, hw);
            let v = vdivq_f64(hy, hw);
            let dx = vsubq_f64(u, x2);
            let dy = vsubq_f64(v, y2);
            let sq = vfmaq_f64(vmulq_f64(dx, dx), dy, dy);
            let mut buf = [0.0f64; 2];
            vst1q_f64(buf.as_mut_ptr(), sq);
            // Commit per-lane (scalar reduction — count++/score+= are data-dependent).
            for k in 0..2 {
                let dd = buf[k];
                if dd.is_finite() && dd <= thresh_sq {
                    *inliers.get_unchecked_mut(idx + k) = true;
                    *count += 1;
                    *score += dd;
                }
            }
            idx += 2;
        }
        idx
    }
}

/// AVX2 mirror of [`score_inliers_h_neon`]. Same H-reprojection math, but
/// 4-lane f64 (`__m256d`) — twice NEON's 2-lane width, halving inner-loop
/// iteration count. `_mm256_fmadd_pd` covers `vfmaq_f64` exactly; the
/// per-lane scalar commit is unchanged because count++/score+= are
/// data-dependent regardless of vector width.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_h_avx2(
    coeffs: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    use std::arch::x86_64::*;
    let (a, b, c, d, e, f, g, hh, ii) = coeffs;
    let a_v = _mm256_set1_pd(a);
    let b_v = _mm256_set1_pd(b);
    let c_v = _mm256_set1_pd(c);
    let d_v = _mm256_set1_pd(d);
    let e_v = _mm256_set1_pd(e);
    let f_v = _mm256_set1_pd(f);
    let g_v = _mm256_set1_pd(g);
    let h_v = _mm256_set1_pd(hh);
    let i_v = _mm256_set1_pd(ii);
    let n = x1_x.len();
    let mut idx = 0usize;
    while idx + 4 <= n {
        let x1 = _mm256_loadu_pd(x1_x.as_ptr().add(idx));
        let y1 = _mm256_loadu_pd(x1_y.as_ptr().add(idx));
        let x2 = _mm256_loadu_pd(x2_x.as_ptr().add(idx));
        let y2 = _mm256_loadu_pd(x2_y.as_ptr().add(idx));
        let hx = _mm256_fmadd_pd(y1, b_v, _mm256_fmadd_pd(x1, a_v, c_v));
        let hy = _mm256_fmadd_pd(y1, e_v, _mm256_fmadd_pd(x1, d_v, f_v));
        let hw = _mm256_fmadd_pd(y1, h_v, _mm256_fmadd_pd(x1, g_v, i_v));
        let u = _mm256_div_pd(hx, hw);
        let v = _mm256_div_pd(hy, hw);
        let dx = _mm256_sub_pd(u, x2);
        let dy = _mm256_sub_pd(v, y2);
        let sq = _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx));
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), sq);
        for k in 0..4 {
            let dd = buf[k];
            if dd.is_finite() && dd <= thresh_sq {
                *inliers.get_unchecked_mut(idx + k) = true;
                *count += 1;
                *score += dd;
            }
        }
        idx += 4;
    }
    idx
}

/// Batched Sampson scorer for F. N correspondences × 1 model. Same structure
/// as `score_inliers_h` — 2-lane f64 NEON FMA path on aarch64, scalar tail.
/// Equivalent to calling `sampson_distance` per-point then thresholding.
#[inline]
pub(super) fn score_inliers_f(
    f_mat: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
) -> (usize, f64) {
    let mut count = 0usize;
    let mut score = 0.0f64;
    score_inliers_f_accumulate(
        f_mat, x1_x, x1_y, x2_x, x2_y, thresh_sq, inliers, &mut count, &mut score,
    );
    (count, score)
}

/// Adds one contiguous Sampson-scoring range to an existing result.
///
/// Keeping `count` and `score` outside the range is significant for bounded
/// scoring: every accepted residual is added in correspondence order, exactly
/// as in the full scorer.
#[inline]
#[allow(clippy::too_many_arguments)]
fn score_inliers_f_accumulate(
    f_mat: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) {
    let n = x1_x.len();
    // F entries (row-major naming, same convention as score_inliers_h).
    let f00 = f_mat.x_axis.x;
    let f01 = f_mat.y_axis.x;
    let f02 = f_mat.z_axis.x;
    let f10 = f_mat.x_axis.y;
    let f11 = f_mat.y_axis.y;
    let f12 = f_mat.z_axis.y;
    let f20 = f_mat.x_axis.z;
    let f21 = f_mat.y_axis.z;
    let f22 = f_mat.z_axis.z;

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut idx = 0usize;

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; all SoA slices and the mask have
    // the same length, and the kernel bounds every vector load/store.
    let mut idx = unsafe {
        score_inliers_f_neon(
            (f00, f01, f02, f10, f11, f12, f20, f21, f22),
            x1_x,
            x1_y,
            x2_x,
            x2_y,
            thresh_sq,
            inliers,
            count,
            score,
        )
    };

    #[cfg(target_arch = "x86_64")]
    let mut idx = if kornia_imgproc::simd::cpu_features().has_avx2
        && kornia_imgproc::simd::cpu_features().has_fma
    {
        // SAFETY: AVX2/FMA support is runtime checked; equally sized SoA
        // slices and mask satisfy the kernel's bounded access requirements.
        unsafe {
            score_inliers_f_avx2(
                (f00, f01, f02, f10, f11, f12, f20, f21, f22),
                x1_x,
                x1_y,
                x2_x,
                x2_y,
                thresh_sq,
                inliers,
                count,
                score,
            )
        }
    } else {
        0usize
    };

    while idx < n {
        let x1 = x1_x[idx];
        let y1 = x1_y[idx];
        let x2 = x2_x[idx];
        let y2 = x2_y[idx];
        let fx1x = f00 * x1 + f01 * y1 + f02;
        let fx1y = f10 * x1 + f11 * y1 + f12;
        let fx1z = f20 * x1 + f21 * y1 + f22;
        let ftx2x = f00 * x2 + f10 * y2 + f20;
        let ftx2y = f01 * x2 + f11 * y2 + f21;
        let err = fx1x * x2 + fx1y * y2 + fx1z;
        let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
        let dd = if denom <= 1e-12 {
            err * err
        } else {
            (err * err) / denom
        };
        if dd <= thresh_sq {
            inliers[idx] = true;
            *count += 1;
            *score += dd;
        }
        idx += 1;
    }
}

/// Scores an F hypothesis in fixed-size chunks, returning `None` when the
/// unscored correspondences cannot make it a strict RANSAC improvement.
///
/// Chunk boundaries are multiples of both the AVX2 and NEON vector widths.
/// Consequently, a non-rejected result follows the same scalar accumulation
/// order and SIMD lanes as [`score_inliers_f`].  Sampson distances admitted
/// into `score` are finite and non-negative, so a partial score can only stay
/// the same or grow as chunks are evaluated.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn score_inliers_f_bounded_masked(
    f_mat: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    best_count: usize,
    best_score: f64,
) -> Option<(usize, f64)> {
    // 64 is divisible by the 4-wide AVX2 and 2-wide NEON f64 paths. It is
    // large enough that the bound check is negligible beside scoring.
    const CHUNK_SIZE: usize = 64;

    let n = x1_x.len();
    let mut start = 0usize;
    let mut count = 0usize;
    let mut score = 0.0f64;
    while start < n {
        let end = (start + CHUNK_SIZE).min(n);
        score_inliers_f_accumulate(
            f_mat,
            &x1_x[start..end],
            &x1_y[start..end],
            &x2_x[start..end],
            &x2_y[start..end],
            thresh_sq,
            &mut inliers[start..end],
            &mut count,
            &mut score,
        );
        let remaining = n - end;
        if count + remaining < best_count
            || (count + remaining == best_count && score >= best_score)
        {
            return None;
        }
        start = end;
    }
    Some((count, score))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn score_inliers_f_bounded_count(
    f_mat: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    best_count: usize,
    best_score: f64,
) -> Option<(usize, f64)> {
    let f = f_score_coefficients(f_mat);

    #[cfg(target_arch = "aarch64")]
    return {
        // SAFETY: NEON is baseline on aarch64 and this scorer only performs
        // bounded loads from equally sized SoA coordinate slices.
        unsafe {
            score_inliers_f_bounded_count_neon(
                f, x1_x, x1_y, x2_x, x2_y, thresh_sq, best_count, best_score,
            )
        }
    };

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if kornia_imgproc::simd::cpu_features().has_avx2
            && kornia_imgproc::simd::cpu_features().has_fma
        {
            // SAFETY: AVX2/FMA support is runtime checked; all coordinate
            // slices have the same length and the kernel bounds every vector
            // load.
            return unsafe {
                score_inliers_f_bounded_count_avx2(
                    f, x1_x, x1_y, x2_x, x2_y, thresh_sq, best_count, best_score,
                )
            };
        }

        score_inliers_f_bounded_count_scalar(
            f, x1_x, x1_y, x2_x, x2_y, thresh_sq, best_count, best_score,
        )
    }
}

#[inline]
fn f_score_coefficients(f_mat: &Mat3F64) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    (
        f_mat.x_axis.x,
        f_mat.y_axis.x,
        f_mat.z_axis.x,
        f_mat.x_axis.y,
        f_mat.y_axis.y,
        f_mat.z_axis.y,
        f_mat.x_axis.z,
        f_mat.y_axis.z,
        f_mat.z_axis.z,
    )
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
fn score_inliers_f_bounded_count_scalar(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    best_count: usize,
    best_score: f64,
) -> Option<(usize, f64)> {
    // 64 is divisible by the 4-wide AVX2 and 2-wide NEON f64 paths.  It is
    // large enough that the bound check is negligible beside scoring.
    const CHUNK_SIZE: usize = 64;

    let n = x1_x.len();
    let mut start = 0usize;
    let mut count = 0usize;
    let mut score = 0.0f64;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    while start < n {
        let end = (start + CHUNK_SIZE).min(n);
        for idx in start..end {
            let fx1x = f00 * x1_x[idx] + f01 * x1_y[idx] + f02;
            let fx1y = f10 * x1_x[idx] + f11 * x1_y[idx] + f12;
            let fx1z = f20 * x1_x[idx] + f21 * x1_y[idx] + f22;
            let ftx2x = f00 * x2_x[idx] + f10 * x2_y[idx] + f20;
            let ftx2y = f01 * x2_x[idx] + f11 * x2_y[idx] + f21;
            let err = fx1x * x2_x[idx] + fx1y * x2_y[idx] + fx1z;
            let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
            let dd = if denom <= 1e-12 {
                err * err
            } else {
                (err * err) / denom
            };
            if dd <= thresh_sq {
                count += 1;
                score += dd;
            }
        }
        let remaining = n - end;
        if count + remaining < best_count
            || (count + remaining == best_count && score >= best_score)
        {
            return None;
        }
        start = end;
    }
    Some((count, score))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_bounded_count_avx2(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    best_count: usize,
    best_score: f64,
) -> Option<(usize, f64)> {
    use std::arch::x86_64::*;
    const CHUNK_SIZE: usize = 64;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    // Broadcast once per hypothesis, rather than once for every pruning chunk.
    let (f00v, f01v, f02v) = (
        _mm256_set1_pd(f00),
        _mm256_set1_pd(f01),
        _mm256_set1_pd(f02),
    );
    let (f10v, f11v, f12v) = (
        _mm256_set1_pd(f10),
        _mm256_set1_pd(f11),
        _mm256_set1_pd(f12),
    );
    let (f20v, f21v, f22v) = (
        _mm256_set1_pd(f20),
        _mm256_set1_pd(f21),
        _mm256_set1_pd(f22),
    );
    let one_v = _mm256_set1_pd(1.0);
    let eps_v = _mm256_set1_pd(1e-12);
    let n = x1_x.len();
    let mut start = 0;
    let mut count = 0;
    let mut score = 0.0;
    while start < n {
        let end = (start + CHUNK_SIZE).min(n);
        let mut idx = start;
        while idx + 4 <= end {
            let x1 = _mm256_loadu_pd(x1_x.as_ptr().add(idx));
            let y1 = _mm256_loadu_pd(x1_y.as_ptr().add(idx));
            let x2 = _mm256_loadu_pd(x2_x.as_ptr().add(idx));
            let y2 = _mm256_loadu_pd(x2_y.as_ptr().add(idx));
            let fx1x = _mm256_fmadd_pd(y1, f01v, _mm256_fmadd_pd(x1, f00v, f02v));
            let fx1y = _mm256_fmadd_pd(y1, f11v, _mm256_fmadd_pd(x1, f10v, f12v));
            let fx1z = _mm256_fmadd_pd(y1, f21v, _mm256_fmadd_pd(x1, f20v, f22v));
            let ftx2x = _mm256_fmadd_pd(y2, f10v, _mm256_fmadd_pd(x2, f00v, f20v));
            let ftx2y = _mm256_fmadd_pd(y2, f11v, _mm256_fmadd_pd(x2, f01v, f21v));
            let err = _mm256_fmadd_pd(y2, fx1y, _mm256_fmadd_pd(x2, fx1x, fx1z));
            let denom = _mm256_fmadd_pd(
                ftx2y,
                ftx2y,
                _mm256_fmadd_pd(
                    ftx2x,
                    ftx2x,
                    _mm256_fmadd_pd(fx1y, fx1y, _mm256_mul_pd(fx1x, fx1x)),
                ),
            );
            let err_sq = _mm256_mul_pd(err, err);
            let denom_ok = _mm256_cmp_pd::<_CMP_GT_OQ>(denom, eps_v);
            let dd = _mm256_blendv_pd(
                err_sq,
                _mm256_div_pd(err_sq, _mm256_blendv_pd(one_v, denom, denom_ok)),
                denom_ok,
            );
            let mut residuals = [0.0; 4];
            _mm256_storeu_pd(residuals.as_mut_ptr(), dd);
            for dd in residuals {
                if dd.is_finite() && dd <= thresh_sq {
                    count += 1;
                    score += dd;
                }
            }
            idx += 4;
        }
        while idx < end {
            let fx1x = f00 * x1_x[idx] + f01 * x1_y[idx] + f02;
            let fx1y = f10 * x1_x[idx] + f11 * x1_y[idx] + f12;
            let fx1z = f20 * x1_x[idx] + f21 * x1_y[idx] + f22;
            let ftx2x = f00 * x2_x[idx] + f10 * x2_y[idx] + f20;
            let ftx2y = f01 * x2_x[idx] + f11 * x2_y[idx] + f21;
            let err = fx1x * x2_x[idx] + fx1y * x2_y[idx] + fx1z;
            let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
            let dd = if denom <= 1e-12 {
                err * err
            } else {
                (err * err) / denom
            };
            if dd <= thresh_sq {
                count += 1;
                score += dd;
            }
            idx += 1;
        }
        let remaining = n - end;
        if count + remaining < best_count
            || (count + remaining == best_count && score >= best_score)
        {
            return None;
        }
        start = end;
    }
    Some((count, score))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_bounded_count_neon(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    best_count: usize,
    best_score: f64,
) -> Option<(usize, f64)> {
    use std::arch::aarch64::*;
    const CHUNK_SIZE: usize = 64;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    // As with AVX2, build this model's vector constants once per root.
    let (f00v, f01v, f02v) = (vdupq_n_f64(f00), vdupq_n_f64(f01), vdupq_n_f64(f02));
    let (f10v, f11v, f12v) = (vdupq_n_f64(f10), vdupq_n_f64(f11), vdupq_n_f64(f12));
    let (f20v, f21v, f22v) = (vdupq_n_f64(f20), vdupq_n_f64(f21), vdupq_n_f64(f22));
    let one_v = vdupq_n_f64(1.0);
    let eps_v = vdupq_n_f64(1e-12);
    let n = x1_x.len();
    let mut start = 0;
    let mut count = 0;
    let mut score = 0.0;
    while start < n {
        let end = (start + CHUNK_SIZE).min(n);
        let mut idx = start;
        while idx + 2 <= end {
            let x1 = vld1q_f64(x1_x.as_ptr().add(idx));
            let y1 = vld1q_f64(x1_y.as_ptr().add(idx));
            let x2 = vld1q_f64(x2_x.as_ptr().add(idx));
            let y2 = vld1q_f64(x2_y.as_ptr().add(idx));
            let fx1x = vfmaq_f64(vfmaq_f64(f02v, x1, f00v), y1, f01v);
            let fx1y = vfmaq_f64(vfmaq_f64(f12v, x1, f10v), y1, f11v);
            let fx1z = vfmaq_f64(vfmaq_f64(f22v, x1, f20v), y1, f21v);
            let ftx2x = vfmaq_f64(vfmaq_f64(f20v, x2, f00v), y2, f10v);
            let ftx2y = vfmaq_f64(vfmaq_f64(f21v, x2, f01v), y2, f11v);
            let err = vfmaq_f64(vfmaq_f64(fx1z, x2, fx1x), y2, fx1y);
            let denom = vfmaq_f64(
                vfmaq_f64(vfmaq_f64(vmulq_f64(fx1x, fx1x), fx1y, fx1y), ftx2x, ftx2x),
                ftx2y,
                ftx2y,
            );
            let err_sq = vmulq_f64(err, err);
            let denom_ok = vcgtq_f64(denom, eps_v);
            let dd = vbslq_f64(
                denom_ok,
                vdivq_f64(err_sq, vbslq_f64(denom_ok, denom, one_v)),
                err_sq,
            );
            let mut residuals = [0.0; 2];
            vst1q_f64(residuals.as_mut_ptr(), dd);
            for dd in residuals {
                if dd.is_finite() && dd <= thresh_sq {
                    count += 1;
                    score += dd;
                }
            }
            idx += 2;
        }
        while idx < end {
            let fx1x = f00 * x1_x[idx] + f01 * x1_y[idx] + f02;
            let fx1y = f10 * x1_x[idx] + f11 * x1_y[idx] + f12;
            let fx1z = f20 * x1_x[idx] + f21 * x1_y[idx] + f22;
            let ftx2x = f00 * x2_x[idx] + f10 * x2_y[idx] + f20;
            let ftx2y = f01 * x2_x[idx] + f11 * x2_y[idx] + f21;
            let err = fx1x * x2_x[idx] + fx1y * x2_y[idx] + fx1z;
            let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
            let dd = if denom <= 1e-12 {
                err * err
            } else {
                (err * err) / denom
            };
            if dd <= thresh_sq {
                count += 1;
                score += dd;
            }
            idx += 1;
        }
        let remaining = n - end;
        if count + remaining < best_count
            || (count + remaining == best_count && score >= best_score)
        {
            return None;
        }
        start = end;
    }
    Some((count, score))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_neon(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    // SAFETY: The caller guarantees NEON and equally sized SoA/mask slices.
    // Vector accesses require at least two remaining entries; the loop checks
    // that bound and stores only those lanes. Scalar tails stay with the caller.
    unsafe {
        use std::arch::aarch64::*;
        let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
        let f00v = vdupq_n_f64(f00);
        let f01v = vdupq_n_f64(f01);
        let f02v = vdupq_n_f64(f02);
        let f10v = vdupq_n_f64(f10);
        let f11v = vdupq_n_f64(f11);
        let f12v = vdupq_n_f64(f12);
        let f20v = vdupq_n_f64(f20);
        let f21v = vdupq_n_f64(f21);
        let f22v = vdupq_n_f64(f22);
        let n = x1_x.len();
        let mut idx = 0usize;
        while idx + 2 <= n {
            let x1 = vld1q_f64(x1_x.as_ptr().add(idx));
            let y1 = vld1q_f64(x1_y.as_ptr().add(idx));
            let x2 = vld1q_f64(x2_x.as_ptr().add(idx));
            let y2 = vld1q_f64(x2_y.as_ptr().add(idx));

            // fx1 = F * [x1; y1; 1]
            let fx1x = vfmaq_f64(vfmaq_f64(f02v, x1, f00v), y1, f01v);
            let fx1y = vfmaq_f64(vfmaq_f64(f12v, x1, f10v), y1, f11v);
            let fx1z = vfmaq_f64(vfmaq_f64(f22v, x1, f20v), y1, f21v);
            // ftx2 = F^T * [x2; y2; 1]  (only x,y components needed for denom)
            let ftx2x = vfmaq_f64(vfmaq_f64(f20v, x2, f00v), y2, f10v);
            let ftx2y = vfmaq_f64(vfmaq_f64(f21v, x2, f01v), y2, f11v);

            // err = [x2; y2; 1] · fx1
            let err = vfmaq_f64(vfmaq_f64(fx1z, x2, fx1x), y2, fx1y);
            // denom = fx1x² + fx1y² + ftx2x² + ftx2y²
            let denom = vfmaq_f64(
                vfmaq_f64(vfmaq_f64(vmulq_f64(fx1x, fx1x), fx1y, fx1y), ftx2x, ftx2x),
                ftx2y,
                ftx2y,
            );
            let err_sq = vmulq_f64(err, err);
            // If denom > 0, use err²/denom; else err². Branchless select via bitwise
            // (denom > 1e-12) mask — otherwise fall back to scalar handling per lane.
            let denom_ok = vcgtq_f64(denom, vdupq_n_f64(1e-12));
            let safe_denom = vbslq_f64(denom_ok, denom, vdupq_n_f64(1.0));
            let div_val = vdivq_f64(err_sq, safe_denom);
            let dd = vbslq_f64(denom_ok, div_val, err_sq);

            let mut buf = [0.0f64; 2];
            vst1q_f64(buf.as_mut_ptr(), dd);
            for k in 0..2 {
                let dd_k = buf[k];
                if dd_k.is_finite() && dd_k <= thresh_sq {
                    *inliers.get_unchecked_mut(idx + k) = true;
                    *count += 1;
                    *score += dd_k;
                }
            }
            idx += 2;
        }
        idx
    }
}

/// AVX2 mirror of [`score_inliers_f_neon`]. Same F Sampson math at 4-lane
/// f64 (`__m256d`). The branchless `denom > 1e-12` masked select uses
/// `_mm256_blendv_pd`, whose argument order is the inverse of NEON's
/// `vbslq_f64` (blendv is `(false, true, mask)`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_avx2(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    use std::arch::x86_64::*;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    let f00v = _mm256_set1_pd(f00);
    let f01v = _mm256_set1_pd(f01);
    let f02v = _mm256_set1_pd(f02);
    let f10v = _mm256_set1_pd(f10);
    let f11v = _mm256_set1_pd(f11);
    let f12v = _mm256_set1_pd(f12);
    let f20v = _mm256_set1_pd(f20);
    let f21v = _mm256_set1_pd(f21);
    let f22v = _mm256_set1_pd(f22);
    let one_v = _mm256_set1_pd(1.0);
    let eps_v = _mm256_set1_pd(1e-12);
    let n = x1_x.len();
    let mut idx = 0usize;
    while idx + 4 <= n {
        let x1 = _mm256_loadu_pd(x1_x.as_ptr().add(idx));
        let y1 = _mm256_loadu_pd(x1_y.as_ptr().add(idx));
        let x2 = _mm256_loadu_pd(x2_x.as_ptr().add(idx));
        let y2 = _mm256_loadu_pd(x2_y.as_ptr().add(idx));

        let fx1x = _mm256_fmadd_pd(y1, f01v, _mm256_fmadd_pd(x1, f00v, f02v));
        let fx1y = _mm256_fmadd_pd(y1, f11v, _mm256_fmadd_pd(x1, f10v, f12v));
        let fx1z = _mm256_fmadd_pd(y1, f21v, _mm256_fmadd_pd(x1, f20v, f22v));
        let ftx2x = _mm256_fmadd_pd(y2, f10v, _mm256_fmadd_pd(x2, f00v, f20v));
        let ftx2y = _mm256_fmadd_pd(y2, f11v, _mm256_fmadd_pd(x2, f01v, f21v));

        let err = _mm256_fmadd_pd(y2, fx1y, _mm256_fmadd_pd(x2, fx1x, fx1z));
        let denom = _mm256_fmadd_pd(
            ftx2y,
            ftx2y,
            _mm256_fmadd_pd(
                ftx2x,
                ftx2x,
                _mm256_fmadd_pd(fx1y, fx1y, _mm256_mul_pd(fx1x, fx1x)),
            ),
        );
        let err_sq = _mm256_mul_pd(err, err);
        let denom_ok = _mm256_cmp_pd::<_CMP_GT_OQ>(denom, eps_v);
        let safe_denom = _mm256_blendv_pd(one_v, denom, denom_ok);
        let div_val = _mm256_div_pd(err_sq, safe_denom);
        let dd = _mm256_blendv_pd(err_sq, div_val, denom_ok);

        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), dd);
        for k in 0..4 {
            let dd_k = buf[k];
            if dd_k.is_finite() && dd_k <= thresh_sq {
                *inliers.get_unchecked_mut(idx + k) = true;
                *count += 1;
                *score += dd_k;
            }
        }
        idx += 4;
    }
    idx
}

/// Flatten `&[Vec2F64]` into two parallel f64 slices (x, y). Called once per
/// RANSAC entry; keeps the inner-loop scorer in contiguous `&[f64]` land.
pub(super) fn split_xy(pts: &[Vec2F64]) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(pts.len());
    let mut ys = Vec::with_capacity(pts.len());
    for p in pts {
        xs.push(p.x);
        ys.push(p.y);
    }
    (xs, ys)
}
