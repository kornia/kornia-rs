//! SoA scoring kernels for two-view RANSAC: Sampson distance for F and
//! transfer error for H, with scalar, NEON and AVX2 paths.

use kornia_algebra::{Mat3F64, Vec2F64};

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

/// Upper bound on the Sampson support of `f_mat`, or `None` once the bound
/// proves it cannot exceed `prune_at` matches.
///
/// The SIMD kernels repeat the exact scorers' `e²` and `d` arithmetic and
/// test `e² <= t'·d` with a slightly inflated `t' = t·(1 + 8ε)` instead of
/// dividing; denominators at or below the 1e-12 guard (or NaN) are always
/// counted and the scalar tail is scored exactly. Every match the exact
/// scorer accepts is therefore counted. Platforms without a SIMD kernel
/// count with the exact scalar residuals instead; thresholds too small for
/// the rounding argument get the trivial bound `n`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn score_inliers_f_upper_bound(
    f_mat: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    prune_at: usize,
) -> Option<usize> {
    let n = x1_x.len();
    if thresh_sq.is_nan() || thresh_sq < F_UPPER_BOUND_MIN_THRESHOLD {
        return Some(n);
    }
    let f = f_score_coefficients(f_mat);

    #[cfg(target_arch = "aarch64")]
    return {
        // SAFETY: NEON is baseline on aarch64 and the kernel only performs
        // bounded loads from equally sized SoA coordinate slices.
        unsafe { score_inliers_f_upper_bound_neon(f, x1_x, x1_y, x2_x, x2_y, thresh_sq, prune_at) }
    };

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if kornia_imgproc::simd::cpu_features().has_avx2
            && kornia_imgproc::simd::cpu_features().has_fma
        {
            // SAFETY: AVX2/FMA support is runtime checked; all coordinate
            // slices have the same length and the kernel bounds every load.
            return unsafe {
                score_inliers_f_upper_bound_avx2(f, x1_x, x1_y, x2_x, x2_y, thresh_sq, prune_at)
            };
        }
        score_inliers_f_upper_bound_scalar(f, x1_x, x1_y, x2_x, x2_y, thresh_sq, prune_at)
    }
}

/// Scalar support bound. The exact scorer is scalar too on these targets,
/// so its own count is the bound; this keeps the cheap count-only rejection
/// of losing roots.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(super) fn score_inliers_f_upper_bound_scalar(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    prune_at: usize,
) -> Option<usize> {
    let n = x1_x.len();
    let mut count = 0;
    let mut start = 0;
    while start < n {
        let end = (start + F_UPPER_BOUND_CHUNK).min(n);
        count += score_inliers_f_tail_count(
            f,
            &x1_x[start..end],
            &x1_y[start..end],
            &x2_x[start..end],
            &x2_y[start..end],
            thresh_sq,
            0,
        );
        if count + (n - end) <= prune_at {
            return None;
        }
        start = end;
    }
    (count > prune_at).then_some(count)
}

/// Matches between pruning checks of [`score_inliers_f_upper_bound`].
const F_UPPER_BOUND_CHUNK: usize = 32;

/// Inflation of the threshold in [`score_inliers_f_upper_bound`]: it covers
/// the three roundings of `e² <= t'·d` against the one of `e²/d <= t`.
const F_UPPER_BOUND_SLACK: f64 = 1.0 + 8.0 * f64::EPSILON;

/// Smallest threshold for which `t'·d` stays normal whenever `d > 1e-12`.
const F_UPPER_BOUND_MIN_THRESHOLD: f64 = 1e-280;

#[inline]
pub(super) fn f_score_coefficients(
    f_mat: &Mat3F64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
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

/// Exact scalar residuals of the tail, counted like the exact scorer.
#[inline]
#[allow(clippy::too_many_arguments)]
fn score_inliers_f_tail_count(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    start: usize,
) -> usize {
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    let mut count = 0;
    for idx in start..x1_x.len() {
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
        count += (dd <= thresh_sq) as usize;
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_upper_bound_avx2(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    prune_at: usize,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
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
    let bound_v = _mm256_set1_pd(thresh_sq * F_UPPER_BOUND_SLACK);
    let eps_v = _mm256_set1_pd(1e-12);
    let n = x1_x.len();
    let simd_end = n & !3;
    let lane_sum = |v: __m256i| -> usize {
        let pair = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        (_mm_cvtsi128_si64(pair) + _mm_extract_epi64::<1>(pair)) as usize
    };
    // Lane counters: a true comparison is all ones, i.e. -1 as an integer.
    let mut possible = _mm256_setzero_si256();
    let mut idx = 0;
    let mut next_check = F_UPPER_BOUND_CHUNK.min(simd_end);
    while idx < simd_end {
        let x1 = _mm256_loadu_pd(x1_x.as_ptr().add(idx));
        let y1 = _mm256_loadu_pd(x1_y.as_ptr().add(idx));
        let x2 = _mm256_loadu_pd(x2_x.as_ptr().add(idx));
        let y2 = _mm256_loadu_pd(x2_y.as_ptr().add(idx));
        // Same operation sequence as `score_inliers_f_avx2`.
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
        let inside = _mm256_cmp_pd::<_CMP_LE_OQ>(err_sq, _mm256_mul_pd(bound_v, denom));
        // `!(d > eps)` also catches NaN denominators.
        let degenerate = _mm256_cmp_pd::<_CMP_NGT_UQ>(denom, eps_v);
        possible = _mm256_sub_epi64(
            possible,
            _mm256_castpd_si256(_mm256_or_pd(inside, degenerate)),
        );
        idx += 4;
        if idx == next_check {
            if lane_sum(possible) + (n - idx) <= prune_at {
                return None;
            }
            next_check = (idx + F_UPPER_BOUND_CHUNK).min(simd_end);
        }
    }
    let count = lane_sum(possible)
        + score_inliers_f_tail_count(f, x1_x, x1_y, x2_x, x2_y, thresh_sq, simd_end);
    (count > prune_at).then_some(count)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_upper_bound_neon(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    prune_at: usize,
) -> Option<usize> {
    use std::arch::aarch64::*;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    let (f00v, f01v, f02v) = (vdupq_n_f64(f00), vdupq_n_f64(f01), vdupq_n_f64(f02));
    let (f10v, f11v, f12v) = (vdupq_n_f64(f10), vdupq_n_f64(f11), vdupq_n_f64(f12));
    let (f20v, f21v, f22v) = (vdupq_n_f64(f20), vdupq_n_f64(f21), vdupq_n_f64(f22));
    let bound_v = vdupq_n_f64(thresh_sq * F_UPPER_BOUND_SLACK);
    let eps_v = vdupq_n_f64(1e-12);
    let n = x1_x.len();
    let simd_end = n & !1;
    // Lane counters: a true comparison is all ones, i.e. -1 as an integer.
    let mut possible = vdupq_n_u64(0);
    let mut idx = 0;
    let mut next_check = F_UPPER_BOUND_CHUNK.min(simd_end);
    while idx < simd_end {
        let x1 = vld1q_f64(x1_x.as_ptr().add(idx));
        let y1 = vld1q_f64(x1_y.as_ptr().add(idx));
        let x2 = vld1q_f64(x2_x.as_ptr().add(idx));
        let y2 = vld1q_f64(x2_y.as_ptr().add(idx));
        // Same operation sequence as `score_inliers_f_neon`.
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
        let inside = vcleq_f64(err_sq, vmulq_f64(bound_v, denom));
        // `!(d > eps)` also catches NaN denominators.
        let degenerate = vmvnq_u32(vreinterpretq_u32_u64(vcgtq_f64(denom, eps_v)));
        possible = vsubq_u64(
            possible,
            vorrq_u64(inside, vreinterpretq_u64_u32(degenerate)),
        );
        idx += 2;
        if idx == next_check {
            if vaddvq_u64(possible) as usize + (n - idx) <= prune_at {
                return None;
            }
            next_check = (idx + F_UPPER_BOUND_CHUNK).min(simd_end);
        }
    }
    let count = vaddvq_u64(possible) as usize
        + score_inliers_f_tail_count(f, x1_x, x1_y, x2_x, x2_y, thresh_sq, simd_end);
    (count > prune_at).then_some(count)
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
