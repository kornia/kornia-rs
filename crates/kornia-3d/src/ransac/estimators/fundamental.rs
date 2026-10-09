//! Fundamental-matrix estimator.
//!
//! Seven-point minimal hypotheses and eight-point inlier refinement share
//! the same SIMD Sampson scoring. The eight-point estimator is also available
//! for callers that need the previous sampling behavior.

use kornia_algebra::{Mat3F64, Vec2F64};

use crate::pose::fundamental_7point_oriented_into;
use crate::pose::{fundamental_8point, sampson_distance};
use crate::ransac::{clamp_pair, Estimator, Match2d2d, ThresholdInlierResult};

/// Seven-point fundamental matrix estimator from 2D-2D pixel correspondences.
///
/// Minimal fitting returns up to three real solutions; inlier sets of eight
/// or more matches are refined with the normalized eight-point solver.
///
/// **Oriented epipolar constraint.** As in DEGENSAC, a minimal solution is
/// discarded when it orients its own seven correspondences inconsistently
/// (Chum, Werner and Matas, ICPR 2004): no camera pair seeing those points in
/// front of both cameras can produce it, so it is never scored.
///
/// **Coordinate convention.** Samples are in raw pixel coordinates; Hartley
/// normalization is applied internally, mirroring the existing solver.
///
/// **Residual.** Sampson distance — the standard first-order epipolar
/// approximation. Squared pixel units, so the matching RANSAC threshold
/// is also squared (e.g. `1.0` ≈ 1 px).
#[derive(Debug, Clone, Copy, Default)]
pub struct FundamentalEstimator;

impl Estimator for FundamentalEstimator {
    type Model = Mat3F64;
    type Sample = Match2d2d;
    const SAMPLE_SIZE: usize = 7;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() == Self::SAMPLE_SIZE {
            let x1 = std::array::from_fn::<_, 7, _>(|i| samples[i].x1);
            let x2 = std::array::from_fn::<_, 7, _>(|i| samples[i].x2);
            let mut models = [Mat3F64::ZERO; 3];
            if let Ok(count) = fundamental_7point_oriented_into(&x1, &x2, &mut models) {
                out.extend_from_slice(&models[..count]);
            }
        } else {
            Fundamental8PointEstimator.fit(samples, out);
        }
    }

    fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
        self.fit(inliers, out);
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sampson_distance(model, &sample.x1, &sample.x2)
    }

    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        Fundamental8PointEstimator.residual_batch(model, samples, out);
    }

    /// Scores in two passes. A division-free pass bounds the support from
    /// above and rejects, every 32 matches, a root whose bound can no longer
    /// exceed the incumbent; most seven-point roots end there without
    /// residual or mask writes. A surviving root is scored exactly in fixed
    /// SIMD-friendly chunks, with the same pruning on its exact count.
    fn threshold_inliers(
        &self,
        model: &Self::Model,
        samples: &[Self::Sample],
        threshold: f64,
        best_inlier_count: usize,
        residuals: &mut [f64],
        inliers_out: &mut Vec<bool>,
    ) -> Option<ThresholdInlierResult> {
        let n = samples.len().min(residuals.len());
        if n != samples.len() {
            return None;
        }

        // Pack once per candidate rather than once per scoring chunk.
        let f = pack_f(model);
        // Most roots lose. A division-free pass bounds their support from
        // above and rejects them without residual or mask writes; only a
        // candidate that might still beat the incumbent pays for the exact
        // pass below, so the outcome matches exact scoring alone.
        if sampson_support_upper_bound(f, samples, threshold, best_inlier_count).is_none() {
            return Some(ThresholdInlierResult::Pruned);
        }
        // The selected backend keeps its vector constants live across the
        // fixed-size chunks and performs the same safe pruning check.
        let Some(count) =
            sampson_residual_batch_threshold(f, samples, residuals, threshold, best_inlier_count)
        else {
            return Some(ThresholdInlierResult::Pruned);
        };

        // Only a candidate that can become the winner needs its full mask.
        inliers_out.resize(n, false);
        for (mask, &residual) in inliers_out.iter_mut().zip(residuals.iter()) {
            *mask = residual < threshold;
        }
        Some(ThresholdInlierResult::Complete(count))
    }
}

/// Eight-point fundamental matrix estimator retaining the original RANSAC path.
///
/// Uses normalized eight-point fitting for both minimal samples and inlier
/// refinement, with the same squared-pixel Sampson residuals as
/// [`FundamentalEstimator`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Fundamental8PointEstimator;

impl Estimator for Fundamental8PointEstimator {
    type Model = Mat3F64;
    type Sample = Match2d2d;
    const SAMPLE_SIZE: usize = 8;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        // Translate AoS samples into the parallel-slice form fundamental_8point
        // expects. SAMPLE_SIZE=8 means we never spill past the stack array;
        // larger inputs (LO-RANSAC refits) fall back to a Vec but only when
        // the driver explicitly passes >8 — they're rare and amortized.
        const STACK_N: usize = 32;
        let n = samples.len();
        if n < Self::SAMPLE_SIZE {
            return;
        }
        if n <= STACK_N {
            let mut x1 = [Vec2F64::ZERO; STACK_N];
            let mut x2 = [Vec2F64::ZERO; STACK_N];
            for (i, s) in samples.iter().enumerate() {
                x1[i] = s.x1;
                x2[i] = s.x2;
            }
            if let Ok(f) = fundamental_8point(&x1[..n], &x2[..n]) {
                out.push(f);
            }
        } else {
            let x1: Vec<Vec2F64> = samples.iter().map(|s| s.x1).collect();
            let x2: Vec<Vec2F64> = samples.iter().map(|s| s.x2).collect();
            if let Ok(f) = fundamental_8point(&x1, &x2) {
                out.push(f);
            }
        }
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sampson_distance(model, &sample.x1, &sample.x2)
    }

    /// LO-friendly refit. `fundamental_8point` is a least-squares solver
    /// that takes any N ≥ 8, so we forward straight to `fit` — the existing
    /// implementation already routes through Householder for n ≤ 64 and
    /// MtM/LDLᵀ otherwise.
    fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
        self.fit(inliers, out);
    }

    /// Dispatcher matching the kornia-imgproc `warp/kernels.rs` convention:
    /// aarch64 → NEON unconditionally (NEON is baseline for the supported
    /// linux-aarch64 target), x86_64 → AVX2+FMA when probed at runtime,
    /// otherwise the portable scalar reference.
    ///
    /// All three paths produce identical results to within FMA reordering
    /// noise (≤ 1e-12 relative) — a unit test pins the equivalence.
    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        // The SIMD kernels below write `out` through raw pointers for every sample;
        // only the first `min(out.len(), samples.len())` entries are computed.
        let (samples, out) = clamp_pair(samples, out);
        sampson_residual_batch(pack_f(model), samples, out);
    }
}

/// Evaluate a batch with the existing residual dispatcher.
#[inline]
fn sampson_residual_batch(f: FPacked, samples: &[Match2d2d], out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is architectural on aarch64-unknown-linux-gnu.
    // `out.len() == samples.len()` holds after the clamp above; the kernel never reads/writes
    // past `samples.len()` (returns `idx`, scalar tail handles the rest).
    unsafe {
        let _ = sampson_residual_batch_neon::<false, false>(f, samples, out, 0.0, 0);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if kornia_imgproc::simd::cpu_features().has_avx2 && kornia_imgproc::simd::cpu_features().has_fma
    {
        // SAFETY: `has_avx2` runtime check; `target_feature(enable=...)`
        // enables AVX2+FMA inside the kernel. Same length invariants.
        unsafe {
            let _ = sampson_residual_batch_avx2::<false, false>(f, samples, out, 0.0, 0);
        }
        return;
    }

    #[allow(unreachable_code)]
    sampson_residual_batch_scalar(f, samples, out);
}

/// Compute residuals and count strict inliers in one pass.
#[inline]
fn sampson_residual_batch_threshold(
    f: FPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    threshold: f64,
    best_inlier_count: usize,
) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is architectural on aarch64. The kernel only processes
    // complete lanes within the equally-sized slices.
    unsafe {
        return sampson_residual_batch_neon::<true, true>(
            f,
            samples,
            out,
            threshold,
            best_inlier_count,
        );
    }

    #[cfg(target_arch = "x86_64")]
    if kornia_imgproc::simd::cpu_features().has_avx2 && kornia_imgproc::simd::cpu_features().has_fma
    {
        // SAFETY: runtime feature checks match the kernel target features;
        // the kernel only processes complete lanes within the slices.
        unsafe {
            return sampson_residual_batch_avx2::<true, true>(
                f,
                samples,
                out,
                threshold,
                best_inlier_count,
            );
        }
    }

    #[allow(unreachable_code)]
    {
        sampson_residual_batch_scalar_threshold_bounded(
            f,
            samples,
            out,
            threshold,
            best_inlier_count,
        )
    }
}

/// Correspondences between pruning checks of the support upper bound.
const UPPER_BOUND_CHUNK: usize = 32;

/// Threshold inflation of the support upper bound. Comparing `e² <= t'·d`
/// with `t' = t·(1 + 8ε)` rounds three times, while the exact `e²/d < t`
/// rounds once; the slack keeps every exact inlier inside the bound.
const UPPER_BOUND_SLACK: f64 = 1.0 + 8.0 * f64::EPSILON;

/// Smallest threshold for which `t'·d` stays normal whenever `d > 1e-12`,
/// so the relative rounding argument above applies.
const UPPER_BOUND_MIN_THRESHOLD: f64 = 1e-280;

/// Upper bound on the number of strict Sampson inliers, or `None` once the
/// bound proves the candidate cannot exceed `prune_at` inliers.
///
/// The SIMD kernels share the exact kernels' `e²` and `d` arithmetic and
/// test `e² <= t'·d` instead of dividing. Denominators at or below the
/// exact path's 1e-12 guard (or NaN) are always counted, and the scalar
/// tail uses exact residuals with `<=`. Every exact inlier is therefore
/// counted, so rejecting on the bound never discards a possible winner.
/// Without a SIMD kernel the exact scalar residuals are counted instead;
/// for thresholds outside the rounding argument the bound is the trivial
/// `samples.len()`.
#[inline]
fn sampson_support_upper_bound(
    f: FPacked,
    samples: &[Match2d2d],
    threshold: f64,
    prune_at: usize,
) -> Option<usize> {
    if threshold.is_nan() || threshold < UPPER_BOUND_MIN_THRESHOLD {
        return Some(samples.len());
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is architectural on aarch64; the kernel reads only
    // complete lane groups within `samples`.
    unsafe {
        return sampson_support_upper_bound_neon(f, samples, threshold, prune_at);
    }

    #[cfg(target_arch = "x86_64")]
    if kornia_imgproc::simd::cpu_features().has_avx2 && kornia_imgproc::simd::cpu_features().has_fma
    {
        // SAFETY: runtime feature checks match the kernel target features;
        // the kernel reads only complete lane groups within `samples`.
        unsafe {
            return sampson_support_upper_bound_avx2(f, samples, threshold, prune_at);
        }
    }

    #[allow(unreachable_code)]
    sampson_support_upper_bound_scalar(f, samples, threshold, prune_at)
}

/// Scalar support bound. The exact pass is scalar too on these targets, so
/// its own residuals bound the count; this keeps the cheap count-only
/// rejection of losing roots.
fn sampson_support_upper_bound_scalar(
    f: FPacked,
    samples: &[Match2d2d],
    threshold: f64,
    prune_at: usize,
) -> Option<usize> {
    let n = samples.len();
    let mut count = 0;
    let mut start = 0;
    while start < n {
        let end = (start + UPPER_BOUND_CHUNK).min(n);
        count += sampson_tail_upper_bound(f, &samples[start..end], threshold);
        if count + (n - end) <= prune_at {
            return None;
        }
        start = end;
    }
    (count > prune_at).then_some(count)
}

/// Exact residuals of the scalar tail, counted with the bound's `<=`.
#[inline]
fn sampson_tail_upper_bound(f: FPacked, tail: &[Match2d2d], threshold: f64) -> usize {
    let mut residuals = [0.0f64; 4];
    let mut count = 0;
    for chunk in tail.chunks(4) {
        sampson_residual_batch_scalar(f, chunk, &mut residuals[..chunk.len()]);
        count += residuals[..chunk.len()]
            .iter()
            .filter(|&&residual| residual <= threshold)
            .count();
    }
    count
}

// ---------------------------------------------------------------------------
// Sampson-residual kernels (scalar reference + NEON + AVX2)
//
// Mirrors the `warp/kernels.rs` layout in kornia-imgproc:
//   - `_scalar`  — portable, always available, single source of numeric truth.
//   - `_neon`    — aarch64 SIMD; baseline feature, no runtime probe.
//   - `_avx2`    — x86_64 SIMD; gated by `simd::cpu_features().has_avx2`.
//
// Each SIMD kernel completes scalar tails internally and returns the inlier
// count, or `None` when bounded scoring proves the remaining samples cannot
// exceed the incumbent.
// ---------------------------------------------------------------------------

/// 9 entries of an F-matrix in row-major order — the natural shape for
/// scalar arithmetic and broadcast-ready for SIMD lane vectors.
type FPacked = (f64, f64, f64, f64, f64, f64, f64, f64, f64);

/// Residuals per fused hard-threshold pass. This keeps SIMD dispatch overhead
/// amortized while allowing weak seven-point roots to be rejected early.
const THRESHOLD_SCORE_CHUNK: usize = 128;

#[inline(always)]
fn pack_f(model: &Mat3F64) -> FPacked {
    (
        model.x_axis.x,
        model.y_axis.x,
        model.z_axis.x,
        model.x_axis.y,
        model.y_axis.y,
        model.z_axis.y,
        model.x_axis.z,
        model.y_axis.z,
        model.z_axis.z,
    )
}

/// Portable scalar Sampson distance — reference for both SIMD backends.
///
/// Computes `err² / denom` where `err = x2ᵀ F x1` and `denom = ‖Fx1‖² +
/// ‖Fᵀx2‖²` (only the x,y components, as in the standard form). Falls
/// back to `err²` when the denominator collapses (mostly on epipoles).
#[inline]
fn sampson_residual_batch_scalar(f: FPacked, samples: &[Match2d2d], out: &mut [f64]) {
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    for (i, s) in samples.iter().enumerate() {
        let (x1, y1) = (s.x1.x, s.x1.y);
        let (x2, y2) = (s.x2.x, s.x2.y);
        let fx1x = f00 * x1 + f01 * y1 + f02;
        let fx1y = f10 * x1 + f11 * y1 + f12;
        let fx1z = f20 * x1 + f21 * y1 + f22;
        let ftx2x = f00 * x2 + f10 * y2 + f20;
        let ftx2y = f01 * x2 + f11 * y2 + f21;
        let err = fx1x * x2 + fx1y * y2 + fx1z;
        let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
        out[i] = if denom <= 1e-12 {
            err * err
        } else {
            err * err / denom
        };
    }
}

/// Scalar tail used after a SIMD kernel has processed the leading multiple
/// of lane-width elements; covers the remaining `samples.len() - start`
/// entries.
#[inline]
fn sampson_residual_batch_scalar_tail(
    f: FPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    start: usize,
) {
    if start >= samples.len() {
        return;
    }
    sampson_residual_batch_scalar(f, &samples[start..], &mut out[start..]);
}

/// Scalar bounded threshold fallback with the same pruning boundaries as SIMD.
fn sampson_residual_batch_scalar_threshold_bounded(
    f: FPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    threshold: f64,
    best_inlier_count: usize,
) -> Option<usize> {
    let mut count = 0usize;
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + THRESHOLD_SCORE_CHUNK).min(samples.len());
        sampson_residual_batch_scalar(f, &samples[start..end], &mut out[start..end]);
        count += out[start..end]
            .iter()
            .filter(|&&residual| residual < threshold)
            .count();
        if count + samples.len() - end <= best_inlier_count {
            return None;
        }
        start = end;
    }
    Some(count)
}

/// 2-lane f64 NEON kernel; loads and arithmetic are shared with the
/// support bound through [`sampson_terms_neon`].
///
/// # Safety
/// - aarch64 architectural (no runtime probe needed); `target_feature` is
///   set to unlock the intrinsics.
/// - `out.len() >= samples.len()`; never reads/writes past either slice.
///   `threshold_inliers` deliberately permits oversized reusable scratch and
///   only accesses the prefix matching `samples.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sampson_residual_batch_neon<const COUNT: bool, const BOUNDED: bool>(
    f: FPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    threshold: f64,
    best_inlier_count: usize,
) -> Option<usize> {
    unsafe {
        use std::arch::aarch64::*;
        let fv = broadcast_f_neon(f);
        let eps = vdupq_n_f64(1e-12);
        let one = vdupq_n_f64(1.0);
        let threshold_v = vdupq_n_f64(threshold);

        let n = samples.len();
        let mut start = 0usize;
        let mut count = 0usize;
        while start < n {
            let end = if BOUNDED {
                (start + THRESHOLD_SCORE_CHUNK).min(n)
            } else {
                n
            };
            let mut idx = start;
            while idx + 2 <= end {
                let (err_sq, denom) =
                    sampson_terms_neon(&fv, samples.as_ptr().add(idx) as *const f64);
                let denom_ok = vcgtq_f64(denom, eps);
                let safe_denom = vbslq_f64(denom_ok, denom, one);
                let div_val = vdivq_f64(err_sq, safe_denom);
                let dd = vbslq_f64(denom_ok, div_val, err_sq);

                vst1q_f64(out.as_mut_ptr().add(idx), dd);
                if COUNT {
                    let mask = vcltq_f64(dd, threshold_v);
                    let mut lanes = [0u64; 2];
                    vst1q_u64(lanes.as_mut_ptr(), mask);
                    count += (lanes[0] != 0) as usize + (lanes[1] != 0) as usize;
                }
                idx += 2;
            }
            if idx < end {
                sampson_residual_batch_scalar_tail(f, &samples[..end], &mut out[..end], idx);
                if COUNT {
                    count += out[idx..end]
                        .iter()
                        .filter(|&&residual| residual < threshold)
                        .count();
                }
            }
            if BOUNDED && count + n - end <= best_inlier_count {
                return None;
            }
            start = end;
        }
        Some(count)
    }
}

/// 4-lane f64 AVX2+FMA kernel; loads and arithmetic are shared with the
/// support bound through [`sampson_terms_avx2`].
///
/// # Safety
/// - Caller has runtime-checked `cpu_features().has_avx2`.
/// - `out.len() >= samples.len()`; `threshold_inliers` deliberately permits
///   oversized reusable scratch and this kernel accesses only the matching
///   sample prefix.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sampson_residual_batch_avx2<const COUNT: bool, const BOUNDED: bool>(
    f: FPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    threshold: f64,
    best_inlier_count: usize,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let fv = broadcast_f_avx2(f);
    let eps = _mm256_set1_pd(1e-12);
    let one = _mm256_set1_pd(1.0);
    let threshold_v = _mm256_set1_pd(threshold);

    let n = samples.len();
    let mut start = 0usize;
    let mut count = 0usize;
    while start < n {
        let end = if BOUNDED {
            (start + THRESHOLD_SCORE_CHUNK).min(n)
        } else {
            n
        };
        let mut idx = start;
        while idx + 4 <= end {
            let (err_sq, denom) = sampson_terms_avx2(&fv, samples.as_ptr().add(idx) as *const f64);
            // Mask: denom > 1e-12. `_mm256_blendv_pd` selects per-lane on the
            // *sign bit* of the mask — `_CMP_GT_OQ` produces all-ones on true,
            // all-zeros on false.
            let denom_ok = _mm256_cmp_pd::<_CMP_GT_OQ>(denom, eps);
            let safe_denom = _mm256_blendv_pd(one, denom, denom_ok);
            let div_val = _mm256_div_pd(err_sq, safe_denom);
            let dd = _mm256_blendv_pd(err_sq, div_val, denom_ok);

            _mm256_storeu_pd(out.as_mut_ptr().add(idx), dd);
            if COUNT {
                let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(dd, threshold_v);
                count += _mm256_movemask_pd(mask).count_ones() as usize;
            }
            idx += 4;
        }
        if idx < end {
            sampson_residual_batch_scalar_tail(f, &samples[..end], &mut out[..end], idx);
            if COUNT {
                count += out[idx..end]
                    .iter()
                    .filter(|&&residual| residual < threshold)
                    .count();
            }
        }
        if BOUNDED && count + n - end <= best_inlier_count {
            return None;
        }
        start = end;
    }
    Some(count)
}

/// Broadcast the nine F entries into NEON lane vectors.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn broadcast_f_neon(f: FPacked) -> [std::arch::aarch64::float64x2_t; 9] {
    use std::arch::aarch64::vdupq_n_f64;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    [f00, f01, f02, f10, f11, f12, f20, f21, f22].map(|value| vdupq_n_f64(value))
}

/// `(e², d)` of the Sampson residual `e²/d` for two consecutive matches.
///
/// `Match2d2d` is `#[repr(C)]` `{x1: Vec2F64, x2: Vec2F64}` → 4 contiguous
/// f64s per match, so `vld4q_f64` deinterleaves two matches into
/// `(x1_x, x1_y, x2_x, x2_y)` lane vectors in a single instruction. Every
/// NEON kernel uses this exact operation sequence.
///
/// # Safety
/// `base` must point to two readable consecutive `Match2d2d` values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sampson_terms_neon(
    f: &[std::arch::aarch64::float64x2_t; 9],
    base: *const f64,
) -> (
    std::arch::aarch64::float64x2_t,
    std::arch::aarch64::float64x2_t,
) {
    use std::arch::aarch64::*;
    let lanes = vld4q_f64(base);
    let (x1, y1, x2, y2) = (lanes.0, lanes.1, lanes.2, lanes.3);
    let fx1x = vfmaq_f64(vfmaq_f64(f[2], x1, f[0]), y1, f[1]);
    let fx1y = vfmaq_f64(vfmaq_f64(f[5], x1, f[3]), y1, f[4]);
    let fx1z = vfmaq_f64(vfmaq_f64(f[8], x1, f[6]), y1, f[7]);
    let ftx2x = vfmaq_f64(vfmaq_f64(f[6], x2, f[0]), y2, f[3]);
    let ftx2y = vfmaq_f64(vfmaq_f64(f[7], x2, f[1]), y2, f[4]);
    let err = vfmaq_f64(vfmaq_f64(fx1z, x2, fx1x), y2, fx1y);
    let denom = vfmaq_f64(
        vfmaq_f64(vfmaq_f64(vmulq_f64(fx1x, fx1x), fx1y, fx1y), ftx2x, ftx2x),
        ftx2y,
        ftx2y,
    );
    (vmulq_f64(err, err), denom)
}

/// NEON support upper bound; see [`sampson_support_upper_bound`].
///
/// # Safety
/// NEON is architectural on aarch64; only complete lane pairs of `samples`
/// are read by vector loads.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sampson_support_upper_bound_neon(
    f: FPacked,
    samples: &[Match2d2d],
    threshold: f64,
    prune_at: usize,
) -> Option<usize> {
    use std::arch::aarch64::*;
    let fv = broadcast_f_neon(f);
    let bound = vdupq_n_f64(threshold * UPPER_BOUND_SLACK);
    let eps = vdupq_n_f64(1e-12);
    let n = samples.len();
    let simd_end = n & !1;
    // Lane counters: a true comparison is all ones, i.e. -1 as an integer.
    let mut possible = vdupq_n_u64(0);
    let mut idx = 0;
    let mut next_check = UPPER_BOUND_CHUNK.min(simd_end);
    while idx < simd_end {
        let (err_sq, denom) = sampson_terms_neon(&fv, samples.as_ptr().add(idx) as *const f64);
        let inside = vcleq_f64(err_sq, vmulq_f64(bound, denom));
        // `!(d > eps)` also catches NaN denominators.
        let degenerate = vmvnq_u32(vreinterpretq_u32_u64(vcgtq_f64(denom, eps)));
        possible = vsubq_u64(
            possible,
            vorrq_u64(inside, vreinterpretq_u64_u32(degenerate)),
        );
        idx += 2;
        if idx == next_check {
            if vaddvq_u64(possible) as usize + (n - idx) <= prune_at {
                return None;
            }
            next_check = (idx + UPPER_BOUND_CHUNK).min(simd_end);
        }
    }
    let count = vaddvq_u64(possible) as usize
        + sampson_tail_upper_bound(f, &samples[simd_end..], threshold);
    (count > prune_at).then_some(count)
}

/// Broadcast the nine F entries into AVX lane vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn broadcast_f_avx2(f: FPacked) -> [std::arch::x86_64::__m256d; 9] {
    use std::arch::x86_64::_mm256_set1_pd;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    [f00, f01, f02, f10, f11, f12, f20, f21, f22].map(|value| _mm256_set1_pd(value))
}

/// `(e², d)` of the Sampson residual `e²/d` for four consecutive matches.
///
/// `Match2d2d` is 32 B = exactly one `__m256d`. Four consecutive matches
/// are 4 × `__m256d` loads; we deinterleave them into the four needed
/// lane-vectors via the standard AVX 4×4 transpose
/// (`unpacklo` / `unpackhi` + two `permute2f128`):
///
/// ```text
///   Loaded:                      After transpose:
///     a = m[0].(x1x x1y x2x x2y)   x1_x = (m0.x1x, m1.x1x, m2.x1x, m3.x1x)
///     b = m[1].(...)               x1_y = (m0.x1y, m1.x1y, m2.x1y, m3.x1y)
///     c = m[2].(...)               x2_x = (m0.x2x, m1.x2x, m2.x2x, m3.x2x)
///     d = m[3].(...)               x2_y = (m0.x2y, m1.x2y, m2.x2y, m3.x2y)
/// ```
///
/// Every AVX2 kernel uses this exact operation sequence.
///
/// # Safety
/// The CPU must support AVX2 and FMA, and `base` must point to four
/// readable consecutive `Match2d2d` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sampson_terms_avx2(
    f: &[std::arch::x86_64::__m256d; 9],
    base: *const f64,
) -> (std::arch::x86_64::__m256d, std::arch::x86_64::__m256d) {
    use std::arch::x86_64::*;
    let a = _mm256_loadu_pd(base);
    let b = _mm256_loadu_pd(base.add(4));
    let c = _mm256_loadu_pd(base.add(8));
    let d = _mm256_loadu_pd(base.add(12));
    let t0 = _mm256_unpacklo_pd(a, b);
    let t1 = _mm256_unpackhi_pd(a, b);
    let t2 = _mm256_unpacklo_pd(c, d);
    let t3 = _mm256_unpackhi_pd(c, d);
    let x1 = _mm256_permute2f128_pd::<0x20>(t0, t2);
    let y1 = _mm256_permute2f128_pd::<0x20>(t1, t3);
    let x2 = _mm256_permute2f128_pd::<0x31>(t0, t2);
    let y2 = _mm256_permute2f128_pd::<0x31>(t1, t3);

    let fx1x = _mm256_fmadd_pd(x1, f[0], _mm256_fmadd_pd(y1, f[1], f[2]));
    let fx1y = _mm256_fmadd_pd(x1, f[3], _mm256_fmadd_pd(y1, f[4], f[5]));
    let fx1z = _mm256_fmadd_pd(x1, f[6], _mm256_fmadd_pd(y1, f[7], f[8]));
    let ftx2x = _mm256_fmadd_pd(x2, f[0], _mm256_fmadd_pd(y2, f[3], f[6]));
    let ftx2y = _mm256_fmadd_pd(x2, f[1], _mm256_fmadd_pd(y2, f[4], f[7]));

    let err = _mm256_fmadd_pd(fx1x, x2, _mm256_fmadd_pd(fx1y, y2, fx1z));
    let denom = _mm256_fmadd_pd(
        ftx2y,
        ftx2y,
        _mm256_fmadd_pd(
            ftx2x,
            ftx2x,
            _mm256_fmadd_pd(fx1y, fx1y, _mm256_mul_pd(fx1x, fx1x)),
        ),
    );
    (_mm256_mul_pd(err, err), denom)
}

/// Sum of the four 64-bit lane counters.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_epi64(v: std::arch::x86_64::__m256i) -> usize {
    use std::arch::x86_64::*;
    let pair = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    (_mm_cvtsi128_si64(pair) + _mm_extract_epi64::<1>(pair)) as usize
}

/// AVX2 support upper bound; see [`sampson_support_upper_bound`].
///
/// # Safety
/// The CPU must support AVX2 and FMA; only complete groups of four
/// `samples` are read by vector loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sampson_support_upper_bound_avx2(
    f: FPacked,
    samples: &[Match2d2d],
    threshold: f64,
    prune_at: usize,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let fv = broadcast_f_avx2(f);
    let bound = _mm256_set1_pd(threshold * UPPER_BOUND_SLACK);
    let eps = _mm256_set1_pd(1e-12);
    let n = samples.len();
    let simd_end = n & !3;
    // Lane counters: a true comparison is all ones, i.e. -1 as an integer.
    let mut possible = _mm256_setzero_si256();
    let mut idx = 0;
    let mut next_check = UPPER_BOUND_CHUNK.min(simd_end);
    while idx < simd_end {
        let (err_sq, denom) = sampson_terms_avx2(&fv, samples.as_ptr().add(idx) as *const f64);
        let inside = _mm256_cmp_pd::<_CMP_LE_OQ>(err_sq, _mm256_mul_pd(bound, denom));
        // `!(d > eps)` also catches NaN denominators.
        let degenerate = _mm256_cmp_pd::<_CMP_NGT_UQ>(denom, eps);
        possible = _mm256_sub_epi64(
            possible,
            _mm256_castpd_si256(_mm256_or_pd(inside, degenerate)),
        );
        idx += 4;
        if idx == next_check {
            if horizontal_sum_epi64(possible) + (n - idx) <= prune_at {
                return None;
            }
            next_check = (idx + UPPER_BOUND_CHUNK).min(simd_end);
        }
    }
    let count = horizontal_sum_epi64(possible)
        + sampson_tail_upper_bound(f, &samples[simd_end..], threshold);
    (count > prune_at).then_some(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;

    /// Smoke test: feed 8 noise-free correspondences from a synthetic geometry,
    /// verify the trait produces an F whose Sampson residuals are ~0 on the
    /// fitting set. Mirrors the existing `fundamental_8point` test but routes
    /// through the trait surface so a future signature change breaks loudly.
    #[test]
    fn fits_and_scores_clean_correspondences() {
        let k_inv_t_e_k_inv = synthetic_pair();
        let mut models = Vec::new();
        let est = FundamentalEstimator;
        est.fit(&k_inv_t_e_k_inv.matches, &mut models);
        assert_eq!(models.len(), 1, "expected exactly one F from 8-pt");
        let f = models[0];
        for m in &k_inv_t_e_k_inv.matches {
            let d = est.residual(&f, m);
            assert!(d < 1e-8, "Sampson residual too large: {d}");
        }
    }

    /// `residual_batch` (whichever kernel the dispatcher picks) must match
    /// the scalar `residual` path element-wise on identical input.
    /// Catches lane-ordering bugs in the AoS→SoA loads (NEON `vld4q_f64`,
    /// AVX2 4×4 transpose) and bad branchless masking on the denom guard.
    #[test]
    fn batch_dispatcher_matches_scalar_residual() {
        let pair = synthetic_pair();
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches, &mut models);
        let f = models[0];

        // Build a longer sample slice (odd N → exercises the scalar tail).
        let mut samples = pair.matches.clone();
        samples.extend(pair.matches.iter().take(3).copied());
        assert_eq!(samples.len() % 2, 1, "odd N to hit the scalar tail");

        let mut batched = vec![0.0f64; samples.len()];
        est.residual_batch(&f, &samples, &mut batched);

        for (i, s) in samples.iter().enumerate() {
            let scalar = est.residual(&f, s);
            // 1e-12 absolute is the NEON↔scalar floor for FMA reordering.
            assert!(
                (batched[i] - scalar).abs() < 1e-12 * scalar.max(1.0).abs(),
                "lane {i}: batched={} scalar={} (Δ={})",
                batched[i],
                scalar,
                batched[i] - scalar
            );
        }
    }

    /// The fused path retains the strict threshold predicate used by
    /// `ThresholdConsensus`, including degenerate and non-finite residuals.
    #[test]
    fn fused_threshold_scoring_keeps_strict_and_nonfinite_semantics() {
        let est = FundamentalEstimator;
        let samples = vec![Match2d2d::new(Vec2F64::new(1.0, 2.0), Vec2F64::new(3.0, 4.0)); 129];
        let mut residuals = vec![0.0; samples.len()];
        let mut mask = Vec::new();

        // The zero matrix gives a degenerate denominator and residual 0.
        // Equality at zero is excluded because the predicate is strict.
        let result =
            est.threshold_inliers(&Mat3F64::ZERO, &samples, 0.0, 0, &mut residuals, &mut mask);
        assert_eq!(result, Some(ThresholdInlierResult::Pruned));
        assert!(mask.iter().take(128).all(|&is_inlier| !is_inlier));

        let result =
            est.threshold_inliers(&Mat3F64::ZERO, &samples, 1.0, 0, &mut residuals, &mut mask);
        assert_eq!(result, Some(ThresholdInlierResult::Complete(samples.len())));
        assert!(mask.iter().all(|&is_inlier| is_inlier));

        let nan_model = Mat3F64::from_cols(
            Vec3F64::new(f64::NAN, 0.0, 0.0),
            Vec3F64::ZERO,
            Vec3F64::ZERO,
        );
        mask.fill(true);
        let result = est.threshold_inliers(
            &nan_model,
            &samples,
            f64::INFINITY,
            0,
            &mut residuals,
            &mut mask,
        );
        assert_eq!(result, Some(ThresholdInlierResult::Pruned));
        assert!(mask.iter().all(|&is_inlier| is_inlier));
    }

    /// The fused counter must agree exactly with the residual dispatcher over
    /// SIMD boundaries, tails, and strict threshold edge cases.
    #[test]
    fn fused_threshold_scoring_matches_batch_masks_and_counts() {
        let pair = synthetic_pair();
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches, &mut models);
        let f = models[0];

        for &n in &[1usize, 2, 3, 4, 5, 127, 128, 129] {
            let samples: Vec<_> = pair.matches.iter().copied().cycle().take(n).collect();
            let mut expected_residuals = vec![0.0; n];
            est.residual_batch(&f, &samples, &mut expected_residuals);
            let max_residual = expected_residuals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            for threshold in [max_residual, f64::INFINITY] {
                let expected_mask: Vec<_> = expected_residuals
                    .iter()
                    .map(|&residual| residual < threshold)
                    .collect();
                let expected_count = expected_mask.iter().filter(|&&inlier| inlier).count();

                let mut residuals = vec![f64::NAN; n];
                let mut mask = vec![false; n];
                let expected_result = if expected_count == 0 {
                    ThresholdInlierResult::Pruned
                } else {
                    ThresholdInlierResult::Complete(expected_count)
                };
                assert_eq!(
                    est.threshold_inliers(&f, &samples, threshold, 0, &mut residuals, &mut mask),
                    Some(expected_result),
                    "threshold={threshold:?}, n={n}"
                );
                if expected_count > 0 {
                    assert_eq!(residuals, expected_residuals, "residuals at n={n}");
                    assert_eq!(mask, expected_mask, "mask at n={n}");
                }
            }

            let mut residuals = vec![f64::NAN; n];
            let mut mask = vec![true; n];
            assert_eq!(
                est.threshold_inliers(&f, &samples, f64::NAN, 0, &mut residuals, &mut mask),
                Some(ThresholdInlierResult::Pruned),
                "NaN threshold must reject every lane at n={n}"
            );
        }
    }

    /// The AVX2 entry point is checked directly so a scalar dispatcher fallback
    /// cannot hide an error in the fused counter.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fused_threshold_kernel_matches_dispatcher() {
        if !kornia_imgproc::simd::cpu_features().has_avx2
            || !kornia_imgproc::simd::cpu_features().has_fma
        {
            return;
        }

        let pair = synthetic_pair();
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches, &mut models);
        let f = models[0];
        let samples: Vec<_> = pair.matches.iter().copied().cycle().take(129).collect();
        let mut expected = vec![0.0; samples.len()];
        est.residual_batch(&f, &samples, &mut expected);
        let threshold = expected.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected_count = expected
            .iter()
            .filter(|&&residual| residual < threshold)
            .count();

        let packed = pack_f(&f);
        let mut actual = vec![f64::NAN; samples.len()];
        // SAFETY: the runtime probe above verifies AVX2 and FMA before calling
        // the target-feature kernel; `actual` has one entry per sample.
        let count = unsafe {
            sampson_residual_batch_avx2::<true, false>(packed, &samples, &mut actual, threshold, 0)
        }
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(count, expected_count);
    }

    /// The NEON entry point is checked directly for the same reason as AVX2.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_threshold_kernel_matches_dispatcher() {
        let pair = synthetic_pair();
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches, &mut models);
        let f = models[0];
        let samples: Vec<_> = pair.matches.iter().copied().cycle().take(129).collect();
        let mut expected = vec![0.0; samples.len()];
        est.residual_batch(&f, &samples, &mut expected);
        let threshold = expected.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected_count = expected
            .iter()
            .filter(|&&residual| residual < threshold)
            .count();

        let packed = pack_f(&f);
        let mut actual = vec![f64::NAN; samples.len()];
        // SAFETY: NEON is architectural on aarch64; `actual` has one entry
        // per sample and therefore satisfies the kernel's slice invariant.
        let count = unsafe {
            sampson_residual_batch_neon::<true, false>(packed, &samples, &mut actual, threshold, 0)
        }
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(count, expected_count);
    }

    /// The division-free support bound must never undercount the exact
    /// strict count, including at thresholds equal to a residual, across
    /// SIMD tails and with degenerate or non-finite correspondences, and it
    /// may only prune a candidate whose exact count cannot win.
    type SupportBound = fn(FPacked, &[Match2d2d], f64, usize) -> Option<usize>;

    #[test]
    fn support_upper_bound_never_undercounts_exact_count() {
        let mut state = 0x0b0e_5eed_u64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
        };
        for trial in 0..60 {
            let n = 1 + trial * 7 % 97;
            let model = Mat3F64::from_cols(
                Vec3F64::new(uniform() - 0.5, uniform() - 0.5, uniform() - 0.5),
                Vec3F64::new(uniform() - 0.5, uniform() - 0.5, uniform() - 0.5),
                Vec3F64::new(uniform() - 0.5, uniform() - 0.5, uniform() - 0.5),
            ) * 1e-3;
            let mut samples: Vec<_> = (0..n)
                .map(|_| {
                    Match2d2d::new(
                        Vec2F64::new(640.0 * uniform(), 480.0 * uniform()),
                        Vec2F64::new(640.0 * uniform(), 480.0 * uniform()),
                    )
                })
                .collect();
            if trial % 5 == 0 {
                samples[n / 2].x1.x = f64::NAN;
            }
            let models = if trial % 7 == 0 {
                vec![model, Mat3F64::ZERO]
            } else {
                vec![model]
            };
            for f in models {
                let mut residuals = vec![0.0; n];
                FundamentalEstimator.residual_batch(&f, &samples, &mut residuals);
                let mut thresholds: Vec<f64> = residuals
                    .iter()
                    .copied()
                    .filter(|r| r.is_finite())
                    .collect();
                thresholds.extend([0.0, 1e-6, 1.0, 1e6]);
                for threshold in thresholds {
                    let exact = residuals.iter().filter(|&&r| r < threshold).count();
                    let bounds: [SupportBound; 2] = [
                        sampson_support_upper_bound,
                        sampson_support_upper_bound_scalar,
                    ];
                    for bound in bounds {
                        let total = bound(pack_f(&f), &samples, threshold, 0);
                        assert!(total.unwrap_or(0) >= exact, "n={n} t={threshold}");
                        for prune_at in [exact.saturating_sub(1), exact, exact + 1] {
                            if bound(pack_f(&f), &samples, threshold, prune_at).is_none() {
                                assert!(exact <= prune_at, "pruned a winner: n={n} t={threshold}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Below-minimal input must yield zero candidate models without panicking.
    #[test]
    fn under_min_samples_yields_no_model() {
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&[], &mut models);
        assert!(models.is_empty());
        let one = Match2d2d::new(Vec2F64::new(0.0, 0.0), Vec2F64::new(0.0, 0.0));
        est.fit(&[one; 6], &mut models);
        assert!(models.is_empty());
    }

    #[test]
    fn seven_point_candidates_and_eight_point_refit() {
        let pair = synthetic_pair();
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches[..7], &mut models);
        assert!((1..=3).contains(&models.len()));
        for f in &models {
            assert!(pair.matches[..7].iter().all(|m| est.residual(f, m) < 1e-8));
        }
        assert!(models
            .iter()
            .any(|f| est.residual(f, &pair.matches[7]) < 1e-8));

        models.clear();
        est.refit(&pair.matches, &mut models);
        let mut baseline = Vec::new();
        Fundamental8PointEstimator.refit(&pair.matches, &mut baseline);
        assert_eq!(
            models, baseline,
            "non-minimal refit must retain the eight-point path"
        );
        models.clear();
        est.refit(&pair.matches[..7], &mut models);
        assert!(!models.is_empty(), "seven-inlier refit must remain valid");
    }

    #[test]
    fn seven_point_ransac_scores_every_root() {
        use crate::ransac::{run, run_parallel, RansacConfig, Sampler, ThresholdConsensus};
        struct FixedSample;
        impl Sampler for FixedSample {
            fn sample(&mut self, _n: usize, out: &mut [usize]) {
                assert_eq!(out.len(), 7);
                for (i, index) in out.iter_mut().enumerate() {
                    *index = i;
                }
            }
        }
        /// Seven-point estimator that returns the root fitting `holdout`
        /// last, so the driver must score every root to find the winner
        /// whatever order the solver produces.
        struct TrueRootLast {
            holdout: Match2d2d,
        }
        impl Estimator for TrueRootLast {
            type Model = Mat3F64;
            type Sample = Match2d2d;
            const SAMPLE_SIZE: usize = FundamentalEstimator::SAMPLE_SIZE;

            fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
                FundamentalEstimator.fit(samples, out);
                out.sort_by_key(|f| FundamentalEstimator.residual(f, &self.holdout) < 1e-8);
            }

            fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
                FundamentalEstimator.residual(model, sample)
            }

            fn residual_batch(
                &self,
                model: &Self::Model,
                samples: &[Self::Sample],
                out: &mut [f64],
            ) {
                FundamentalEstimator.residual_batch(model, samples, out);
            }

            fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
                FundamentalEstimator.refit(inliers, out);
            }

            fn threshold_inliers(
                &self,
                model: &Self::Model,
                samples: &[Self::Sample],
                threshold: f64,
                best_inlier_count: usize,
                residuals: &mut [f64],
                inliers_out: &mut Vec<bool>,
            ) -> Option<ThresholdInlierResult> {
                FundamentalEstimator.threshold_inliers(
                    model,
                    samples,
                    threshold,
                    best_inlier_count,
                    residuals,
                    inliers_out,
                )
            }
        }
        let pair = synthetic_pair();
        let estimator = TrueRootLast {
            holdout: pair.matches[7],
        };
        let mut hypotheses = Vec::new();
        estimator.fit(&pair.matches[..7], &mut hypotheses);
        assert!(hypotheses.len() > 1);
        assert!(
            estimator.residual(&hypotheses[0], &pair.matches[7]) > 1e-8,
            "fixture must require scoring a later root to fit the holdout"
        );
        let cfg = RansacConfig {
            max_iters: 1,
            lo_every: 1,
            ..Default::default()
        };
        let consensus = ThresholdConsensus { threshold: 1e-8 };
        for result in [
            run(
                &estimator,
                &consensus,
                &mut FixedSample,
                &pair.matches,
                &cfg,
            ),
            run_parallel(
                &estimator,
                &consensus,
                &mut FixedSample,
                &pair.matches,
                &cfg,
            ),
        ] {
            assert!(result.model.is_some());
            assert_eq!(result.inlier_count(), pair.matches.len());
            assert_eq!(
                result.num_iters, 1,
                "iterations count samples, rather than roots"
            );
        }
    }

    struct Pair {
        matches: Vec<Match2d2d>,
    }

    fn synthetic_pair() -> Pair {
        let k_fx = 500.0;
        let k_fy = 500.0;
        let k_cx = 320.0;
        let k_cy = 240.0;
        let angle = 0.1_f64;
        let r = [
            [angle.cos(), 0.0, -angle.sin()],
            [0.0, 1.0, 0.0],
            [angle.sin(), 0.0, angle.cos()],
        ];
        let t = [1.0_f64, 0.0, 0.2];

        let pts = [
            Vec3F64::new(-0.5, -0.3, 4.0),
            Vec3F64::new(0.4, -0.2, 3.5),
            Vec3F64::new(-0.3, 0.5, 5.0),
            Vec3F64::new(0.6, 0.4, 4.5),
            Vec3F64::new(-0.1, -0.6, 3.0),
            Vec3F64::new(0.2, 0.3, 6.0),
            Vec3F64::new(-0.4, 0.1, 3.8),
            Vec3F64::new(0.5, -0.5, 4.2),
        ];

        let mut matches = Vec::with_capacity(pts.len());
        for p in &pts {
            let u1 = k_fx * p.x / p.z + k_cx;
            let v1 = k_fy * p.y / p.z + k_cy;
            let pc2 = [
                r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
                r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
                r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
            ];
            let u2 = k_fx * pc2[0] / pc2[2] + k_cx;
            let v2 = k_fy * pc2[1] / pc2[2] + k_cy;
            matches.push(Match2d2d::new(Vec2F64::new(u1, v1), Vec2F64::new(u2, v2)));
        }
        Pair { matches }
    }
}
