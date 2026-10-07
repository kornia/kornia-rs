//! Fundamental-matrix estimator.
//!
//! Seven-point minimal hypotheses and eight-point inlier refinement share
//! the same SIMD Sampson scoring. The eight-point estimator is also available
//! for callers that need the previous sampling behavior.

use kornia_algebra::{Mat3F64, Vec2F64};

use crate::pose::fundamental_7point_into;
use crate::pose::{fundamental_8point, sampson_distance};
use crate::ransac::{clamp_pair, Estimator, Match2d2d, ThresholdInlierResult};

/// Seven-point fundamental matrix estimator from 2D-2D pixel correspondences.
///
/// Minimal fitting returns up to three real solutions; inlier sets of eight
/// or more matches are refined with the normalized eight-point solver.
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
            if let Ok(count) = fundamental_7point_into(&x1, &x2, &mut models) {
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

    /// Scores in fixed SIMD-friendly chunks. A losing seven-point root often
    /// needs only the first chunk: after it cannot exceed the incumbent,
    /// RANSAC can skip both the remaining residuals and mask writes.
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

/// 2-lane f64 NEON kernel.
///
/// `Match2d2d` is `#[repr(C)]` `{x1: Vec2F64, x2: Vec2F64}` → 4 contiguous
/// f64s per match. Two consecutive matches are exactly the 8 f64s
/// `vld4q_f64` reads and deinterleaves into `(x1_x, x1_y, x2_x, x2_y)`
/// lane-vectors — perfect AoS→SoA load in a single instruction.
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
                let base = samples.as_ptr().add(idx) as *const f64;
                let lanes = vld4q_f64(base);
                let x1 = lanes.0;
                let y1 = lanes.1;
                let x2 = lanes.2;
                let y2 = lanes.3;

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

/// 4-lane f64 AVX2+FMA kernel.
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
            let base = samples.as_ptr().add(idx) as *const f64;
            // Each Match2d2d is 32 B = exactly one __m256d. Load 4 of them.
            let a = _mm256_loadu_pd(base);
            let b = _mm256_loadu_pd(base.add(4));
            let c = _mm256_loadu_pd(base.add(8));
            let d = _mm256_loadu_pd(base.add(12));
            // 4×4 transpose: per-128b-lane unpack, then cross-lane permute.
            let t0 = _mm256_unpacklo_pd(a, b);
            let t1 = _mm256_unpackhi_pd(a, b);
            let t2 = _mm256_unpacklo_pd(c, d);
            let t3 = _mm256_unpackhi_pd(c, d);
            let x1 = _mm256_permute2f128_pd::<0x20>(t0, t2);
            let y1 = _mm256_permute2f128_pd::<0x20>(t1, t3);
            let x2 = _mm256_permute2f128_pd::<0x31>(t0, t2);
            let y2 = _mm256_permute2f128_pd::<0x31>(t1, t3);

            let fx1x = _mm256_fmadd_pd(x1, f00v, _mm256_fmadd_pd(y1, f01v, f02v));
            let fx1y = _mm256_fmadd_pd(x1, f10v, _mm256_fmadd_pd(y1, f11v, f12v));
            let fx1z = _mm256_fmadd_pd(x1, f20v, _mm256_fmadd_pd(y1, f21v, f22v));
            let ftx2x = _mm256_fmadd_pd(x2, f00v, _mm256_fmadd_pd(y2, f10v, f20v));
            let ftx2y = _mm256_fmadd_pd(x2, f01v, _mm256_fmadd_pd(y2, f11v, f21v));

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
            let err_sq = _mm256_mul_pd(err, err);
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
        let pair = synthetic_pair();
        let mut hypotheses = Vec::new();
        FundamentalEstimator.fit(&pair.matches[..7], &mut hypotheses);
        assert!(hypotheses.len() > 1);
        assert!(
            FundamentalEstimator.residual(&hypotheses[0], &pair.matches[7]) > 1e-8,
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
                &FundamentalEstimator,
                &consensus,
                &mut FixedSample,
                &pair.matches,
                &cfg,
            ),
            run_parallel(
                &FundamentalEstimator,
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
