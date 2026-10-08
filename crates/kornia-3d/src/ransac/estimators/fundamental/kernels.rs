//! AoS Sampson residual kernels and strict-threshold bounded scoring.
//!
//! Only estimator dispatch belongs in the parent module. This layer owns
//! architecture detection, matrix packing, SIMD lanes and scalar tails.

use crate::ransac::Match2d2d;
use kornia_algebra::Mat3F64;

/// Evaluate a batch with the existing residual dispatcher.
#[inline]
pub(super) fn sampson_residual_batch(f: FPacked, samples: &[Match2d2d], out: &mut [f64]) {
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
pub(super) fn sampson_residual_batch_threshold(
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
pub(super) type FPacked = (f64, f64, f64, f64, f64, f64, f64, f64, f64);

/// Residuals per fused hard-threshold pass. This keeps SIMD dispatch overhead
/// amortized while allowing weak seven-point roots to be rejected early.
const THRESHOLD_SCORE_CHUNK: usize = 128;

#[inline(always)]
pub(super) fn pack_f(model: &Mat3F64) -> FPacked {
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
pub(super) unsafe fn sampson_residual_batch_neon<const COUNT: bool, const BOUNDED: bool>(
    f: FPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    threshold: f64,
    best_inlier_count: usize,
) -> Option<usize> {
    // SAFETY: The caller guarantees NEON support and sufficient output space.
    // Match2d2d has four contiguous f64 fields; each vld4q load is guarded by
    // two remaining samples and each output store writes the same two lanes.
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
pub(super) unsafe fn sampson_residual_batch_avx2<const COUNT: bool, const BOUNDED: bool>(
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
