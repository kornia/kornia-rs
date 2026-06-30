//! Homography estimator (4-point minimal solver).
//!
//! Wraps [`crate::pose::homography::homography_4pt2d`], the 8×8 LU-based
//! minimal solver kept hot for inside-RANSAC use. Non-minimal refits (LO
//! step, final inlier-set polish) should call
//! `crate::pose::homography::homography_dlt` directly.

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

use crate::pose::{homography_4pt2d, homography_dlt};
use crate::ransac::{Estimator, Match2d2d};

/// Estimator for a planar homography from 2D-2D pixel correspondences.
///
/// **Coordinate convention.** Pixel coordinates. The minimal solver does
/// not Hartley-normalize internally — the 4-point system is small enough
/// that LU on the raw 8×8 stays well-conditioned for typical image scales.
///
/// **Residual.** One-sided forward transfer error squared, i.e.
/// `‖x2 - π(H · [x1, 1])‖²` in pixel units. Symmetric transfer error is
/// available as a future variant; the asymmetric form is the OpenCV
/// `findHomography` default and matches what most downstream code expects.
#[derive(Debug, Clone, Copy, Default)]
pub struct HomographyEstimator;

impl Estimator for HomographyEstimator {
    type Model = Mat3F64;
    type Sample = Match2d2d;
    const SAMPLE_SIZE: usize = 4;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() < Self::SAMPLE_SIZE {
            return;
        }
        let x1 = [
            [samples[0].x1.x, samples[0].x1.y],
            [samples[1].x1.x, samples[1].x1.y],
            [samples[2].x1.x, samples[2].x1.y],
            [samples[3].x1.x, samples[3].x1.y],
        ];
        let x2 = [
            [samples[0].x2.x, samples[0].x2.y],
            [samples[1].x2.x, samples[1].x2.y],
            [samples[2].x2.x, samples[2].x2.y],
            [samples[3].x2.x, samples[3].x2.y],
        ];
        let mut h = [[0.0f64; 3]; 3];
        if homography_4pt2d(&x1, &x2, &mut h).is_ok() {
            out.push(Mat3F64::from_cols(
                Vec3F64::new(h[0][0], h[1][0], h[2][0]),
                Vec3F64::new(h[0][1], h[1][1], h[2][1]),
                Vec3F64::new(h[0][2], h[1][2], h[2][2]),
            ));
        }
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        let x1h = Vec3F64::new(sample.x1.x, sample.x1.y, 1.0);
        let mapped = *model * x1h;
        // Behind-the-plane / numerically degenerate H · x → ∞ residual so
        // the consensus step rejects it without poisoning the score.
        if mapped.z.abs() < 1e-12 {
            return f64::INFINITY;
        }
        let dx = mapped.x / mapped.z - sample.x2.x;
        let dy = mapped.y / mapped.z - sample.x2.y;
        dx * dx + dy * dy
    }

    /// LO-friendly refit on a (typically larger) inlier set.
    ///
    /// The minimal `homography_4pt2d` solver doesn't accept N > 4 (its
    /// 8×8 LU is fixed-size by design), so for over-determined input we
    /// switch to `homography_dlt`'s SVD-of-AᵀA path. The minimal case
    /// still flows through `fit` for consistency with the rest of the
    /// trait surface.
    fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
        let n = inliers.len();
        if n < Self::SAMPLE_SIZE {
            return;
        }
        if n == Self::SAMPLE_SIZE {
            self.fit(inliers, out);
            return;
        }
        let x1: Vec<Vec2F64> = inliers.iter().map(|s| s.x1).collect();
        let x2: Vec<Vec2F64> = inliers.iter().map(|s| s.x2).collect();
        if let Ok(h) = homography_dlt(&x1, &x2) {
            out.push(h);
        }
    }

    /// Dispatcher mirroring the FundamentalEstimator pattern:
    /// aarch64 → NEON (`vld4q_f64` AoS→SoA), x86_64 → AVX2+FMA when probed,
    /// otherwise the portable scalar reference. All three paths produce
    /// byte-equal results to FMA-reordering noise.
    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        debug_assert_eq!(out.len(), samples.len());
        let h = pack_h(model);

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let idx = transfer_error_batch_neon(h, samples, out);
            transfer_error_batch_scalar_tail(h, samples, out, idx);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe {
                let idx = transfer_error_batch_avx2(h, samples, out);
                transfer_error_batch_scalar_tail(h, samples, out, idx);
            }
            return;
        }

        #[allow(unreachable_code)]
        transfer_error_batch_scalar(h, samples, out);
    }
}

// ---------------------------------------------------------------------------
// Forward-transfer-error kernels (scalar + NEON + AVX2)
//
// Same dispatcher convention as fundamental.rs: scalar reference is the
// numeric ground truth; SIMD paths return the index processed and the
// dispatcher invokes the scalar tail for the remainder.
// ---------------------------------------------------------------------------

/// 9 entries of `H` in row-major order — broadcast-ready for SIMD lanes.
type HPacked = (f64, f64, f64, f64, f64, f64, f64, f64, f64);

#[inline(always)]
fn pack_h(model: &Mat3F64) -> HPacked {
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

/// Portable scalar transfer error — single source of numeric truth.
#[inline]
fn transfer_error_batch_scalar(h: HPacked, samples: &[Match2d2d], out: &mut [f64]) {
    let (h00, h01, h02, h10, h11, h12, h20, h21, h22) = h;
    for (i, s) in samples.iter().enumerate() {
        let (x, y) = (s.x1.x, s.x1.y);
        let mx = h00 * x + h01 * y + h02;
        let my = h10 * x + h11 * y + h12;
        let mz = h20 * x + h21 * y + h22;
        if mz.abs() < 1e-12 {
            out[i] = f64::INFINITY;
            continue;
        }
        let inv_z = 1.0 / mz;
        let dx = mx * inv_z - s.x2.x;
        let dy = my * inv_z - s.x2.y;
        out[i] = dx * dx + dy * dy;
    }
}

#[inline]
fn transfer_error_batch_scalar_tail(
    h: HPacked,
    samples: &[Match2d2d],
    out: &mut [f64],
    start: usize,
) {
    if start >= samples.len() {
        return;
    }
    transfer_error_batch_scalar(h, &samples[start..], &mut out[start..]);
}

/// 2-lane f64 NEON kernel — same `vld4q_f64` AoS→SoA trick used by F.
///
/// # Safety
/// - aarch64 architectural; `target_feature(neon)` for the intrinsics.
/// - `out.len() >= samples.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn transfer_error_batch_neon(h: HPacked, samples: &[Match2d2d], out: &mut [f64]) -> usize {
    unsafe {
        use std::arch::aarch64::*;
        let (h00, h01, h02, h10, h11, h12, h20, h21, h22) = h;
        let h00v = vdupq_n_f64(h00);
        let h01v = vdupq_n_f64(h01);
        let h02v = vdupq_n_f64(h02);
        let h10v = vdupq_n_f64(h10);
        let h11v = vdupq_n_f64(h11);
        let h12v = vdupq_n_f64(h12);
        let h20v = vdupq_n_f64(h20);
        let h21v = vdupq_n_f64(h21);
        let h22v = vdupq_n_f64(h22);
        let eps = vdupq_n_f64(1e-12);
        let inf = vdupq_n_f64(f64::INFINITY);

        let n = samples.len();
        let mut idx = 0usize;
        while idx + 2 <= n {
            let base = samples.as_ptr().add(idx) as *const f64;
            let lanes = vld4q_f64(base);
            let x1 = lanes.0;
            let y1 = lanes.1;
            let x2 = lanes.2;
            let y2 = lanes.3;

            let mx = vfmaq_f64(vfmaq_f64(h02v, x1, h00v), y1, h01v);
            let my = vfmaq_f64(vfmaq_f64(h12v, x1, h10v), y1, h11v);
            let mz = vfmaq_f64(vfmaq_f64(h22v, x1, h20v), y1, h21v);

            // Behind-plane mask: |mz| > eps (use bitcast-abs via subtraction trick).
            let abs_mz = vabsq_f64(mz);
            let mz_ok = vcgtq_f64(abs_mz, eps);
            // Avoid divide-by-zero: replace mz with 1.0 in unsafe lanes.
            let safe_mz = vbslq_f64(mz_ok, mz, vdupq_n_f64(1.0));
            let inv_z = vdivq_f64(vdupq_n_f64(1.0), safe_mz);
            let dx = vsubq_f64(vmulq_f64(mx, inv_z), x2);
            let dy = vsubq_f64(vmulq_f64(my, inv_z), y2);
            let dd = vfmaq_f64(vmulq_f64(dx, dx), dy, dy);
            // Replace bad-mz lanes with +∞ (matches scalar branch).
            let result = vbslq_f64(mz_ok, dd, inf);

            vst1q_f64(out.as_mut_ptr().add(idx), result);
            idx += 2;
        }
        idx
    }
}

/// 4-lane f64 AVX2+FMA kernel — same 4×4 transpose trick as F.
///
/// # Safety
/// - Caller has runtime-checked `cpu_features().has_avx2`.
/// - `out.len() >= samples.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn transfer_error_batch_avx2(h: HPacked, samples: &[Match2d2d], out: &mut [f64]) -> usize {
    use std::arch::x86_64::*;
    let (h00, h01, h02, h10, h11, h12, h20, h21, h22) = h;
    let h00v = _mm256_set1_pd(h00);
    let h01v = _mm256_set1_pd(h01);
    let h02v = _mm256_set1_pd(h02);
    let h10v = _mm256_set1_pd(h10);
    let h11v = _mm256_set1_pd(h11);
    let h12v = _mm256_set1_pd(h12);
    let h20v = _mm256_set1_pd(h20);
    let h21v = _mm256_set1_pd(h21);
    let h22v = _mm256_set1_pd(h22);
    let one = _mm256_set1_pd(1.0);
    let eps = _mm256_set1_pd(1e-12);
    let inf = _mm256_set1_pd(f64::INFINITY);
    // |x| via and-NOT on the sign bit.
    let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffffi64));

    let n = samples.len();
    let mut idx = 0usize;
    while idx + 4 <= n {
        let base = samples.as_ptr().add(idx) as *const f64;
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

        let mx = _mm256_fmadd_pd(x1, h00v, _mm256_fmadd_pd(y1, h01v, h02v));
        let my = _mm256_fmadd_pd(x1, h10v, _mm256_fmadd_pd(y1, h11v, h12v));
        let mz = _mm256_fmadd_pd(x1, h20v, _mm256_fmadd_pd(y1, h21v, h22v));

        let abs_mz = _mm256_and_pd(mz, abs_mask);
        let mz_ok = _mm256_cmp_pd::<_CMP_GT_OQ>(abs_mz, eps);
        let safe_mz = _mm256_blendv_pd(one, mz, mz_ok);
        let inv_z = _mm256_div_pd(one, safe_mz);
        let dx = _mm256_sub_pd(_mm256_mul_pd(mx, inv_z), x2);
        let dy = _mm256_sub_pd(_mm256_mul_pd(my, inv_z), y2);
        let dd = _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx));
        let result = _mm256_blendv_pd(inf, dd, mz_ok);

        _mm256_storeu_pd(out.as_mut_ptr().add(idx), result);
        idx += 4;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec2F64;

    /// Build 4 correspondences from a known H, fit through the trait, verify
    /// transfer error is ~0 on the fitting set.
    #[test]
    fn fits_and_scores_clean_correspondences() {
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.05, 0.0),
            Vec3F64::new(0.03, 0.95, 0.0),
            Vec3F64::new(7.0, -3.0, 1.0),
        );
        let pts = [
            Vec2F64::new(10.0, 20.0),
            Vec2F64::new(100.0, 30.0),
            Vec2F64::new(80.0, 200.0),
            Vec2F64::new(20.0, 180.0),
        ];
        let matches: Vec<Match2d2d> = pts
            .iter()
            .map(|p| {
                let mapped = h_true * Vec3F64::new(p.x, p.y, 1.0);
                let mp = Vec2F64::new(mapped.x / mapped.z, mapped.y / mapped.z);
                Match2d2d::new(*p, mp)
            })
            .collect();

        let est = HomographyEstimator;
        let mut models = Vec::new();
        est.fit(&matches, &mut models);
        assert_eq!(models.len(), 1);

        let h = models[0];
        for m in &matches {
            let r = est.residual(&h, m);
            assert!(r < 1e-10, "transfer error too large: {r}");
        }
    }

    /// SIMD `residual_batch` must match scalar `residual` element-wise.
    /// Catches lane-ordering / mask bugs on either NEON or AVX2 paths.
    #[test]
    fn batch_dispatcher_matches_scalar_residual() {
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.05, 0.0),
            Vec3F64::new(0.03, 0.95, 0.0),
            Vec3F64::new(7.0, -3.0, 1.0),
        );
        // Mix of regular + adversarial cases (point that maps near the
        // vanishing line → tiny `mz`, exercises the infinity branch).
        let pts = [
            Vec2F64::new(10.0, 20.0),
            Vec2F64::new(100.0, 30.0),
            Vec2F64::new(80.0, 200.0),
            Vec2F64::new(20.0, 180.0),
            Vec2F64::new(-50.0, 40.0),
            Vec2F64::new(300.0, -200.0),
            Vec2F64::new(0.0, 0.0),
        ];
        let matches: Vec<Match2d2d> = pts
            .iter()
            .map(|p| Match2d2d::new(*p, Vec2F64::new(0.0, 0.0)))
            .collect();
        assert_eq!(matches.len() % 2, 1, "odd N to exercise scalar tail");

        let est = HomographyEstimator;
        let mut batched = vec![0.0f64; matches.len()];
        est.residual_batch(&h_true, &matches, &mut batched);
        for (i, m) in matches.iter().enumerate() {
            let scalar = est.residual(&h_true, m);
            if scalar.is_finite() {
                assert!(
                    (batched[i] - scalar).abs() < 1e-12 * scalar.max(1.0).abs(),
                    "lane {i}: batched={} scalar={}",
                    batched[i],
                    scalar
                );
            } else {
                assert!(
                    !batched[i].is_finite(),
                    "lane {i}: scalar=∞ but batched={}",
                    batched[i]
                );
            }
        }
    }

    /// Below-minimal sample → no model.
    #[test]
    fn under_min_samples_yields_no_model() {
        let est = HomographyEstimator;
        let mut models = Vec::new();
        est.fit(&[], &mut models);
        assert!(models.is_empty());
    }
}
