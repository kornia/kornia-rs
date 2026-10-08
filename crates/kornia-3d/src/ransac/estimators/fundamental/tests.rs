use super::kernels::*;
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
    let result = est.threshold_inliers(&Mat3F64::ZERO, &samples, 0.0, 0, &mut residuals, &mut mask);
    assert_eq!(result, Some(ThresholdInlierResult::Pruned));
    assert!(mask.iter().take(128).all(|&is_inlier| !is_inlier));

    let result = est.threshold_inliers(&Mat3F64::ZERO, &samples, 1.0, 0, &mut residuals, &mut mask);
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
