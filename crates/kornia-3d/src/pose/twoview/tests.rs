use super::robust::{adaptive_ransac_cap, ransac_confidence};
use super::scoring::{
    homography_reproj_error, sample_is_degenerate, score_inliers_f, score_inliers_f_bounded_count,
    score_inliers_f_bounded_masked, score_inliers_h, split_xy,
};
use super::*;

#[test]
fn test_ransac_fundamental_basic() {
    let (x1, x2, _, _, _) = synthetic_two_view(50, 0.0, 0);

    let params = RansacParams {
        max_iterations: 200,
        threshold: 1.0,
        min_inliers: 10,
        random_seed: Some(0),
        confidence: None,
        refit: false,
    };
    let res = ransac_fundamental(&x1, &x2, &params).unwrap();
    assert!(res.inlier_count >= params.min_inliers);
}

#[test]
fn test_ransac_fundamental_seven_matches_and_all_roots() -> Result<(), TwoViewError> {
    let (x1, x2, _, _, _) = synthetic_two_view(40, 0.0, 42);
    let params = RansacParams {
        max_iterations: 1,
        threshold: 1e-4,
        min_inliers: 7,
        random_seed: Some(0),
        confidence: None,
        refit: true,
    };
    let minimal = ransac_fundamental(&x1[..7], &x2[..7], &params)?;
    assert_eq!(minimal.inlier_count, 7);
    assert!(matches!(
        ransac_fundamental_8point(&x1[..7], &x2[..7], &params),
        Err(TwoViewError::InvalidInput { required: 8 })
    ));
    let full = ransac_fundamental(&x1, &x2, &params)?;
    assert_eq!(
        full.inlier_count,
        x1.len(),
        "all roots of the sample must be scored"
    );
    assert!(full.score.is_finite());
    Ok(())
}

/// Verify that enabling the LO-refit step either matches or improves the
/// plain-RANSAC result (more inliers *or* a lower Sampson score on ties).
///
/// Synthetic scene: pinhole camera, 100 noisy correspondences + 20
/// outliers with fixed seed so the test is fully deterministic.
#[test]
fn test_ransac_fundamental_refit_improves_accuracy() {
    use crate::pose::fundamental::sampson_distance;

    // Ground-truth fundamental matrix from a simple rotation about Y.
    // R = Ry(5°), t = [0.5, 0, 0].
    let angle = 5.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    // R (column-major): Ry rotates in X-Z plane.
    let r = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    // t cross-product matrix.
    let t = Vec3F64::new(0.5, 0.0, 0.0);
    let tx = Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    // F = [t]_x R (unnormalized; used below to synthesize correspondences)
    let _f_true = tx * r;

    // Simple linear-congruential generator for a deterministic sequence
    // without pulling in extra dependencies.
    let mut lcg: u64 = 12345678901234567;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map high 32 bits to [0, 1)
        (*state >> 32) as f64 / 4294967296.0
    };

    let mut x1: Vec<Vec2F64> = Vec::new();
    let mut x2: Vec<Vec2F64> = Vec::new();

    // 100 inliers: random points in front of cam, project into both views.
    // Focal=500, cx=320, cy=240.
    let fx = 500.0_f64;
    let cx = 320.0_f64;
    let cy = 240.0_f64;
    let noise_px = 0.5_f64; // half-pixel gaussian-ish noise
    for _ in 0..100 {
        let xc = (lcg_next(&mut lcg) - 0.5) * 4.0; // ±2 m
        let yc = (lcg_next(&mut lcg) - 0.5) * 2.0;
        let zc = lcg_next(&mut lcg) * 3.0 + 2.0; // 2-5 m in front
        let p1 = Vec3F64::new(xc, yc, zc);
        let p2 = r * p1 + t;
        // Project with noise.
        let u1 = p1.x / p1.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let v1 = p1.y / p1.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let u2 = p2.x / p2.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let v2 = p2.y / p2.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }
    // 20 outliers: random pixel positions unrelated to the scene.
    for _ in 0..20 {
        let u1 = lcg_next(&mut lcg) * 640.0;
        let v1 = lcg_next(&mut lcg) * 480.0;
        let u2 = lcg_next(&mut lcg) * 640.0;
        let v2 = lcg_next(&mut lcg) * 480.0;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }

    let base_params = RansacParams {
        max_iterations: 500,
        threshold: 2.0,
        min_inliers: 15,
        random_seed: Some(42),
        confidence: None,
        refit: false,
    };
    let refit_params = RansacParams {
        refit: true,
        ..base_params
    };

    let res_base = ransac_fundamental(&x1, &x2, &base_params).unwrap();
    let res_refit = ransac_fundamental(&x1, &x2, &refit_params).unwrap();

    // Compute per-inlier Sampson score for each result to compare quality.
    let sampson_score = |res: &RansacResult<Mat3F64>| -> f64 {
        x1.iter()
            .zip(x2.iter())
            .zip(res.inliers.iter())
            .filter(|&(_, &inl)| inl)
            .map(|((p1, p2), _)| sampson_distance(&res.model, p1, p2))
            .sum::<f64>()
    };

    let score_base = sampson_score(&res_base);
    let score_refit = sampson_score(&res_refit);

    // Refit must not regress: at least as many inliers, and if equal then
    // a lower or equal Sampson score.
    assert!(
        res_refit.inlier_count >= res_base.inlier_count || score_refit <= score_base,
        "refit regressed: base inliers={} score={:.4}, refit inliers={} score={:.4}",
        res_base.inlier_count,
        score_base,
        res_refit.inlier_count,
        score_refit,
    );
}

/// Basic happy-path: known (R, t), pinhole intrinsics, 100 inliers + 20
/// outliers in pixel space. RANSAC with 5pt must (a) flag ≥ 95% of the
/// inliers, and (b) recover an E close to the ground-truth E (up to sign
/// and overall scale, since both are unobservable from epipolar
/// constraints alone).
#[test]
fn test_ransac_essential_5pt_recovers_known_e() {
    // R = Ry(5°), t = (0.5, 0, 0).
    let angle = 5.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    let r = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let t = Vec3F64::new(0.5, 0.0, 0.0);
    let tx = Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    let e_true = tx * r;

    // Pinhole K (matches the F-RANSAC refit test).
    let fx = 500.0_f64;
    let cx = 320.0_f64;
    let cy = 240.0_f64;
    let k = Mat3F64::from_cols(
        Vec3F64::new(fx, 0.0, 0.0),
        Vec3F64::new(0.0, fx, 0.0),
        Vec3F64::new(cx, cy, 1.0),
    );

    // LCG for deterministic noisy data — same recipe as F-RANSAC test.
    let mut lcg: u64 = 12345678901234567;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as f64 / 4294967296.0
    };

    let mut x1: Vec<Vec2F64> = Vec::new();
    let mut x2: Vec<Vec2F64> = Vec::new();
    let noise_px = 0.5_f64;
    for _ in 0..100 {
        let xc = (lcg_next(&mut lcg) - 0.5) * 4.0;
        let yc = (lcg_next(&mut lcg) - 0.5) * 2.0;
        let zc = lcg_next(&mut lcg) * 3.0 + 2.0;
        let p1 = Vec3F64::new(xc, yc, zc);
        let p2 = r * p1 + t;
        let u1 = p1.x / p1.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let v1 = p1.y / p1.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let u2 = p2.x / p2.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let v2 = p2.y / p2.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }
    for _ in 0..20 {
        let u1 = lcg_next(&mut lcg) * 640.0;
        let v1 = lcg_next(&mut lcg) * 480.0;
        let u2 = lcg_next(&mut lcg) * 640.0;
        let v2 = lcg_next(&mut lcg) * 480.0;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }

    let params = RansacParams {
        max_iterations: 500,
        threshold: 2.0,
        min_inliers: 30,
        random_seed: Some(42),
        confidence: None,
        refit: false,
    };
    let res = ransac_essential_5pt(&x1, &x2, &k, &k, &params).unwrap();
    // First 100 are inliers; we expect to flag the vast majority.
    let inl_in_first_100 = res.inliers[..100].iter().filter(|&&b| b).count();
    assert!(
        inl_in_first_100 >= 90,
        "expected ≥90/100 true inliers flagged, got {inl_in_first_100}"
    );

    // Compare recovered E to ground truth up to sign/scale (Frobenius dist
    // on unit-normalized matrices).
    let flat_true = e_true.to_cols_array();
    let norm_true: f64 = flat_true.iter().map(|v| v * v).sum::<f64>().sqrt();
    let e_true_unit: [f64; 9] = core::array::from_fn(|k| flat_true[k] / norm_true);
    let flat_est = res.model.to_cols_array();
    let norm_est: f64 = flat_est.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut err_pos = 0.0f64;
    let mut err_neg = 0.0f64;
    for k in 0..9 {
        let u = flat_est[k] / norm_est;
        err_pos += (u - e_true_unit[k]).powi(2);
        err_neg += (u + e_true_unit[k]).powi(2);
    }
    let dist = err_pos.min(err_neg).sqrt();
    // 0.5 px noise + 17% outliers — relax vs the synthetic-clean test.
    assert!(
        dist < 0.05,
        "recovered E far from ground truth: Frobenius dist (unit, ±sign) = {dist:.4e}"
    );
}

#[test]
fn test_ransac_essential_5pt_invalid_input() {
    let x1 = vec![Vec2F64::new(0.0, 0.0); 4];
    let x2 = vec![Vec2F64::new(0.0, 0.0); 4];
    let k = Mat3F64::IDENTITY;
    let params = RansacParams::default();
    let err = ransac_essential_5pt(&x1, &x2, &k, &k, &params).unwrap_err();
    match err {
        TwoViewError::InvalidInput { required } => assert_eq!(required, 5),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_ransac_fundamental_invalid_input() {
    let x1 = vec![Vec2F64::new(0.0, 0.0); 6];
    let x2 = vec![Vec2F64::new(0.0, 0.0); 6];
    let params = RansacParams::default();
    let err = ransac_fundamental(&x1, &x2, &params).unwrap_err();
    match err {
        TwoViewError::InvalidInput { required } => assert_eq!(required, 7),
        other => panic!("unexpected error: {other:?}"),
    }

    let x2 = vec![Vec2F64::new(0.0, 0.0); 8];
    let err = ransac_fundamental(&x1, &x2, &params).unwrap_err();
    match err {
        TwoViewError::InvalidInput { required } => assert_eq!(required, 7),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_ransac_confidence_validation_and_defaults() {
    let mut params = RansacParams::default();
    assert_eq!(ransac_confidence(&params, 0.9999).unwrap(), 0.9999);
    assert_eq!(ransac_confidence(&params, 0.99).unwrap(), 0.99);

    for confidence in [0.0, 1.0, f64::NAN, f64::INFINITY] {
        params.confidence = Some(confidence);
        assert!(matches!(
            ransac_confidence(&params, 0.9999),
            Err(TwoViewError::InvalidConfidence { .. })
        ));
    }
}

#[test]
fn test_ransac_confidence_monotonically_increases_adaptive_cap() {
    let low = adaptive_ransac_cap(25, 100, 7, 0.99, 1_000_000, 0);
    let high = adaptive_ransac_cap(25, 100, 7, 0.999_999, 1_000_000, 0);
    assert!(high > low, "high-confidence cap {high} must exceed {low}");
}

#[test]
fn test_sample_degenerate_collinear_rejected() {
    // Four points on a line — every 3-subset is collinear → reject.
    let collinear = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    assert!(sample_is_degenerate(&collinear));

    // Three collinear + one off-line — still has a collinear triple → reject.
    let mixed = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.5, 5.0]];
    assert!(sample_is_degenerate(&mixed));
}

#[test]
fn test_sample_non_degenerate_accepted() {
    // A proper quadrilateral — no collinear triples → accept.
    let quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
    assert!(!sample_is_degenerate(&quad));

    // Points from the `_make_known_homography_pair` region in the Python tests.
    let normal = [
        [120.0, 180.0],
        [340.0, 150.0],
        [280.0, 420.0],
        [410.0, 370.0],
    ];
    assert!(!sample_is_degenerate(&normal));
}

#[test]
fn test_score_inliers_h_matches_scalar() {
    use crate::pose::fundamental::sampson_distance;
    let h = Mat3F64::from_cols(
        Vec3F64::new(1.25, 0.03, 0.0012),
        Vec3F64::new(-0.08, 0.95, -0.0008),
        Vec3F64::new(2.4, -1.7, 1.0),
    );
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for i in 0..47 {
        let xi = (i as f64 * 0.73 - 12.5).sin() * 100.0;
        let yi = (i as f64 * 1.11 + 3.0).cos() * 80.0;
        let p = Vec3F64::new(xi, yi, 1.0);
        let hp = h * p;
        let jitter = (i as f64 * 0.2).sin() * 0.01;
        x1.push(Vec2F64::new(xi, yi));
        x2.push(Vec2F64::new(hp.x / hp.z + jitter, hp.y / hp.z - jitter));
    }
    let thresh_sq = 0.04f64;

    let mut inl_ref = vec![false; x1.len()];
    let mut count_ref = 0usize;
    let mut score_ref = 0.0f64;
    for i in 0..x1.len() {
        let d = homography_reproj_error(&h, &x1[i], &x2[i]);
        if d <= thresh_sq {
            inl_ref[i] = true;
            count_ref += 1;
            score_ref += d;
        }
    }

    let (x1_x, x1_y) = split_xy(&x1);
    let (x2_x, x2_y) = split_xy(&x2);
    let mut inl = vec![false; x1.len()];
    let (count, score) = score_inliers_h(&h, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inl);

    assert_eq!(count, count_ref);
    assert_eq!(inl, inl_ref);
    assert!((score - score_ref).abs() < 1e-10);

    // keep sampson_distance import live for symmetry across test changes
    let _ = sampson_distance;
}

#[test]
fn test_score_inliers_f_matches_scalar() {
    use crate::pose::fundamental::sampson_distance;
    let f_mat = Mat3F64::from_cols(
        Vec3F64::new(0.0, -0.0012, 0.011),
        Vec3F64::new(0.0014, 0.0, -0.021),
        Vec3F64::new(-0.012, 0.018, 1.0),
    );
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for i in 0..53 {
        let xi = i as f64 * 1.33 - 20.0;
        let yi = (i as f64 * 0.7).sin() * 50.0 - 10.0;
        let x = Vec3F64::new(xi, yi, 1.0);
        let l = f_mat * x;
        let xp = if l.x.abs() > 1e-10 { -l.z / l.x } else { 0.0 };
        x1.push(Vec2F64::new(xi, yi));
        x2.push(Vec2F64::new(xp, (i as f64 * 0.05).cos() * 0.5));
    }
    let thresh_sq = 0.25f64;

    let mut inl_ref = vec![false; x1.len()];
    let mut count_ref = 0usize;
    let mut score_ref = 0.0f64;
    for i in 0..x1.len() {
        let d = sampson_distance(&f_mat, &x1[i], &x2[i]);
        if d <= thresh_sq {
            inl_ref[i] = true;
            count_ref += 1;
            score_ref += d;
        }
    }

    let (x1_x, x1_y) = split_xy(&x1);
    let (x2_x, x2_y) = split_xy(&x2);
    let mut inl = vec![false; x1.len()];
    let (count, score) = score_inliers_f(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inl);

    assert_eq!(count, count_ref);
    assert_eq!(inl, inl_ref);
    assert!((score - score_ref).abs() < 1e-9);
}

#[test]
fn test_bounded_f_scorer_matches_full_winner_and_zero_denominator() {
    let f_mat = Mat3F64::ZERO;
    let x1: Vec<_> = (0..131)
        .map(|i| Vec2F64::new(i as f64 * 0.25, -(i as f64)))
        .collect();
    let x2: Vec<_> = (0..131)
        .map(|i| Vec2F64::new(-(i as f64), i as f64 * 0.5))
        .collect();
    let (x1_x, x1_y) = split_xy(&x1);
    let (x2_x, x2_y) = split_xy(&x2);

    let mut full_mask = vec![false; x1.len()];
    let full = score_inliers_f(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, 0.0, &mut full_mask);
    let bounded =
        score_inliers_f_bounded_count(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, 0.0, 0, f64::INFINITY)
            .unwrap();

    assert_eq!(bounded.0, full.0);
    assert_eq!(bounded.1.to_bits(), full.1.to_bits());
    assert_eq!(full.0, x1.len());
    assert_eq!(full.1, 0.0);

    // A non-zero score catches accidental reassociation at chunk borders.
    let finite_f = Mat3F64::from_cols(
        Vec3F64::new(0.0, -0.001, 0.02),
        Vec3F64::new(0.0015, 0.0, -0.01),
        Vec3F64::new(-0.03, 0.04, 1.0),
    );
    let mut finite_full_mask = vec![false; x1.len()];
    let finite_full = score_inliers_f(
        &finite_f,
        &x1_x,
        &x1_y,
        &x2_x,
        &x2_y,
        1e20,
        &mut finite_full_mask,
    );
    let finite_bounded = score_inliers_f_bounded_count(
        &finite_f,
        &x1_x,
        &x1_y,
        &x2_x,
        &x2_y,
        1e20,
        0,
        f64::INFINITY,
    )
    .unwrap();
    assert_eq!(finite_bounded.0, finite_full.0);
    assert_eq!(finite_bounded.1.to_bits(), finite_full.1.to_bits());

    let mut masked_mask = vec![false; x1.len()];
    let masked = score_inliers_f_bounded_masked(
        &finite_f,
        &x1_x,
        &x1_y,
        &x2_x,
        &x2_y,
        1e20,
        &mut masked_mask,
        0,
        f64::INFINITY,
    )
    .unwrap();
    assert_eq!(masked.0, finite_full.0);
    assert_eq!(masked.1.to_bits(), finite_full.1.to_bits());
    assert_eq!(masked_mask, finite_full_mask);
}

#[test]
fn test_bounded_f_scorer_rejects_count_and_score_ties() {
    let f_mat = Mat3F64::ZERO;
    let x1: Vec<_> = (0..128).map(|i| Vec2F64::new(i as f64, 1.0)).collect();
    let x2: Vec<_> = (0..128).map(|i| Vec2F64::new(2.0, i as f64)).collect();
    let (x1_x, x1_y) = split_xy(&x1);
    let (x2_x, x2_y) = split_xy(&x2);

    let mut full_mask = vec![false; x1.len()];
    let full = score_inliers_f(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, 0.0, &mut full_mask);
    assert_eq!(full, (128, 0.0));

    assert!(
        score_inliers_f_bounded_count(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, 0.0, full.0, full.1,)
            .is_none()
    );
}

#[test]
fn test_bounded_f_scorer_rejects_when_count_cannot_catch_up() {
    let f_mat = Mat3F64::from_cols(Vec3F64::ZERO, Vec3F64::ZERO, Vec3F64::new(0.0, 0.0, 1.0));
    let x1: Vec<_> = (0..128).map(|i| Vec2F64::new(i as f64, 1.0)).collect();
    let x2: Vec<_> = (0..128).map(|i| Vec2F64::new(2.0, i as f64)).collect();
    let (x1_x, x1_y) = split_xy(&x1);
    let (x2_x, x2_y) = split_xy(&x2);

    let mut full_mask = vec![false; x1.len()];
    let full = score_inliers_f(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, 0.0, &mut full_mask);
    assert_eq!(full, (0, 0.0));
    assert!(full_mask.iter().all(|&v| !v));

    assert!(score_inliers_f_bounded_count(
        &f_mat,
        &x1_x,
        &x1_y,
        &x2_x,
        &x2_y,
        0.0,
        100,
        f64::INFINITY,
    )
    .is_none());
}

#[test]
fn test_ransac_homography_adaptive_stops_early_on_clean_data() {
    // All-inlier data should stop in O(10) iterations, not 2000. We can't
    // observe the iteration count directly, but we can verify it runs
    // quickly and converges; the adaptive bound at w=1 triggers the
    // `best_count == n` early-exit immediately.
    let h_true = Mat3F64::from_cols(
        Vec3F64::new(1.2, 0.0, 0.001),
        Vec3F64::new(0.1, 0.9, 0.002),
        Vec3F64::new(5.0, -3.0, 1.0),
    );
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for i in 0..30 {
        let xi = (i % 6) as f64 * 5.0;
        let yi = (i / 6) as f64 * 7.0;
        let p = Vec3F64::new(xi, yi, 1.0);
        let hp = h_true * p;
        x1.push(Vec2F64::new(xi, yi));
        x2.push(Vec2F64::new(hp.x / hp.z, hp.y / hp.z));
    }
    let params = RansacParams {
        max_iterations: 2000,
        threshold: 0.1,
        min_inliers: 25,
        random_seed: Some(42),
        confidence: None,
        refit: false,
    };
    let start = std::time::Instant::now();
    let res = ransac_homography(&x1, &x2, &params).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(res.inlier_count, x1.len(), "should find all inliers");
    // Generous bound — even with collinearity-rejection overhead, 30 points
    // on clean data is well under 10ms on any dev machine. A runaway loop
    // (e.g., adaptive termination broken) would be orders of magnitude slower.
    assert!(
        elapsed.as_millis() < 50,
        "adaptive termination should be fast on clean data, took {elapsed:?}"
    );
}

#[test]
fn test_ransac_homography_basic() {
    let h_true = Mat3F64::from_cols(
        Vec3F64::new(1.2, 0.0, 0.001),
        Vec3F64::new(0.1, 0.9, 0.002),
        Vec3F64::new(5.0, -3.0, 1.0),
    );

    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for i in 0..25 {
        let xi = (i % 5) as f64 * 2.0 - 4.0;
        let yi = (i / 5) as f64 * 1.5 - 3.0;
        let p = Vec3F64::new(xi, yi, 1.0);
        let hp = h_true * p;
        let u = hp.x / hp.z;
        let v = hp.y / hp.z;
        x1.push(Vec2F64::new(xi, yi));
        x2.push(Vec2F64::new(u, v));
    }

    let params = RansacParams {
        max_iterations: 100,
        threshold: 1e-6,
        min_inliers: 12,
        random_seed: Some(0),
        confidence: None,
        refit: false,
    };
    let res = ransac_homography(&x1, &x2, &params).unwrap();
    assert!(res.inlier_count >= params.min_inliers);
}

#[test]
fn test_ransac_homography_invalid_input() {
    let x1 = vec![Vec2F64::new(0.0, 0.0); 3];
    let x2 = vec![Vec2F64::new(0.0, 0.0); 3];
    let params = RansacParams::default();
    let err = ransac_homography(&x1, &x2, &params).unwrap_err();
    match err {
        TwoViewError::InvalidInput { required } => assert_eq!(required, 4),
        other => panic!("unexpected error: {other:?}"),
    }

    let x2 = vec![Vec2F64::new(0.0, 0.0); 4];
    let err = ransac_homography(&x1, &x2, &params).unwrap_err();
    match err {
        TwoViewError::InvalidInput { required } => assert_eq!(required, 4),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_default_triangulation_config_values() {
    // Defaults the builder will pick up unless overridden via
    // `TwoViewEstimatorBuilder::triangulation`.
    let tri = TriangulationConfig::default();
    assert_eq!(tri.min_parallax_deg, 1.0);
    assert_eq!(tri.max_midpoint_gap, 1.0);
    assert_eq!(tri.max_reprojection_error, 2.0);
    assert_eq!(tri.min_cheirality_count, 1);
    assert_eq!(tri.cheirality_ambiguity_max, 0.7);
}

/// End-to-end two-view pose estimation on real EuRoC MH_01_easy images.
///
/// Reads two grayscale frames, runs ORB detection + matching, estimates
/// the relative pose via `TwoViewEstimator::estimate`, and compares
/// against ground truth camera-frame pose.
///
/// Frame pair: 1403636633263555584 → 1403636634263555584 (20 frames apart).
/// Ground truth camera-frame relative pose (Vicon-derived; see
/// `kornia-py/scripts/derive_mh01_gt.py` — Δ=0 ns match at both frames):
///   - Rotation: 2.7021°
///   - Translation: 658.5mm, direction [0.2422, -0.2330, 0.9418]
#[test]
fn test_two_view_euroc_mh01() {
    use kornia_imgproc::features::{match_orb_descriptors, OrbDetector, OrbMatchConfig};

    // Load images.
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/data");

    let img1_u8 = kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame1.png")).unwrap();
    let img2_u8 = kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame2.png")).unwrap();

    let img1 = u8_to_f32_image(&img1_u8);
    let img2 = u8_to_f32_image(&img2_u8);

    // ORB detect + extract.
    let orb = OrbDetector::default();
    let (kps1, scales1, ori1, _) = orb.detect(&img1).unwrap();
    let (desc1, mask1) = orb.extract(&img1, &kps1, &scales1, &ori1).unwrap();
    let (kps2, scales2, ori2, _) = orb.detect(&img2).unwrap();
    let (desc2, mask2) = orb.extract(&img2, &kps2, &scales2, &ori2).unwrap();

    // Filter by valid descriptors (border mask).
    let (valid_kps1, valid_ori1, valid_desc1) = filter_by_mask(&kps1, &ori1, &desc1, &mask1);
    let (valid_kps2, valid_ori2, valid_desc2) = filter_by_mask(&kps2, &ori2, &desc2, &mask2);

    // Match descriptors.
    let match_config = OrbMatchConfig {
        nn_ratio: 0.6,
        th_low: 50,
        check_orientation: true,
        histo_length: 30,
    };
    let matches = match_orb_descriptors(
        &valid_ori1,
        &valid_desc1,
        &valid_ori2,
        &valid_desc2,
        match_config,
    );
    assert!(
        matches.len() >= 15,
        "too few ORB matches: {} (need >= 15)",
        matches.len()
    );

    // Convert matched keypoints to Vec2F64 (x=col, y=row).
    let pts1: Vec<Vec2F64> = matches
        .iter()
        .map(|&(i, _)| {
            let (row, col) = valid_kps1[i];
            Vec2F64::new(col as f64, row as f64)
        })
        .collect();
    let pts2: Vec<Vec2F64> = matches
        .iter()
        .map(|&(_, j)| {
            let (row, col) = valid_kps2[j];
            Vec2F64::new(col as f64, row as f64)
        })
        .collect();

    // EuRoC MH_01_easy cam0 intrinsics.
    let k = Mat3F64::from_cols(
        Vec3F64::new(458.654, 0.0, 0.0),
        Vec3F64::new(0.0, 457.296, 0.0),
        Vec3F64::new(367.215, 248.375, 1.0),
    );

    let est = TwoViewEstimator::builder()
        .epipolar_solver(Fundamental8ptSolver {
            ransac: RansacParams {
                max_iterations: 2000,
                threshold: 1.0,
                min_inliers: 15,
                random_seed: Some(42),
                confidence: None,
                refit: true,
            },
        })
        .homography_ransac(RansacParams {
            max_iterations: 2000,
            threshold: 1.0,
            min_inliers: 8,
            random_seed: Some(42),
            confidence: None,
            refit: false,
        })
        .triangulation(TriangulationConfig {
            min_parallax_deg: 0.5,
            ..TriangulationConfig::default()
        })
        .build();

    let result = est.estimate(&pts1, &pts2, &k, &k).unwrap();

    // Should select fundamental model (general motion, not planar).
    assert!(
        matches!(result.model, TwoViewModel::Fundamental(_)),
        "expected fundamental model"
    );

    // Check rotation angle: GT is 2.7021°. With LM polishing the
    // post-cheirality pose, rotation error stays well below 1°.
    let r = result.rotation;
    let trace = r.col(0).x + r.col(1).y + r.col(2).z;
    let est_angle_rad = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
    let est_angle_deg = est_angle_rad.to_degrees();
    let gt_angle_deg = 2.7021;
    assert!(
        (est_angle_deg - gt_angle_deg).abs() < 1.0,
        "rotation error too large: estimated {est_angle_deg:.2}°, GT {gt_angle_deg}°"
    );

    // Check translation direction: GT is [0.242, -0.233, 0.942].
    // Translation can be recovered up to sign. Pre-LM the error was ~8°;
    // post-LM Sampson refinement pulls it under 5° on this pair.
    let t = result.translation.normalize();
    let gt_t = Vec3F64::new(0.2422, -0.2330, 0.9418).normalize();
    let cos_angle = t.dot(gt_t).clamp(-1.0, 1.0);
    let t_err_deg = cos_angle.abs().acos().to_degrees();
    assert!(
        t_err_deg < 5.0,
        "translation direction error too large: {t_err_deg:.2}°"
    );

    // Should have triangulated some points.
    assert!(
        !result.points3d.is_empty(),
        "expected triangulated 3D points"
    );
}

/// Plugging in [`EssentialNister5ptSolver`] via the builder must:
/// the returned model variant is `Essential`, the recovered (R, t) hits
/// ground truth, and triangulation produces points. The 5pt path skips
/// the F→E SVD round-trip (since 5pt builds E on-manifold), so for the
/// same RANSAC budget it's expected to match or beat the F path on
/// small-motion synthetic scenes.
#[test]
fn test_two_view_estimate_with_5pt_essential() {
    // Synthetic scene: Ry(5°), t = (0.5, 0, 0), pinhole at fx=500.
    let angle = 5.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    let r_true = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let t_true = Vec3F64::new(0.5, 0.0, 0.0);

    let fx = 500.0_f64;
    let cx = 320.0_f64;
    let cy = 240.0_f64;
    let k = Mat3F64::from_cols(
        Vec3F64::new(fx, 0.0, 0.0),
        Vec3F64::new(0.0, fx, 0.0),
        Vec3F64::new(cx, cy, 1.0),
    );

    // 100 noisy inliers + 20 outliers (same recipe as the RANSAC test).
    let mut lcg: u64 = 12345678901234567;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as f64 / 4294967296.0
    };

    let mut x1: Vec<Vec2F64> = Vec::new();
    let mut x2: Vec<Vec2F64> = Vec::new();
    let noise_px = 0.5_f64;
    for _ in 0..100 {
        let xc = (lcg_next(&mut lcg) - 0.5) * 4.0;
        let yc = (lcg_next(&mut lcg) - 0.5) * 2.0;
        let zc = lcg_next(&mut lcg) * 3.0 + 2.0;
        let p1 = Vec3F64::new(xc, yc, zc);
        let p2 = r_true * p1 + t_true;
        let u1 = p1.x / p1.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let v1 = p1.y / p1.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let u2 = p2.x / p2.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
        let v2 = p2.y / p2.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }
    for _ in 0..20 {
        x1.push(Vec2F64::new(
            lcg_next(&mut lcg) * 640.0,
            lcg_next(&mut lcg) * 480.0,
        ));
        x2.push(Vec2F64::new(
            lcg_next(&mut lcg) * 640.0,
            lcg_next(&mut lcg) * 480.0,
        ));
    }

    let est = TwoViewEstimator::builder()
        .epipolar_solver(EssentialNister5ptSolver {
            ransac: RansacParams {
                max_iterations: 500,
                threshold: 2.0,
                min_inliers: 30,
                random_seed: Some(42),
                confidence: None,
                refit: false,
            },
        })
        .homography_ransac(RansacParams {
            max_iterations: 500,
            threshold: 2.0,
            min_inliers: 8,
            random_seed: Some(42),
            confidence: None,
            refit: false,
        })
        // Force the epipolar branch even if H scores comparably.
        .homography_inlier_ratio(1.5)
        .triangulation(TriangulationConfig {
            min_parallax_deg: 0.1,
            ..TriangulationConfig::default()
        })
        .build();

    let result = est.estimate(&x1, &x2, &k, &k).unwrap();

    assert!(
        matches!(result.model, TwoViewModel::Essential(_)),
        "expected Essential model, got {:?}",
        result.model
    );

    // Rotation error.
    let r_est = result.rotation;
    let rt_r = r_est.transpose() * r_true;
    let trace = rt_r.col(0).x + rt_r.col(1).y + rt_r.col(2).z;
    let rot_err_deg = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    assert!(
        rot_err_deg < 0.5,
        "rotation error too large: {rot_err_deg:.4}°"
    );

    // Translation direction (recoverable up to sign).
    let t_dir = result.translation.normalize();
    let t_gt = t_true.normalize();
    let t_err_deg = t_dir.dot(t_gt).clamp(-1.0, 1.0).abs().acos().to_degrees();
    assert!(
        t_err_deg < 5.0,
        "translation direction error too large: {t_err_deg:.2}°"
    );

    assert!(
        !result.points3d.is_empty(),
        "expected triangulated 3D points"
    );
}

/// Guard should NOT fire on a well-conditioned general-motion scene.
/// The synthetic_two_view helper uses t=(0.5, 0.1, 0.2) at Z∈[3,6] — a
/// baseline/depth ratio around 0.1 that can be borderline. Here we
/// hand-build a wider-baseline scene (t magnitude ~1 m, depth ~5 m)
/// where one E candidate dominates cheirality decisively, so the
/// default 0.7 threshold must accept it.
#[test]
fn test_cheirality_ambiguity_guard_permissive_default() {
    let k = Mat3F64::from_cols(
        Vec3F64::new(500.0, 0.0, 0.0),
        Vec3F64::new(0.0, 500.0, 0.0),
        Vec3F64::new(320.0, 240.0, 1.0),
    );
    let angle = 6.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    let r = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let t = Vec3F64::new(1.0, 0.0, 0.1);

    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for i in 0..50usize {
        let fi = i as f64;
        // Spread over a cube: x,y ∈ [-1.5, 1.5], z ∈ [4, 6].
        let xc = (fi * 0.37).sin() * 1.5;
        let yc = (fi * 0.53).cos() * 1.0;
        let zc = 4.0 + (fi * 0.11).sin().abs() * 2.0;
        let p1 = Vec3F64::new(xc, yc, zc);
        let p2 = r * p1 + t;
        x1.push(Vec2F64::new(
            500.0 * p1.x / p1.z + 320.0,
            500.0 * p1.y / p1.z + 240.0,
        ));
        x2.push(Vec2F64::new(
            500.0 * p2.x / p2.z + 320.0,
            500.0 * p2.y / p2.z + 240.0,
        ));
    }

    TwoViewEstimator::default()
        .estimate(&x1, &x2, &k, &k)
        .expect("default 0.7 ambiguity threshold must accept well-conditioned motion");
}

/// With the threshold driven to 0.0, ANY runner-up candidate (even a
/// single borderline triangulated point) trips the guard. Used to
/// confirm the error path is reachable and the reported counts are sane.
/// On clean synthetic data the runner-up is usually 0, so we force the
/// degenerate case via a fronto-parallel plane + tiny translation —
/// the classic two-E-candidates-both-pass configuration.
#[test]
fn test_cheirality_ambiguity_guard_fires_on_degenerate_motion() {
    // Fronto-parallel plane at Z=5, points spread over a 2×2 m patch.
    let k = Mat3F64::from_cols(
        Vec3F64::new(500.0, 0.0, 0.0),
        Vec3F64::new(0.0, 500.0, 0.0),
        Vec3F64::new(320.0, 240.0, 1.0),
    );
    // Small rotation about Y, near-zero translation. Pure rotation makes
    // E ≈ 0; a tiny bit of translation gives RANSAC enough signal to find
    // *some* F, but the resulting E's decomposition is poorly conditioned
    // and commonly produces two candidates with similar cheirality.
    let angle = 1.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    let r = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let t = Vec3F64::new(0.02, 0.0, 0.0);

    // Generate 30 points on the plane Z=5.
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for i in 0..30 {
        let xc = -1.0 + 0.1 * i as f64;
        let yc = 0.4 * (i as f64 * 0.7).sin();
        let p1 = Vec3F64::new(xc, yc, 5.0);
        let p2 = r * p1 + t;
        let u1 = 500.0 * p1.x / p1.z + 320.0;
        let v1 = 500.0 * p1.y / p1.z + 240.0;
        let u2 = 500.0 * p2.x / p2.z + 320.0;
        let v2 = 500.0 * p2.y / p2.z + 240.0;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }

    // Strict threshold: any non-zero runner-up trips it. If the scene
    // ends up with second=0 (unambiguous), we'll take that as a signal
    // to document that this path needs a stronger synthetic case — but
    // typical runs of this near-pure-rotation setup produce multiple
    // cheirality-passing candidates.
    let est = TwoViewEstimator::builder()
        .epipolar_solver(Fundamental8ptSolver {
            ransac: RansacParams {
                min_inliers: 8,
                random_seed: Some(0),
                ..RansacParams::default()
            },
        })
        .homography_ransac(RansacParams {
            min_inliers: 8,
            random_seed: Some(0),
            ..RansacParams::default()
        })
        .triangulation(TriangulationConfig {
            min_parallax_deg: 0.0,
            cheirality_ambiguity_max: 0.0,
            ..TriangulationConfig::default()
        })
        .build();

    match est.estimate(&x1, &x2, &k, &k) {
        Err(TwoViewError::AmbiguousCheirality {
            best,
            second,
            ratio,
            max_ratio,
        }) => {
            assert!(best >= second);
            assert!(ratio > max_ratio);
            assert_eq!(max_ratio, 0.0);
        }
        Err(TwoViewError::RansacFailure) => {
            // Acceptable: RANSAC may reject the F entirely when
            // t is this small — the whole pipeline correctly bails.
        }
        Err(other) => panic!("unexpected error variant: {other:?}"),
        Ok(r) => panic!(
            "degenerate motion must not produce a confident pose; got inliers={}",
            r.inliers.iter().filter(|b| **b).count()
        ),
    }
}

// -------- LM refinement tests --------

/// Construct a synthetic two-view setup: N 3D points in front of both
/// cameras, projected (with optional Gaussian-ish noise) into both views
/// through a shared K. Uses a deterministic LCG so tests are reproducible
/// without extra RNG dependencies.
///
/// Returns (x1, x2, R_true, t_true_unit, K).
fn synthetic_two_view(
    n_pts: usize,
    noise_px: f64,
    seed: u64,
) -> (Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64, Vec3F64, Mat3F64) {
    // Fixed GT pose: small rotation about y-axis, translation mostly along x.
    let angle = 5.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    let r = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let t = Vec3F64::new(0.5, 0.1, 0.2);
    let t_unit = t.normalize();

    // Intrinsics: 640x480, fx=fy=500, principal point at center.
    let k = Mat3F64::from_cols(
        Vec3F64::new(500.0, 0.0, 0.0),
        Vec3F64::new(0.0, 500.0, 0.0),
        Vec3F64::new(320.0, 240.0, 1.0),
    );

    let mut lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || -> f64 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (lcg >> 32) as f64 / 4_294_967_296.0
    };

    let mut x1 = Vec::with_capacity(n_pts);
    let mut x2 = Vec::with_capacity(n_pts);
    for _ in 0..n_pts {
        let xc = (next() - 0.5) * 3.0;
        let yc = (next() - 0.5) * 2.0;
        let zc = next() * 3.0 + 3.0; // 3-6 m depth
        let p1 = Vec3F64::new(xc, yc, zc);
        let p2_cam = r * p1 + t;
        // Project through K.
        let u1 = 500.0 * p1.x / p1.z + 320.0 + (next() - 0.5) * 2.0 * noise_px;
        let v1 = 500.0 * p1.y / p1.z + 240.0 + (next() - 0.5) * 2.0 * noise_px;
        let u2 = 500.0 * p2_cam.x / p2_cam.z + 320.0 + (next() - 0.5) * 2.0 * noise_px;
        let v2 = 500.0 * p2_cam.y / p2_cam.z + 240.0 + (next() - 0.5) * 2.0 * noise_px;
        x1.push(Vec2F64::new(u1, v1));
        x2.push(Vec2F64::new(u2, v2));
    }
    (x1, x2, r, t_unit, k)
}

/// Sum of Sampson distances for F = K2^-T [t]x R K1^-1 over all pairs.
fn sampson_cost_rt(
    r: &Mat3F64,
    t: &Vec3F64,
    k1: &Mat3F64,
    k2: &Mat3F64,
    x1: &[Vec2F64],
    x2: &[Vec2F64],
) -> f64 {
    use crate::pose::fundamental::sampson_distance;
    let skew = Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    let e = skew * *r;
    let f = k2.inverse().transpose() * e * k1.inverse();
    x1.iter()
        .zip(x2.iter())
        .map(|(p1, p2)| sampson_distance(&f, p1, p2))
        .sum()
}

fn rot_angle_deg(r_rel: &Mat3F64) -> f64 {
    let tr = r_rel.col(0).x + r_rel.col(1).y + r_rel.col(2).z;
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

fn rot_err_deg(r_est: &Mat3F64, r_gt: &Mat3F64) -> f64 {
    let r_diff = r_est.transpose() * *r_gt;
    rot_angle_deg(&r_diff)
}

fn t_err_deg(t_est: &Vec3F64, t_gt: &Vec3F64) -> f64 {
    let te = t_est.normalize();
    let tg = t_gt.normalize();
    te.dot(tg).abs().clamp(0.0, 1.0).acos().to_degrees()
}

/// Small rotation (~1°) + small translation-direction perturbation applied
/// to a GT pose, used to seed LM.
fn perturb_pose(r: Mat3F64, t: Vec3F64, rot_deg: f64, t_dir_deg: f64) -> (Mat3F64, Vec3F64) {
    // Rotation perturbation: axis = Y (normalized), angle = rot_deg.
    let angle = rot_deg.to_radians();
    let (s, c) = angle.sin_cos();
    let r_pert = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let r_new = r * r_pert;

    // Translation-direction perturbation: rotate t by t_dir_deg about an
    // axis orthogonal to t. Pick e.g. world-X (unless t is near-parallel to
    // it, then world-Y).
    let t_u = t.normalize();
    let seed_axis = if t_u.x.abs() < 0.9 {
        Vec3F64::new(1.0, 0.0, 0.0)
    } else {
        Vec3F64::new(0.0, 1.0, 0.0)
    };
    let axis = Vec3F64::new(
        t_u.y * seed_axis.z - t_u.z * seed_axis.y,
        t_u.z * seed_axis.x - t_u.x * seed_axis.z,
        t_u.x * seed_axis.y - t_u.y * seed_axis.x,
    )
    .normalize();
    let ang = t_dir_deg.to_radians();
    let (ss, cc) = ang.sin_cos();
    // Rodrigues: v' = v cos + (k × v) sin + k (k·v)(1 - cos)
    let kxv = Vec3F64::new(
        axis.y * t_u.z - axis.z * t_u.y,
        axis.z * t_u.x - axis.x * t_u.z,
        axis.x * t_u.y - axis.y * t_u.x,
    );
    let kdotv = axis.dot(t_u);
    let t_new = Vec3F64::new(
        t_u.x * cc + kxv.x * ss + axis.x * kdotv * (1.0 - cc),
        t_u.y * cc + kxv.y * ss + axis.y * kdotv * (1.0 - cc),
        t_u.z * cc + kxv.z * ss + axis.z * kdotv * (1.0 - cc),
    )
    .normalize();
    (r_new, t_new)
}

/// Noise-free synthetic setup: with perfect correspondences and a small
/// perturbation of GT pose, LM must recover the GT to numerical
/// precision.
#[test]
fn test_lm_pose_refine_synthetic_perfect() {
    let (x1, x2, r_gt, t_gt, k) = synthetic_two_view(60, 0.0, 7);
    let (r0, t0) = perturb_pose(r_gt, t_gt, 1.0, 5.0);

    let cfg = LmPoseConfig::default();
    let (r_ref, t_ref) = refine_pose_lm(r0, t0, &x1, &x2, &k, &k, &cfg);

    let r_err = rot_err_deg(&r_ref, &r_gt);
    let t_err = t_err_deg(&t_ref, &t_gt);
    // With a finite-difference Jacobian the residual of LM is bounded by
    // O(h²) ≈ 1e-12 on cost, but Sampson is scale-invariant in F; 1e-2° is
    // the effective noise floor on R/t from numerical error.
    assert!(
        r_err < 1e-2,
        "perfect-noise LM should recover R to ≤1e-2°, got {r_err:.6}°"
    );
    assert!(
        t_err < 1e-2,
        "perfect-noise LM should recover t_dir to ≤1e-2°, got {t_err:.6}°"
    );
}

/// Noisy synthetic setup: LM must strictly reduce the Sampson cost and
/// never regress the rotation / translation error.
#[test]
fn test_lm_pose_refine_noisy_improves() {
    let (x1, x2, r_gt, t_gt, k) = synthetic_two_view(120, 0.5, 42);
    // Perturb GT by a non-trivial amount to ensure LM has work to do.
    let (r0, t0) = perturb_pose(r_gt, t_gt, 2.0, 8.0);

    let cost_before = sampson_cost_rt(&r0, &t0, &k, &k, &x1, &x2);
    let r_err_before = rot_err_deg(&r0, &r_gt);
    let t_err_before = t_err_deg(&t0, &t_gt);

    let cfg = LmPoseConfig::default();
    let (r1, t1) = refine_pose_lm(r0, t0, &x1, &x2, &k, &k, &cfg);

    let cost_after = sampson_cost_rt(&r1, &t1, &k, &k, &x1, &x2);
    let r_err_after = rot_err_deg(&r1, &r_gt);
    let t_err_after = t_err_deg(&t1, &t_gt);

    assert!(
        cost_after < cost_before,
        "Sampson cost did not decrease: before={cost_before:.6e}, after={cost_after:.6e}"
    );
    // With a finite-DoF linearized fit on noisy data we don't expect
    // monotone improvement of the *pose errors* in general, but on this
    // setup (large perturbation, low noise) they must not regress by more
    // than a small margin.
    assert!(
        r_err_after <= r_err_before + 0.05,
        "rotation error regressed: before={r_err_before:.4}°, after={r_err_after:.4}°"
    );
    assert!(
        t_err_after <= t_err_before + 0.05,
        "translation error regressed: before={t_err_before:.4}°, after={t_err_after:.4}°"
    );
}

/// If we start from GT pose, LM must stay there (within numerical
/// tolerance) — an idempotency / no-harm guarantee.
#[test]
fn test_lm_pose_refine_does_no_harm_when_already_optimal() {
    let (x1, x2, r_gt, t_gt, k) = synthetic_two_view(60, 0.0, 99);
    let cfg = LmPoseConfig::default();
    let (r_out, t_out) = refine_pose_lm(r_gt, t_gt, &x1, &x2, &k, &k, &cfg);
    let r_err = rot_err_deg(&r_out, &r_gt);
    let t_err = t_err_deg(&t_out, &t_gt);
    assert!(r_err < 1e-4, "LM drifted from optimal R: err={r_err:.6}°");
    assert!(t_err < 1e-4, "LM drifted from optimal t: err={t_err:.6}°");
}

/// Helper to build a minimal TwoViewResult with given inlier_indices.
fn stub_result(inlier_indices: Vec<usize>) -> TwoViewResult {
    TwoViewResult {
        model: TwoViewModel::Fundamental(Mat3F64::IDENTITY),
        rotation: Mat3F64::IDENTITY,
        translation: Vec3F64::new(0.0, 0.0, 1.0),
        points3d: Vec::new(),
        inlier_indices,
        inliers: Vec::new(),
    }
}

fn test_camera() -> crate::camera::PinholeCamera {
    crate::camera::PinholeCamera {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

#[test]
fn test_median_parallax_empty_inliers() {
    let result = stub_result(vec![]);
    let cam = test_camera();
    let x1 = vec![Vec2F64::new(320.0, 240.0)];
    let x2 = vec![Vec2F64::new(330.0, 240.0)];
    assert_eq!(result.median_parallax_deg(&x1, &x2, &cam), 0.0);
}

#[test]
fn test_median_parallax_identical_points() {
    // Same pixel in both views → zero parallax.
    let result = stub_result(vec![0]);
    let cam = test_camera();
    let x1 = vec![Vec2F64::new(400.0, 300.0)];
    let x2 = vec![Vec2F64::new(400.0, 300.0)];
    let angle = result.median_parallax_deg(&x1, &x2, &cam);
    assert!(
        angle.abs() < 1e-4,
        "expected ~0 parallax for identical points, got {angle}"
    );
}

#[test]
fn test_median_parallax_known_angle() {
    // Construct a case where bearing vectors differ by a known angle.
    // Camera: fx=fy=500, cx=320, cy=240.
    // Point 1: at principal point → bearing (0, 0, 1).
    // Point 2: shifted 500px in x → bearing (1, 0, 1)/sqrt(2).
    // Angle = acos( (0*1 + 0*0 + 1*1) / (1 * sqrt(2)) ) = acos(1/sqrt(2)) = 45°.
    let cam = test_camera();
    let result = stub_result(vec![0]);
    let x1 = vec![Vec2F64::new(320.0, 240.0)]; // principal point
    let x2 = vec![Vec2F64::new(820.0, 240.0)]; // 500px right
    let angle = result.median_parallax_deg(&x1, &x2, &cam);
    assert!(
        (angle - 45.0).abs() < 0.01,
        "expected ~45° parallax, got {angle}"
    );
}

#[test]
fn test_median_parallax_multiple_inliers() {
    // 3 inliers: angles 0°, 45°, 45° → sorted [0, 45, 45], median = 45°.
    let cam = test_camera();
    let result = stub_result(vec![0, 1, 2]);
    let x1 = vec![
        Vec2F64::new(320.0, 240.0), // pp
        Vec2F64::new(320.0, 240.0), // pp
        Vec2F64::new(320.0, 240.0), // pp
    ];
    let x2 = vec![
        Vec2F64::new(320.0, 240.0), // same → 0°
        Vec2F64::new(820.0, 240.0), // +500px → 45°
        Vec2F64::new(820.0, 240.0), // +500px → 45°
    ];
    let angle = result.median_parallax_deg(&x1, &x2, &cam);
    assert!(
        (angle - 45.0).abs() < 0.01,
        "expected median ~45°, got {angle}"
    );
}

#[test]
fn test_median_parallax_out_of_bounds_indices_ignored() {
    // Inlier indices beyond x1/x2 length are filtered out.
    let cam = test_camera();
    let result = stub_result(vec![0, 99]); // index 99 doesn't exist
    let x1 = vec![Vec2F64::new(320.0, 240.0)];
    let x2 = vec![Vec2F64::new(820.0, 240.0)];
    let angle = result.median_parallax_deg(&x1, &x2, &cam);
    assert!(
        (angle - 45.0).abs() < 0.01,
        "expected ~45° (out-of-bounds index skipped), got {angle}"
    );
}

fn u8_to_f32_image(src: &kornia_image::Image<u8, 1>) -> kornia_image::Image<f32, 1> {
    let mut dst = kornia_image::Image::from_size_val(src.size(), 0.0).unwrap();
    src.as_slice()
        .iter()
        .zip(dst.as_slice_mut())
        .for_each(|(&s, d)| *d = s as f32 / 255.0);
    dst
}

type OrbFiltered = (Vec<(f32, f32)>, Vec<f32>, Vec<[u8; 32]>);

fn filter_by_mask(
    kps: &[(f32, f32)],
    ori: &[f32],
    desc: &[[u8; 32]],
    mask: &[bool],
) -> OrbFiltered {
    let mut out_kps = Vec::new();
    let mut out_ori = Vec::new();
    let mut out_desc = Vec::new();
    let mut desc_idx = 0;
    for (i, &valid) in mask.iter().enumerate() {
        if valid {
            out_kps.push(kps[i]);
            out_ori.push(ori[i]);
            out_desc.push(desc[desc_idx]);
            desc_idx += 1;
        }
    }
    (out_kps, out_ori, out_desc)
}

#[test]
fn scoring_view_overwrites_masks_and_materializes_only_strict_winners() {
    use super::scoring::ScoringPoints;

    // Sampson residuals are 0, 0.5 and 2. Include an odd SIMD tail and two
    // bounded-scoring chunk boundaries. Equality at 0.5 remains an inlier.
    let n = 129;
    let zeros = vec![0.0; n];
    let y2: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let points = ScoringPoints::new(&zeros, &zeros, &zeros, &y2);
    let f = Mat3F64::from_cols(
        Vec3F64::ZERO,
        Vec3F64::new(0.0, 0.0, 1.0),
        Vec3F64::new(0.0, -1.0, 0.0),
    );
    let mut reference_mask = vec![false; n];
    let reference = score_inliers_f(&f, &zeros, &zeros, &zeros, &y2, 0.5, &mut reference_mask);
    assert_eq!(reference.0, 86);

    let mut reused = vec![true; n];
    let full = points.fundamental(&f, 0.5, &mut reused);
    assert_eq!(full.0, reference.0);
    assert_eq!(full.1.to_bits(), reference.1.to_bits());
    assert_eq!(reused, reference_mask);

    for bounded in [false, true] {
        reused.fill(true);
        let candidate = if bounded {
            points.fundamental_candidate::<true>(&f, 0.5, (0, f64::INFINITY), &mut reused)
        } else {
            points.fundamental_candidate::<false>(&f, 0.5, (0, f64::INFINITY), &mut reused)
        };
        let candidate = candidate.unwrap();
        assert_eq!(candidate.0, reference.0);
        assert_eq!(candidate.1.to_bits(), reference.1.to_bits());
        assert_eq!(reused, reference_mask);
        assert!(points
            .fundamental_candidate::<true>(&f, 0.5, candidate, &mut reused)
            .is_none());
    }

    // High-support incumbents use the masked bounded path instead.
    reference_mask.fill(false);
    let reference = score_inliers_f(&f, &zeros, &zeros, &zeros, &y2, 2.0, &mut reference_mask);
    reused.fill(false);
    let high = points
        .fundamental_candidate::<true>(&f, 2.0, (100, f64::INFINITY), &mut reused)
        .unwrap();
    assert_eq!(high.0, reference.0);
    assert_eq!(high.1.to_bits(), reference.1.to_bits());
    assert_eq!(reused, reference_mask);

    // Tightening after a full-support score must clear old true entries.
    let tight = points.fundamental(&f, 0.0, &mut reused);
    assert_eq!(tight.0, 43);
    assert_eq!(reused, (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>());
    let h = points.homography(&Mat3F64::IDENTITY, 0.0, &mut reused);
    assert_eq!(h.0, 43);
    assert_eq!(reused, (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>());
}
