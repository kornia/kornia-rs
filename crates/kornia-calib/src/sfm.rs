//! Tags-free (feature-driven) multi-camera calibration via incremental structure-from-motion.
//!
//! Natural-feature tracks — not a tag — drive the geometry. A best-connected camera pair bootstraps
//! the reconstruction from the two-view essential matrix, remaining cameras register by PnP against
//! the growing point cloud, and a bundle adjustment polishes everything. The reconstruction is
//! recovered **up to scale** (the fundamental monocular ambiguity); a single metric tag then fixes
//! that one scalar — the tag is a *scale bar*, nothing else. Output poses are `T_world_cam` in the
//! reference camera's frame, metric ONLY when a tag actually anchored the scale: supplying a tag is
//! not sufficient, since it must also be seen by two registered views and triangulate. See
//! [`crate::ScaleSource`], which reports which case a given result is.
//!
//! Everything except the incremental orchestration is reused: `ransac_essential_5pt` +
//! `decompose_essential` (bootstrap relative pose), [`kornia_3d::pose::triangulate_matched_points`],
//! [`kornia_3d::pnp::solve_pnp_ransac`] (register a new camera into the cloud), and
//! [`kornia_3d::ba_schur::bundle_adjust_schur`].

use std::collections::{HashMap, HashSet};

use kornia_3d::ba::{BaMotionPrior, BaObservation, BaParams, BaPosePrior, BaResult};
use kornia_3d::ba_schur::bundle_adjust_schur_with_all_priors;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{solve_pnp_ransac, PnPMethod, RansacParams as PnpRansacParams};
use kornia_3d::pose::{
    decompose_essential, ransac_essential_5pt, triangulate_matched_points, Pose3d,
    RansacParams as TvRp, TriangulationConfig,
};
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64};

use crate::error::CalibError;
use crate::types::{
    CalibConfig, CameraStats, FeatureTrack, Observation, Point, Reconstruction, ScaleSource,
    TagObservation,
};

/// Convert an f32 PnP rotation/translation (world→cam) into an f64 [`Pose3d`].
fn pose_from_pnp(r: Mat3AF32, t: Vec3AF32) -> Pose3d {
    let a = r.to_cols_array(); // column-major [f32; 9]
    let rot = Mat3F64::from_cols_array(&[
        a[0] as f64,
        a[1] as f64,
        a[2] as f64,
        a[3] as f64,
        a[4] as f64,
        a[5] as f64,
        a[6] as f64,
        a[7] as f64,
        a[8] as f64,
    ]);
    Pose3d::new(rot, Vec3F64::new(t.x as f64, t.y as f64, t.z as f64))
}

/// Reproject a world point into a camera and return the normalized-coordinate residual, or `None`
/// behind the camera.
fn norm_residual(pose: &Pose3d, p_world: Vec3F64, n: Vec2F64) -> Option<f64> {
    let pc = pose.transform_point(&p_world);
    if pc.z <= 1e-6 {
        return None;
    }
    Some(((pc.x / pc.z - n.x).powi(2) + (pc.y / pc.z - n.y).powi(2)).sqrt())
}

/// Reconstruct from feature tracks, returning the MAP as well as the camera poses.
///
/// Bootstrap from the best-supported view pair, register the rest by PnP against the growing
/// cloud, bundle-adjust, and optionally anchor the metric scale from a tag.
///
/// Returns the map, not just the cameras. If only the rig geometry is wanted, convert with
/// `RigCalibration::from(..)` -- see that `From` impl for exactly what the conversion drops.
///
/// # Arguments
///
/// * `cameras` - intrinsics per view, indexed the same way as `FeatureTrack::obs`. Distortion is
///   applied when normalising observations.
/// * `tags_for_scale` - optional metric anchor. Empty means the result is up to scale; the first
///   tag's frame otherwise fixes the world gauge and its known side length fixes the metric.
/// * `tracks` - multi-view feature tracks. A track needs at least two views to triangulate, and
///   the reconstruction needs enough shared tracks between views to register them.
/// * `config` - solver settings; see [`CalibConfig`].
/// * `obs_depth` - optional metric depth per observation, shaped exactly like `tracks`:
///   `obs_depth[i][j]` is the depth of `tracks[i].obs[j]`, `None` where none is available. Pass
///   `None` for the classic depth-free solve. Ignored unless
///   [`CalibConfig::depth_prior_rel_sigma`] is positive.
///
/// # Depth
///
/// Monocular reprojection is exactly scale-invariant, so the gauge has one free DoF that the
/// optimiser navigates by numerical accident. That is the root of two failures a fiducial-free
/// walkthrough cannot otherwise escape:
///
/// 1. **No metric scale.** Depth residuals observe absolute scale directly, so the reconstruction
///    lands in metres with no tag in the scene.
/// 2. **Scale drift.** Along a no-revisit walk the reconstruction's scale wanders — the measured
///    symptom is rooms late in a clip reconstructing several times larger than early ones. A
///    per-observation depth prior pins EVERY segment of the chain, not just a global average.
///
/// Depths are a soft prior, not a constraint: they are robustified, sigma-weighted by
/// [`CalibConfig::depth_prior_rel_sigma`], and (by default) re-gauged per view — see
/// [`CalibConfig::depth_per_keyframe_scale`].
///
/// # Returns
///
/// A [`Reconstruction`]: the registered view poses (`T_world_cam`, `None` where a view could not
/// be registered), the triangulated points, the track each point came from, the observations that
/// survived the solve, per-view statistics, and where the metric scale came from.
///
/// # Errors
///
/// [`CalibError`] if the inputs cannot produce a reconstruction — too few views or tracks, no
/// viable bootstrap pair, or a bundle-adjustment failure.
///
/// # Example
///
/// ```no_run
/// use kornia_calib::{reconstruct, CalibConfig, FeatureTrack, ScaleSource};
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_algebra::Vec2F64;
///
/// # fn main() -> Result<(), kornia_calib::CalibError> {
/// let cam = || PinholeCamera { fx: 600.0, fy: 600.0, cx: 320.0, cy: 240.0, ..PinholeCamera::IDENTITY };
/// let cameras = vec![cam(), cam(), cam()];
///
/// // One scene point seen by three views. Real input comes from a matcher.
/// let tracks = vec![FeatureTrack {
///     obs: vec![
///         (0, Vec2F64::new(320.0, 240.0)),
///         (1, Vec2F64::new(300.0, 241.0)),
///         (2, Vec2F64::new(340.0, 239.0)),
///     ],
/// }];
///
/// // `None` for the depth argument: no metric depth measurements, so the map is up to scale.
/// let recon = reconstruct(&cameras, &[], &tracks, &CalibConfig::new(0.1), None)?;
///
/// // No tag was supplied, so the map is honestly up to scale rather than silently "metric".
/// assert_eq!(recon.scale, ScaleSource::UpToScale);
///
/// // Carry per-track data onto the map without re-matching -- each point names its own track.
/// for point in &recon.points {
///     let _world_position = point.position;
///     let _source_track = &tracks[point.track_id];
/// }
/// # Ok(())
/// # }
/// ```
pub fn reconstruct(
    cameras: &[PinholeCamera],
    tags_for_scale: &[TagObservation],
    tracks: &[FeatureTrack],
    config: &CalibConfig,
    obs_depth: Option<&[Vec<Option<f64>>]>,
) -> Result<Reconstruction, CalibError> {
    let n_cams = cameras.len();
    let idcam = PinholeCamera::IDENTITY;
    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };

    // Per track: normalized observation per camera (undistort + K⁻¹). Raw pixels stay in `tracks`.
    // MUTABLE because `refine_intrinsics` rewrites these in place once it has fitted a correction.
    let mut norm: Vec<Vec<(usize, Vec2F64)>> = tracks
        .iter()
        .map(|t| {
            t.obs
                .iter()
                .map(|(c, uv)| (*c, cameras[*c].normalize(*uv)))
                .collect()
        })
        .collect();
    // Parallel to `norm`: metric depth per observation, or `None`. All-`None` when the feature is
    // off, so every downstream site indexes it unconditionally instead of branching on whether
    // depth exists. A caller whose depth array is the wrong shape simply gets `None` per lookup.
    let norm_depth: Vec<Vec<Option<f32>>> = match obs_depth {
        Some(d) if config.depth_prior_rel_sigma > 0.0 => d
            .iter()
            .map(|t| t.iter().map(|x| x.map(|v| v as f32)).collect())
            .collect(),
        _ => norm.iter().map(|t| vec![None; t.len()]).collect(),
    };

    // Count shared tracks per camera pair to choose the bootstrap pair.
    let mut pair_count: HashMap<(usize, usize), usize> = HashMap::new();
    for obs in &norm {
        for i in 0..obs.len() {
            for j in (i + 1)..obs.len() {
                let (a, b) = (obs[i].0.min(obs[j].0), obs[i].0.max(obs[j].0));
                *pair_count.entry((a, b)).or_insert(0) += 1;
            }
        }
    }
    // Most-shared-track pair; deterministic tie-break on the smaller (a, b) so the whole
    // reconstruction (world frame, seed cloud, growth order) is reproducible run-to-run — HashMap
    // iteration order is randomized and must not decide the bootstrap.
    let &(a0, b0) = pair_count
        .iter()
        .max_by(|x, y| x.1.cmp(y.1).then_with(|| y.0.cmp(x.0)))
        .map(|(p, _)| p)
        .ok_or(CalibError::NoReferenceTagView)?;

    // --- Bootstrap: two-view essential matrix on the best pair → poses (world = cam a0), s = 1. ---
    let (mut x1, mut x2) = (Vec::new(), Vec::new());
    for t in tracks {
        let pa = t.obs.iter().find(|(c, _)| *c == a0);
        let pb = t.obs.iter().find(|(c, _)| *c == b0);
        if let (Some((_, ua)), Some((_, ub))) = (pa, pb) {
            x1.push(*ua);
            x2.push(*ub);
        }
    }
    // Calibrated 5-point Nistér essential + explicit cheirality vote. We drive the essential arm
    // directly (not the full `TwoViewEstimator`, whose F/H model selection can pick a degenerate
    // homography on a converging rig): a calibrated rig always wants the essential.
    let n1: Vec<Vec2F64> = x1.iter().map(|p| cameras[a0].normalize(*p)).collect();
    let n2: Vec<Vec2F64> = x2.iter().map(|p| cameras[b0].normalize(*p)).collect();
    // `ransac_essential_5pt` only normalizes with K⁻¹ — it does NOT undistort. Feed UNDISTORTED
    // pixels so the essential arm matches the undistorted-normalized coords every other stage uses
    // (cheirality vote, triangulation, BA); K⁻¹·undistort(px) == normalize(px). The 2.0 px threshold
    // stays valid in undistorted pixel space. (Raw pixels here would bias R/t on a distorted lens.)
    let x1u: Vec<Vec2F64> = x1.iter().map(|p| cameras[a0].undistort(p.x, p.y)).collect();
    let x2u: Vec<Vec2F64> = x2.iter().map(|p| cameras[b0].undistort(p.x, p.y)).collect();
    let rp = TvRp {
        max_iterations: 2000,
        threshold: 2.0,
        min_inliers: 8,
        random_seed: Some(0),
        refit: true,
    };
    let ess = ransac_essential_5pt(
        &x1u,
        &x2u,
        &cameras[a0].intrinsic_matrix(),
        &cameras[b0].intrinsic_matrix(),
        &rp,
    )
    .map_err(|e| CalibError::BundleAdjust(format!("essential bootstrap: {e:?}")))?;
    let cands = decompose_essential(&ess.model)
        .ok_or_else(|| CalibError::BundleAdjust("essential decomposition failed".into()))?;
    // Lenient triangulation for the cheirality vote (count points in front of BOTH cameras).
    let tvote = TriangulationConfig {
        min_parallax_deg: 0.0,
        max_reprojection_error: 1e9,
        min_cheirality_count: 0,
        ..Default::default()
    };
    let mut best = (0usize, Pose3d::IDENTITY);
    for (r, t) in cands {
        let pb = Pose3d::new(r, t); // world(=a0) → b, unit translation
        let mut cnt = 0usize;
        for k in 0..n1.len() {
            if let Ok(pts) = triangulate_matched_points(
                &[n1[k]],
                &[n2[k]],
                &Pose3d::IDENTITY,
                &pb,
                &idcam,
                &tvote,
            ) {
                if let Some(p) = pts.first() {
                    if p.position.z > 0.0 && pb.transform_point(&p.position).z > 0.0 {
                        cnt += 1;
                    }
                }
            }
        }
        if cnt > best.0 {
            best = (cnt, pb);
        }
    }
    if best.0 == 0 {
        return Err(CalibError::BundleAdjust(
            "essential cheirality: no valid pose".into(),
        ));
    }

    let mut poses: Vec<Option<Pose3d>> = vec![None; n_cams];
    poses[a0] = Some(Pose3d::IDENTITY);
    poses[b0] = Some(best.1); // T_b0_a0, unit translation

    // Triangulate every track visible in the bootstrap pair → seed the point cloud (world = a0 frame).
    let mut point3d: HashMap<usize, Vec3F64> = HashMap::new();
    triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);

    // --- Incremental grow: register the unplaced camera with the most 2D↔3D links via PnP. ---
    // PnP (nondeterministic EPnP-RANSAC) can transiently fail for one camera while others remain
    // solvable, so a failure marks just that camera unregisterable and the loop keeps growing —
    // NOT aborting every remaining camera.
    let mut pnp_failed: HashSet<usize> = HashSet::new();
    loop {
        // For each unplaced camera, gather (world_point, normalized_pixel) from tracks with a 3D point.
        let mut best: Option<(usize, Vec<Vec3F64>, Vec<Vec2F64>)> = None;
        for c in 0..n_cams {
            if poses[c].is_some() || pnp_failed.contains(&c) {
                continue;
            }
            let (mut wp, mut ip) = (Vec::new(), Vec::new());
            for (ti, obs) in norm.iter().enumerate() {
                if let (Some(p), Some((_, uv))) =
                    (point3d.get(&ti), obs.iter().find(|(cc, _)| *cc == c))
                {
                    wp.push(*p);
                    ip.push(*uv);
                }
            }
            if wp.len() >= 4 && best.as_ref().is_none_or(|(_, w, _)| wp.len() > w.len()) {
                best = Some((c, wp, ip));
            }
        }
        let Some((c, wp, ip)) = best else { break };

        let world: Vec<Vec3AF32> = wp
            .iter()
            .map(|p| Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32))
            .collect();
        let image: Vec<Vec2F32> = ip
            .iter()
            .map(|p| Vec2F32::new(p.x as f32, p.y as f32))
            .collect();
        let pnp = solve_pnp_ransac(
            &world,
            &image,
            &Mat3AF32::IDENTITY, // normalized coords ⇒ identity intrinsics
            None,
            PnPMethod::EPnPDefault,
            &PnpRansacParams {
                // Normalized units, tied to the same threshold the rest of the pipeline calls an
                // outlier bound. Defaults to 0.01, the value this was hardcoded to.
                reproj_threshold_px: config.max_reprojection_error as f32,
                // Seed the sampler. `RansacParams::default()` leaves `random_seed: None`, which
                // draws from the thread RNG, so registration was NONDETERMINISTIC: identical
                // inputs produced different reconstructions run to run (measured on 40 keyframes
                // of EuRoC MH01 — 12 / 30 / 39 cameras registered over three runs of the same
                // command). Because a transient PnP failure marks a camera unregisterable, that
                // randomness changes the final map, not merely its timing. The seed varies per
                // camera so different views still draw different sample sequences.
                random_seed: Some(0x00C0_FFEE ^ c as u64),
                ..Default::default()
            },
        );
        // Accepting a registration is not the same as PnP returning `Ok`. `solve_pnp_ransac`
        // succeeds whenever it finds *a* consensus, however small, so an unguarded `Ok` arm admits
        // cameras whose pose is fitted to a handful of correspondences. That is not a
        // self-correcting mistake: `triangulate_new` immediately creates 3D points FROM the bad
        // pose, and those points feed the next camera's PnP, so one bad registration propagates.
        // Measured on EuRoC MH01 the symptom was global RMSE rising WITH the registered count —
        // 34 cameras at 10.4 px, 38 at 23.2 px — the opposite of a healthy incremental SfM.
        let accepted = pnp.as_ref().ok().and_then(|r| {
            let pose = pose_from_pnp(r.pose.rotation, r.pose.translation);
            // Score BOTH the returned (refit) pose and the RANSAC consensus, and keep the larger.
            // `inliers` is classified against the PRE-refit minimal-sample model, and the refit can
            // move either way, so trusting one alone under-counts: a solid 107-inlier consensus has
            // been observed reclassifying to under 30 against its own refit, silently losing a good
            // view. Both counts use the same threshold, so they are directly comparable.
            let n_refit = wp
                .iter()
                .zip(ip.iter())
                .filter(|(w, i)| {
                    norm_residual(&pose, **w, **i)
                        .is_some_and(|e| e <= config.max_reprojection_error)
                })
                .count();
            (n_refit.max(r.inliers.len()) >= config.min_registration_inliers).then_some(pose)
        });
        match accepted {
            Some(pose) => {
                poses[c] = Some(pose);
                if let Some(cb) = config.progress.as_ref() {
                    cb(poses.iter().filter(|p| p.is_some()).count(), n_cams);
                }
                let before = point3d.len();
                triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);
                // A camera that could not register earlier may register comfortably NOW: each
                // success triangulates new points, so the 2D-3D evidence available to the remaining
                // views grows. Without this, `min_registration_inliers` would turn every transient
                // rejection into a permanent one and strictly REDUCE the registered count versus
                // having no gate at all — the gate is only safe in the presence of the retry.
                //
                // Gated on the cloud ACTUALLY growing, so a registration that added no evidence
                // does not buy a re-run of every rejected camera against an unchanged map.
                //
                // This terminates: the set is cleared only when the cloud grows, the cloud is
                // bounded by the track count, and between clears each iteration either registers a
                // camera (at most `n_cams` times) or removes one from consideration.
                if point3d.len() > before {
                    pnp_failed.clear();
                }
            }
            // PnP failed outright, or its consensus was too thin to trust. Either way this camera
            // cannot register against the CURRENT map; try the others and revisit it after the
            // cloud has grown.
            None => {
                pnp_failed.insert(c);
            }
        }
    }

    // --- Bundle adjustment: all track points free, the reference camera (a0) fixed to anchor gauge. ---
    // Iterate the triangulated points in TRACK ORDER, not `HashMap` order. Rust's default hasher
    // is randomised per process, so `for (ti, p) in &point3d` yields a different order every run.
    // That was harmless while these only fed bundle adjustment and the RMS statistics, all
    // permutation-invariant -- but `Reconstruction` publishes `points`, `point_track_id` and
    // `observations`, so the order becomes part of the contract. A caller diffing two runs,
    // snapshotting a map, or holding point indices across a rebuild would otherwise see churn
    // with no cause. This file already guards the same hazard for the bootstrap pair.
    let mut track_ids: Vec<usize> = point3d.keys().copied().collect();
    track_ids.sort_unstable();

    // `pt_index[ti]` is just `ti`'s position in `track_ids`, so it is stable across BA rounds and
    // the published `points` order matches the BA input order in every one of them.
    let pt_index: HashMap<usize, usize> =
        track_ids.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let point_track_id: Vec<usize> = track_ids.clone();
    let mut kept_obs: Vec<Observation> = Vec::new();
    for ti in &track_ids {
        for (j, (c, _)) in norm[*ti].iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            // `norm[ti]` is built by mapping over `tracks[ti].obs`, so index j lines up and the
            // raw pixel is recoverable without re-normalising.
            kept_obs.push(Observation {
                view: *c,
                point: pt_index[ti],
                pixel: tracks[*ti].obs[j].1,
            });
        }
    }

    let mut res = global_ba(
        &poses,
        &point3d,
        &track_ids,
        &norm,
        &norm_depth,
        &idcam,
        a0,
        config,
    )?;

    // --- Alternating intrinsics refinement (COLMAP refines focal/distortion INSIDE its BA; this is
    // the alternating equivalent, which needs no solver surgery). The solver sees NORMALIZED
    // coordinates, so a focal error is a single global scale `gamma` on them and lens distortion a
    // radial/tangential polynomial. Against the current map that model is LINEAR in the stacked
    // unknowns, so one closed-form least squares per round corrects the camera; the second bundle
    // adjustment below then re-settles the geometry against it.
    let camera_correction = if config.refine_intrinsics {
        let fit = fit_camera_correction(&res, &pt_index, &norm, &poses);
        if let Some((gamma, k1, k2, p1, p2)) = fit {
            for track in norm.iter_mut() {
                for (_, n) in track.iter_mut() {
                    let (x, y) = (n.x, n.y);
                    let r2 = x * x + y * y;
                    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                    n.x = gamma * (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x));
                    n.y = gamma * (y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
                }
            }
            // Feed the first solve's state back in, then RE-SOLVE against the corrected
            // observations. Without this second pass the refinement would be computed, reported,
            // and never reflected in the poses this function returns — the observations it rewrote
            // would go unread, since nothing downstream of here solves anything.
            for (c, p) in poses.iter_mut().enumerate() {
                if p.is_some() {
                    *p = Some(res.poses[c]);
                }
            }
            for (ti, pidx) in &pt_index {
                if let Some(v) = res.points.get(*pidx) {
                    point3d.insert(*ti, *v);
                }
            }
            res = global_ba(
                &poses,
                &point3d,
                &track_ids,
                &norm,
                &norm_depth,
                &idcam,
                a0,
                config,
            )?;
        }
        fit
    } else {
        None
    };

    let registered: Vec<bool> = poses.iter().map(|p| p.is_some()).collect();

    // --- Metric scale from the tag: triangulate its corners, compare to the known side length. ---
    // Scaling the world by `s` (points ×s AND world→cam translation ×s) leaves reprojection unchanged,
    // so we compute per-camera stats on the UNSCALED BA result and only scale the output translation.
    // `None` when the tag could NOT anchor the scale -- unset size, seen by fewer than two
    // REGISTERED cameras, or its corners failed to triangulate. Those are ordinary outcomes, not
    // corner cases: a camera that cannot register via PnP is skipped by design, so "a tag was
    // supplied but the views that saw it never registered" happens in normal use. Reporting
    // `ScaleSource::Tag` off `tags_for_scale.first()` would then claim metric for a map still at
    // the identity scale -- exactly the ambiguity this type exists to remove.
    let anchored = tag_scale(
        tags_for_scale,
        cameras,
        &res.poses,
        &registered,
        &idcam,
        &tcfg,
        config.tag_size_m,
    );
    let scale = anchored.unwrap_or(1.0);

    // Pixels-per-normalized-unit correction for the reported RMS.
    //
    // `refine_intrinsics` rewrites `norm` IN PLACE into the true camera's normalized frame, where
    // fx_true = fx_assumed / gamma. Every RMS below converts residuals to pixels with the ASSUMED
    // fx from `cameras`, so without this the reported number is gamma times too large — and this
    // is the number the callers quote as their acceptance metric, so a silent few-percent bias in
    // it is worse than a loud one.
    let focal_scale = camera_correction.map_or(1.0, |(gamma, ..)| 1.0 / gamma.max(1e-9));

    // Per-camera reprojection RMS (pixels); analytical covariance is tag-oriented so stays `None`.
    // Use the BA-optimized points (`res.points`), NOT the pre-BA cloud: BA moves points as free
    // variables, so evaluating the pre-BA cloud under post-BA poses would report a stale residual.
    // `pt_index` indexes both identically (same order as the BA input).
    let per_camera = (0..n_cams)
        .map(|c| {
            feature_stats(
                focal_scale,
                c,
                &res.poses,
                &registered,
                &res.points,
                &pt_index,
                &norm,
                &cameras[c],
            )
        })
        .collect();
    let reproj_rmse_px = global_reproj_rmse(
        focal_scale,
        &res.poses,
        &registered,
        &res.points,
        &pt_index,
        &norm,
        cameras,
    );

    // Output `T_world_cam` (camera→world): invert the metric world→cam (translation scaled).
    let out_poses: Vec<Option<Pose3d>> = (0..n_cams)
        .map(|c| {
            registered[c].then(|| {
                Pose3d::new(res.poses[c].rotation, res.poses[c].translation * scale).inverse()
            })
        })
        .collect();
    // Points share the world frame with the output poses, so they take the same metric scale.
    // Paired with their track id here rather than returned as parallel vectors, so the two cannot
    // come apart in a caller's hands.
    let out_points: Vec<Point> = res
        .points
        .iter()
        .zip(&point_track_id)
        .map(|(p, &track_id)| Point {
            position: *p * scale,
            track_id,
        })
        .collect();

    Ok(Reconstruction {
        views: out_poses,
        points: out_points,
        observations: kept_obs,
        reproj_rmse_px,
        per_view: per_camera,
        scale: match (anchored, tags_for_scale.first()) {
            (Some(_), Some(t)) => ScaleSource::Tag {
                id: t.tag_id,
                size_m: config.tag_size_m,
            },
            _ => ScaleSource::UpToScale,
        },
        camera_correction,
    })
}
/// The reference reprojection sigma every prior family is deflated by, in NORMALISED units.
///
/// Bundle adjustment here runs against `PinholeCamera::IDENTITY`, so its reprojection residuals are
/// in normalised-camera units while the prior sigmas the caller supplies are quoted in physical
/// ones (metres, unit-vector components). Dividing each prior sigma by this value puts every family
/// on one numeric scale, which is what lets a single Huber knee gate them coherently.
///
/// `max_reprojection_error / 2` because that threshold reads as 2σ. Derived in ONE place because
/// four call sites must agree exactly: the depth measurement's sigma, the up prior's, the motion
/// prior's, and the robust knee that gates all three. If they drift, the knee stops matching the
/// residuals it is meant to gate and the priors are silently mis-weighted rather than wrong in any
/// way a test would show.
fn reproj_sigma(config: &CalibConfig) -> f64 {
    (config.max_reprojection_error / 2.0).max(1e-6)
}

/// Depth measurement + already-deflated sigma for observation `j` of track `ti`, or `(None, 0.0)`
/// when there is no usable depth there.
///
/// The returned sigma is DEFLATED into reprojection units, not the raw metric sigma. The solver's
/// reprojection residuals are unwhitened normalized-camera values (implicit σ = 1 normalized unit —
/// an entire focal length!), while its depth residuals divide by their sigma. Passing the honest
/// metric sigma therefore makes each depth row carry orders of magnitude more cost than a
/// reprojection row, and the solve goes depth-dominated no matter how loose the relative sigma
/// looks — measured: `rel_sigma` 5.0 (≈ no confidence at all) still halved registration.
/// Multiplying by `1/σ_r` — where `σ_r` is the reprojection noise scale in normalized units, the
/// threshold read as 2σ — puts both families in the same implicit unit.
fn depth_fields(
    norm_depth: &[Vec<Option<f32>>],
    ti: usize,
    j: usize,
    config: &CalibConfig,
) -> (Option<f32>, f32) {
    let rel_sigma = config.depth_prior_rel_sigma;
    match norm_depth.get(ti).and_then(|t| t.get(j)).copied().flatten() {
        Some(d) if rel_sigma > 0.0 && d > 0.0 => {
            let sigma_r = reproj_sigma(config) as f32;
            (Some(d), (rel_sigma as f32) * d / sigma_r)
        }
        _ => (None, 0.0),
    }
}

/// Robust per-view depth gauge: the median of `z_map / d_pred` over that view's depth
/// observations, re-gauged against the median view and clamped. `1.0` where a view lacks enough
/// pairs to fit.
///
/// # Why per view rather than one global scale
///
/// Learned depth is not gauge-stable frame to frame. On forward motion a per-frame scale error is
/// indistinguishable from along-axis translation, so a single global scale hands every wander of
/// the network straight to the trajectory: the residual is real, the solver dutifully moves the
/// camera, every frame. That is a drift generator.
///
/// # Scale only, not the affine `s·d + t`
///
/// A free intercept absorbs bas-relief compression instead of correcting it — measured: sparse
/// depth spanned a ratio of 1.13 across a view where the network saw 2.13, and an affine fit
/// reproduced that flattening faithfully. It also makes the map's ABSOLUTE scale unobservable from
/// depth, which defeats the point of supplying depth at all.
///
/// # Gauge
///
/// The scales are normalised by their own median, so this re-gauges views RELATIVE to each other
/// without moving the map as a whole. Without that, the reconstruction would be free to breathe
/// every time bundle adjustment ran.
fn fit_depth_scales(
    poses: &[Option<Pose3d>],
    point3d: &HashMap<usize, Vec3F64>,
    track_ids: &[usize],
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    n_cams: usize,
) -> Vec<f64> {
    /// Below this many depth pairs a median is noise, and a wrong per-view gauge is worse than the
    /// global one it replaces.
    const MIN_PAIRS: usize = 12;

    let mut per_cam: Vec<Vec<f64>> = vec![Vec::new(); n_cams];
    for ti in track_ids {
        let Some(p) = point3d.get(ti) else { continue };
        for (j, (c, _)) in norm[*ti].iter().enumerate() {
            // `.get(ti)`, not `[ti]`: the contract on `obs_depth` is that a wrong-shaped array
            // simply yields `None` per lookup, and `depth_fields` already honours that. Indexing
            // here made a caller who supplied depth for only the first N tracks panic instead —
            // the one shape of malformed input the documentation explicitly invites.
            let (Some(pose), Some(d)) = (
                &poses[*c],
                norm_depth
                    .get(*ti)
                    .and_then(|t| t.get(j))
                    .copied()
                    .flatten(),
            ) else {
                continue;
            };
            let z = pose.transform_point(p).z;
            if z > 1e-9 && d > 0.0 {
                per_cam[*c].push(z / d as f64); // map units per network unit
            }
        }
    }
    // Median, not mean: a depth network hallucinates at occlusion boundaries and on mirrors, and
    // those observations are a fat tail, not Gaussian noise.
    let median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.len() < MIN_PAIRS {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = v[v.len() / 2];
        (m.is_finite() && m > 1e-9).then_some(m)
    };
    let mut scales: Vec<Option<f64>> = per_cam.iter_mut().map(median).collect();

    let mut fitted: Vec<f64> = scales.iter().flatten().copied().collect();
    if fitted.len() < 2 {
        return vec![1.0; n_cams];
    }
    fitted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let anchor = fitted[fitted.len() / 2];
    for s in scales.iter_mut().flatten() {
        *s /= anchor;
    }
    // A view whose scale is wildly off has a broken pose or a broken depth map, not a gauge
    // offset; trusting its fit would let it drag the solve. Fall back to neutral.
    scales
        .into_iter()
        .map(|s| match s {
            Some(v) if (0.5..2.0).contains(&v) => v,
            _ => 1.0,
        })
        .collect()
}

/// Per-view up priors, or `None` when [`CalibConfig::up_prior_sigma`] is off.
///
/// The camera-frame direction asserted to be up is image-up `(0, −1, 0)` — "the camera was held
/// roughly upright" — so `up_prior_sigma` must be chosen to match how much that is actually
/// believed.
///
/// World up is a GAUGE choice and has to agree with the anchor camera `a0`, whose pose is held
/// fixed: `up_world = R_a0ᵀ · (0,−1,0)`. Any other choice fights the one pose the solve cannot move.
fn up_priors(
    poses: &[Option<Pose3d>],
    a0: usize,
    config: &CalibConfig,
) -> Option<Vec<Option<BaPosePrior>>> {
    if config.up_prior_sigma <= 0.0 {
        return None;
    }
    const UP_CAM: [f64; 3] = [0.0, -1.0, 0.0];
    let up_world: [f32; 3] = match poses.get(a0).and_then(|p| *p) {
        Some(pa) => {
            let w = pa.rotation.transpose() * Vec3F64::new(UP_CAM[0], UP_CAM[1], UP_CAM[2]);
            [w.x as f32, w.y as f32, w.z as f32]
        }
        None => [0.0, -1.0, 0.0],
    };
    // Deflated into reprojection units for the same reason as `depth_fields`: bundle adjustment
    // runs against `PinholeCamera::IDENTITY`, so its reprojection residuals are in NORMALISED units
    // while this sigma is quoted in unit-vector units. Passing it raw made the prior `1/σ_r` times
    // stiffer than every other term in the same solve — 360× at fx 1440 with an 8 px threshold,
    // where 1° of pitch deviation cost what a 489 px reprojection error would.
    let sigma_r = reproj_sigma(config);
    Some(
        poses
            .iter()
            .map(|p| {
                p.as_ref().map(|_| {
                    BaPosePrior::orientation_only(
                        up_world,
                        (config.up_prior_sigma / sigma_r) as f32,
                    )
                })
            })
            .collect(),
    )
}

/// Constant-velocity motion priors over consecutive REGISTERED triplets, or `None` when off.
///
/// Consecutive in VIEW-INDEX order (video keyframes are time-ordered), `alpha` from the index
/// spacing, and triplets spanning more than [`MAX_TRIPLET_SPAN`] indices skipped — a bridge across
/// a long unregistered stretch is not a constant-velocity hypothesis worth asserting. Sigmas are
/// deflated into reprojection units exactly like the depth and up priors.
fn motion_priors_for(poses: &[Option<Pose3d>], config: &CalibConfig) -> Option<Vec<BaMotionPrior>> {
    /// Widest index gap a triplet may span before it stops being a plausible motion hypothesis.
    const MAX_TRIPLET_SPAN: usize = 12;

    if config.motion_prior_sigma <= 0.0 {
        return None;
    }
    let sigma_r = reproj_sigma(config);
    let sp = (config.motion_prior_sigma / sigma_r) as f32;
    let so = (0.5 * config.motion_prior_sigma / sigma_r) as f32;
    let reg: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    let out: Vec<BaMotionPrior> = reg
        .windows(3)
        .filter(|w| w[2] - w[0] <= MAX_TRIPLET_SPAN)
        .map(|w| BaMotionPrior {
            i0: w[0],
            i1: w[1],
            i2: w[2],
            alpha: (w[1] - w[0]) as f32 / (w[2] - w[0]) as f32,
            position_sigma: sp,
            orientation_sigma: so,
        })
        .collect();
    (!out.is_empty()).then_some(out)
}

/// One global bundle adjustment over every triangulated point and every registered view, with the
/// anchor camera `a0` fixed to hold the gauge.
///
/// Callable more than once on the same problem, which is what `refine_intrinsics` needs: it
/// rewrites `norm` in place and the geometry then has to re-settle against the corrected camera.
#[allow(clippy::too_many_arguments)]
fn global_ba(
    poses: &[Option<Pose3d>],
    point3d: &HashMap<usize, Vec3F64>,
    track_ids: &[usize],
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    idcam: &PinholeCamera,
    a0: usize,
    config: &CalibConfig,
) -> Result<BaResult, CalibError> {
    // Re-fit the per-view depth gauge against the CURRENT geometry before every solve. Alternating
    // rather than joint: the scales are closed-form medians, so each pass refines the other.
    let depth_scale = if config.depth_per_keyframe_scale {
        fit_depth_scales(poses, point3d, track_ids, norm, norm_depth, poses.len())
    } else {
        vec![1.0; poses.len()]
    };
    let mut points: Vec<Vec3F64> = Vec::with_capacity(track_ids.len());
    let mut obs: Vec<BaObservation> = Vec::new();
    for (pidx, ti) in track_ids.iter().enumerate() {
        // `track_ids` is derived from `point3d`'s keys and points are only ever updated in place,
        // so every id resolves. Failing loudly rather than skipping matters: a skip would shift
        // every later `pidx` off its point and silently solve a scrambled problem.
        let p = point3d.get(ti).ok_or_else(|| {
            CalibError::BundleAdjust(format!("track {ti} has no triangulated point"))
        })?;
        points.push(*p);
        for (j, (c, nrm)) in norm[*ti].iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            let (depth_meas, depth_sigma) = depth_fields(norm_depth, *ti, j, config);
            obs.push(BaObservation {
                pose_idx: *c,
                point_idx: pidx,
                pixel: [nrm.x as f32, nrm.y as f32],
                fixed_pose: *c == a0, // reference camera fixed → gauge anchor
                fixed_point: false,
                // This view's own gauge baked into the measurement, so the residual measures the
                // shape the network got right rather than the scale it got wrong.
                depth_meas: depth_meas.map(|d| d * depth_scale[*c] as f32),
                depth_sigma,
            });
        }
    }
    let poses_ba: Vec<Pose3d> = poses
        .iter()
        .map(|p| p.unwrap_or(Pose3d::IDENTITY))
        .collect();
    bundle_adjust_schur_with_all_priors(
        &poses_ba,
        &points,
        &obs,
        idcam,
        &BaParams {
            max_iterations: config.max_iterations,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            // Depth AND motion residuals are both deflated by `reproj_sigma` (see `depth_fields`
            // and `motion_priors_for`), so their Huber knee is 1.345 × that scale, squared.
            //
            // Gated on EITHER family being armed, not just depth: `ba_schur` uses this one knee for
            // both, so a config with motion priors but no depth would fall back to the reprojection
            // knee and gate motion residuals at ~2σ instead of 1.345σ — tighter than intended, and
            // silent.
            depth_robust_scale_sq: if config.depth_prior_rel_sigma > 0.0
                || config.motion_prior_sigma > 0.0
            {
                let sr = (1.345 * reproj_sigma(config)) as f32;
                sr * sr
            } else {
                0.0
            },
            ..Default::default()
        },
        up_priors(poses, a0, config).as_deref(),
        motion_priors_for(poses, config).as_deref(),
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))
}

/// Closed-form OpenCV-model intrinsics correction `(gamma, k1, k2, p1, p2)` fitted against the
/// current reconstruction, or `None` when the fit is singular or outside its sanity bounds.
///
/// Linear in the stacked unknowns `beta = (gamma, gamma·k1, gamma·k2, gamma·p1, gamma·p2)`:
///
/// ```text
///   u_x = b0·x + b1·x·r² + b2·x·r⁴ + b3·(2xy)     + b4·(r²+2x²)
///   u_y = b0·y + b1·y·r² + b2·y·r⁴ + b3·(r²+2y²)  + b4·(2xy)
/// ```
///
/// so the whole set costs one least squares. The tangential terms are what a focal-only fit can
/// never see: a decentred lens bends verticals asymmetrically, and forcing that into `k1` leaves a
/// residue that looks like scene curvature.
fn fit_camera_correction(
    res: &BaResult,
    pt_index: &HashMap<usize, usize>,
    norm: &[Vec<(usize, Vec2F64)>],
    poses: &[Option<Pose3d>],
) -> Option<(f64, f64, f64, f64, f64)> {
    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    // DETERMINISM. These are floating-point normal equations, and float addition is not
    // associative, so the ORDER of accumulation changes the solved correction. `pt_index` is a
    // `HashMap` and Rust randomises hash iteration per process, so summing in iteration order
    // yields a different `(gamma, k1, k2, p1, p2)` — hence different intrinsics, hence a different
    // reconstruction — from bit-identical input.
    let mut ordered: Vec<(&usize, &usize)> = pt_index.iter().collect();
    ordered.sort_unstable_by_key(|(ti, _)| **ti);
    for (ti, pidx) in ordered {
        let Some(p) = res.points.get(*pidx) else {
            continue;
        };
        for (c, n) in &norm[*ti] {
            if poses[*c].is_none() {
                continue;
            }
            let pc = res.poses[*c].transform_point(p);
            if pc.z <= 1e-9 {
                continue;
            }
            let (u, v) = (pc.x / pc.z, pc.y / pc.z);
            let (x, y) = (n.x, n.y);
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let rows = [
                ([x, x * r2, x * r4, 2.0 * x * y, r2 + 2.0 * x * x], u),
                ([y, y * r2, y * r4, r2 + 2.0 * y * y, 2.0 * x * y], v),
            ];
            for (basis, target) in rows {
                for i in 0..5 {
                    for j in 0..5 {
                        ata[i][j] += basis[i] * basis[j];
                    }
                    atb[i] += basis[i] * target;
                }
            }
        }
    }
    // Try the full 5-parameter fit first; when it is singular or out of bounds fall back to the
    // (gamma, k1) subproblem rather than applying a fit the geometry cannot support — thin tracks
    // make the r⁴ and tangential columns nearly collinear on a narrow-FOV rig.
    let full = solve_sym5(&ata, &atb).and_then(|b| {
        let gamma = b[0];
        if gamma.abs() < 1e-9 {
            return None;
        }
        let (k1, k2, p1, p2) = (b[1] / gamma, b[2] / gamma, b[3] / gamma, b[4] / gamma);
        ((0.7..1.3).contains(&gamma)
            && (-0.3..0.3).contains(&k1)
            && (-0.1..0.1).contains(&k2)
            && (-0.05..0.05).contains(&p1)
            && (-0.05..0.05).contains(&p2))
        .then_some((gamma, k1, k2, p1, p2))
    });
    full.or_else(|| {
        let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[0][1];
        if det.abs() <= 1e-12 {
            return None;
        }
        let gamma = (atb[0] * ata[1][1] - atb[1] * ata[0][1]) / det;
        let gk1 = (atb[1] * ata[0][0] - atb[0] * ata[0][1]) / det;
        let k1 = if gamma.abs() > 1e-9 { gk1 / gamma } else { 0.0 };
        // Sanity bounds: a fit outside them means the MAP is wrong, not the camera, and applying
        // it would let geometry errors masquerade as optics.
        ((0.7..1.3).contains(&gamma) && (-0.3..0.3).contains(&k1))
            .then_some((gamma, k1, 0.0, 0.0, 0.0))
    })
}

/// Solve the symmetric positive-semidefinite `5×5` system `A x = b` by Gaussian elimination with
/// partial pivoting. `None` when a pivot collapses — for the intrinsics fit that means the
/// distortion columns are collinear and the caller falls back to the two-parameter model.
fn solve_sym5(a: &[[f64; 5]; 5], b: &[f64; 5]) -> Option<[f64; 5]> {
    let mut m = [[0.0f64; 6]; 5];
    for i in 0..5 {
        m[i][..5].copy_from_slice(&a[i]);
        m[i][5] = b[i];
    }
    for col in 0..5 {
        let piv = (col..5).max_by(|&i, &j| {
            m[i][col]
                .abs()
                .partial_cmp(&m[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        for row in (col + 1)..5 {
            let f = m[row][col] / m[col][col];
            for k in col..6 {
                m[row][k] -= f * m[col][k];
            }
        }
    }
    let mut x = [0.0f64; 5];
    for i in (0..5).rev() {
        let mut s = m[i][5];
        for j in (i + 1)..5 {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

/// Triangulate every not-yet-reconstructed track that has ≥2 placed cameras, adding it to `point3d`.
fn triangulate_new(
    point3d: &mut HashMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    poses: &[Option<Pose3d>],
    idcam: &PinholeCamera,
    tcfg: &TriangulationConfig,
) {
    for (ti, obs) in norm.iter().enumerate() {
        if point3d.contains_key(&ti) {
            continue;
        }
        let placed: Vec<(usize, Vec2F64)> = obs
            .iter()
            .copied()
            .filter(|(c, _)| poses[*c].is_some())
            .collect();
        if placed.len() < 2 {
            continue;
        }
        // Widest-baseline pair among placed cameras.
        let centers: Vec<Vec3F64> = placed
            .iter()
            .map(|(c, _)| poses[*c].unwrap().inverse().translation)
            .collect();
        let mut best = (0usize, 1usize, -1.0f64);
        for i in 0..placed.len() {
            for j in (i + 1)..placed.len() {
                let d = (centers[i] - centers[j]).length();
                if d > best.2 {
                    best = (i, j, d);
                }
            }
        }
        let (ca, ua) = placed[best.0];
        let (cb, ub) = placed[best.1];
        if let Ok(pts) = triangulate_matched_points(
            &[ua],
            &[ub],
            &poses[ca].unwrap(),
            &poses[cb].unwrap(),
            idcam,
            tcfg,
        ) {
            if pts.len() == 1 {
                point3d.insert(ti, pts[0].position);
            }
        }
    }
}

/// Metric scale factor `tag_size / reconstructed_side` from the reference tag's triangulated corners.
/// Returns `1.0` (leave up-to-scale) when the tag can't be triangulated or the tag size is unset.
#[allow(clippy::too_many_arguments)]
fn tag_scale(
    tags: &[TagObservation],
    cameras: &[PinholeCamera],
    poses_w2c: &[Pose3d],
    registered: &[bool],
    idcam: &PinholeCamera,
    tcfg: &TriangulationConfig,
    tag_size_m: f64,
) -> Option<f64> {
    if tag_size_m <= 0.0 {
        return None;
    }
    let tag = tags.first()?;
    let seers: Vec<usize> = tag
        .per_camera
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| registered[*c])
        .collect();
    if seers.len() < 2 {
        return None;
    }
    // Widest-baseline placed pair seeing the tag (same pair for all 4 corners).
    let centers: Vec<Vec3F64> = seers
        .iter()
        .map(|c| poses_w2c[*c].inverse().translation)
        .collect();
    let mut best = (0usize, 1usize, -1.0f64);
    for i in 0..seers.len() {
        for j in (i + 1)..seers.len() {
            let d = (centers[i] - centers[j]).length();
            if d > best.2 {
                best = (i, j, d);
            }
        }
    }
    let (ca, cb) = (seers[best.0], seers[best.1]);
    let cca = tag.per_camera.iter().find(|(c, _)| *c == ca).unwrap().1;
    let ccb = tag.per_camera.iter().find(|(c, _)| *c == cb).unwrap().1;
    let mut world: [Option<Vec3F64>; 4] = [None; 4];
    for k in 0..4 {
        let na = cameras[ca].normalize(cca[k]);
        let nb = cameras[cb].normalize(ccb[k]);
        if let Ok(pts) =
            triangulate_matched_points(&[na], &[nb], &poses_w2c[ca], &poses_w2c[cb], idcam, tcfg)
        {
            if pts.len() == 1 {
                world[k] = Some(pts[0].position);
            }
        }
    }
    // Aruco winding (TL,TR,BR,BL): all four edges are one tag side. Average the available edges.
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for (a, b) in edges {
        if let (Some(pa), Some(pb)) = (world[a], world[b]) {
            sum += (pa - pb).length();
            cnt += 1;
        }
    }
    let recon_side = sum / cnt as f64;
    if cnt == 0 || recon_side < 1e-9 {
        return None;
    }
    Some(tag_size_m / recon_side)
}

/// Per-camera reprojection RMS (pixels) for the feature path; analytical covariance fields left `None`.
/// Reprojection is scale-invariant, so the UNSCALED world→cam BA poses are used directly.
#[allow(clippy::too_many_arguments)]
fn feature_stats(
    focal_scale: f64,
    camera: usize,
    poses_w2c: &[Pose3d],
    registered: &[bool],
    points: &[Vec3F64],
    pt_index: &HashMap<usize, usize>,
    norm: &[Vec<(usize, Vec2F64)>],
    cam: &PinholeCamera,
) -> CameraStats {
    if !registered[camera] {
        return CameraStats::unconstrained(camera, false, 0);
    }
    let pose = poses_w2c[camera]; // world→cam
    let (mut se, mut num) = (0.0f64, 0usize);
    for (ti, obs) in norm.iter().enumerate() {
        let Some(&pidx) = pt_index.get(&ti) else {
            continue;
        };
        let Some((_, n)) = obs.iter().find(|(c, _)| *c == camera) else {
            continue;
        };
        if let Some(r) = norm_residual(&pose, points[pidx], *n) {
            // `focal_scale` carries the intrinsics refinement: it rewrote `norm` into the TRUE
            // camera's normalized frame, where fx_true = fx_assumed / gamma, so converting with
            // the ASSUMED fx would report every residual gamma times too large. 1.0 when
            // `refine_intrinsics` is off.
            se += (r * cam.fx * focal_scale).powi(2); // r Euclidean in normalized units; fx≈fy
            num += 1;
        }
    }
    let mut s = CameraStats::unconstrained(camera, true, num);
    if num > 0 {
        s.reproj_rmse_px = (se / num as f64).sqrt();
    }
    s
}

fn global_reproj_rmse(
    focal_scale: f64,
    poses_w2c: &[Pose3d],
    registered: &[bool],
    points: &[Vec3F64],
    pt_index: &HashMap<usize, usize>,
    norm: &[Vec<(usize, Vec2F64)>],
    cameras: &[PinholeCamera],
) -> f64 {
    let (mut se, mut num) = (0.0f64, 0usize);
    for (ti, obs) in norm.iter().enumerate() {
        let Some(&pidx) = pt_index.get(&ti) else {
            continue;
        };
        for (c, n) in obs {
            if !registered[*c] {
                continue;
            }
            if let Some(r) = norm_residual(&poses_w2c[*c], points[pidx], *n) {
                se += (r * cameras[*c].fx * focal_scale).powi(2);
                num += 1;
            }
        }
    }
    if num > 0 {
        (se / num as f64).sqrt()
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RigCalibration;

    fn pinhole(f: f64) -> PinholeCamera {
        PinholeCamera {
            fx: f,
            fy: f,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
    fn rot(yaw: f64, pitch: f64) -> Mat3F64 {
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(sy * sp, cp, cy * sp),
            Vec3F64::new(sy * cp, -sp, cy * cp),
        )
    }
    fn project(pw: Vec3F64, pose_w2c: &Pose3d, k: &PinholeCamera) -> Vec2F64 {
        let pc = pose_w2c.transform_point(&pw);
        Vec2F64::new(k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy)
    }

    #[test]
    fn recovers_metric_extrinsics_features_drive_geometry_tag_scales() {
        // 3 cameras (world→cam) viewing a textured cloud ~2 m away from oblique angles; a 10 cm tag is
        // the ONLY metric reference. Features drive the geometry; the tag fixes scale.
        let cams = [pinhole(500.0), pinhole(500.0), pinhole(500.0)];
        // Cameras CONVERGE on the cloud (~(0,0,1.4)): the left cam yaws right, the right cam yaws left,
        // so all three actually see the shared region (as a real overlapping-FOV rig does).
        let gt = [
            Pose3d::new(rot(0.0, 0.05), Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.40, 0.05), Vec3F64::new(-0.6, 0.0, 0.10)),
            Pose3d::new(rot(-0.40, 0.05), Vec3F64::new(0.6, 0.0, 0.15)),
        ];
        // Only add an observation when the point is IN FRONT and inside the 640x480 image — real
        // feature tracks only exist where a camera actually sees the point.
        let (w, hgt) = (640.0, 480.0);
        let visible = |p: Vec3F64, c: usize| -> Option<Vec2F64> {
            let pc = gt[c].transform_point(&p);
            if pc.z <= 0.1 {
                return None;
            }
            let uv = project(p, &gt[c], &cams[c]);
            (uv.x >= 0.0 && uv.x < w && uv.y >= 0.0 && uv.y < hgt).then_some(uv)
        };
        let mut tracks: Vec<FeatureTrack> = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let x = -0.5 + 0.111 * i as f64;
                let y = -0.5 + 0.111 * j as f64;
                let z = 1.4 + 0.5 * ((i * 5 + j) as f64 * 0.7).sin() + 0.05 * (i as f64 - j as f64);
                let p = Vec3F64::new(x, y, z);
                let obs: Vec<(usize, Vec2F64)> = (0..3)
                    .filter_map(|c| visible(p, c).map(|uv| (c, uv)))
                    .collect();
                if obs.len() >= 2 {
                    tracks.push(FeatureTrack { obs });
                }
            }
        }
        // 10 cm tag (aruco winding TL,TR,BR,BL) centred at (0,0,2).
        let s = 0.10;
        let corners = [
            Vec3F64::new(-s / 2.0, s / 2.0, 1.4),
            Vec3F64::new(s / 2.0, s / 2.0, 1.4),
            Vec3F64::new(s / 2.0, -s / 2.0, 1.4),
            Vec3F64::new(-s / 2.0, -s / 2.0, 1.4),
        ];
        let tag = TagObservation {
            tag_id: 0,
            per_camera: (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(corners[0], &gt[c], &cams[c]),
                            project(corners[1], &gt[c], &cams[c]),
                            project(corners[2], &gt[c], &cams[c]),
                            project(corners[3], &gt[c], &cams[c]),
                        ],
                    )
                })
                .collect(),
        };

        let cfg = CalibConfig::new(s);
        let cal = match reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg, None)
            .map(RigCalibration::from)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERR: {e:?}");
                panic!("{e:?}");
            }
        };
        assert!(
            cal.poses.iter().all(|p| p.is_some()),
            "all cameras registered"
        );

        // Camera-to-camera baselines (gauge-invariant, metric) must match ground truth. Output poses
        // are T_world_cam (camera→world) → translation is the camera centre in world.
        let rc: Vec<Vec3F64> = cal.poses.iter().map(|p| p.unwrap().translation).collect();
        let gc: Vec<Vec3F64> = gt.iter().map(|p| p.inverse().translation).collect();
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let r = (rc[i] - rc[j]).length();
            let g = (gc[i] - gc[j]).length();
            assert!(
                (r - g).abs() < 0.02,
                "baseline {i}-{j}: recovered {r:.4} vs gt {g:.4}"
            );
        }
        assert!(
            cal.reproj_rmse_px >= 0.0 && cal.reproj_rmse_px < 1.0,
            "reproj {}",
            cal.reproj_rmse_px
        );

        // The MAP must be metric too, not just the poses. `reconstruct` runs the same solve, so
        // the points it returns have to share the poses' scale -- this is the assertion the
        // no-tag consistency test cannot make, because there `scale == 1.0` and dropping the
        // scaling changes nothing.
        let recon = reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg, None)
            .expect("same solve must succeed");
        assert_eq!(
            recon.scale,
            ScaleSource::Tag {
                id: 0,
                size_m: 0.10
            }
        );
        assert!(!recon.points.is_empty());

        // Reproject every observation through its own view and point: metric points against
        // metric poses land on the measured pixel, unscaled points against scaled poses do not.
        let mut worst: f64 = 0.0;
        for o in &recon.observations {
            let w2c = recon.views[o.view].expect("registered").inverse();
            worst = worst.max(
                (project(recon.points[o.point].position, &w2c, &cams[o.view]) - o.pixel).length(),
            );
        }
        assert!(
            worst < 2.0,
            "map is not in the same metric frame as the poses: worst reprojection {worst:.3} px"
        );

        // And the cloud spans a physically sensible extent -- the synthetic points live within
        // ~1.2 m of each other, which an unscaled or mis-scaled map would not reproduce.
        let (lo, hi) = recon
            .points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.position.z), hi.max(p.position.z))
            });
        assert!(
            (0.5..4.0).contains(&lo) && (0.5..4.0).contains(&hi),
            "point depths {lo:.2}..{hi:.2} m are not the metric scene"
        );
    }

    /// `reconstruct` must return a map that is CONSISTENT with the poses it returns.
    ///
    /// The pre-M1 API computed all of this and discarded it, so this test could not be written
    /// against it at all -- there was no map to check. Asserting the points are
    /// merely non-empty would be weak, so this reprojects every returned observation through its
    /// own view and its own point and requires it to land on the measured pixel.
    #[test]
    fn reconstruct_returns_a_map_consistent_with_its_poses() {
        // Same synthetic rig as the calibration test above: three views of a point cloud.
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
            Pose3d::new(rot(-0.22, 0.05), Vec3F64::new(0.28, 0.01, 0.03)),
        ];
        let world: Vec<Vec3F64> = (0..24)
            .map(|i| {
                let (a, b) = ((i % 6) as f64, (i / 6) as f64);
                Vec3F64::new(
                    -0.25 + 0.1 * a,
                    -0.15 + 0.1 * b,
                    1.4 + 0.05 * ((i % 5) as f64),
                )
            })
            .collect();

        let tracks: Vec<FeatureTrack> = world
            .iter()
            .map(|pw| FeatureTrack {
                obs: (0..3)
                    .map(|c| (c, project(*pw, &poses_gt[c], &cams[c])))
                    .collect(),
            })
            .collect();

        let cfg = CalibConfig::new(0.1);
        let recon =
            reconstruct(&cams, &[], &tracks, &cfg, None).expect("synthetic scene must solve");

        // The map came back at all -- the whole point of M1.
        assert!(!recon.points.is_empty(), "no points returned");
        assert!(!recon.observations.is_empty(), "no observations returned");
        assert!(
            recon.points.iter().all(|p| p.track_id < tracks.len()),
            "every point must name a real input track"
        );
        // No tag was supplied, so the map is honestly up to scale rather than silently "metric".
        assert_eq!(recon.scale, ScaleSource::UpToScale);

        // Every track id must index the input, and every observation must index the map.
        for o in &recon.observations {
            assert!(o.view < recon.views.len(), "view {} out of range", o.view);
            assert!(
                o.point < recon.points.len(),
                "point {} out of range",
                o.point
            );
            assert!(
                recon.views[o.view].is_some(),
                "an observation names view {} which was never registered",
                o.view
            );
        }

        // The real assertion: reprojecting each returned point through its own returned pose must
        // reproduce the measured pixel. This fails if points, poses and observations are not
        // mutually consistent -- e.g. if the points were left in the pre-scaling frame, or if
        // point_track_id were misaligned with points.
        let mut worst: f64 = 0.0;
        for o in &recon.observations {
            let t_world_cam = recon.views[o.view].expect("checked above");
            let w2c = t_world_cam.inverse();
            let uv = project(recon.points[o.point].position, &w2c, &cams[o.view]);
            worst = worst.max((uv - o.pixel).length());
        }
        assert!(
            worst < 1.0,
            "returned map is not consistent with its poses: worst reprojection {worst:.3} px"
        );
    }

    /// A tag that could not anchor the scale must NOT be reported as the scale source.
    ///
    /// `tag_scale` falls back to the identity in ordinary situations -- an unset `tag_size_m`,
    /// a tag seen by fewer than two REGISTERED views, corners that fail to triangulate. Deciding
    /// `ScaleSource` from `tags_for_scale.first()` reported `Tag { .. }` in all of them, so the
    /// caller was told the map is metric while the points sat at the identity scale. That is the
    /// exact ambiguity `ScaleSource` exists to remove, so it is worth a test: the old
    /// `reference_tag_id` carried the same lie and nothing caught it.
    #[test]
    fn a_tag_that_cannot_anchor_is_reported_as_up_to_scale() {
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
            Pose3d::new(rot(-0.22, 0.05), Vec3F64::new(0.28, 0.01, 0.03)),
        ];
        let world: Vec<Vec3F64> = (0..24)
            .map(|i| {
                let (a, b) = ((i % 6) as f64, (i / 6) as f64);
                Vec3F64::new(
                    -0.25 + 0.1 * a,
                    -0.15 + 0.1 * b,
                    1.4 + 0.05 * ((i % 5) as f64),
                )
            })
            .collect();
        let tracks: Vec<FeatureTrack> = world
            .iter()
            .map(|pw| FeatureTrack {
                obs: (0..3)
                    .map(|c| (c, project(*pw, &poses_gt[c], &cams[c])))
                    .collect(),
            })
            .collect();

        // A tag seen by ONE view only: `tag_scale` needs two registered seers, so it cannot
        // anchor. The tag is supplied, so the old rule would have claimed `Tag { id: 3, .. }`.
        let s = 0.05;
        let corners = [
            Vec3F64::new(-s, s, 1.5),
            Vec3F64::new(s, s, 1.5),
            Vec3F64::new(s, -s, 1.5),
            Vec3F64::new(-s, -s, 1.5),
        ];
        let tag = TagObservation {
            tag_id: 3,
            per_camera: vec![(
                0,
                [
                    project(corners[0], &poses_gt[0], &cams[0]),
                    project(corners[1], &poses_gt[0], &cams[0]),
                    project(corners[2], &poses_gt[0], &cams[0]),
                    project(corners[3], &poses_gt[0], &cams[0]),
                ],
            )],
        };

        let cfg = CalibConfig::new(2.0 * s);
        let recon = reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg, None)
            .expect("the feature tracks alone must still solve");

        assert_eq!(
            recon.scale,
            ScaleSource::UpToScale,
            "a tag seen by one view cannot anchor scale, so the map must not claim to be metric"
        );

        // `tag_scale` has four fallback paths, and reporting `Tag { .. }` for ANY of them is the
        // bug. The case above is "seen by fewer than two registered views"; cover the other three.

        // (a) Size unset: nothing to convert reconstruction units into.
        let mut two_view = tag.clone();
        two_view.per_camera.push((
            1,
            [
                project(corners[0], &poses_gt[1], &cams[1]),
                project(corners[1], &poses_gt[1], &cams[1]),
                project(corners[2], &poses_gt[1], &cams[1]),
                project(corners[3], &poses_gt[1], &cams[1]),
            ],
        ));
        let unset = reconstruct(
            &cams,
            std::slice::from_ref(&two_view),
            &tracks,
            &CalibConfig::new(0.0),
            None,
        )
        .expect("solves");
        assert_eq!(
            unset.scale,
            ScaleSource::UpToScale,
            "tag_size_m = 0 cannot anchor scale"
        );

        // (b) Degenerate corners: all four coincide, so the reconstructed side is ~0 and the
        // implied scale would be a division by nothing.
        let degenerate = TagObservation {
            tag_id: 3,
            per_camera: (0..2)
                .map(|c| {
                    let uv = project(corners[0], &poses_gt[c], &cams[c]);
                    (c, [uv, uv, uv, uv])
                })
                .collect(),
        };
        let deg = reconstruct(
            &cams,
            std::slice::from_ref(&degenerate),
            &tracks,
            &cfg,
            None,
        )
        .expect("solves");
        assert_eq!(
            deg.scale,
            ScaleSource::UpToScale,
            "a tag whose corners coincide has no measurable side, so it cannot anchor scale"
        );

        // (c) Corners that cannot triangulate: put them behind the cameras, so cheirality rejects
        // every one and no side is reconstructed at all.
        let behind: [Vec3F64; 4] = [
            Vec3F64::new(-s, s, -1.0),
            Vec3F64::new(s, s, -1.0),
            Vec3F64::new(s, -s, -1.0),
            Vec3F64::new(-s, -s, -1.0),
        ];
        let untriangulable = TagObservation {
            tag_id: 3,
            per_camera: (0..2)
                .map(|c| {
                    (
                        c,
                        [
                            project(behind[0], &poses_gt[c], &cams[c]),
                            project(behind[1], &poses_gt[c], &cams[c]),
                            project(behind[2], &poses_gt[c], &cams[c]),
                            project(behind[3], &poses_gt[c], &cams[c]),
                        ],
                    )
                })
                .collect(),
        };
        let untri = reconstruct(
            &cams,
            std::slice::from_ref(&untriangulable),
            &tracks,
            &cfg,
            None,
        )
        .expect("solves");
        assert_eq!(
            untri.scale,
            ScaleSource::UpToScale,
            "corners that fail to triangulate cannot anchor scale"
        );
    }

    /// `RigCalibration::from` inherits the scale honesty, and that CHANGES its behaviour.
    ///
    /// Before this, `reference_tag_id` was `tags_for_scale.first()`'s id whenever a tag was
    /// supplied — even when `tag_scale` had fallen back to the identity, so a caller could read a
    /// real tag id off a map that was still up to scale. It is now `0` unless the tag actually
    /// anchored.
    ///
    /// That is a deliberate fix, not a regression, but it is a behaviour change on the EXISTING
    /// entry point rather than only on the new one, so it is asserted here at the
    /// `RigCalibration` level and not merely via `ScaleSource`.
    #[test]
    fn rig_calibration_reports_no_reference_tag_when_the_tag_did_not_anchor() {
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
            Pose3d::new(rot(-0.22, 0.05), Vec3F64::new(0.28, 0.01, 0.03)),
        ];
        let tracks: Vec<FeatureTrack> = (0..24)
            .map(|i| {
                let (a, b) = ((i % 6) as f64, (i / 6) as f64);
                let p = Vec3F64::new(
                    -0.25 + 0.1 * a,
                    -0.15 + 0.1 * b,
                    1.4 + 0.05 * ((i % 5) as f64),
                );
                FeatureTrack {
                    obs: (0..3)
                        .map(|c| (c, project(p, &poses_gt[c], &cams[c])))
                        .collect(),
                }
            })
            .collect();

        // Tag id 9, seen by ONE view: `tag_scale` needs two registered seers, so it cannot anchor.
        let s = 0.05;
        let corners = [
            Vec3F64::new(-s, s, 1.5),
            Vec3F64::new(s, s, 1.5),
            Vec3F64::new(s, -s, 1.5),
            Vec3F64::new(-s, -s, 1.5),
        ];
        let tag = TagObservation {
            tag_id: 9,
            per_camera: vec![(
                0,
                [
                    project(corners[0], &poses_gt[0], &cams[0]),
                    project(corners[1], &poses_gt[0], &cams[0]),
                    project(corners[2], &poses_gt[0], &cams[0]),
                    project(corners[3], &poses_gt[0], &cams[0]),
                ],
            )],
        };

        let cal = reconstruct(
            &cams,
            std::slice::from_ref(&tag),
            &tracks,
            &CalibConfig::new(2.0 * s),
            None,
        )
        .map(RigCalibration::from)
        .expect("the feature tracks alone must still solve");

        assert_eq!(
            cal.reference_tag_id, 0,
            "tag 9 was supplied but could not anchor scale, so it must not be reported as the \
             reference — that would tell the caller the map is metric when it is not"
        );
    }

    /// The published map order must not depend on `HashMap` iteration order.
    #[test]
    fn map_order_is_deterministic_and_sorted_by_track() {
        let cams = vec![pinhole(600.0), pinhole(600.0)];
        let poses_gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.25, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
        ];
        let tracks: Vec<FeatureTrack> = (0..30)
            .map(|i| {
                let p = Vec3F64::new(
                    -0.3 + 0.02 * i as f64,
                    -0.1 + 0.01 * (i % 7) as f64,
                    1.3 + 0.03 * (i % 5) as f64,
                );
                FeatureTrack {
                    obs: (0..2)
                        .map(|c| (c, project(p, &poses_gt[c], &cams[c])))
                        .collect(),
                }
            })
            .collect();
        let cfg = CalibConfig::new(0.1);

        let a = reconstruct(&cams, &[], &tracks, &cfg, None).expect("solves");
        let b = reconstruct(&cams, &[], &tracks, &cfg, None).expect("solves");

        let ids = |r: &Reconstruction| r.points.iter().map(|p| p.track_id).collect::<Vec<_>>();
        let (ia, ib) = (ids(&a), ids(&b));
        assert!(!ia.is_empty(), "need points to order");
        assert!(
            ia.windows(2).all(|w| w[0] < w[1]),
            "points must be published in ascending track order, got {:?}",
            &ia[..ia.len().min(8)]
        );
        assert_eq!(
            ia, ib,
            "two runs over identical input must publish the same map order"
        );
    }

    /// Three converging views of a textured cloud, at a SPECIFIED metric scale.
    ///
    /// The shape is scale-free, so `scale` changes nothing about how well the geometry
    /// reconstructs — only where the true metric sits relative to the bootstrap's unit baseline.
    /// Returns `(cameras, world→cam ground truth, tracks)`.
    fn converging_rig(f: f64, scale: f64) -> (Vec<PinholeCamera>, Vec<Pose3d>, Vec<FeatureTrack>) {
        let cams = vec![pinhole(f), pinhole(f), pinhole(f)];
        let gt = vec![
            Pose3d::new(rot(0.0, 0.05), Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.40, 0.05), Vec3F64::new(-0.6, 0.0, 0.10) * scale),
            Pose3d::new(rot(-0.40, 0.05), Vec3F64::new(0.6, 0.0, 0.15) * scale),
        ];
        let (w, h) = (640.0, 480.0);
        let mut tracks: Vec<FeatureTrack> = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let x = -0.5 + 0.111 * i as f64;
                let y = -0.5 + 0.111 * j as f64;
                let z = 1.4 + 0.5 * ((i * 5 + j) as f64 * 0.7).sin() + 0.05 * (i as f64 - j as f64);
                let p = Vec3F64::new(x, y, z) * scale;
                let obs: Vec<(usize, Vec2F64)> = (0..3)
                    .filter_map(|c| {
                        let pc = gt[c].transform_point(&p);
                        if pc.z <= 0.1 * scale {
                            return None;
                        }
                        let uv = project(p, &gt[c], &cams[c]);
                        (uv.x >= 0.0 && uv.x < w && uv.y >= 0.0 && uv.y < h).then_some((c, uv))
                    })
                    .collect();
                if obs.len() >= 2 {
                    tracks.push(FeatureTrack { obs });
                }
            }
        }
        (cams, gt, tracks)
    }

    /// Ground-truth metric depth for every observation, shaped exactly like `tracks`.
    fn gt_depths(
        gt: &[Pose3d],
        tracks: &[FeatureTrack],
        cams: &[PinholeCamera],
    ) -> Vec<Vec<Option<f64>>> {
        // Recover each track's world point by intersecting its first two rays under the GT poses,
        // which for noise-free synthetic data is exact; then take its z in each observing view.
        tracks
            .iter()
            .map(|t| {
                let (c0, uv0) = t.obs[0];
                let (c1, uv1) = t.obs[1];
                let pts = triangulate_matched_points(
                    &[cams[c0].normalize(uv0)],
                    &[cams[c1].normalize(uv1)],
                    &gt[c0],
                    &gt[c1],
                    &PinholeCamera::IDENTITY,
                    &TriangulationConfig {
                        min_parallax_deg: 0.0,
                        max_reprojection_error: 1e9,
                        min_cheirality_count: 0,
                        ..Default::default()
                    },
                )
                .expect("synthetic rays must intersect");
                let p = pts[0].position;
                t.obs
                    .iter()
                    .map(|(c, _)| Some(gt[*c].transform_point(&p).z))
                    .collect()
            })
            .collect()
    }

    /// Median of `z_reconstructed / d_measured` over every surviving observation: `1.0` exactly
    /// when the map is at the metric scale the depths assert.
    fn median_depth_ratio(
        recon: &Reconstruction,
        tracks: &[FeatureTrack],
        depths: &[Vec<Option<f64>>],
    ) -> f64 {
        let mut r: Vec<f64> = Vec::new();
        for o in &recon.observations {
            let ti = recon.points[o.point].track_id;
            let Some(j) = tracks[ti].obs.iter().position(|(c, _)| *c == o.view) else {
                continue;
            };
            let Some(d) = depths[ti][j] else { continue };
            let w2c = recon.views[o.view].expect("registered").inverse();
            let z = w2c.transform_point(&recon.points[o.point].position).z;
            if d > 1e-9 {
                r.push(z / d);
            }
        }
        assert!(!r.is_empty(), "no observation carried a depth to compare");
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r[r.len() / 2]
    }

    /// **Depth measurements fix the scale of an otherwise scale-free reconstruction.**
    ///
    /// This is the whole point of the `obs_depth` argument. A feature-only monocular solve is
    /// determined only up to a similarity: the bootstrap fixes the gauge by giving its seed pair a
    /// UNIT baseline, so the map's absolute size is an artifact of that convention and has nothing
    /// to do with the scene. Here the true seed baseline is deliberately far from 1, so the
    /// depth-free control lands several times too large — and the depth priors have to pull it back.
    ///
    /// The control is not decoration. Every assertion below would also pass on a build where the
    /// depth arguments were accepted and silently ignored, if the scene happened to be near unit
    /// scale; asserting that the SAME scene reconstructs at the wrong scale without depth is what
    /// makes this a test of the depth term rather than of the geometry.
    #[test]
    fn depth_priors_fix_the_scale_of_an_otherwise_scale_free_reconstruction() {
        // Scale 0.4: the widest camera-centre baseline is ~0.5 m, so the unit-baseline convention
        // inflates the depth-free map by roughly 2x.
        let (cams, gt, tracks) = converging_rig(500.0, 0.4);
        let depths = gt_depths(&gt, &tracks, &cams);

        // Control: no depth. Same tracks, same config -- only the argument differs.
        let free = reconstruct(&cams, &[], &tracks, &CalibConfig::new(0.0), None).expect("solves");
        let free_ratio = median_depth_ratio(&free, &tracks, &depths);
        assert!(
            free_ratio > 1.5,
            "the depth-free control must be visibly off-scale for this test to mean anything, \
             got ratio {free_ratio:.4}"
        );

        // With depth. `depth_prior_rel_sigma` is what ARMS the feature: at 0.0 the depths are
        // ignored no matter what is passed.
        let cfg = CalibConfig {
            depth_prior_rel_sigma: 0.15,
            ..CalibConfig::new(0.0)
        };
        let metric = reconstruct(&cams, &[], &tracks, &cfg, Some(&depths)).expect("solves");
        let ratio = median_depth_ratio(&metric, &tracks, &depths);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "depth priors must bring the map to the metric scale they assert: ratio {ratio:.4} \
             (depth-free control {free_ratio:.4})"
        );

        // Metric SCALE, not merely metric depths: the camera centres must now sit at their true
        // separations, which is the quantity a downstream consumer actually uses.
        let centre = |p: &Option<Pose3d>| p.expect("registered").translation;
        let gt_centre = |p: &Pose3d| p.inverse().translation;
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let got = (centre(&metric.views[i]) - centre(&metric.views[j])).length();
            let want = (gt_centre(&gt[i]) - gt_centre(&gt[j])).length();
            assert!(
                (got - want).abs() < 0.05 * want,
                "baseline {i}-{j} is {got:.4} m, ground truth {want:.4} m"
            );
        }

        // And the map is still HONEST about where that scale came from: no tag anchored it, so the
        // reported `ScaleSource` stays `UpToScale` even though the numbers are now metric. Depth is
        // a soft prior, not a fiducial, and `ScaleSource` names fiducials.
        assert_eq!(metric.scale, ScaleSource::UpToScale);
    }

    /// Passing depths while `depth_prior_rel_sigma` is 0 must change NOTHING.
    ///
    /// The knob, not the argument, arms the feature -- so a caller that wires depth through before
    /// deciding to trust it gets exactly the solve it had. Bit-identical, not merely close.
    #[test]
    fn depths_are_inert_until_the_sigma_arms_them() {
        let (cams, gt, tracks) = converging_rig(500.0, 0.4);
        let depths = gt_depths(&gt, &tracks, &cams);
        let cfg = CalibConfig::new(0.0);

        let without = reconstruct(&cams, &[], &tracks, &cfg, None).expect("solves");
        let with = reconstruct(&cams, &[], &tracks, &cfg, Some(&depths)).expect("solves");

        assert_eq!(without.points.len(), with.points.len());
        for (a, b) in without.points.iter().zip(&with.points) {
            assert_eq!(a.position, b.position, "depth leaked into a disarmed solve");
        }
    }

    /// `min_registration_inliers` decides whether a thinly-supported view is admitted.
    ///
    /// Views 0 and 1 share 80 tracks and bootstrap the map; view 2 sees only 20 of them, so its
    /// PnP consensus cannot exceed 20 however good the pose is. The default gate of 30 must refuse
    /// it; lowering the gate must admit it. Without the gate there is nothing between a 4-point
    /// consensus and a registration, and a bad registration is not self-correcting -- points get
    /// triangulated FROM it and then feed the next view's PnP.
    #[test]
    fn min_registration_inliers_gates_a_thinly_supported_view() {
        let cams = vec![pinhole(600.0), pinhole(600.0), pinhole(600.0)];
        let gt = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
            Pose3d::new(rot(0.30, 0.0), Vec3F64::new(-0.35, 0.0, 0.02)),
            Pose3d::new(rot(-0.28, 0.04), Vec3F64::new(0.33, 0.01, 0.03)),
        ];
        let tracks: Vec<FeatureTrack> = (0..100)
            .map(|i| {
                let p = Vec3F64::new(
                    -0.30 + 0.006 * i as f64,
                    -0.20 + 0.004 * (i % 11) as f64,
                    1.30 + 0.05 * (i % 7) as f64,
                );
                // Only the last 20 tracks are visible to view 2.
                let seen: &[usize] = if i < 80 { &[0, 1] } else { &[0, 1, 2] };
                FeatureTrack {
                    obs: seen
                        .iter()
                        .map(|c| (*c, project(p, &gt[*c], &cams[*c])))
                        .collect(),
                }
            })
            .collect();

        let gated = reconstruct(&cams, &[], &tracks, &CalibConfig::new(0.0), None).expect("solves");
        assert_eq!(
            CalibConfig::new(0.0).min_registration_inliers,
            30,
            "this test is calibrated against the documented default"
        );
        assert!(
            gated.views[2].is_none(),
            "20 correspondences cannot clear a 30-inlier gate, so view 2 must stay unregistered"
        );
        assert!(gated.views[0].is_some() && gated.views[1].is_some());

        let open = reconstruct(
            &cams,
            &[],
            &tracks,
            &CalibConfig {
                min_registration_inliers: 15,
                ..CalibConfig::new(0.0)
            },
            None,
        )
        .expect("solves");
        assert!(
            open.views[2].is_some(),
            "lowering the gate below the available support must admit the view"
        );
    }

    /// `refine_intrinsics` must REPORT its fit and then ACT on it.
    ///
    /// Two independent claims, with different failure modes:
    ///
    /// 1. The fit reaches the caller on [`Reconstruction::camera_correction`]. Refinement applies
    ///    itself by rewriting normalized observations in place, so a caller that never sees the fit
    ///    keeps storing the camera it guessed — one that did not produce these poses.
    /// 2. The fit CHANGES THE POSES. Refinement runs after a bundle adjustment; unless a second
    ///    solve follows it, the corrected observations are never read by anything that moves
    ///    geometry, and the feature is an expensive no-op that still reports a number.
    ///
    /// # What this test does NOT claim, and why
    ///
    /// It does not claim the refinement recovers a known FOCAL error, because on a synthetic of
    /// this size it cannot and no honest assertion would pass. The fit regresses the map's
    /// predicted projections onto the observations, so it can only see error that bundle
    /// adjustment failed to absorb — and a focal error is very nearly absorbable, since scaling
    /// normalized coordinates stretches space in `x` and `y` at fixed `z`. MEASURED here: handing
    /// the solver 320 for a true focal of 400 (a 25% error) left `plain.reproj_rmse_px = 0.636`
    /// and `gamma = 0.9994` on a three-view converging rig, and 0.9994 again on a five-view rig
    /// orbiting through ±52° of yaw. Bundle adjustment had swallowed all of it. The production
    /// evidence for focal recovery is a 3208-frame handheld clip with hundreds of cameras, which
    /// is not a unit test.
    ///
    /// Radial DISTORTION is a different matter: it is a nonlinear image warp, not a projective
    /// one, so a rigid reconstruction cannot absorb it and it leaves the radially-systematic
    /// residual this fit is shaped to see. So the scene injects barrel distortion the solver is
    /// not told about, and the assertion is that the fit opposes it.
    #[test]
    fn refine_intrinsics_reports_its_fit_and_re_solves_against_it() {
        const K1_TRUE: f64 = 0.25;
        let (cams, _, clean) = converging_rig(500.0, 1.0);
        // Barrel-distort every pixel. The cameras handed to the solver keep `k1 = 0`, so
        // `normalize` leaves the distortion in the observations for the refinement to find.
        let tracks: Vec<FeatureTrack> = clean
            .iter()
            .map(|t| FeatureTrack {
                obs: t
                    .obs
                    .iter()
                    .map(|(c, uv)| {
                        let (x, y) = ((uv.x - 320.0) / 500.0, (uv.y - 240.0) / 500.0);
                        let s = 1.0 + K1_TRUE * (x * x + y * y);
                        (
                            *c,
                            Vec2F64::new(320.0 + 500.0 * x * s, 240.0 + 500.0 * y * s),
                        )
                    })
                    .collect(),
            })
            .collect();

        let plain = reconstruct(&cams, &[], &tracks, &CalibConfig::new(0.0), None).expect("solves");
        assert_eq!(
            plain.camera_correction, None,
            "refinement is off by default, so there is no fit to report"
        );

        let refined = reconstruct(
            &cams,
            &[],
            &tracks,
            &CalibConfig {
                refine_intrinsics: true,
                ..CalibConfig::new(0.0)
            },
            None,
        )
        .expect("solves");

        let (gamma, k1, ..) = refined
            .camera_correction
            .expect("injected distortion must produce a reportable fit");
        assert!(
            k1 < -0.005,
            "the fit must UNDO the injected barrel distortion (k1_true = +{K1_TRUE}), so its own \
             k1 has to be negative; got {k1:.5}"
        );
        // The focal was correct here, so the refinement must not invent a focal error while it is
        // busy fitting distortion. `gamma` inverts into focal: fx_true = fx_assumed / gamma.
        assert!(
            (0.95..1.05).contains(&gamma),
            "a correctly-assumed focal must survive the fit intact; gamma={gamma:.4} implies \
             fx_true = {:.1} against the true 500",
            500.0 / gamma
        );

        // The second solve is the load-bearing half. Without it the poses would be BIT-IDENTICAL
        // to the unrefined run, because nothing after the refinement re-optimises anything: the
        // rewritten observations would be reported on and then dropped on the floor.
        let moved = plain
            .views
            .iter()
            .zip(&refined.views)
            .filter_map(|(a, b)| Some((a.as_ref()?.translation - b.as_ref()?.translation).length()))
            .fold(0.0f64, f64::max);
        assert!(
            moved > 1e-6,
            "the refinement was reported but never re-solved against: poses are unchanged"
        );
        assert!(
            refined.views.iter().all(|v| v.is_some()),
            "the re-solve must not cost a registration"
        );
    }

    /// The up and motion priors reach the solver at a strength that does not overrule the images.
    ///
    /// Both sigmas are quoted in their own physical units and then DEFLATED into the normalized
    /// reprojection units bundle adjustment actually works in. Skipping that deflation is not a
    /// subtle mis-tuning: it makes each prior row `1/sigma_r` times stiffer than every reprojection
    /// row in the same solve (360x at a phone focal), and the solve then fits the assumption
    /// instead of the scene. So the assertion is that turning both priors ON, at their documented
    /// production values, leaves a good reconstruction good.
    #[test]
    fn up_and_motion_priors_do_not_overrule_the_image_evidence() {
        // Views are index-ordered along the rig, which is what makes a motion prior meaningful.
        let (cams, gt, tracks) = converging_rig(500.0, 1.0);
        let baseline =
            reconstruct(&cams, &[], &tracks, &CalibConfig::new(0.0), None).expect("solves");
        let with_priors = reconstruct(
            &cams,
            &[],
            &tracks,
            &CalibConfig {
                up_prior_sigma: 0.25,
                motion_prior_sigma: 0.1,
                ..CalibConfig::new(0.0)
            },
            None,
        )
        .expect("solves");

        assert!(
            with_priors.views.iter().all(|v| v.is_some()),
            "the priors must not cost a registration"
        );
        // Baselines are gauge-invariant up to the one global scale the bootstrap fixed, so compare
        // their RATIOS against ground truth.
        let c = |r: &Reconstruction, i: usize| r.views[i].expect("registered").translation;
        let g = |i: usize| gt[i].inverse().translation;
        let want = (g(0) - g(2)).length() / (g(0) - g(1)).length();
        for (name, r) in [("baseline", &baseline), ("with priors", &with_priors)] {
            let got = (c(r, 0) - c(r, 2)).length() / (c(r, 0) - c(r, 1)).length();
            assert!(
                (got - want).abs() < 0.05 * want,
                "{name}: baseline ratio {got:.4} vs ground truth {want:.4}"
            );
        }
        assert!(
            with_priors.reproj_rmse_px < 1.0,
            "priors drove the fit away from the pixels: rmse {:.3} px",
            with_priors.reproj_rmse_px
        );
    }

    /// The reported pixel RMS must be in the corrected camera's pixels, not the assumed one's.
    ///
    /// `refine_intrinsics` rewrites the observations into the TRUE camera's normalized frame,
    /// where `fx_true = fx_assumed / gamma`. Converting those residuals to pixels with the
    /// ASSUMED `fx` — which is what `cameras` still holds — scales every reported number by
    /// gamma. It is a small factor, which is exactly why it is worth pinning: `reproj_rmse_px` is
    /// the number callers quote as their acceptance metric, and a silent few-percent bias in an
    /// acceptance metric is worse than a loud one.
    #[test]
    fn reported_rmse_is_in_corrected_pixels() {
        const K1_TRUE: f64 = 0.25;
        let (cams, _, clean) = converging_rig(500.0, 1.0);
        let tracks: Vec<FeatureTrack> = clean
            .iter()
            .map(|t| FeatureTrack {
                obs: t
                    .obs
                    .iter()
                    .map(|(c, uv)| {
                        let (x, y) = ((uv.x - 320.0) / 500.0, (uv.y - 240.0) / 500.0);
                        let s = 1.0 + K1_TRUE * (x * x + y * y);
                        (
                            *c,
                            Vec2F64::new(320.0 + 500.0 * x * s, 240.0 + 500.0 * y * s),
                        )
                    })
                    .collect(),
            })
            .collect();

        let refined = reconstruct(
            &cams,
            &[],
            &tracks,
            &CalibConfig {
                refine_intrinsics: true,
                ..CalibConfig::new(0.0)
            },
            None,
        )
        .expect("solves");

        let (gamma, ..) = refined
            .camera_correction
            .expect("injected distortion must produce a fit");
        assert!(
            (gamma - 1.0).abs() > 1e-9,
            "test is vacuous unless the fit actually moved gamma (got {gamma})"
        );

        // Reproduce the conversion by hand from the same reported per-camera numbers, and check
        // the global figure is consistent with them under the CORRECTED focal. If the global RMS
        // had been left in assumed-fx pixels while the per-camera ones were corrected (or vice
        // versa), this ratio would sit at gamma rather than 1.
        let n: usize = refined.per_view.iter().map(|s| s.num_obs).sum();
        assert!(n > 0, "no observations to check");
        let pooled: f64 = refined
            .per_view
            .iter()
            .filter(|s| s.num_obs > 0)
            .map(|s| s.reproj_rmse_px.powi(2) * s.num_obs as f64)
            .sum::<f64>()
            / n as f64;
        let pooled = pooled.sqrt();
        assert!(
            (pooled - refined.reproj_rmse_px).abs() <= 1e-6 * refined.reproj_rmse_px.max(1.0),
            "global RMS {:.6} disagrees with the pooled per-camera RMS {pooled:.6} — the two \
             conversions are not using the same focal",
            refined.reproj_rmse_px
        );
    }
}
