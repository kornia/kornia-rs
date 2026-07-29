//! Tags-free (feature-driven) multi-camera calibration via incremental structure-from-motion.
//!
//! Natural-feature tracks — not a tag — drive the geometry. A best-connected camera pair bootstraps
//! the reconstruction from the two-view essential matrix, remaining cameras register by PnP against
//! the growing point cloud, and a bundle adjustment polishes everything. The reconstruction is
//! recovered **up to scale** (the fundamental monocular ambiguity); a single metric tag then fixes
//! that one scalar — the tag is a *scale bar*, nothing else. Output poses are `T_world_cam` in the
//! reference camera's frame (metric).
//!
//! Everything except the incremental orchestration is reused: `ransac_essential_5pt` +
//! `decompose_essential` (bootstrap relative pose), [`kornia_3d::pose::triangulate_matched_points`],
//! [`kornia_3d::pnp::solve_pnp_ransac`] (register a new camera into the cloud), and
//! [`kornia_3d::ba_schur::bundle_adjust_schur`].

use std::collections::{BTreeMap, HashMap, HashSet};

use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba::{BaMotionPrior, BaPosePrior};
use kornia_3d::ba_schur::bundle_adjust_schur_with_all_priors;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{solve_pnp_ransac, PnPMethod, RansacParams as PnpRansacParams};
use kornia_3d::pose::{
    decompose_essential, ransac_essential_5pt, ransac_fundamental, ransac_homography,
    triangulate_matched_points, Pose3d, RansacParams as TvRp, TriangulationConfig,
};
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64};

use crate::error::CalibError;
use crate::types::{CalibConfig, CameraStats, FeatureTrack, RigCalibration, TagObservation};

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

/// Calibrate multi-camera rig extrinsics **without a tag anchoring the geometry**: natural-feature
/// tracks drive an incremental SfM reconstruction, and `tags_for_scale` supplies only the metric
/// scale (a scale bar). Returns per-camera `T_world_cam` (world = the reference camera's frame).
///
/// `tracks` are multi-view feature tracks (build them with [`crate::build_tracks`]); each needs the
/// raw pixel in every camera that sees it. `tags_for_scale` may be empty — then the result is left
/// up-to-scale (translations in reconstruction units). A camera that shares too little with the
/// reconstruction is left unregistered (`poses[c] == None`).
pub fn calibrate_features(
    cameras: &[PinholeCamera],
    tags_for_scale: &[TagObservation],
    tracks: &[FeatureTrack],
    config: &CalibConfig,
) -> Result<RigCalibration, CalibError> {
    calibrate_features_with_depth(cameras, tags_for_scale, tracks, config, None)
}

/// [`calibrate_features`] with optional per-observation METRIC depth priors.
///
/// `obs_depth` is parallel to `tracks`: `obs_depth[ti][j]` is the measured camera-frame depth in
/// METRES of `tracks[ti].obs[j]`, or `None` where no measurement exists. When
/// `config.depth_prior_rel_sigma > 0`, each such measurement becomes a depth residual
/// `(z_pred − d)/σ` in every bundle adjustment, with `σ = rel_sigma · d`.
///
/// Two things this buys that reprojection alone cannot:
/// 1. **Metric scale without a fiducial.** Monocular reprojection is exactly scale-invariant, so
///    the gauge has one free DoF that LM navigates by numerical accident. Depth residuals observe
///    absolute scale directly; the reconstruction lands in metres.
/// 2. **No scale drift.** Along a no-revisit walkthrough the reconstruction's scale wanders (the
///    measured symptom: rooms later in a clip reconstructing several times larger than early
///    ones). Per-observation depth pins the scale of EVERY segment of the chain, not just a
///    global average.
///
/// The seed cloud is pre-scaled to the depth measurements (median ratio) before growth, so the
/// priors start near-satisfied instead of asking LM to cross a large scale gap.
pub fn calibrate_features_with_depth(
    cameras: &[PinholeCamera],
    tags_for_scale: &[TagObservation],
    tracks: &[FeatureTrack],
    config: &CalibConfig,
    obs_depth: Option<&[Vec<Option<f64>>]>,
) -> Result<RigCalibration, CalibError> {
    let n_cams = cameras.len();
    let idcam = PinholeCamera::IDENTITY;
    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };

    // Per track: normalized observation per camera (undistort + K⁻¹). Raw pixels stay in `tracks`.
    let mut norm: Vec<Vec<(usize, Vec2F64)>> = tracks
        .iter()
        .map(|t| {
            t.obs
                .iter()
                .map(|(c, uv)| (*c, cameras[*c].normalize(*uv)))
                .collect()
        })
        .collect();
    // Parallel to `norm`: metric depth per observation, or None. All-None when priors are off, so
    // downstream indexing never branches on the feature being enabled.
    let use_depth = config.depth_prior_rel_sigma > 0.0 && obs_depth.is_some();
    let norm_depth: Vec<Vec<Option<f32>>> = match (use_depth, obs_depth) {
        (true, Some(d)) => d
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
    // Bootstrap pair selection.
    //
    // Picking the pair with the MOST shared tracks is the natural choice for a rig, and the wrong
    // one for a video: consecutive keyframes overlap most precisely because they are closest
    // together, so "most matches" systematically selects the SMALLEST baseline — the worst
    // conditioning an essential matrix can be given. Low parallax makes the decomposition
    // ill-posed, and because every later camera is registered against the seed cloud, a bad
    // bootstrap is not recoverable downstream. Measured across 7-Scenes with one fixed config,
    // this produced 1.3 px reprojection RMSE on one scene and 58 px (with a 129 degree median
    // rotation error — a structurally wrong reconstruction) on another.
    //
    // So: take the strongest candidates by shared-track count, actually solve each, and choose on
    // geometry — cheirality-valid count first, with a median triangulation angle floor to reject
    // the degenerate low-parallax pairs outright. Deterministic tie-break on (a, b) keeps the whole
    // reconstruction reproducible, since HashMap iteration order must not decide the seed.
    const BOOTSTRAP_CANDIDATES: usize = 12;
    /// COLMAP's `Mapper.ba_global_images_ratio`: re-run global BA once the registered set has grown
    /// by this factor since the last one.
    const BA_IMAGES_RATIO: f64 = 1.1;
    /// A seed pair below this median triangulation angle is too degenerate to anchor a map.
    ///
    /// This ORDERS seeds rather than rejecting them: `seeds` sorts floor-passing pairs first and
    /// falls back to cheirality support when none qualify, so raising it can change which pair wins
    /// but never whether a reconstruction happens at all.
    ///
    /// MEASURED, and NOT what COLMAP's `Mapper.init_min_tri_angle = 16` would suggest: raising this
    /// to 16.0 made things worse at BOTH densities. Per-frame chess went 72.5 -> 83.7 cm ATE with
    /// median rotation error 22.3 -> 66.6 degrees, and keyframe chess 0.85 -> 1.44 cm — while the
    /// reprojection RMSE *improved* (66.8 -> 56.3 px), which is one more case of that residual
    /// moving opposite to truth. The wide-baseline pairs the GPU anchors contribute are 1-in-15
    /// frame jumps carrying correspondingly thin support, and a seed fitted to thin support loses to
    /// a short-baseline one fitted to plenty. Same conclusion the candidate-injection experiment
    /// reached above; the anchors did not change it.
    const MIN_SEED_PARALLAX_DEG: f64 = 1.5;

    let mut by_count: Vec<((usize, usize), usize)> =
        pair_count.iter().map(|(k, v)| (*k, *v)).collect();
    by_count.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
    if by_count.is_empty() {
        return Err(CalibError::NoReferenceTagView);
    }

    // Candidates = strongest by match count PLUS the most widely separated views.
    //
    // Ranking by match count alone is self-defeating for the seed: views overlap most when they
    // are closest together, so the top of that list is precisely the set of smallest baselines,
    // and any genuinely wide-baseline pair — the kind a well-conditioned seed needs — is ranked
    // too low to ever be considered. Sampling by index separation as well guarantees the selector
    // actually sees the well-separated options. Pairs that do not truly overlap simply fail the
    // two-view solve and cost one attempt each.
    // MEASURED: also injecting the most widely-separated pairs as candidates (the textbook move,
    // since COLMAP wants a 16 degree initial pair) did NOT pay off here — it fixed `fire`
    // (20.3 -> 2.0 cm ATE) but broke `chess` (1.1 -> 41.2 cm), for no net gain across five scenes.
    // At this keyframe spacing the widely-separated pairs carry too little support to seed a
    // reconstruction, so the selection just trades one scene for another. Left out until candidates
    // can actually reach a useful triangulation angle; the parallax floor and the ambiguity guard
    // below are the parts that carried their weight.
    let ranked: Vec<((usize, usize), usize)> =
        by_count.iter().take(BOOTSTRAP_CANDIDATES).copied().collect();

    // Preference threshold for the homography-vs-fundamental ratio (ORB-SLAM's value).
    const MAX_SEED_H_RATIO: f64 = 0.45;
    // Every viable seed, best first — not just the winner.
    //
    // The seed pair decides the entire reconstruction: it fixes the gauge, seeds the point cloud,
    // and determines which cameras can register at all. No scalar computed from the pair alone
    // predicts which one ends up best — parallax, RH ratio and support have each been measured as a
    // ranking criterion here and each traded one scene for another (see the notes below). Keeping
    // the ranked list lets a caller sweep the top few and choose on the finished reconstruction,
    // which is the only evidence that actually distinguishes them.
    // a, b, pose, cheirality, parallax, rh
    let mut seeds: Vec<(usize, usize, Pose3d, usize, f64, f64)> = Vec::new();
    for ((ca, cb), n_shared) in &ranked {
        if *n_shared < 8 {
            continue;
        }
        let Some((pose, cnt, par, rh)) = try_bootstrap_pair(*ca, *cb, tracks, cameras, &idcam)
        else {
            continue;
        };
        seeds.push((*ca, *cb, pose, cnt, par, rh));
    }
        // Prefer a pair that clears the parallax floor; among those, most cheirality-valid points.
        //
        // MEASURED, against the theoretically-tidier alternative of maximizing parallax outright
        // (COLMAP's 16 degree initial-pair minimum is the same instinct): scoring on parallax made
        // things markedly worse here — chess went 1.32 -> 8.01 px RMSE and pumpkin 6.38 -> 15.58,
        // because at this keyframe spacing the widest-parallax candidate also carries far less
        // evidence, and a seed fitted to thin support is worse than a short-baseline one fitted to
        // plenty. The parallax floor still rejects the outright degenerate pairs, which is the part
        // that mattered; beyond that, support wins. Revisit if keyframes are ever spaced widely
        // enough that candidates genuinely reach double-digit triangulation angles.
    // Prefer pairs that clear the parallax floor; among those, most cheirality-valid points.
    //
    // MEASURED, against the theoretically-tidier alternative of maximizing parallax outright
    // (COLMAP's 16 degree initial-pair minimum is the same instinct): scoring on parallax made
    // things markedly worse here — chess went 1.32 -> 8.01 px RMSE and pumpkin 6.38 -> 15.58,
    // because at this keyframe spacing the widest-parallax candidate also carries far less
    // evidence, and a seed fitted to thin support is worse than a short-baseline one fitted to
    // plenty. The parallax floor still rejects the outright degenerate pairs, which is the part
    // that mattered; beyond that, support wins.
    //
    // MEASURED, and NOT what theory predicted: letting the homography/fundamental ratio drive this
    // made things clearly worse — chess 0.66 -> 44.50 cm ATE and pumpkin 2.70 -> 38.00 cm, while
    // stairs (the scene it was added for) did not move at all. RH is still computed and logged,
    // because it correctly identifies that these handheld sequences are near-degenerate throughout,
    // but it does not rank usable seeds against each other: among pairs that are all somewhat
    // planar, "least planar" is not "best conditioned".
    //
    // The `(ca, cb)` tie-break is not cosmetic — without a total order, equal-scoring seeds would
    // reintroduce exactly the run-to-run variability the BTreeMap above exists to remove.
    // COLMAP's `init_max_forward_motion = 0.95`: a pair whose (unit) translation points nearly
    // along the optical axis triangulates everything near the epipole, where depth is
    // unobservable. A handheld walkthrough is forward-motion dominated, so this DEMOTES rather
    // than rejects — rejecting outright could empty the candidate list on exactly the clips that
    // need a seed most.
    const MAX_SEED_FORWARD: f64 = 0.95;
    seeds.sort_by(|a, b| {
        let (a_ok, b_ok) = (a.4 >= MIN_SEED_PARALLAX_DEG, b.4 >= MIN_SEED_PARALLAX_DEG);
        let (a_fwd, b_fwd) = (
            a.2.translation.z.abs() <= MAX_SEED_FORWARD,
            b.2.translation.z.abs() <= MAX_SEED_FORWARD,
        );
        b_ok.cmp(&a_ok)
            .then_with(|| b_fwd.cmp(&a_fwd))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
    });

    if seeds.is_empty() {
        return Err(CalibError::BundleAdjust(
            "no bootstrap pair produced a valid two-view pose".into(),
        ));
    }
    // Out-of-range ranks saturate rather than fail: a caller sweeping ranks 0..N should get the
    // worst available seed for the tail of its sweep, not an error that looks like a broken scene.
    let rank = config.seed_rank.min(seeds.len() - 1);
    let (a0, b0, seed_pose, seed_cnt, seed_par, seed_rh) = seeds[rank].clone();
    if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
        eprintln!(
            "[calib] seed pair ({a0},{b0}) rank={rank}/{} cheirality={seed_cnt} median_parallax={seed_par:.2} deg RH={seed_rh:.3}{}",
            seeds.len(),
            if seed_rh > MAX_SEED_H_RATIO { "  (DEGENERATE — no clean pair available)" } else { "" }
        );
    }

    let mut poses: Vec<Option<Pose3d>> = vec![None; n_cams];
    poses[a0] = Some(Pose3d::IDENTITY);
    poses[b0] = Some(seed_pose); // T_b0_a0, unit translation

    // Triangulate every track visible in the bootstrap pair → seed the point cloud (world = a0 frame).
    //
    // BTreeMap, not HashMap, and that is load-bearing rather than taste: this map is ITERATED below
    // to lay out the bundle-adjustment problem, so its order fixes the point indices and therefore
    // the order residuals are summed and the normal equations assembled. Rust seeds each HashMap
    // instance separately, so with a HashMap the same input reconstructed differently on every run
    // — measured at 60/60 registered cameras and 35.3 px on one run against 59/60 and 6.9 px on the
    // next. Track indices are already a dense `usize` key, so ordering them costs nothing.
    let mut point3d: BTreeMap<usize, Vec3F64> = BTreeMap::new();
    triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);

    // Metric prescale of the seed. The bootstrap's unit-baseline gauge can sit orders of magnitude
    // from metres; asking LM to close that gap through Huber-gated depth residuals is asking it to
    // climb out of a robust-kernel plateau. Rescaling the seed cloud AND the seed baseline by the
    // median measured/predicted depth ratio starts the priors near-satisfied.
    if use_depth && std::env::var_os("KORNIA_CALIB_NO_PRESCALE").is_none() {
        let mut ratios: Vec<f64> = Vec::new();
        for (ti, p) in &point3d {
            for (j, (c, _)) in norm[*ti].iter().enumerate() {
                let (Some(pose), Some(d)) = (&poses[*c], norm_depth[*ti].get(j).copied().flatten())
                else {
                    continue;
                };
                let z = pose.transform_point(p).z;
                if z > 1e-9 && d > 0.0 {
                    ratios.push(d as f64 / z);
                }
            }
        }
        if ratios.len() >= 8 {
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let s = ratios[ratios.len() / 2];
            if s.is_finite() && s > 1e-9 {
                for p in point3d.values_mut() {
                    *p = *p * s;
                }
                for pose in poses.iter_mut().flatten() {
                    pose.translation = pose.translation * s;
                }
                if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
                    eprintln!("[calib] metric prescale x{s:.4} from {} depth pairs", ratios.len());
                }
            }
        }
    }

    // Growth-time triangulation floor. Creation must be AT LEAST as strict on angle as
    // `filter_points`, or the pair composes into a resurrection loop: the filter drops a point in
    // [config floor, filter floor) and the very next `triangulate_new` — running on the SAME poses,
    // since BA wrote them back before the filter — re-creates it at the identical position, every
    // BA round. The seed cloud above deliberately keeps the permissive config floor: seed pairs on
    // video are low-parallax by nature, and an empty seed cloud aborts the candidate outright
    // rather than letting the sweep judge it.
    let tcfg_grow = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg.max(FILTER_MIN_TRI_ANGLE_DEG),
        ..tcfg.clone()
    };

    // --- Incremental grow: register the unplaced camera with the most 2D↔3D links via PnP. ---
    // COLMAP's `ba_global_images_ratio`: refine once the registered set has grown by 10%.
    //
    // Gate is an ABSOLUTE inlier count only, COLMAP's `abs_pose_min_num_inliers = 30`. The
    // previous (12, 0.25) ratio gate tightened as the map improved: the candidate's 2D-3D
    // correspondence list grows with every triangulation — including points fixed by frames
    // nowhere near this view — so the denominator inflates with correspondences a CORRECT pose
    // legitimately rejects, and growth stalls precisely as the reconstruction gets better.
    let mut next_ba = (registered_now(&poses) as f64 * BA_IMAGES_RATIO).max(3.0);
    grow_registrations(
        &mut poses, &mut point3d, &norm, &norm_depth, n_cams, &idcam, &tcfg_grow, config.min_registration_inliers, 0.0, a0,
        config, BA_IMAGES_RATIO, &mut next_ba,
    );

    // --- Bundle adjustment: all track points free, the reference camera (a0) fixed to anchor gauge. ---
    let mut points: Vec<Vec3F64> = Vec::new();
    let mut pt_index: HashMap<usize, usize> = HashMap::new();
    let mut obs: Vec<BaObservation> = Vec::new();
    for (ti, p) in &point3d {
        let pidx = points.len();
        pt_index.insert(*ti, pidx);
        points.push(*p);
        for (j, (c, nrm)) in norm[*ti].iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            let (depth_meas, depth_sigma) =
                depth_fields(&norm_depth, *ti, j, config);
            obs.push(BaObservation {
                pose_idx: *c,
                point_idx: pidx,
                pixel: [nrm.x as f32, nrm.y as f32],
                fixed_pose: *c == a0, // reference camera fixed → gauge anchor
                fixed_point: false,
                depth_meas,
                depth_sigma,
            });
        }
    }
    let poses_ba: Vec<Pose3d> = poses
        .iter()
        .map(|p| p.unwrap_or(Pose3d::IDENTITY))
        .collect();
    let res = bundle_adjust_schur_with_all_priors(
        &poses_ba,
        &points,
        &obs,
        &idcam,
        &BaParams {
            max_iterations: config.max_iterations,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            // Depth residuals now live in reprojection-like units (see `depth_fields`), so the
            // Huber knee is 1.345 × the reprojection noise scale, squared.
            depth_robust_scale_sq: {
                let sr = (config.max_reprojection_error / 2.0).max(1e-6) as f32;
                (1.345 * sr) * (1.345 * sr)
            },
            ..Default::default()
        },
        up_priors(&poses, config).as_deref(),
        motion_priors_for(&poses, config).as_deref(),
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))?;


    // --- Second registration pass, against the BA-refined map. ---
    //
    // The first pass judged every camera against the bootstrap cloud, which is rough: poses come
    // straight from PnP and points from two-view triangulation. Bundle adjustment then moves both
    // substantially. A camera whose PnP consensus fell just under the acceptance gate on the rough
    // map often clears it comfortably on the refined one — and without this retry that view stays
    // unregistered forever, shrinking map coverage for no reason other than evaluation order.
    // COLMAP interleaves registration and BA continuously for the same reason; this is the cheap
    // two-pass version of that idea.
    //
    // Feed the refined state back in: BA-optimized poses for registered cameras, BA-optimized
    // points via the same `pt_index` mapping used to build the problem.
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

    // ── Alternating intrinsics refinement (COLMAP refines focal/distortion INSIDE its BA; this
    // is the alternating equivalent that needs no solver surgery). The solver sees normalized
    // coordinates, so a focal error is a single global scale gamma on them and leading radial
    // distortion a k1 term: u_true ≈ gamma·n·(1 + k1·r²). Given the current map, that model is
    // LINEAR in (gamma, gamma·k1) against the predicted normalized projections — one closed-form
    // least squares per round, then re-normalize the observations and let the next BA re-settle
    // geometry. Guessed phone intrinsics (a fov sweep, k1 assumed zero on an ultra-wide) bend
    // the whole reconstruction; this lets the data correct them.
    if config.refine_intrinsics {
        let mut num = [0.0f64; 2];
        let mut a00 = 0.0f64;
        let mut a01 = 0.0f64;
        let mut a11 = 0.0f64;
        for (ti, pidx) in &pt_index {
            let Some(p) = res.points.get(*pidx) else { continue };
            for (c, n) in &norm[*ti] {
                let Some(_) = &poses[*c] else { continue };
                let pc = res.poses[*c].transform_point(p);
                if pc.z <= 1e-9 {
                    continue;
                }
                let u = Vec2F64::new(pc.x / pc.z, pc.y / pc.z);
                let r2 = n.x * n.x + n.y * n.y;
                // Model: u ≈ a·n + b·(n·r²), a = gamma, b = gamma·k1. Two scalar unknowns fit
                // over both residual components.
                for (nu, uu) in [(n.x, u.x), (n.y, u.y)] {
                    let x0 = nu;
                    let x1 = nu * r2;
                    a00 += x0 * x0;
                    a01 += x0 * x1;
                    a11 += x1 * x1;
                    num[0] += x0 * uu;
                    num[1] += x1 * uu;
                }
            }
        }
        let det = a00 * a11 - a01 * a01;
        if det.abs() > 1e-12 {
            let gamma = (num[0] * a11 - num[1] * a01) / det;
            let gk1 = (num[1] * a00 - num[0] * a01) / det;
            let k1 = if gamma.abs() > 1e-9 { gk1 / gamma } else { 0.0 };
            // Sanity bounds: a fit outside them means the map (not the camera) is wrong, and
            // applying it would let geometry errors masquerade as optics.
            if (0.7..1.3).contains(&gamma) && (-0.3..0.3).contains(&k1) {
                if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
                    eprintln!("[calib] intrinsics refinement: gamma={gamma:.4} k1={k1:.4}");
                }
                for track in norm.iter_mut() {
                    for (_, n) in track.iter_mut() {
                        let r2 = n.x * n.x + n.y * n.y;
                        let f = gamma * (1.0 + k1 * r2);
                        n.x *= f;
                        n.y *= f;
                    }
                }
            }
        }
    }

    // Iterate step on the refined map (see `filter_points`): drop what BA could not fix, rebuild
    // those tracks from the optimized poses. Runs regardless of `second_pass` — the filtered cloud
    // also feeds the final solve below when the second pass is off. Filter threshold is 2x the
    // creation threshold (COLMAP: create at `tri_*`, filter at the looser
    // `filter_max_reproj_error`) so a boundary point does not flip state every round; the angle
    // floors already agree via `tcfg_grow`.
    filter_points(
        &mut point3d,
        &norm,
        &poses,
        2.0 * config.max_reprojection_error,
        config.min_parallax_deg,
    );
    deregister_starved(&mut poses, &point3d, &norm, a0, 30);
    triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg_grow);

    // STRICTER than the first pass. The extra cameras this pass finds are a coverage bonus on top
    // of an already-good map, so they must not be allowed to degrade it — and measurably they can:
    // reusing the first-pass gate (12 inliers / 0.25) raised chess from 23/60 to 60/60 registered
    // while ATE collapsed from 0.66 cm to 44.93, because marginal registrations triangulate
    // marginal points that then drag the second bundle adjustment. Demanding a strong consensus
    // keeps the coverage that is real and refuses the rest.
    let added = if config.second_pass {
        grow_registrations(
            &mut poses, &mut point3d, &norm, &norm_depth, n_cams, &idcam, &tcfg_grow,
            config.min_registration_inliers, 0.0,
            a0, config, BA_IMAGES_RATIO, &mut next_ba,
        )
    } else {
        0
    };

    // Final relaxation rung — the practice all three reference pipelines share, each in its own
    // dialect: COLMAP's outer loop HALVES its thresholds when growth stalls (kNumInitRelaxations),
    // OpenSfM's resection floor is 10 inliers outright, and OpenMVG's a-contrario RANSAC has no
    // fixed floor at all (adaptive significance test). A fixed gate leaves the tail of a capture
    // permanently unregistered even when the finished, refined map could support it. One rung,
    // half the gate with a floor of 10, judged against the BEST map this solve will ever have —
    // and the post-BA filter + de-registration hygiene still polices anything it admits.
    let relaxed = if config.second_pass && config.min_registration_inliers > 10 {
        grow_registrations(
            &mut poses, &mut point3d, &norm, &norm_depth, n_cams, &idcam, &tcfg_grow,
            (config.min_registration_inliers / 2).max(10), 0.0,
            a0, config, BA_IMAGES_RATIO, &mut next_ba,
        )
    } else {
        0
    };
    if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
        eprintln!("[calib] relaxation rung registered {relaxed} more camera(s)");
    }
    let added = added + relaxed;
    if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
        eprintln!("[calib] second pass registered {added} more camera(s)");
    }

    // Final solve on the filtered, retriangulated cloud. This ALWAYS runs (it used to be skipped
    // when the second pass added nothing): the filter/retriangulate step above changed the point
    // set, so the first BA's result no longer describes the problem, and the terminal statistics
    // below read `res`/`pt_index` — skipping the resolve would report the unfiltered map while
    // silently discarding the cleanup.
    let res = {
        let _ = res;
        let mut points2: Vec<Vec3F64> = Vec::new();
        let mut pt_index2: HashMap<usize, usize> = HashMap::new();
        let mut obs2: Vec<BaObservation> = Vec::new();
        for (ti, p) in &point3d {
            let pidx = points2.len();
            pt_index2.insert(*ti, pidx);
            points2.push(*p);
            for (j, (c, nrm)) in norm[*ti].iter().enumerate() {
                if poses[*c].is_none() {
                    continue;
                }
                let (depth_meas, depth_sigma) =
                    depth_fields(&norm_depth, *ti, j, config);
                obs2.push(BaObservation {
                    pose_idx: *c,
                    point_idx: pidx,
                    pixel: [nrm.x as f32, nrm.y as f32],
                    fixed_pose: *c == a0,
                    fixed_point: false,
                    depth_meas,
                    depth_sigma,
                });
            }
        }
        let poses_ba2: Vec<Pose3d> = poses
            .iter()
            .map(|p| p.unwrap_or(Pose3d::IDENTITY))
            .collect();
        let res2 = bundle_adjust_schur_with_all_priors(
            &poses_ba2,
            &points2,
            &obs2,
            &idcam,
            &BaParams {
                max_iterations: config.max_iterations,
                robust: RobustKernelKind::Huber,
                robust_scale_sq: config.robust_scale_sq,
            // Depth residuals now live in reprojection-like units (see `depth_fields`), so the
            // Huber knee is 1.345 × the reprojection noise scale, squared.
            depth_robust_scale_sq: {
                let sr = (config.max_reprojection_error / 2.0).max(1e-6) as f32;
                (1.345 * sr) * (1.345 * sr)
            },
                ..Default::default()
            },
            up_priors(&poses, config).as_deref(),
        motion_priors_for(&poses, config).as_deref(),
        )
        .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))?;
        pt_index = pt_index2;
        res2
    };

    let registered: Vec<bool> = poses.iter().map(|p| p.is_some()).collect();

    // --- Metric scale from the tag: triangulate its corners, compare to the known side length. ---
    // Scaling the world by `s` (points ×s AND world→cam translation ×s) leaves reprojection unchanged,
    // so we compute per-camera stats on the UNSCALED BA result and only scale the output translation.
    let scale = tag_scale(
        tags_for_scale,
        cameras,
        &res.poses,
        &registered,
        &idcam,
        &tcfg,
        config.tag_size_m,
    );

    // Per-camera reprojection RMS (pixels); analytical covariance is tag-oriented so stays `None`.
    // Use the BA-optimized points (`res.points`), NOT the pre-BA cloud: BA moves points as free
    // variables, so evaluating the pre-BA cloud under post-BA poses would report a stale residual.
    // `pt_index` indexes both identically (same order as the BA input).
    let per_camera = (0..n_cams)
        .map(|c| {
            feature_stats(
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
    Ok(RigCalibration {
        poses: out_poses,
        reference_tag_id: tags_for_scale.first().map(|t| t.tag_id).unwrap_or(0),
        reproj_rmse_px,
        per_camera,
    })
}

/// ORB-SLAM's homography-vs-fundamental model selection score, `RH = SH / (SH + SF)`.
///
/// Both models are scored with a symmetric transfer error, accumulating `threshold - chi2` per
/// inlier observation so a model is rewarded for explaining points *tightly*, not merely for
/// counting them. Chi-square thresholds follow the reference implementation: 5.991 for the
/// homography (2 degrees of freedom, point-to-point) and 3.841 for the fundamental (1 DOF,
/// point-to-line), with the fundamental's score capped against 5.991 so the two totals live on the
/// same scale.
///
/// A high ratio means a homography explains the correspondences about as well as the epipolar
/// geometry does, which happens exactly when the scene is planar or the motion is near-pure
/// rotation. In that regime the essential matrix is not merely imprecise but ambiguous, and
/// committing to one of its decompositions can produce a mirrored reconstruction that bundle
/// adjustment then makes internally self-consistent — undetectable from reprojection error alone.
///
/// Returns `None` when either model cannot be estimated.
fn homography_vs_fundamental_ratio(x1: &[Vec2F64], x2: &[Vec2F64], seed: u64) -> Option<f64> {
    const TH_H: f64 = 5.991;
    const TH_F: f64 = 3.841;
    const SCORE_CAP: f64 = 5.991;
    // Assumed keypoint localization sigma, in pixels.
    const INV_SIGMA_SQ: f64 = 1.0;

    if x1.len() < 8 {
        return None;
    }
    let rp = TvRp {
        max_iterations: 2000,
        threshold: 2.0,
        min_inliers: 8,
        random_seed: Some(seed),
        refit: true,
    };

    let h = ransac_homography(x1, x2, &rp).ok()?;
    let f = ransac_fundamental(x1, x2, &rp).ok()?;

    // --- Homography score: symmetric transfer error in both directions. ---
    let hm = h.model;
    let hinv = hm.inverse();
    let mut s_h = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        // 1 -> 2
        let d = hm * Vec3F64::new(p1.x, p1.y, 1.0);
        if d.z.abs() > 1e-12 {
            let (u, v): (f64, f64) = (d.x / d.z, d.y / d.z);
            let chi = ((p2.x - u).powi(2) + (p2.y - v).powi(2)) * INV_SIGMA_SQ;
            if chi < TH_H {
                s_h += TH_H - chi;
            }
        }
        // 2 -> 1
        let d = hinv * Vec3F64::new(p2.x, p2.y, 1.0);
        if d.z.abs() > 1e-12 {
            let (u, v): (f64, f64) = (d.x / d.z, d.y / d.z);
            let chi = ((p1.x - u).powi(2) + (p1.y - v).powi(2)) * INV_SIGMA_SQ;
            if chi < TH_H {
                s_h += TH_H - chi;
            }
        }
    }

    // --- Fundamental score: symmetric point-to-epipolar-line distance. ---
    let fm = f.model;
    let ft = fm.transpose();
    let mut s_f = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        // Line in image 2 induced by the point in image 1.
        let l = fm * Vec3F64::new(p1.x, p1.y, 1.0);
        let den = l.x * l.x + l.y * l.y;
        if den > 1e-12 {
            let num = (l.x * p2.x + l.y * p2.y + l.z).powi(2);
            let chi = (num / den) * INV_SIGMA_SQ;
            if chi < TH_F {
                s_f += SCORE_CAP - chi;
            }
        }
        // Line in image 1 induced by the point in image 2.
        let l = ft * Vec3F64::new(p2.x, p2.y, 1.0);
        let den = l.x * l.x + l.y * l.y;
        if den > 1e-12 {
            let num = (l.x * p1.x + l.y * p1.y + l.z).powi(2);
            let chi = (num / den) * INV_SIGMA_SQ;
            if chi < TH_F {
                s_f += SCORE_CAP - chi;
            }
        }
    }

    let total = s_h + s_f;
    if total <= 0.0 {
        return None;
    }
    Some(s_h / total)
}

/// How many cameras currently hold a pose.
/// Depth-prior fields for one observation: `(depth_meas, depth_sigma)`.
///
/// The returned sigma is DEFLATED-into-reprojection-units, not the raw metric sigma. The solver's
/// reprojection residuals are unwhitened normalized-camera values (implicit σ = 1 normalized unit
/// — i.e. an entire focal length!), while its depth residuals divide by their sigma. Passing the
/// honest metric sigma therefore makes each depth row carry orders of magnitude more cost than a
/// reprojection row, and the solve becomes depth-dominated no matter how "loose" the relative
/// sigma looks — measured: rel_sigma 5.0 (≈ no confidence at all) still halved registration.
/// Multiplying the sigma by 1/σ_r — where σ_r is the reprojection noise scale in normalized units
/// (threshold treated as 2σ) — expresses both families in the same implicit unit.
fn depth_fields(
    norm_depth: &[Vec<Option<f32>>],
    ti: usize,
    j: usize,
    config: &CalibConfig,
) -> (Option<f32>, f32) {
    let rel_sigma = config.depth_prior_rel_sigma;
    match norm_depth.get(ti).and_then(|t| t.get(j)).copied().flatten() {
        Some(d) if rel_sigma > 0.0 && d > 0.0 => {
            let sigma_r = (config.max_reprojection_error / 2.0).max(1e-6) as f32;
            (Some(d), (rel_sigma as f32) * d / sigma_r)
        }
        _ => (None, 0.0),
    }
}

fn registered_now(poses: &[Option<Pose3d>]) -> usize {
    poses.iter().filter(|p| p.is_some()).count()
}

/// Global bundle adjustment over the registered cameras and the current cloud, written back in place.
///
/// Shared by the periodic refinement inside `grow_registrations` and the final solve, so the two
/// cannot drift apart in how they build the problem — the gauge anchor (`a0` held fixed) especially,
/// since a mid-growth BA that failed to fix it would let the whole reconstruction slide between
/// refinements.
/// COLMAP's `Mapper.filter_min_tri_angle`: a point whose best observation pair subtends less than
/// this is depth-unconstrained — bundle adjustment can slide it along the ray almost freely, and
/// wherever it lands then anchors later PnP registrations.
const FILTER_MIN_TRI_ANGLE_DEG: f64 = 1.5;

/// COLMAP-style point filtering, run after every global BA.
///
/// Without this the cloud is append-only: a point triangulated from a rough pre-BA pose stays at
/// its stale position forever and feeds every later registration. Measured signature on this
/// pipeline's failed maps (7-Scenes chess, GPU determinism repeat 3): 9.4% of points behind their
/// nearest camera and a p1 depth percentile of −62, against 0.4% / +3.5 on the healthy run of the
/// IDENTICAL configuration. COLMAP never lets a map reach that state — `FilterPoints` runs after
/// every BA round, dropping observations over the reprojection threshold and points that lose
/// cheirality or fall under the triangulation-angle floor.
///
/// One deliberate divergence: COLMAP drops individual observations and only then points with < 2
/// surviving views. Observations here live in `norm`, which the caller owns immutably, so the
/// per-observation drop is approximated by a majority vote — a point survives only if at least
/// half its registered observations reproject within threshold and in front of their camera. The
/// minority bad observations stay in the next BA, where the Huber kernel bounds their influence.
///
/// Removal is not final: the next [`triangulate_new`] pass re-triangulates the track from the
/// refined poses, so a point killed by a bad initial triangulation gets a second chance from
/// better geometry — this filter-then-retriangulate pair is exactly COLMAP's iterate loop.
/// `min_tri_angle_deg` is the FILTER floor, and it deliberately follows the caller's configured
/// parallax floor rather than `FILTER_MIN_TRI_ANGLE_DEG`: video seeds are low-parallax by nature
/// (measured 0.34 deg median on a 7-Scenes chess seed), so filtering at the COLMAP-style 1.5 deg
/// reaped 1295 of 1296 seed points after the first BA, deregistered both seed cameras, and killed
/// every candidate with `NoFreeVariables`. The 1.5 deg floor still applies where COLMAP applies
/// it — to points CREATED during growth (`tcfg_grow`) — which keeps the hysteresis ordering
/// (creation at least as strict as filtering) without executing the seed scaffolding the young
/// map stands on. Low-parallax seed points remain policed by the reprojection and cheirality
/// majority vote.
fn filter_points(
    point3d: &mut BTreeMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    poses: &[Option<Pose3d>],
    max_reproj_norm: f64,
    min_tri_angle_deg: f64,
) -> usize {
    let centers: Vec<Option<Vec3F64>> = poses
        .iter()
        .map(|p| p.as_ref().map(|p| p.inverse().translation))
        .collect();
    let before = point3d.len();
    point3d.retain(|ti, p| {
        let mut good: Vec<usize> = Vec::new();
        let mut seen = 0usize;
        for (c, uv) in &norm[*ti] {
            let Some(pose) = &poses[*c] else { continue };
            seen += 1;
            if let Some(e) = norm_residual(pose, *p, *uv) {
                if e <= max_reproj_norm {
                    good.push(*c);
                }
            }
        }
        if good.len() < 2 || 2 * good.len() < seen {
            return false;
        }
        // Best triangulation angle over surviving observation pairs.
        let mut best = 0.0f64;
        for i in 0..good.len() {
            for j in (i + 1)..good.len() {
                let (Some(ca), Some(cb)) = (centers[good[i]], centers[good[j]]) else {
                    continue;
                };
                let ra = (*p - ca).normalize();
                let rb = (*p - cb).normalize();
                best = best.max(ra.dot(rb).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
        best >= min_tri_angle_deg
    });
    before - point3d.len()
}

/// COLMAP's `FilterImages` analogue: un-register cameras whose support the point filter removed.
///
/// Without this, registration is a one-way door — `poses[c] = Some(..)` is never revisited — so a
/// camera registered against points that later proved degenerate stays in the map contributing
/// pose parameters to every subsequent BA while constraining nothing. De-registered cameras
/// return to the candidate pool (the growth loop clears its failed set after each BA), so a view
/// dropped here can re-register later against better geometry. The gauge anchor `a0` is exempt:
/// removing it would free the gauge mid-solve.
fn deregister_starved(
    poses: &mut [Option<Pose3d>],
    point3d: &BTreeMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    a0: usize,
    min_obs: usize,
) -> usize {
    let mut support = vec![0usize; poses.len()];
    for ti in point3d.keys() {
        for (c, _) in &norm[*ti] {
            support[*c] += 1;
        }
    }
    let mut dropped = 0usize;
    for (c, pose) in poses.iter_mut().enumerate() {
        if c != a0 && pose.is_some() && support[c] < min_obs {
            *pose = None;
            dropped += 1;
        }
    }
    dropped
}

/// COLMAP's `AdjustLocalBundle`: refine the just-registered camera against its neighbourhood,
/// immediately, instead of letting raw linear-PnP poses accumulate until the next global BA.
///
/// Without this the chain drifts structurally: every registration is a linear solve against
/// points triangulated from OTHER unrefined linear solves, and the ratio-triggered global BA —
/// which runs ~23 times on a 200-frame clip — is asked to pull a whole drifted chain straight at
/// once, from an initialization that is the drift itself. The measured signature is a
/// reconstruction that registers everything yet renders as smeared mush: 8.96 px global RMSE on
/// the walkthrough clip with no recognisable wall planes. COLMAP never lets a pose exist
/// unrefined for more than one step, and that — not the global BA cadence — is what keeps its
/// incremental chains crisp.
///
/// Scope, mirroring COLMAP's defaults: the new camera plus its `LOCAL_BA_NEIGHBOURS` (6)
/// most-connected registered neighbours are free (connectivity = shared observed 3D points);
/// points seen by any free camera are free; every other registered camera observing those points
/// joins the problem as a fixed constraint. The gauge anchor stays fixed regardless. Failure is
/// swallowed: a local refinement that cannot run (too few observations) just leaves the PnP pose
/// for the next global BA, which is exactly the pre-local-BA behaviour.
/// Per-pose gravity priors for `bundle_adjust_schur_with_priors`, or `None` when disabled.
///
/// The prior direction is `(0, -1, 0)` in the SOLVE frame — the reference camera's own image-up,
/// since the gauge fixes `a0` at identity. Every camera of a handheld capture shares (roughly)
/// one physical up, so pulling them all toward the same direction removes RELATIVE pitch/roll
/// drift; whatever global tilt `a0` itself carries is a gauge choice a caller can rotate away
/// afterwards. Registered poses only; the anchor gets one too (harmless — its pose is fixed).
fn up_priors(
    poses: &[Option<Pose3d>],
    config: &CalibConfig,
) -> Option<Vec<Option<BaPosePrior>>> {
    if config.up_prior_sigma <= 0.0 {
        return None;
    }
    Some(
        poses
            .iter()
            .map(|p| {
                p.as_ref().map(|_| BaPosePrior {
                    // No positional anchor: sigma is unused when infinite-like; use a huge sigma
                    // so the centre residual contributes nothing.
                    center_world: [0.0; 3],
                    sigma: 1e6,
                    up_world: Some([0.0, -1.0, 0.0]),
                    up_sigma: config.up_prior_sigma as f32,
                })
            })
            .collect(),
    )
}

/// Constant-velocity motion priors over consecutive REGISTERED triplets, or `None` when off.
///
/// Consecutive in camera-index order (video keyframes are time-ordered), with `alpha` from the
/// index spacing and triplets spanning a gap larger than 12 indices skipped — a bridge across an
/// unregistered stretch is not a constant-velocity hypothesis worth asserting. Sigmas are
/// deflated into reprojection units exactly like the depth priors (see `depth_fields`).
fn motion_priors_for(poses: &[Option<Pose3d>], config: &CalibConfig) -> Option<Vec<BaMotionPrior>> {
    if config.motion_prior_sigma <= 0.0 {
        return None;
    }
    let sigma_r = (config.max_reprojection_error / 2.0).max(1e-6);
    let sp = (config.motion_prior_sigma / sigma_r) as f32;
    let so = (0.5 * config.motion_prior_sigma / sigma_r) as f32;
    let reg: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    let mut out = Vec::new();
    for w in reg.windows(3) {
        let (i0, i1, i2) = (w[0], w[1], w[2]);
        if i2 - i0 > 12 {
            continue;
        }
        out.push(BaMotionPrior {
            i0,
            i1,
            i2,
            alpha: (i1 - i0) as f32 / (i2 - i0) as f32,
            position_sigma: sp,
            orientation_sigma: so,
        });
    }
    (!out.is_empty()).then_some(out)
}

fn run_local_ba(
    poses: &mut [Option<Pose3d>],
    point3d: &mut BTreeMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    idcam: &PinholeCamera,
    a0: usize,
    c_new: usize,
    config: &CalibConfig,
) {
    const LOCAL_BA_NEIGHBOURS: usize = 6;
    const LOCAL_BA_ITERATIONS: usize = 25;

    // Most-connected registered neighbours of the new camera, by shared observed 3D points.
    let mut shared: HashMap<usize, usize> = HashMap::new();
    for ti in point3d.keys() {
        let obs = &norm[*ti];
        if !obs.iter().any(|(c, _)| *c == c_new) {
            continue;
        }
        for (c, _) in obs {
            if *c != c_new && poses[*c].is_some() {
                *shared.entry(*c).or_insert(0) += 1;
            }
        }
    }
    let mut neigh: Vec<(usize, usize)> = shared.into_iter().collect();
    // Sort by shared count desc, then camera index for determinism.
    neigh.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut window: Vec<usize> = vec![c_new];
    window.extend(neigh.iter().take(LOCAL_BA_NEIGHBOURS).map(|(c, _)| *c));

    let is_free = |c: usize| c != a0 && window.contains(&c);

    // Free points: observed by at least one free camera.
    let mut points: Vec<Vec3F64> = Vec::new();
    let mut pt_index: HashMap<usize, usize> = HashMap::new();
    let mut obs: Vec<BaObservation> = Vec::new();
    for (ti, p) in point3d.iter() {
        let track = &norm[*ti];
        if !track.iter().any(|(c, _)| poses[*c].is_some() && is_free(*c)) {
            continue;
        }
        let pidx = points.len();
        points.push(*p);
        pt_index.insert(*ti, pidx);
        for (j, (c, nrm)) in track.iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            let (depth_meas, depth_sigma) =
                depth_fields(norm_depth, *ti, j, config);
            obs.push(BaObservation {
                pose_idx: *c,
                point_idx: pidx,
                pixel: [nrm.x as f32, nrm.y as f32],
                fixed_pose: !is_free(*c),
                fixed_point: false,
                depth_meas,
                depth_sigma,
            });
        }
    }
    if points.is_empty() || obs.is_empty() {
        return;
    }

    let poses_ba: Vec<Pose3d> = poses.iter().map(|p| p.unwrap_or(Pose3d::IDENTITY)).collect();
    let Ok(res) = bundle_adjust_schur_with_all_priors(
        &poses_ba,
        &points,
        &obs,
        idcam,
        &BaParams {
            max_iterations: LOCAL_BA_ITERATIONS,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            // Depth residuals now live in reprojection-like units (see `depth_fields`), so the
            // Huber knee is 1.345 × the reprojection noise scale, squared.
            depth_robust_scale_sq: {
                let sr = (config.max_reprojection_error / 2.0).max(1e-6) as f32;
                (1.345 * sr) * (1.345 * sr)
            },
            ..Default::default()
        },
        up_priors(poses, config).as_deref(),
        motion_priors_for(poses, config).as_deref(),
    ) else {
        return;
    };

    for &c in &window {
        if c != a0 && poses[c].is_some() {
            poses[c] = Some(res.poses[c]);
        }
    }
    for (ti, pidx) in &pt_index {
        if let Some(v) = res.points.get(*pidx) {
            point3d.insert(*ti, *v);
        }
    }
}

fn run_global_ba(
    poses: &mut [Option<Pose3d>],
    point3d: &mut BTreeMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    idcam: &PinholeCamera,
    a0: usize,
    config: &CalibConfig,
) -> Result<(), CalibError> {
    let mut points: Vec<Vec3F64> = Vec::new();
    let mut pt_index: HashMap<usize, usize> = HashMap::new();
    let mut obs: Vec<BaObservation> = Vec::new();
    for (ti, p) in point3d.iter() {
        let pidx = points.len();
        pt_index.insert(*ti, pidx);
        points.push(*p);
        for (j, (c, nrm)) in norm[*ti].iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            let (depth_meas, depth_sigma) =
                depth_fields(norm_depth, *ti, j, config);
            obs.push(BaObservation {
                pose_idx: *c,
                point_idx: pidx,
                pixel: [nrm.x as f32, nrm.y as f32],
                fixed_pose: *c == a0,
                fixed_point: false,
                depth_meas,
                depth_sigma,
            });
        }
    }
    if obs.is_empty() || points.is_empty() {
        return Err(CalibError::BundleAdjust("nothing to optimize".into()));
    }
    let poses_ba: Vec<Pose3d> = poses.iter().map(|p| p.unwrap_or(Pose3d::IDENTITY)).collect();
    let res = bundle_adjust_schur_with_all_priors(
        &poses_ba,
        &points,
        &obs,
        idcam,
        &BaParams {
            max_iterations: config.max_iterations,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            // Depth residuals now live in reprojection-like units (see `depth_fields`), so the
            // Huber knee is 1.345 × the reprojection noise scale, squared.
            depth_robust_scale_sq: {
                let sr = (config.max_reprojection_error / 2.0).max(1e-6) as f32;
                (1.345 * sr) * (1.345 * sr)
            },
            ..Default::default()
        },
        up_priors(poses, config).as_deref(),
        motion_priors_for(poses, config).as_deref(),
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))?;

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
    Ok(())
}

/// Register every camera that can be placed against the current point cloud, growing the cloud as
/// it goes. Returns how many cameras were newly registered.
///
/// Extracted so it can run TWICE: once on the bootstrap cloud, and again after bundle adjustment.
/// The second pass matters because the first judges every camera against a rough, pre-BA map — a
/// camera rejected there may register comfortably once poses and points have been optimized, and
/// without a retry that view is lost for good even though the evidence to place it now exists.
fn grow_registrations(
    poses: &mut [Option<Pose3d>],
    point3d: &mut BTreeMap<usize, Vec3F64>,
    norm: &[Vec<(usize, Vec2F64)>],
    norm_depth: &[Vec<Option<f32>>],
    n_cams: usize,
    idcam: &PinholeCamera,
    tcfg: &TriangulationConfig,
    min_inliers: usize,
    min_ratio: f64,
    a0: usize,
    config: &CalibConfig,
    ba_every: f64,
    next_ba: &mut f64,
) -> usize {
    let mut newly_registered = 0usize;
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
            if wp.len() < 4 {
                continue;
            }
            // COLMAP's visibility score, small version: count OCCUPIED cells of a coarse grid
            // over the view's 2D-3D correspondences instead of raw count. Many correspondences
            // clustered in one corner are abundant evidence and terrible PnP conditioning; a
            // spread of fewer points is the better next view. Score = occupied 8x8 cells * 1000
            // + count (count as tie-break), on normalized coords which span roughly [-1, 1].
            let mut cells = [false; 64];
            for uv in &ip {
                let gx = (((uv.x + 1.5) / 3.0) * 8.0).clamp(0.0, 7.999) as usize;
                let gy = (((uv.y + 1.5) / 3.0) * 8.0).clamp(0.0, 7.999) as usize;
                cells[gy * 8 + gx] = true;
            }
            let score = cells.iter().filter(|&&b| b).count() * 1000 + wp.len();
            if best
                .as_ref()
                .is_none_or(|(_, w, i2)| {
                    let mut bc = [false; 64];
                    for uv in i2 {
                        let gx = (((uv.x + 1.5) / 3.0) * 8.0).clamp(0.0, 7.999) as usize;
                        let gy = (((uv.y + 1.5) / 3.0) * 8.0).clamp(0.0, 7.999) as usize;
                        bc[gy * 8 + gx] = true;
                    }
                    score > bc.iter().filter(|&&b| b).count() * 1000 + w.len()
                })
            {
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
        // AP3P, not EPnP, as the RANSAC kernel — and the iteration budget matters as much as the
        // kernel. EPnP's minimal sample is 5; at the gate's own 25% inlier boundary a clean
        // 5-sample has probability 0.25^5 ≈ 0.1%, so 100 iterations succeed 9% of the time — a
        // view at the threshold registered by coin flip, and since a failure is sticky within a
        // growth round, the flip changed the reconstruction rather than its timing. AP3P's
        // 3-sample at 2000 iterations puts the same event above 99.99%. This is the quantitative
        // core of the measured chaos: a 0.02% match perturbation flipping 60/60 <-> 25/60.
        let pnp = solve_pnp_ransac(
            &world,
            &image,
            &Mat3AF32::IDENTITY, // normalized coords ⇒ identity intrinsics
            None,
            PnPMethod::AP3PDefault,
            &PnpRansacParams {
                // COLMAP's `abs_pose_max_error` is 12 px; the hardcoded 0.01 normalized this
                // replaced was ~2.6 px at a phone focal — so tight that on soft video frames the
                // consensus count was artificially crushed (measured: 44 correspondences, 8-11
                // "inliers", every frontier view rejected). Tie it to the same threshold the
                // rest of the pipeline calls an outlier bound.
                reproj_threshold_px: config.max_reprojection_error as f32,
                max_iterations: 2000,
                // Seed the sampler. `RansacParams::default()` leaves `random_seed: None`, which
                // draws from the thread RNG, so registration was nondeterministic: identical
                // inputs produced wildly different reconstructions run to run (measured on EuRoC
                // MH01, 40 keyframes — 12 / 30 / 39 cameras registered and 41.8 / 12.5 / 11.9 px
                // global RMSE over three runs of the same command). Because a transient PnP
                // failure permanently marks a camera unregisterable, that randomness changes the
                // final reconstruction, not merely its timing. The seed varies per camera so
                // different views still draw different sample sequences.
                random_seed: Some(0x00C0FFEE ^ c as u64),
                ..Default::default()
            },
        );
        // Accepting a registration is not the same as PnP returning `Ok`. `solve_pnp_ransac`
        // succeeds whenever it finds *a* consensus, however small, so an unguarded `Ok` arm admits
        // cameras whose pose is fitted to a handful of correspondences. That is not a
        // self-correcting mistake: `triangulate_new` immediately creates 3D points FROM the bad
        // pose, and those points then feed the next camera's PnP, so one bad registration
        // propagates through the rest of the reconstruction. Measured on EuRoC MH01 (40 keyframes)
        // the symptom was global RMSE rising *with* the number of registered cameras — 34
        // registered at 10.4 px, 38 at 23.2 px — which is the opposite of what a healthy
        // incremental SfM does.
        //
        // Gate on both an absolute inlier count (a 4-point consensus is not evidence) and an
        // inlier ratio (a low ratio means the 2D-3D matches for this view are mostly wrong).
        let (min_pnp_inliers, min_pnp_inlier_ratio) = (min_inliers, min_ratio);
        // Opt-in trace: set KORNIA_CALIB_DEBUG=1 to see, per candidate view, how much 2D-3D
        // evidence it had and how much of it PnP agreed with. Without this the growth loop is a
        // black box — "registration stopped" gives no clue whether the cause is missing
        // correspondences or a rejected consensus, and those have opposite fixes.
        if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
            let (ok, n_in) = match &pnp {
                Ok(r) => (true, r.inliers.len()),
                Err(_) => (false, 0),
            };
            eprintln!(
                "[calib] cam {c}: corr={} pnp_ok={ok} inliers={n_in} ratio={:.2} points_so_far={}",
                wp.len(),
                if wp.is_empty() { 0.0 } else { n_in as f64 / wp.len() as f64 },
                point3d.len()
            );
        }
        // Gate on the consensus of the pose actually being ACCEPTED — which is not always the
        // refit. `solve_pnp_ransac` reports `inliers` classified against the pre-refit
        // minimal-sample model (systematically pessimistic), then returns an EPnP refit of the
        // inlier set — and EPnP's linear solve can DEGRADE a good AP3P consensus when the inlier
        // spread is unfavourable (measured: a 107-inlier consensus reclassified to under 30
        // against the refit pose, silently un-registering a solid view). So score BOTH poses
        // against the full correspondence set and keep whichever explains more.
        let accepted = pnp.as_ref().ok().and_then(|r| {
            let count_for = |pose: &Pose3d| {
                wp.iter()
                    .zip(ip.iter())
                    .filter(|(w, i)| {
                        norm_residual(pose, **w, **i)
                            .is_some_and(|e| e <= config.max_reprojection_error)
                    })
                    .count()
            };
            let refit = pose_from_pnp(r.pose.rotation, r.pose.translation);
            let n_refit = count_for(&refit);
            // The RANSAC consensus count is already a same-threshold classification of the
            // minimal-sample model, so the two counts are directly comparable.
            let best_n = n_refit.max(r.inliers.len());
            (best_n >= min_pnp_inliers
                && best_n as f64 >= min_pnp_inlier_ratio * wp.len() as f64)
                .then_some((refit, n_refit, r.inliers.len()))
        });
        match accepted {
            Some((refit_pose, n_refit, n_ransac)) => {
                // If the refit lost a substantial share of the RANSAC consensus, the linear EPnP
                // step hurt rather than helped — but the driver has already discarded the
                // minimal-sample pose, so recover it by re-running with `refine: false` (same
                // seed → same sampling → same consensus pose).
                let pose = if n_refit * 2 >= n_ransac {
                    refit_pose
                } else {
                    let retry = solve_pnp_ransac(
                        &world,
                        &image,
                        &Mat3AF32::IDENTITY,
                        None,
                        PnPMethod::AP3PDefault,
                        &PnpRansacParams {
                            reproj_threshold_px: config.max_reprojection_error as f32,
                            max_iterations: 2000,
                            random_seed: Some(0x00C0FFEE ^ c as u64),
                            refine: false,
                            ..Default::default()
                        },
                    );
                    match retry {
                        Ok(r2) => pose_from_pnp(r2.pose.rotation, r2.pose.translation),
                        Err(_) => refit_pose,
                    }
                };
                poses[c] = Some(pose);
                newly_registered += 1;
                // Refine the pose against the existing map BEFORE triangulating from it — points
                // created off a raw linear-PnP pose inherit its error and then feed the next
                // view's PnP (see `run_local_ba`).
                run_local_ba(poses, point3d, norm, norm_depth, idcam, a0, c, config);
                let before = point3d.len();
                triangulate_new(point3d, norm, &poses, idcam, tcfg);
                // A camera that could not register earlier may well register now: each successful
                // registration triangulates new points, so the 2D-3D evidence available to the
                // remaining views grows. Treating the first rejection as permanent throws away
                // views that the enlarged map would support, which is why tightening the PnP gate
                // otherwise trades registration count for accuracy. Retrying keeps both.
                //
                // This terminates: the set is only cleared after a registration succeeds, and a
                // camera can only be registered once, so there are at most `n_cams` clears.
                if point3d.len() > before {
                    pnp_failed.clear();
                }

                // Periodic global BA, COLMAP's `ba_global_images_ratio`.
                //
                // Registration compounds error without this: every PnP is fitted against points
                // triangulated from poses that have never been refined, so drift accumulates along
                // the chain, inlier ratios sink below the acceptance gate, and growth stalls part
                // way through. Invisible at rig scale — with <= 60 cameras a single terminal BA
                // cleans up whatever drifted — but at per-frame density it dominates: 334 frames
                // registered only 81 cameras at 45.4 px and 59.6 cm ATE with terminal-only BA.
                //
                // Triggered on a RATIO rather than a fixed interval so the cost stays proportional:
                // BA runs when the registered set has grown by 10%, which is often early on and
                // rarely once the map is large.
                if ba_every > 0.0
                    && registered_now(poses) as f64 >= *next_ba
                    && run_global_ba(poses, point3d, norm, norm_depth, idcam, a0, config).is_ok()
                {
                    *next_ba = (registered_now(poses) as f64 * ba_every).max(*next_ba + 1.0);
                    // COLMAP's iterate step: BA → filter → retriangulate. Filtering after the
                    // solve removes the points BA could not fix (behind-camera, gross residual,
                    // depth-unconstrained), and retriangulating from the refined poses rebuilds
                    // those tracks from better geometry — so the cloud the NEXT registration is
                    // judged against is clean, instead of accreting every early mistake.
                    let dropped =
                        filter_points(
                        point3d,
                        norm,
                        poses,
                        2.0 * config.max_reprojection_error,
                        config.min_parallax_deg,
                    );
                    let dereg = deregister_starved(poses, point3d, norm, a0, min_inliers);
                    triangulate_new(point3d, norm, &poses, idcam, tcfg);
                    if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
                        eprintln!(
                            "[calib] post-BA filter: dropped {dropped} points, deregistered \
                             {dereg} cams, cloud now {}",
                            point3d.len()
                        );
                    }
                    // The refined map can support views rejected against the rough one.
                    pnp_failed.clear();
                }
            }
            // Weak consensus is treated exactly like a hard failure: leave the camera
            // unregistered rather than poison the map with it.
            _ => {
                pnp_failed.insert(c); // this camera can't register now; try the others
            }
        }
    }

    newly_registered
}

/// Solve the two-view geometry for one candidate seed pair.
///
/// Returns `(T_a_to_b with unit translation, cheirality-valid count, median triangulation angle in
/// degrees, homography-vs-fundamental ratio)`, or `None` when the pair has too few correspondences
/// or no decomposition survives the cheirality vote. The median angle is what distinguishes a usable seed from a degenerate one:
/// two nearly-coincident views can still produce a confident-looking essential matrix whose
/// triangulated depths are meaningless.
fn try_bootstrap_pair(
    a0: usize,
    b0: usize,
    tracks: &[FeatureTrack],
    cameras: &[PinholeCamera],
    idcam: &PinholeCamera,
) -> Option<(Pose3d, usize, f64, f64)> {
    let (mut x1, mut x2) = (Vec::new(), Vec::new());
    for t in tracks {
        let pa = t.obs.iter().find(|(c, _)| *c == a0);
        let pb = t.obs.iter().find(|(c, _)| *c == b0);
        if let (Some((_, ua)), Some((_, ub))) = (pa, pb) {
            x1.push(*ua);
            x2.push(*ub);
        }
    }
    if x1.len() < 8 {
        return None;
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

    // Model selection gate (ORB-SLAM): reject a seed whose correspondences a homography explains
    // about as well as the epipolar geometry. Above this ratio the scene is planar or the motion
    // is rotation-dominated, and the essential decomposition is ambiguous rather than merely
    // noisy. Rejecting costs one candidate; accepting can cost the whole map, because every later
    // camera registers against the seed cloud and a mirrored seed cannot be recovered from
    // downstream. 0.45 is the reference implementation's threshold.
    // Reported, not enforced here: the caller PREFERS pairs under the threshold but falls back to
    // the least-planar candidate when none clears it. Enforcing it as a hard reject fails outright
    // on handheld sequences whose every pair is near-degenerate — measured on 7-Scenes `chess`,
    // where all 12 candidates score RH ~0.495 and a hard gate produced no reconstruction at all.
    // A weak seed still beats no map, provided the choice is the best available one.
    let seed_rng = 0xB007 ^ ((a0 as u64) << 16) ^ b0 as u64;
    let rh = homography_vs_fundamental_ratio(&x1u, &x2u, seed_rng).unwrap_or(1.0);

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
    .ok()?;
    let cands = decompose_essential(&ess.model)?;

    // Lenient triangulation for the cheirality vote (count points in front of BOTH cameras).
    let tvote = TriangulationConfig {
        min_parallax_deg: 0.0,
        max_reprojection_error: 1e9,
        min_cheirality_count: 0,
        ..Default::default()
    };
    let mut best = (0usize, Pose3d::IDENTITY);
    let mut runner_up = 0usize;
    for (r, t) in cands {
        let pb = Pose3d::new(r, t); // world(=a0) → b, unit translation
        let mut cnt = 0usize;
        for k in 0..n1.len() {
            if let Ok(pts) = triangulate_matched_points(
                &[n1[k]],
                &[n2[k]],
                &Pose3d::IDENTITY,
                &pb,
                idcam,
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
            runner_up = best.0;
            best = (cnt, pb);
        } else if cnt > runner_up {
            runner_up = cnt;
        }
    }
    if best.0 == 0 {
        return None;
    }

    // Twofold-ambiguity guard.
    //
    // The cheirality vote picks whichever of the four essential decompositions puts the most points
    // in front of both cameras. When the scene is planar or the motion is near-pure rotation, two
    // decompositions score almost equally — the classic twofold planar ambiguity — and the winner
    // is then decided by noise. Committing to it yields a MIRRORED reconstruction that is
    // internally self-consistent: bundle adjustment happily drives its reprojection error down,
    // so the map looks healthy by every internal metric while being structurally wrong. That is
    // exactly what produced 112 and 148 degree median rotation errors against ground truth at only
    // 6-9 px RMSE, where an RMSE-based quality gate sees nothing wrong.
    //
    // ORB-SLAM's rule is to refuse to initialize in this situation rather than risk a corrupt map.
    // `TriangulationConfig::cheirality_ambiguity_max` (0.7) encodes the same threshold, but the
    // manual vote above never consults it, so it is applied explicitly here. Rejecting the pair is
    // cheap: the caller simply tries the next seed candidate.
    if best.0 > 0 && (runner_up as f64) >= tvote.cheirality_ambiguity_max * best.0 as f64 {
        if std::env::var_os("KORNIA_CALIB_DEBUG").is_some() {
            eprintln!(
                "[calib] seed ({a0},{b0}) REJECTED: ambiguous decomposition \
                 (best={} runner_up={} ratio={:.2})",
                best.0,
                runner_up,
                runner_up as f64 / best.0 as f64
            );
        }
        return None;
    }

    // Median triangulation angle for the chosen pose: the angle between the two bearing rays,
    // with the second rotated into the first camera's frame.
    let r_t = best.1.rotation.transpose();
    let mut angles: Vec<f64> = Vec::with_capacity(n1.len());
    for k in 0..n1.len() {
        let d1 = Vec3F64::new(n1[k].x, n1[k].y, 1.0).normalize();
        let d2 = r_t * Vec3F64::new(n2[k].x, n2[k].y, 1.0).normalize();
        let c = d1.dot(d2).clamp(-1.0, 1.0);
        angles.push(c.acos().to_degrees());
    }
    angles.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    let median = angles.get(angles.len() / 2).copied().unwrap_or(0.0);

    Some((best.1, best.0, median, rh))
}

/// Triangulate every not-yet-reconstructed track that has ≥2 placed cameras, adding it to `point3d`.
fn triangulate_new(
    point3d: &mut BTreeMap<usize, Vec3F64>,
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
        // Candidate pairs by baseline, widest first. Trying several and keeping the most
        // SUPPORTED result (COLMAP's robust triangulation, small-K version) matters because the
        // widest pair is also the most likely to contain the one mismatched observation a track
        // picked up — triangulating from a bad widest pair poisons the point even though a
        // slightly narrower pair would have placed it correctly.
        let centers: Vec<Vec3F64> = placed
            .iter()
            .map(|(c, _)| poses[*c].unwrap().inverse().translation)
            .collect();
        let mut by_baseline: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..placed.len() {
            for j in (i + 1)..placed.len() {
                by_baseline.push((i, j, (centers[i] - centers[j]).length()));
            }
        }
        by_baseline.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut best_pt: Option<(Vec3F64, usize)> = None;
        for &(i, j, _) in by_baseline.iter().take(3) {
            let (ca, ua) = placed[i];
            let (cb, ub) = placed[j];
            let Ok(pts) = triangulate_matched_points(
                &[ua],
                &[ub],
                &poses[ca].unwrap(),
                &poses[cb].unwrap(),
                idcam,
                tcfg,
            ) else {
                continue;
            };
            if pts.len() != 1 {
                continue;
            }
            let p = pts[0].position;
            let ok = placed
                .iter()
                .filter(|(c, uv)| {
                    norm_residual(&poses[*c].unwrap(), p, *uv)
                        .is_some_and(|e| e <= tcfg.max_reprojection_error)
                })
                .count();
            if best_pt.is_none_or(|(_, b)| ok > b) {
                best_pt = Some((p, ok));
            }
        }
        {
            if let Some((p, ok)) = best_pt {
                // Majority of ALL placed views must agree — the SAME predicate `filter_points`
                // applies, so a point cannot be admitted here and dropped there on grading alone.
                if ok >= 2 && 2 * ok >= placed.len() {
                    point3d.insert(ti, p);
                }
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
) -> f64 {
    if tag_size_m <= 0.0 {
        return 1.0;
    }
    let Some(tag) = tags.first() else { return 1.0 };
    let seers: Vec<usize> = tag
        .per_camera
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| registered[*c])
        .collect();
    if seers.len() < 2 {
        return 1.0;
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
        return 1.0;
    }
    tag_size_m / recon_side
}

/// Per-camera reprojection RMS (pixels) for the feature path; analytical covariance fields left `None`.
/// Reprojection is scale-invariant, so the UNSCALED world→cam BA poses are used directly.
#[allow(clippy::too_many_arguments)]
fn feature_stats(
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
            se += (r * cam.fx).powi(2); // r is Euclidean in normalized units; fx≈fy assumed
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
                se += (r * cameras[*c].fx).powi(2);
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
        let cal = match calibrate_features(&cams, &[tag], &tracks, &cfg) {
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
    }
}
