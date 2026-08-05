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

use kornia_3d::ba::{BaObservation, BaParams};
use kornia_3d::ba_schur::bundle_adjust_schur;
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
/// let recon = reconstruct(&cameras, &[], &tracks, &CalibConfig::new(0.1))?;
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
) -> Result<Reconstruction, CalibError> {
    let n_cams = cameras.len();
    let idcam = PinholeCamera::IDENTITY;
    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };

    // Per track: normalized observation per camera (undistort + K⁻¹). Raw pixels stay in `tracks`.
    let norm: Vec<Vec<(usize, Vec2F64)>> = tracks
        .iter()
        .map(|t| {
            t.obs
                .iter()
                .map(|(c, uv)| (*c, cameras[*c].normalize(*uv)))
                .collect()
        })
        .collect();

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
                reproj_threshold_px: 0.01, // normalized units
                ..Default::default()
            },
        );
        match pnp {
            Ok(r) => {
                poses[c] = Some(pose_from_pnp(r.pose.rotation, r.pose.translation));
                triangulate_new(&mut point3d, &norm, &poses, &idcam, &tcfg);
            }
            Err(_) => {
                pnp_failed.insert(c); // this camera can't register now; try the others
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

    let mut points: Vec<Vec3F64> = Vec::new(); // BA input; paired with track ids on output
    let mut pt_index: HashMap<usize, usize> = HashMap::new();
    let mut obs: Vec<BaObservation> = Vec::new();
    let mut kept_obs: Vec<Observation> = Vec::new();
    let mut point_track_id: Vec<usize> = Vec::new();
    for ti in &track_ids {
        let p = &point3d[ti];
        let pidx = points.len();
        pt_index.insert(*ti, pidx);
        points.push(*p);
        point_track_id.push(*ti);
        for (j, (c, nrm)) in norm[*ti].iter().enumerate() {
            if poses[*c].is_none() {
                continue;
            }
            // `norm[ti]` is built by mapping over `tracks[ti].obs`, so index j lines up and the
            // raw pixel is recoverable without re-normalising.
            kept_obs.push(Observation {
                view: *c,
                point: pidx,
                pixel: tracks[*ti].obs[j].1,
            });
            obs.push(BaObservation {
                pose_idx: *c,
                point_idx: pidx,
                pixel: [nrm.x as f32, nrm.y as f32],
                fixed_pose: *c == a0, // reference camera fixed → gauge anchor
                fixed_point: false,
                depth_meas: None,
                depth_sigma: 1.0,
            });
        }
    }
    let poses_ba: Vec<Pose3d> = poses
        .iter()
        .map(|p| p.unwrap_or(Pose3d::IDENTITY))
        .collect();
    let res = bundle_adjust_schur(
        &poses_ba,
        &points,
        &obs,
        &idcam,
        &BaParams {
            max_iterations: config.max_iterations,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            ..Default::default()
        },
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))?;

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
    })
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
        let cal = match reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg)
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
        let recon = reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg)
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
        let recon = reconstruct(&cams, &[], &tracks, &cfg).expect("synthetic scene must solve");

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
        let recon = reconstruct(&cams, std::slice::from_ref(&tag), &tracks, &cfg)
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
        let deg =
            reconstruct(&cams, std::slice::from_ref(&degenerate), &tracks, &cfg).expect("solves");
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
        let untri = reconstruct(&cams, std::slice::from_ref(&untriangulable), &tracks, &cfg)
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

        let a = reconstruct(&cams, &[], &tracks, &cfg).expect("solves");
        let b = reconstruct(&cams, &[], &tracks, &cfg).expect("solves");

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
}
