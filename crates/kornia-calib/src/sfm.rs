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

use std::collections::HashMap;

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
    let &(a0, b0) = pair_count
        .iter()
        .max_by_key(|(_, &n)| n)
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
    let rp = TvRp {
        max_iterations: 2000,
        threshold: 2.0,
        min_inliers: 8,
        random_seed: Some(0),
        refit: true,
    };
    let ess = ransac_essential_5pt(
        &x1,
        &x2,
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
    loop {
        // For each unplaced camera, gather (world_point, normalized_pixel) from tracks with a 3D point.
        let mut best: Option<(usize, Vec<Vec3F64>, Vec<Vec2F64>)> = None;
        for c in 0..n_cams {
            if poses[c].is_some() {
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
            Err(_) => break, // no more cameras can be registered
        }
    }

    // --- Bundle adjustment: all track points free, the reference camera (a0) fixed to anchor gauge. ---
    let mut points: Vec<Vec3F64> = Vec::new();
    let mut pt_index: HashMap<usize, usize> = HashMap::new();
    let mut obs: Vec<BaObservation> = Vec::new();
    for (ti, p) in &point3d {
        let pidx = points.len();
        pt_index.insert(*ti, pidx);
        points.push(*p);
        for (c, nrm) in &norm[*ti] {
            if poses[*c].is_none() {
                continue;
            }
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
    let per_camera = (0..n_cams)
        .map(|c| {
            feature_stats(
                c,
                &res.poses,
                &registered,
                &points,
                &pt_index,
                &norm,
                &cameras[c],
            )
        })
        .collect();
    let reproj_rmse_px =
        global_reproj_rmse(&res.poses, &registered, &points, &pt_index, &norm, cameras);

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
