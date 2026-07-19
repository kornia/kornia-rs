//! Build the bundle-adjustment problem: fixed reference-tag corners plus free
//! triangulated points from the other tags and the feature matches.

use std::collections::HashMap;

use kornia_3d::ba::BaObservation;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{triangulate_matched_points, Pose3d, TriangulationConfig};
use kornia_algebra::{Vec2F64, Vec3F64};

use crate::board::AprilGridBoard;
use crate::init::{identity_camera, normalize};
use crate::types::{CalibConfig, FeatureMatch, FeatureTrack, TagObservation};

fn ba_obs(pose_idx: usize, point_idx: usize, n: Vec2F64, fixed_point: bool) -> BaObservation {
    BaObservation {
        pose_idx,
        point_idx,
        pixel: [n.x as f32, n.y as f32],
        fixed_pose: false,
        fixed_point,
        depth_meas: None,
        depth_sigma: 1.0,
    }
}

/// Assemble `(points, observations)` for bundle adjustment. World frame = the
/// reference tag's frame: its 4 corners are FIXED (gauge + metric scale); every
/// other tag corner and feature point is a FREE point triangulated from two
/// registered cameras.
pub(crate) fn assemble_problem(
    cameras: &[PinholeCamera],
    tags: &[TagObservation],
    features: &[FeatureMatch],
    poses: &[Pose3d],
    have: &[bool],
    ref_ti: usize,
    config: &CalibConfig,
) -> (Vec<Vec3F64>, Vec<BaObservation>) {
    let idcam = identity_camera();
    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };
    let h = config.tag_size_m / 2.0;

    let mut points: Vec<Vec3F64> = Vec::new();
    let mut obs: Vec<BaObservation> = Vec::new();

    // Reference tag: 4 FIXED corners at the known origin square (aruco winding).
    let ref_corners = [
        Vec3F64::new(-h, h, 0.0),
        Vec3F64::new(h, h, 0.0),
        Vec3F64::new(h, -h, 0.0),
        Vec3F64::new(-h, -h, 0.0),
    ];
    let base = points.len();
    points.extend_from_slice(&ref_corners);
    for (cam_idx, corners) in &tags[ref_ti].per_camera {
        for k in 0..4 {
            let n = normalize(&cameras[*cam_idx], corners[k]);
            obs.push(ba_obs(*cam_idx, base + k, n, true));
        }
    }

    // Other tags: turn the observed RIGID arrangement into fixed metric anchors. Measure each tag's
    // 4 world corners via per-camera PnP averaged into the reference-tag frame (no grid layout
    // assumed — the observed geometry IS the board), then FIX them like the reference tag. Fixed,
    // spatially-spread corners are what constrain rotation and what the per-camera observability
    // covariance counts — so a multi-tag rig reports strong instead of the single-tag weak-rotation
    // trap. Falls back to per-corner FREE triangulation when a tag can't be measured (no registered
    // camera sees it).
    for (ti, tag) in tags.iter().enumerate() {
        if ti == ref_ti {
            continue;
        }
        if let Some(world_corners) =
            crate::init::measure_tag_corners(cameras, tag, poses, have, config)
        {
            let base = points.len();
            points.extend_from_slice(&world_corners);
            for (cam_idx, corners) in &tag.per_camera {
                if !have[*cam_idx] {
                    continue;
                }
                for k in 0..4 {
                    let n = normalize(&cameras[*cam_idx], corners[k]);
                    obs.push(ba_obs(*cam_idx, base + k, n, true)); // FIXED measured board corner
                }
            }
            continue;
        }
        // Fallback: per-corner FREE point triangulated from two registered cameras that see it.
        for k in 0..4 {
            let seers: Vec<(usize, Vec2F64)> = tag
                .per_camera
                .iter()
                .filter(|(c, _)| have[*c])
                .map(|(c, corners)| (*c, corners[k]))
                .collect();
            if seers.len() < 2 {
                continue;
            }
            let (ca, ua) = seers[0];
            let (cb, ub) = seers[1];
            let na = normalize(&cameras[ca], ua);
            let nb = normalize(&cameras[cb], ub);
            let Ok(pts) =
                triangulate_matched_points(&[na], &[nb], &poses[ca], &poses[cb], &idcam, &tcfg)
            else {
                continue;
            };
            if pts.len() != 1 {
                continue;
            }
            let pidx = points.len();
            points.push(pts[0].position);
            for (c, uv) in &seers {
                let n = normalize(&cameras[*c], *uv);
                obs.push(ba_obs(*c, pidx, n, false));
            }
        }
    }

    // Feature matches: triangulate each pair with the init poses (per-point, so
    // the observation mapping is exact).
    for f in features {
        if !have[f.cam_a] || !have[f.cam_b] {
            continue;
        }
        let na = normalize(&cameras[f.cam_a], f.uv_a);
        let nb = normalize(&cameras[f.cam_b], f.uv_b);
        let Ok(pts) = triangulate_matched_points(
            &[na],
            &[nb],
            &poses[f.cam_a],
            &poses[f.cam_b],
            &idcam,
            &tcfg,
        ) else {
            continue;
        };
        if pts.len() != 1 {
            continue;
        }
        let pidx = points.len();
        points.push(pts[0].position);
        obs.push(ba_obs(f.cam_a, pidx, na, false));
        obs.push(ba_obs(f.cam_b, pidx, nb, false));
    }

    (points, obs)
}

/// Assemble `(points, observations)` for the RIGID-BOARD path. World = board frame: EVERY observed
/// board tag corner is a FIXED point at its known board-geometry position (one point per
/// `(tag_id, corner)`, observed by all registered cameras that see it) — this is what pins the gauge +
/// metric scale and constrains rotation. Feature matches remain FREE triangulated points that add
/// robustness and cross-camera coverage.
pub(crate) fn assemble_board(
    cameras: &[PinholeCamera],
    tags: &[TagObservation],
    tracks: &[FeatureTrack],
    board: &AprilGridBoard,
    poses: &[Pose3d],
    have: &[bool],
    config: &CalibConfig,
) -> (Vec<Vec3F64>, Vec<BaObservation>) {
    let idcam = identity_camera();
    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };

    let mut points: Vec<Vec3F64> = Vec::new();
    let mut obs: Vec<BaObservation> = Vec::new();

    // Fixed board corners: one point per (tag_id, corner_idx), observed by every registered camera
    // that sees it.
    let mut corner_pt: HashMap<(u16, usize), usize> = HashMap::new();
    for tag in tags {
        let Some(object) = board.object_points(tag.tag_id) else {
            continue; // tag not on the board
        };
        for k in 0..4 {
            let pidx = *corner_pt.entry((tag.tag_id, k)).or_insert_with(|| {
                let i = points.len();
                points.push(object[k]);
                i
            });
            for (c, corners) in &tag.per_camera {
                if !have[*c] {
                    continue;
                }
                let n = normalize(&cameras[*c], corners[k]);
                obs.push(ba_obs(*c, pidx, n, true)); // fixed board point
            }
        }
    }

    // Feature tracks: ONE free point per track, triangulated from the widest-baseline pair of
    // registered cameras (best conditioned), with every registered observation then attached.
    for track in tracks {
        let regs: Vec<(usize, Vec2F64)> = track
            .obs
            .iter()
            .copied()
            .filter(|(c, _)| *c < have.len() && have[*c])
            .collect();
        if regs.len() < 2 {
            continue;
        }
        // Widest-baseline registered pair (camera centre = pose.inverse().translation).
        let mut best = (0usize, 1usize, -1.0f64);
        for i in 0..regs.len() {
            for j in (i + 1)..regs.len() {
                let ci = poses[regs[i].0].inverse().translation;
                let cj = poses[regs[j].0].inverse().translation;
                let d = (ci - cj).length();
                if d > best.2 {
                    best = (i, j, d);
                }
            }
        }
        let (ca, ua) = regs[best.0];
        let (cb, ub) = regs[best.1];
        let na = normalize(&cameras[ca], ua);
        let nb = normalize(&cameras[cb], ub);
        let Ok(pts) =
            triangulate_matched_points(&[na], &[nb], &poses[ca], &poses[cb], &idcam, &tcfg)
        else {
            continue;
        };
        if pts.len() != 1 {
            continue;
        }
        let pidx = points.len();
        points.push(pts[0].position);
        for (c, uv) in &regs {
            let n = normalize(&cameras[*c], *uv);
            obs.push(ba_obs(*c, pidx, n, false));
        }
    }

    (points, obs)
}
