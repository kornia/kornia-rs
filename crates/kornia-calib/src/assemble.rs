//! Build the bundle-adjustment problem: fixed reference-tag corners plus free
//! triangulated points from the other tags and the feature matches.

use std::collections::HashMap;

use kornia_3d::ba::BaObservation;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{triangulate_matched_points, Pose3d, TriangulationConfig};
use kornia_algebra::{Vec2F64, Vec3F64};

use crate::board::BoardGeometry;
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

/// Triangulate one FREE point from the registered camera pair `(ca, ua)`/`(cb, ub)`, push it, and
/// attach a free observation for every `(camera, pixel)` in `attach`. Returns `false` (adding
/// nothing) if triangulation fails or is degenerate. Shared by every free-point site (feature tracks,
/// feature matches, and the non-measurable-tag fallback).
#[allow(clippy::too_many_arguments)]
fn add_free_point(
    points: &mut Vec<Vec3F64>,
    obs: &mut Vec<BaObservation>,
    cameras: &[PinholeCamera],
    poses: &[Pose3d],
    tcfg: &TriangulationConfig,
    (ca, ua): (usize, Vec2F64),
    (cb, ub): (usize, Vec2F64),
    attach: &[(usize, Vec2F64)],
) -> bool {
    let na = cameras[ca].normalize(ua);
    let nb = cameras[cb].normalize(ub);
    let Ok(pts) = triangulate_matched_points(
        &[na],
        &[nb],
        &poses[ca],
        &poses[cb],
        &PinholeCamera::IDENTITY,
        tcfg,
    ) else {
        return false;
    };
    if pts.len() != 1 {
        return false;
    }
    let pidx = points.len();
    points.push(pts[0].position);
    for (c, uv) in attach {
        obs.push(ba_obs(*c, pidx, cameras[*c].normalize(*uv), false));
    }
    true
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
            let n = cameras[*cam_idx].normalize(corners[k]);
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
                    let n = cameras[*cam_idx].normalize(corners[k]);
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
            add_free_point(
                &mut points,
                &mut obs,
                cameras,
                poses,
                &tcfg,
                seers[0],
                seers[1],
                &seers,
            );
        }
    }

    // Feature matches: one free point per pair (the pair IS the observation set).
    for f in features {
        if !have[f.cam_a] || !have[f.cam_b] {
            continue;
        }
        add_free_point(
            &mut points,
            &mut obs,
            cameras,
            poses,
            &tcfg,
            (f.cam_a, f.uv_a),
            (f.cam_b, f.uv_b),
            &[(f.cam_a, f.uv_a), (f.cam_b, f.uv_b)],
        );
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
    board: &BoardGeometry,
    poses: &[Pose3d],
    have: &[bool],
    config: &CalibConfig,
) -> (Vec<Vec3F64>, Vec<BaObservation>) {
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
                let n = cameras[*c].normalize(corners[k]);
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
        // Widest-baseline registered pair. Camera centres (pose.inverse().translation) are computed
        // ONCE per track here, not O(n²) inside the pair search.
        let centers: Vec<Vec3F64> = regs
            .iter()
            .map(|(c, _)| poses[*c].inverse().translation)
            .collect();
        let mut best = (0usize, 1usize, -1.0f64);
        for i in 0..regs.len() {
            for j in (i + 1)..regs.len() {
                let d = (centers[i] - centers[j]).length();
                if d > best.2 {
                    best = (i, j, d);
                }
            }
        }
        add_free_point(
            &mut points,
            &mut obs,
            cameras,
            poses,
            &tcfg,
            regs[best.0],
            regs[best.1],
            &regs,
        );
    }

    (points, obs)
}
