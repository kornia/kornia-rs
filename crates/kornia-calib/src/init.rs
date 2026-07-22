//! Pose initialization: reference-tag selection, per-camera planar pose
//! (2-fold ambiguity), and branch disambiguation via feature reprojection.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{triangulate_matched_points, Pose3d, TriangulationConfig};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_apriltag::pose::estimate_tag_pose;

use crate::board::BoardGeometry;
use crate::error::CalibError;
use crate::types::{CalibConfig, FeatureMatch, TagObservation};

/// Result of initialization: per-camera init poses (world→cam), a per-camera
/// "registered" mask, and the index of the reference tag in `tags`.
pub(crate) struct Init {
    pub poses: Vec<Pose3d>,
    pub have: Vec<bool>,
    pub ref_ti: usize,
}

/// Select the reference tag (seen by the most cameras), estimate each camera's
/// two planar tag poses, and pick the branch combination that best explains the
/// feature matches (a single small planar tag alone can't resolve rotation).
pub(crate) fn init_poses(
    cameras: &[PinholeCamera],
    tags: &[TagObservation],
    features: &[FeatureMatch],
    config: &CalibConfig,
) -> Result<Init, CalibError> {
    let n_cams = cameras.len();
    let ref_ti = tags
        .iter()
        .enumerate()
        .max_by_key(|(_, t)| t.per_camera.len())
        .map(|(i, _)| i)
        .ok_or(CalibError::NoTags)?;

    let h = config.tag_size_m / 2.0;
    // estimate_tag_pose expects object/image points in (BL, BR, TR, TL) order;
    // our tag corners are aruco-wound (TL, TR, BR, BL) — reverse to match.
    let object_pts = [
        Vec3F64::new(-h, -h, 0.0),
        Vec3F64::new(h, -h, 0.0),
        Vec3F64::new(h, h, 0.0),
        Vec3F64::new(-h, h, 0.0),
    ];

    let mut branches: Vec<(usize, [Pose3d; 2])> = Vec::new();
    for (cam_idx, corners) in &tags[ref_ti].per_camera {
        let cam = &cameras[*cam_idx];
        // Undistort corners so the planar pose init is distortion-correct too.
        let image_pts = [
            cam.undistort(corners[3].x, corners[3].y),
            cam.undistort(corners[2].x, corners[2].y),
            cam.undistort(corners[1].x, corners[1].y),
            cam.undistort(corners[0].x, corners[0].y),
        ];
        let result = estimate_tag_pose(&object_pts, &image_pts, cam, 50)?;
        branches.push((*cam_idx, [result.best.pose, result.second.pose]));
    }
    if branches.is_empty() {
        return Err(CalibError::NoReferenceTagView);
    }

    let tcfg = TriangulationConfig {
        min_parallax_deg: config.min_parallax_deg,
        max_reprojection_error: config.max_reprojection_error,
        ..Default::default()
    };
    let idcam = PinholeCamera::IDENTITY;

    // Score every branch combination by feature reprojection under its init.
    // Wrong planar branches make non-coplanar features triangulate/reproject
    // badly, which the score below heavily penalizes. >6 cams: fall back to
    // each camera's lower-error branch (2^nb combos is too many).
    let nb = branches.len();
    let ncomb = if nb <= 6 { 1u32 << nb } else { 1 };
    // Correspondences for branch scoring: the XFeat feature matches PLUS every non-reference tag
    // corner as an exact cross-camera match. The tag corners let the multi-tag rigid geometry
    // disambiguate the planar branch even with NO natural features (the correct branch combo is the
    // only one under which the shared corners triangulate + reproject consistently). One tag alone
    // still can't be disambiguated — that's the documented single-planar-tag degeneracy.
    let mut corr: Vec<(usize, Vec2F64, usize, Vec2F64)> = features
        .iter()
        .map(|f| (f.cam_a, f.uv_a, f.cam_b, f.uv_b))
        .collect();
    for (ti, tag) in tags.iter().enumerate() {
        if ti == ref_ti {
            continue;
        }
        for k in 0..4 {
            let seers: Vec<(usize, Vec2F64)> =
                tag.per_camera.iter().map(|(c, cs)| (*c, cs[k])).collect();
            for j in 1..seers.len() {
                corr.push((seers[0].0, seers[0].1, seers[j].0, seers[j].1));
            }
        }
    }
    // Normalized coords depend only on (camera, pixel), not the branch combo — compute once
    // (undistort is an iterative Brown-Conrady inversion, so recomputing it per combo is waste).
    let corr_norm: Vec<(Vec2F64, Vec2F64)> = corr
        .iter()
        .map(|&(ca, ua, cb, ub)| (cameras[ca].normalize(ua), cameras[cb].normalize(ub)))
        .collect();
    let mut best_combo = (f64::INFINITY, 0u32);
    let mut ps = vec![Pose3d::IDENTITY; n_cams];
    let mut hv = vec![false; n_cams];
    for combo in 0..ncomb {
        ps.iter_mut().for_each(|p| *p = Pose3d::IDENTITY);
        hv.iter_mut().for_each(|h| *h = false);
        for (bi, (cam_idx, brs)) in branches.iter().enumerate() {
            ps[*cam_idx] = brs[((combo >> bi) & 1) as usize];
            hv[*cam_idx] = true;
        }
        let mut err = 0.0f64;
        let mut cnt = 0usize;
        for (i, &(ca, _, cb, _)) in corr.iter().enumerate() {
            if !hv[ca] || !hv[cb] {
                continue;
            }
            let (na, nb2) = corr_norm[i];
            cnt += 2;
            let tri = triangulate_matched_points(&[na], &[nb2], &ps[ca], &ps[cb], &idcam, &tcfg);
            let pts = match &tri {
                Ok(p) if p.len() == 1 => p,
                _ => {
                    err += 8.0;
                    continue;
                }
            };
            let pw = pts[0].position;
            for (ci, nn) in [(ca, na), (cb, nb2)] {
                let pc = ps[ci].transform_point(&pw);
                if pc.z <= 1e-6 {
                    err += 4.0;
                } else {
                    err += (pc.x / pc.z - nn.x).powi(2) + (pc.y / pc.z - nn.y).powi(2);
                }
            }
        }
        let score = if cnt > 0 { err / cnt as f64 } else { 0.0 };
        if score < best_combo.0 {
            best_combo = (score, combo);
        }
    }

    let mut have = vec![false; n_cams];
    let mut poses = vec![Pose3d::IDENTITY; n_cams];
    for (bi, (cam_idx, brs)) in branches.iter().enumerate() {
        poses[*cam_idx] = brs[((best_combo.1 >> bi) & 1) as usize];
        have[*cam_idx] = true;
    }

    Ok(Init {
        poses,
        have,
        ref_ti,
    })
}

/// Measure a tag's 4 world corners (aruco winding TL,TR,BR,BL) directly from observations, WITHOUT
/// assuming any grid layout. PnP each registered camera that sees the tag (`estimate_tag_pose`),
/// map the tag square into that camera, lift to the world (reference-tag) frame via the camera's
/// init pose, and average across cameras. This turns an arbitrary *rigid* multi-tag arrangement into
/// fixed metric anchors: the observed geometry itself becomes the "board", so no rows/cols/spacing/
/// orientation must be specified and mirrored/rotated boards just work. `None` if no registered
/// camera sees the tag or every PnP fails.
pub(crate) fn measure_tag_corners(
    cameras: &[PinholeCamera],
    tag: &TagObservation,
    poses: &[Pose3d],
    have: &[bool],
    config: &CalibConfig,
) -> Option<[Vec3F64; 4]> {
    let h = config.tag_size_m / 2.0;
    // estimate_tag_pose object frame is (BL, BR, TR, TL); our corners are aruco (TL, TR, BR, BL).
    let object_pts = [
        Vec3F64::new(-h, -h, 0.0), // BL
        Vec3F64::new(h, -h, 0.0),  // BR
        Vec3F64::new(h, h, 0.0),   // TR
        Vec3F64::new(-h, h, 0.0),  // TL
    ];
    // aruco corner k maps to object index [TL,TR,BR,BL] = object[3,2,1,0].
    let aruco_from_object = [3usize, 2, 1, 0];
    // World corners of a candidate tag→cam pose, lifted into the world (reference-tag) frame.
    let world_corners = |t_cam_tag: &Pose3d, cam_to_world: &Pose3d| -> [Vec3F64; 4] {
        let mut w = [Vec3F64::ZERO; 4];
        for (ak, &oi) in aruco_from_object.iter().enumerate() {
            w[ak] = cam_to_world.transform_point(&t_cam_tag.transform_point(&object_pts[oi]));
        }
        w
    };
    let set_dist = |a: &[Vec3F64; 4], b: &[Vec3F64; 4]| -> f64 {
        (0..4).map(|k| (a[k] - b[k]).length()).sum()
    };

    // Per registered camera, BOTH planar-pose branches lifted to world corners. A planar tag has a
    // 2-fold pose ambiguity; `best` is usually right but flips near fronto-parallel, so we resolve it
    // by cross-camera consistency rather than trusting `best` blindly.
    let mut cands: Vec<[[Vec3F64; 4]; 2]> = Vec::new();
    for (c, corners) in &tag.per_camera {
        if !have[*c] {
            continue;
        }
        let cam = &cameras[*c];
        let image_pts = [
            cam.undistort(corners[3].x, corners[3].y),
            cam.undistort(corners[2].x, corners[2].y),
            cam.undistort(corners[1].x, corners[1].y),
            cam.undistort(corners[0].x, corners[0].y),
        ];
        let Ok(result) = estimate_tag_pose(&object_pts, &image_pts, cam, 50) else {
            continue;
        };
        let cam_to_world = poses[*c].inverse();
        cands.push([
            world_corners(&result.best.pose, &cam_to_world),
            world_corners(&result.second.pose, &cam_to_world),
        ]);
    }
    if cands.is_empty() {
        return None;
    }

    // Anchor = the camera whose two branches disagree MOST (the most head-on view — least ambiguous,
    // so its `best` is the most trustworthy). Then each camera contributes the branch closest to the
    // anchor's `best`, and we average the agreeing branches.
    let anchor_i = (0..cands.len())
        .max_by(|&a, &b| {
            set_dist(&cands[a][0], &cands[a][1])
                .partial_cmp(&set_dist(&cands[b][0], &cands[b][1]))
                .unwrap()
        })
        .unwrap();
    let anchor = cands[anchor_i][0];
    let mut acc = [Vec3F64::ZERO; 4];
    for cand in &cands {
        let pick = if set_dist(&cand[0], &anchor) <= set_dist(&cand[1], &anchor) {
            &cand[0]
        } else {
            &cand[1]
        };
        for k in 0..4 {
            acc[k] += pick[k];
        }
    }
    let inv = 1.0 / cands.len() as f64;
    Some(acc.map(|v| v * inv))
}

/// Mean squared reprojection error (px²) of every board corner this camera observes, under a
/// candidate `T_cam_board` pose. Points behind the camera are heavily penalised.
fn board_reproj_err(
    cam: &PinholeCamera,
    t_cam_board: &Pose3d,
    board: &BoardGeometry,
    obs: &[(u16, [Vec2F64; 4])],
) -> f64 {
    let mut e = 0.0f64;
    let mut n = 0usize;
    for (tid, corners) in obs {
        let Some(op) = board.object_points(*tid) else {
            continue;
        };
        for k in 0..4 {
            let pc = t_cam_board.transform_point(&op[k]);
            n += 1;
            if pc.z <= 1e-6 {
                e += 1e6;
                continue;
            }
            let u = cam.fx * pc.x / pc.z + cam.cx;
            let v = cam.fy * pc.y / pc.z + cam.cy;
            // Compare in UNDISTORTED pixel space (the projection above is plain pinhole) so a distorted
            // camera doesn't bias the branch ranking.
            let obs = cam.undistort(corners[k].x, corners[k].y);
            e += (u - obs.x).powi(2) + (v - obs.y).powi(2);
        }
    }
    if n > 0 {
        e / n as f64
    } else {
        f64::INFINITY
    }
}

/// Initialise each camera's `T_cam_board` pose (world = board frame) from its observed board tags.
///
/// For every visible board tag, both planar-pose branches from [`estimate_tag_pose`] give a candidate
/// board pose (`T_cam_tagcentre ∘ translate(-tag_centre)`); the candidate with the lowest reprojection
/// over ALL of that camera's board corners is kept. A multi-tag board resolves the planar 2-fold
/// ambiguity by consensus — no cross-camera `2^N` search and no feature matches required.
pub(crate) fn init_poses_board(
    cameras: &[PinholeCamera],
    tags: &[TagObservation],
    board: &BoardGeometry,
) -> (Vec<Pose3d>, Vec<bool>) {
    let n_cams = cameras.len();
    let mut poses = vec![Pose3d::IDENTITY; n_cams];
    let mut have = vec![false; n_cams];

    // Regroup observations per camera (only board tags).
    let mut per_cam: Vec<Vec<(u16, [Vec2F64; 4])>> = vec![Vec::new(); n_cams];
    for t in tags {
        if !board.contains(t.tag_id) {
            continue;
        }
        for (c, corners) in &t.per_camera {
            if *c < n_cams {
                per_cam[*c].push((t.tag_id, *corners));
            }
        }
    }

    for c in 0..n_cams {
        let cam = &cameras[c];
        let obs = &per_cam[c];
        if obs.is_empty() {
            continue;
        }
        let mut best: Option<(f64, Pose3d)> = None;
        for (tid, corners) in obs {
            let Some(op) = board.object_points(*tid) else {
                continue;
            };
            // Per-tag centred object frame (no uniform tag-size assumption): estimate_tag_pose wants
            // corners about the tag centre in (BL, BR, TR, TL) order = board corners (TL,TR,BR,BL)
            // reversed, minus the centre.
            let center = (op[0] + op[1] + op[2] + op[3]) * 0.25;
            let object_centered = [
                op[3] - center, // BL
                op[2] - center, // BR
                op[1] - center, // TR
                op[0] - center, // TL
            ];
            // Undistort corners; feed reversed (aruco TL,TR,BR,BL → BL,BR,TR,TL).
            let img = [
                cam.undistort(corners[3].x, corners[3].y),
                cam.undistort(corners[2].x, corners[2].y),
                cam.undistort(corners[1].x, corners[1].y),
                cam.undistort(corners[0].x, corners[0].y),
            ];
            let Ok(pair) = estimate_tag_pose(&object_centered, &img, cam, 50) else {
                continue;
            };
            let t_object_board = Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(-center.x, -center.y, -center.z),
            );
            for cand in [pair.best.pose, pair.second.pose] {
                let t_cam_board = cand.compose(&t_object_board);
                let err = board_reproj_err(cam, &t_cam_board, board, obs);
                if best.as_ref().is_none_or(|(e, _)| err < *e) {
                    best = Some((err, t_cam_board));
                }
            }
        }
        if let Some((_, p)) = best {
            poses[c] = p;
            have[c] = true;
        }
    }

    (poses, have)
}
