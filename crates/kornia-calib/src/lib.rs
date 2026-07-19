#![deny(missing_docs)]
// Tag corners are fixed 4-element arrays indexed by a shared `k` across several
// arrays; range loops read clearer than zipped iterators here.
#![allow(clippy::needless_range_loop)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Entry points
//!
//! - [`calibrate_apriltag`] — extrinsics from free AprilTag corners (+ optional feature matches). A
//!   second tag is auto-measured into fixed board geometry (per-camera PnP averaged into the
//!   reference-tag frame) so any rigid multi-tag arrangement constrains rotation without a layout.
//! - [`calibrate_board`] — extrinsics from a known rigid [`AprilGridBoard`] (every corner a fixed
//!   metric anchor) plus optional multi-view feature tracks ([`build_tracks`]).
//! - [`calibrate_multishot`] — multi-shot self-calibration: **refines each camera's focal**
//!   ([`estimate_focal`], Zhang median across shots) and averages the extrinsics over N board poses
//!   for a real empirical covariance. Single-view focal is unobservable (the pose absorbs it); N
//!   views break that coupling.
//!
//! Bundle adjustment runs on the Schur-complement solver ([`kornia_3d::ba_schur`]) so cost scales
//! with the camera count, not the (often >1000) feature-track points. Per-camera pose covariance +
//! observability (the min-eigenvalue that flags the single-tag rotation degeneracy) is in
//! [`RigCalibration::per_camera`].
//!
//! # Conventions
//!
//! - **Output pose = `T_world_cam`** — maps a point from the camera OPTICAL
//!   frame into the world frame. The world frame is the reference tag's frame.
//!   Bundle adjustment optimizes `world→cam` poses internally; the result is
//!   inverted before it is returned.
//! - **Camera optical frame = OpenCV**: x-right, y-down, z-forward. Any z-up
//!   re-gauging is a caller concern.
//! - **Corner winding = aruco `(TL, TR, BR, BL)`** on input, in raw (possibly
//!   distorted) pixels. Distortion is removed internally via each camera's model.
//! - **f64** geometry throughout; the bundle-adjustment pixel storage is f32.
//! - **Overlapping FOV**: only cameras that observe the reference tag are
//!   calibrated. A single planar tag under-constrains rotation — supply ≥2 tags
//!   or feature matches for a well-constrained solve.

mod assemble;
mod covariance;
mod error;
mod init;
mod intrinsics;
mod multishot;
mod tracks;
mod types;

pub use error::CalibError;
pub use intrinsics::estimate_focal;
pub use multishot::{calibrate_multishot, MultiShotCalibration, MultiShotConfig, Shot};
pub use tracks::{build_tracks, TrackEdge};
pub use types::{
    CalibConfig, CameraStats, FeatureMatch, FeatureTrack, RigCalibration, TagObservation,
};

// Re-exported so callers can name calibration poses without depending on
// `kornia-3d` directly — this crate is the calibration facade.
pub use kornia_3d::pose::Pose3d;

use kornia_3d::ba::{BaObservation, BaParams};
// Schur-complement BA: the point block (feature tracks — often >1000) is eliminated per LM iteration
// via the reduced camera system, so cost scales with the handful of camera poses, not the point count.
// The non-Schur `bundle_adjust` is O((6C+3N)³)-ish and took ~50s on ~1100 tracks; this is seconds. It
// honours the same Huber/Cauchy IRLS + fixed-pose/point flags, so robustness is unchanged.
use kornia_3d::ba_schur::bundle_adjust_schur;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::Vec3F64;
// AprilGridBoard is defined in kornia-apriltag (a tag-layout concept); imported for the board path,
// NOT re-exported — callers depend on kornia-apriltag directly for it.
use kornia_apriltag::board::AprilGridBoard;

/// Normalized reprojection residual norm of one observation under a solved (poses, points).
/// `None` if the point is behind the camera.
fn obs_residual(o: &BaObservation, poses: &[Pose3d], points: &[Vec3F64]) -> Option<f64> {
    let pc = poses[o.pose_idx].transform_point(&points[o.point_idx]);
    if pc.z <= 1e-6 {
        return None;
    }
    let dx = pc.x / pc.z - o.pixel[0] as f64;
    let dy = pc.y / pc.z - o.pixel[1] as f64;
    Some((dx * dx + dy * dy).sqrt())
}

/// X84 outlier threshold `median + k·1.4826·MAD`. Returns `+∞` (no removal) when there are too few
/// samples or the spread is numerically zero (a near-perfect fit — nothing to reject).
fn x84_threshold(res: &mut [f64], k: f64) -> f64 {
    if res.len() < 10 {
        return f64::INFINITY;
    }
    res.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = res[res.len() / 2];
    let mut dev: Vec<f64> = res.iter().map(|r| (r - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = dev[dev.len() / 2];
    if mad < 1e-9 {
        return f64::INFINITY;
    }
    med + k * 1.4826 * mad
}

/// Calibrate multi-camera rig extrinsics from AprilTag corners (and optional
/// feature matches) via depth-free reprojection bundle adjustment.
///
/// `cameras[i]` holds the known intrinsics (and distortion) of camera `i`; the
/// camera indices used in `tags` and `features` index into this slice. Returns
/// per-camera `T_world_cam` poses (world = reference-tag frame) and the final
/// reprojection RMS in pixels.
pub fn calibrate_apriltag(
    cameras: &[PinholeCamera],
    tags: &[TagObservation],
    features: &[FeatureMatch],
    config: &CalibConfig,
) -> Result<RigCalibration, CalibError> {
    if tags.is_empty() {
        return Err(CalibError::NoTags);
    }

    // 1. init poses (reference tag + branch disambiguation)
    let init = init::init_poses(cameras, tags, features, config)?;

    // 2. assemble fixed/free points + observations
    let (points, obs) = assemble::assemble_problem(
        cameras,
        tags,
        features,
        &init.poses,
        &init.have,
        init.ref_ti,
        config,
    );

    // 3. bundle adjust + finalize (shared with the board path).
    finalize(
        cameras,
        &init.poses,
        &points,
        &obs,
        &init.have,
        tags[init.ref_ti].tag_id,
        config,
    )
}

/// Shared tail: run bundle adjustment on the identity BA camera, compute per-camera-focal reprojection
/// RMS in pixels, and return `T_world_cam` per camera (`None` for unregistered cameras).
fn finalize(
    cameras: &[PinholeCamera],
    poses_init: &[Pose3d],
    points: &[Vec3F64],
    obs: &[kornia_3d::ba::BaObservation],
    have: &[bool],
    reference_tag_id: u16,
    config: &CalibConfig,
) -> Result<RigCalibration, CalibError> {
    let idcam = PinholeCamera::IDENTITY;

    // Pass 1: Huber warm-start over ALL observations.
    let res1 = bundle_adjust_schur(
        poses_init,
        points,
        obs,
        &idcam,
        &BaParams {
            max_iterations: config.max_iterations,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: config.robust_scale_sq,
            ..Default::default()
        },
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))?;

    // X84 removal of FREE-point (feature) observations only — the fixed board/tag corners are the
    // gauge and are always kept. On a near-perfect fit (MAD≈0) nothing is removed.
    let mut free_res: Vec<f64> = obs
        .iter()
        .filter(|o| !o.fixed_point)
        .filter_map(|o| obs_residual(o, &res1.poses, &res1.points))
        .collect();
    let thr = x84_threshold(&mut free_res, config.x84_k);
    let obs2: Vec<BaObservation> = obs
        .iter()
        .filter(|o| {
            o.fixed_point || obs_residual(o, &res1.poses, &res1.points).is_some_and(|r| r <= thr)
        })
        .copied()
        .collect();

    // Pass 2: Cauchy redescender on the surviving observations, warm-started from pass 1.
    let res = bundle_adjust_schur(
        &res1.poses,
        &res1.points,
        &obs2,
        &idcam,
        &BaParams {
            max_iterations: config.max_iterations,
            robust: RobustKernelKind::Cauchy,
            robust_scale_sq: config.robust_scale_sq,
            ..Default::default()
        },
    )
    .map_err(|e| CalibError::BundleAdjust(format!("{e:?}")))?;

    // Reprojection RMS in PIXELS over the surviving observations: residuals are normalized (identity
    // BA camera) — scale EACH by ITS OWN camera's focal, never a global mean.
    let mut se = 0.0f64;
    let mut nobs = 0usize;
    for o in &obs2 {
        let pw = res.points[o.point_idx];
        let pc = res.poses[o.pose_idx].transform_point(&pw);
        if pc.z <= 1e-6 {
            continue; // cheirality: skip points behind this camera
        }
        let (px, py) = (pc.x / pc.z, pc.y / pc.z);
        let k = &cameras[o.pose_idx];
        let dx = (px - o.pixel[0] as f64) * k.fx;
        let dy = (py - o.pixel[1] as f64) * k.fy;
        se += dx * dx + dy * dy;
        nobs += 1;
    }
    let reproj_rmse_px = if nobs > 0 {
        (se / nobs as f64).sqrt()
    } else {
        -1.0
    };

    let poses: Vec<Option<Pose3d>> = (0..cameras.len())
        .map(|c| {
            if have[c] {
                Some(res.poses[c].inverse())
            } else {
                None
            }
        })
        .collect();

    // Per-camera covariance + observability from the fixed-point (board/gauge) reprojections.
    let per_camera = covariance::camera_stats(cameras, &res.poses, &obs2, &res.points, have);

    Ok(RigCalibration {
        poses,
        reference_tag_id,
        reproj_rmse_px,
        per_camera,
    })
}

/// Calibrate a multi-camera rig from a KNOWN rigid AprilGrid [`AprilGridBoard`] (and optional feature
/// matches).
///
/// The board is the world frame: every observed tag corner is a fixed metric anchor (no free reference
/// frame, no `2^N` branch search), which constrains rotation far better than a single tag. Returns
/// per-camera `T_world_cam` (world = board frame). Feature `tracks` (build them with
/// [`build_tracks`]) are optional and add cross-camera coverage / robustness.
pub fn calibrate_board(
    cameras: &[PinholeCamera],
    tags: &[TagObservation],
    tracks: &[FeatureTrack],
    board: &AprilGridBoard,
    config: &CalibConfig,
) -> Result<RigCalibration, CalibError> {
    if tags.is_empty() {
        return Err(CalibError::NoTags);
    }
    let (poses, have) = init::init_poses_board(cameras, tags, board);
    if !have.iter().any(|&h| h) {
        return Err(CalibError::NoReferenceTagView);
    }
    // Informational only (world IS the board, not a tag): the lowest observed board tag id.
    let ref_id = tags
        .iter()
        .filter(|t| board.contains(t.tag_id))
        .map(|t| t.tag_id)
        .min()
        .unwrap_or(0);
    let (points, obs) =
        assemble::assemble_board(cameras, tags, tracks, board, &poses, &have, config);
    finalize(cameras, &poses, &points, &obs, &have, ref_id, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

    fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
        PinholeCamera {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Project a world point through a `T_world_cam` pose + pinhole K → pixel.
    fn project(pw: Vec3F64, t_world_cam: &Pose3d, k: &PinholeCamera) -> Vec2F64 {
        let wc = t_world_cam.inverse(); // world→cam
        let pc = wc.transform_point(&pw);
        Vec2F64::new(k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy)
    }

    fn yaw_pose(yaw: f64, t: Vec3F64) -> Pose3d {
        let r = Mat3F64::from_cols(
            Vec3F64::new(yaw.cos(), 0.0, -yaw.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(yaw.sin(), 0.0, yaw.cos()),
        );
        Pose3d::new(r, t)
    }

    /// aruco-wound (TL, TR, BR, BL) square of half-size `s` at z=0 in a tag frame.
    fn square(s: f64) -> [Vec3F64; 4] {
        [
            Vec3F64::new(-s, s, 0.0),
            Vec3F64::new(s, s, 0.0),
            Vec3F64::new(s, -s, 0.0),
            Vec3F64::new(-s, -s, 0.0),
        ]
    }

    fn corners_px(w: &[Vec3F64; 4], pose: &Pose3d, k: &PinholeCamera) -> [Vec2F64; 4] {
        [
            project(w[0], pose, k),
            project(w[1], pose, k),
            project(w[2], pose, k),
            project(w[3], pose, k),
        ]
    }

    /// Rotation angle (rad) and translation error between two relative poses.
    fn rel_error(est_a: &Pose3d, est_b: &Pose3d, true_a: &Pose3d, true_b: &Pose3d) -> (f64, f64) {
        let rel_est = Pose3d::between(&est_a.inverse(), &est_b.inverse());
        let rel_true = Pose3d::between(&true_a.inverse(), &true_b.inverse());
        let dt = (rel_est.translation - rel_true.translation).length();
        // R·R̂ᵀ ≈ I  →  angle from the trace.
        let m = rel_est.rotation * rel_true.rotation.transpose();
        let trace = m.to_cols_array()[0] + m.to_cols_array()[4] + m.to_cols_array()[8];
        let dr = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        (dr, dt)
    }

    #[test]
    fn recovers_two_camera_pose_no_depth() {
        // Two cameras ~36 cm apart, nearly parallel, both looking +z at a tag ~2.2 m away.
        let ka = pinhole(384.0, 384.0, 320.0, 180.0);
        let kb = pinhole(516.0, 516.0, 320.0, 180.0);
        let pose_a = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.2));
        let pose_b = yaw_pose(0.05, Vec3F64::new(0.35, 0.02, -2.15));

        let s = 0.08;
        let sq = square(s);
        let tag = TagObservation {
            tag_id: 0,
            per_camera: vec![
                (0, corners_px(&sq, &pose_a, &ka)),
                (1, corners_px(&sq, &pose_b, &kb)),
            ],
        };

        // Non-coplanar scene points → feature matches (resolve the rotation).
        let pts = [
            Vec3F64::new(-0.4, 0.3, 0.4),
            Vec3F64::new(0.5, -0.2, 0.9),
            Vec3F64::new(0.1, 0.4, -0.2),
            Vec3F64::new(-0.3, -0.35, 0.2),
            Vec3F64::new(0.45, 0.25, 0.6),
            Vec3F64::new(-0.1, -0.1, 1.2),
            Vec3F64::new(0.3, 0.1, 0.0),
            Vec3F64::new(-0.45, 0.05, 0.8),
        ];
        let features: Vec<FeatureMatch> = pts
            .iter()
            .map(|p| FeatureMatch {
                cam_a: 0,
                cam_b: 1,
                uv_a: project(*p, &pose_a, &ka),
                uv_b: project(*p, &pose_b, &kb),
            })
            .collect();

        let cfg = CalibConfig::new(2.0 * s);
        let out = calibrate_apriltag(&[ka, kb], &[tag], &features, &cfg).unwrap();
        assert!(
            out.reproj_rmse_px >= 0.0 && out.reproj_rmse_px < 0.5,
            "rms {}",
            out.reproj_rmse_px
        );
        let (dr, dt) = rel_error(
            out.poses[0].as_ref().unwrap(),
            out.poses[1].as_ref().unwrap(),
            &pose_a,
            &pose_b,
        );
        assert!(dt < 0.02, "baseline error {dt} m");
        assert!(dr < 0.02, "rotation error {dr} rad");
    }

    // Regression for the "all tags pinned to one origin" bug: TWO tags at
    // DIFFERENT physical positions, both seen by both cameras, NO features.
    #[test]
    fn two_tags_at_different_positions_recover_pose() {
        let ka = pinhole(384.0, 384.0, 320.0, 180.0);
        let kb = pinhole(516.0, 516.0, 320.0, 180.0);
        let pose_a = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.2));
        let pose_b = yaw_pose(0.08, Vec3F64::new(0.35, 0.02, -2.15));

        let s = 0.08;
        let sq = square(s);
        let off = |c: Vec3F64, d: [f64; 3]| Vec3F64::new(c.x + d[0], c.y + d[1], c.z + d[2]);
        let a_world = sq;
        let d = [0.5, 0.05, 0.2];
        let b_world = [off(sq[0], d), off(sq[1], d), off(sq[2], d), off(sq[3], d)];

        let tag_a = TagObservation {
            tag_id: 0,
            per_camera: vec![
                (0, corners_px(&a_world, &pose_a, &ka)),
                (1, corners_px(&a_world, &pose_b, &kb)),
            ],
        };
        let tag_b = TagObservation {
            tag_id: 1,
            per_camera: vec![
                (0, corners_px(&b_world, &pose_a, &ka)),
                (1, corners_px(&b_world, &pose_b, &kb)),
            ],
        };

        let cfg = CalibConfig::new(2.0 * s);
        let out = calibrate_apriltag(&[ka, kb], &[tag_a, tag_b], &[], &cfg).unwrap();
        assert!(
            out.reproj_rmse_px >= 0.0 && out.reproj_rmse_px < 1.0,
            "rms {}",
            out.reproj_rmse_px
        );
        let (dr, dt) = rel_error(
            out.poses[0].as_ref().unwrap(),
            out.poses[1].as_ref().unwrap(),
            &pose_a,
            &pose_b,
        );
        assert!(dt < 0.03, "baseline error {dt} m");
        assert!(dr < 0.03, "rotation error {dr} rad");
    }

    #[test]
    fn rebased_puts_reference_at_gauge() {
        let p0 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.2));
        let p1 = yaw_pose(0.1, Vec3F64::new(0.3, 0.0, -2.1));
        let cal = RigCalibration {
            poses: vec![Some(p0), Some(p1), None],
            reference_tag_id: 0,
            reproj_rmse_px: 0.0,
            per_camera: vec![],
        };

        // No gauge → the reference camera maps to identity.
        let based = cal.rebased(1, None).unwrap();
        let r1 = based[1].unwrap();
        assert!((r1.translation).length() < 1e-9);
        let m = r1.rotation * Mat3F64::IDENTITY.transpose();
        let trace = m.to_cols_array()[0] + m.to_cols_array()[4] + m.to_cols_array()[8];
        assert!(
            (trace - 3.0).abs() < 1e-9,
            "reference rotation should be identity"
        );

        // Relative geometry preserved: based[0] == p1⁻¹ ∘ p0.
        let expect0 = p1.inverse().compose(&p0);
        let b0 = based[0].unwrap();
        assert!((b0.translation - expect0.translation).length() < 1e-9);

        // Unsolved camera stays None; unsolved reference errors.
        assert!(based[2].is_none());
        assert!(cal.rebased(2, None).is_err());
    }

    #[test]
    fn board_recovers_three_camera_rig_tightly() {
        // A rigid 3x3 AprilGrid removes the single-tag rotation degeneracy: every corner is a fixed
        // anchor, so rotation is recovered near-exactly with NO feature matches.
        let ks = [
            pinhole(400.0, 400.0, 320.0, 240.0),
            pinhole(450.0, 450.0, 320.0, 240.0),
            pinhole(500.0, 500.0, 320.0, 240.0),
        ];
        let poses = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.5)),
            yaw_pose(0.10, Vec3F64::new(-0.30, 0.0, -2.4)),
            yaw_pose(-0.08, Vec3F64::new(0.30, -0.05, -2.6)),
        ];
        let board = AprilGridBoard::new(3, 3, 0.08, 0.02); // extent ~0.28 m

        let mut tags = Vec::new();
        for id in 0..9u16 {
            let op = board.object_points(id).unwrap();
            let per_camera: Vec<(usize, [Vec2F64; 4])> = (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(op[0], &poses[c], &ks[c]),
                            project(op[1], &poses[c], &ks[c]),
                            project(op[2], &poses[c], &ks[c]),
                            project(op[3], &poses[c], &ks[c]),
                        ],
                    )
                })
                .collect();
            tags.push(TagObservation {
                tag_id: id,
                per_camera,
            });
        }

        let cfg = CalibConfig::new(board.tag_size_m);
        let out = calibrate_board(&ks, &tags, &[], &board, &cfg).unwrap();
        assert!(
            out.reproj_rmse_px >= 0.0 && out.reproj_rmse_px < 0.5,
            "rms {}",
            out.reproj_rmse_px
        );
        for c in 1..3 {
            let (dr, dt) = rel_error(
                out.poses[0].as_ref().unwrap(),
                out.poses[c].as_ref().unwrap(),
                &poses[0],
                &poses[c],
            );
            assert!(dt < 0.01, "cam {c} baseline error {dt} m");
            assert!(dr < 0.005, "cam {c} rotation error {dr} rad"); // tight: board constrains rotation
        }
    }

    #[test]
    fn observability_flags_single_tag_degeneracy() {
        // The non-dimensionalized pose-Hessian min-eigenvalue is a pure-geometry observability
        // metric. A rigid board (spread fixed corners) is far better observed than a single planar
        // tag (4 coplanar corners — the in-plane-tilt degeneracy this whole system fixes).
        let ks = [
            pinhole(400.0, 400.0, 320.0, 240.0),
            pinhole(450.0, 450.0, 320.0, 240.0),
            pinhole(500.0, 500.0, 320.0, 240.0),
        ];
        let poses = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.5)),
            yaw_pose(0.10, Vec3F64::new(-0.30, 0.0, -2.4)),
            yaw_pose(-0.08, Vec3F64::new(0.30, -0.05, -2.6)),
        ];
        let board = AprilGridBoard::new(3, 3, 0.08, 0.02);
        let mut tags = Vec::new();
        for id in 0..9u16 {
            let op = board.object_points(id).unwrap();
            let per_camera: Vec<(usize, [Vec2F64; 4])> = (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(op[0], &poses[c], &ks[c]),
                            project(op[1], &poses[c], &ks[c]),
                            project(op[2], &poses[c], &ks[c]),
                            project(op[3], &poses[c], &ks[c]),
                        ],
                    )
                })
                .collect();
            tags.push(TagObservation {
                tag_id: id,
                per_camera,
            });
        }
        let cfg = CalibConfig::new(board.tag_size_m);
        let board_cal = calibrate_board(&ks, &tags, &[], &board, &cfg).unwrap();
        let board_min_eig = board_cal
            .per_camera
            .iter()
            .filter(|s| s.registered)
            .map(|s| s.min_eigenvalue)
            .fold(f64::INFINITY, f64::min);
        // All board cameras are well-observed.
        assert_eq!(board_cal.per_camera.len(), 3);
        for s in &board_cal.per_camera {
            assert!(s.registered && s.num_obs > 0);
            assert!(s.min_eigenvalue > 0.0 && s.rot_sigma_deg.is_finite());
        }

        // A single centered planar tag at the world origin → only 4 coplanar fixed corners.
        let sq = square(board.tag_size_m / 2.0);
        let one_tag = TagObservation {
            tag_id: 0,
            per_camera: (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(sq[0], &poses[c], &ks[c]),
                            project(sq[1], &poses[c], &ks[c]),
                            project(sq[2], &poses[c], &ks[c]),
                            project(sq[3], &poses[c], &ks[c]),
                        ],
                    )
                })
                .collect(),
        };
        let tag_cal = calibrate_apriltag(&ks, &[one_tag], &[], &cfg).unwrap();
        let tag_min_eig = tag_cal
            .per_camera
            .iter()
            .filter(|s| s.registered)
            .map(|s| s.min_eigenvalue)
            .fold(f64::INFINITY, f64::min);

        assert!(
            board_min_eig > tag_min_eig,
            "board observability {board_min_eig} should beat single-tag {tag_min_eig}"
        );
    }

    #[test]
    fn x84_threshold_math() {
        // Too few samples → no removal.
        assert_eq!(x84_threshold(&mut [1.0, 2.0, 3.0], 2.5), f64::INFINITY);
        // Constant residuals (MAD = 0) → no removal (near-perfect fit).
        assert_eq!(x84_threshold(&mut [0.5; 20], 2.5), f64::INFINITY);
        // Spread data → finite cutoff above the median.
        let mut d: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let thr = x84_threshold(&mut d, 2.5);
        assert!(thr.is_finite() && thr > 0.9, "thr {thr}");
    }

    #[test]
    fn board_tolerates_outlier_track() {
        // Board + many good feature tracks + one gross-outlier track (garbage pixel in cam 2) →
        // the two-pass robust BA still recovers rotation tightly.
        let ks = [
            pinhole(400.0, 400.0, 320.0, 240.0),
            pinhole(450.0, 450.0, 320.0, 240.0),
            pinhole(500.0, 500.0, 320.0, 240.0),
        ];
        let poses = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.5)),
            yaw_pose(0.10, Vec3F64::new(-0.30, 0.0, -2.4)),
            yaw_pose(-0.08, Vec3F64::new(0.30, -0.05, -2.6)),
        ];
        let board = AprilGridBoard::new(3, 3, 0.08, 0.02);
        let mut tags = Vec::new();
        for id in 0..9u16 {
            let op = board.object_points(id).unwrap();
            let per_camera: Vec<(usize, [Vec2F64; 4])> = (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(op[0], &poses[c], &ks[c]),
                            project(op[1], &poses[c], &ks[c]),
                            project(op[2], &poses[c], &ks[c]),
                            project(op[3], &poses[c], &ks[c]),
                        ],
                    )
                })
                .collect();
            tags.push(TagObservation {
                tag_id: id,
                per_camera,
            });
        }

        // 12 good 3-view tracks (deterministic scene points).
        let mut tracks: Vec<FeatureTrack> = (0..12)
            .map(|i| {
                let t = i as f64;
                let p = Vec3F64::new(
                    0.12 * (t * 0.7).sin(),
                    0.15 * (t * 1.3).cos(),
                    0.35 + 0.04 * t.sin(),
                );
                FeatureTrack {
                    obs: (0..3).map(|c| (c, project(p, &poses[c], &ks[c]))).collect(),
                }
            })
            .collect();
        // One outlier track: correct in cams 0/1, garbage pixel in cam 2.
        let po = Vec3F64::new(0.0, 0.0, 0.4);
        tracks.push(FeatureTrack {
            obs: vec![
                (0, project(po, &poses[0], &ks[0])),
                (1, project(po, &poses[1], &ks[1])),
                (2, Vec2F64::new(600.0, 40.0)),
            ],
        });

        let cfg = CalibConfig::new(board.tag_size_m);
        let out = calibrate_board(&ks, &tags, &tracks, &board, &cfg).unwrap();
        for c in 1..3 {
            let (dr, dt) = rel_error(
                out.poses[0].as_ref().unwrap(),
                out.poses[c].as_ref().unwrap(),
                &poses[0],
                &poses[c],
            );
            assert!(dt < 0.01, "cam {c} baseline error {dt} m");
            assert!(dr < 0.005, "cam {c} rotation error {dr} rad");
        }
    }

    #[test]
    fn board_with_feature_tracks_still_recovers() {
        // Multi-view tracks are consumed alongside the board without breaking the solve (the board
        // dominates; tracks add coverage). Exercises build_tracks + calibrate_board(tracks).
        let ks = [
            pinhole(400.0, 400.0, 320.0, 240.0),
            pinhole(450.0, 450.0, 320.0, 240.0),
            pinhole(500.0, 500.0, 320.0, 240.0),
        ];
        let poses = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.5)),
            yaw_pose(0.10, Vec3F64::new(-0.30, 0.0, -2.4)),
            yaw_pose(-0.08, Vec3F64::new(0.30, -0.05, -2.6)),
        ];
        let board = AprilGridBoard::new(3, 3, 0.08, 0.02);

        let mut tags = Vec::new();
        for id in 0..9u16 {
            let op = board.object_points(id).unwrap();
            let per_camera: Vec<(usize, [Vec2F64; 4])> = (0..3)
                .map(|c| {
                    (
                        c,
                        [
                            project(op[0], &poses[c], &ks[c]),
                            project(op[1], &poses[c], &ks[c]),
                            project(op[2], &poses[c], &ks[c]),
                            project(op[3], &poses[c], &ks[c]),
                        ],
                    )
                })
                .collect();
            tags.push(TagObservation {
                tag_id: id,
                per_camera,
            });
        }

        // Non-coplanar scene points → pairwise edges across all 3 cameras → 3-view tracks.
        let pts3d = [
            Vec3F64::new(0.10, 0.10, 0.4),
            Vec3F64::new(0.20, 0.05, -0.3),
            Vec3F64::new(-0.05, 0.20, 0.5),
        ];
        let mut edges = Vec::new();
        for (pi, p) in pts3d.iter().enumerate() {
            for &(a, b) in &[(0usize, 1usize), (1, 2), (0, 2)] {
                edges.push(TrackEdge {
                    cam_a: a,
                    kpt_a: pi as u32,
                    uv_a: project(*p, &poses[a], &ks[a]),
                    cam_b: b,
                    kpt_b: pi as u32,
                    uv_b: project(*p, &poses[b], &ks[b]),
                });
            }
        }
        let tracks = build_tracks(&edges);
        assert_eq!(tracks.len(), 3);
        assert!(tracks.iter().all(|t| t.obs.len() == 3));

        let cfg = CalibConfig::new(board.tag_size_m);
        let out = calibrate_board(&ks, &tags, &tracks, &board, &cfg).unwrap();
        assert!(
            out.reproj_rmse_px >= 0.0 && out.reproj_rmse_px < 0.5,
            "rms {}",
            out.reproj_rmse_px
        );
        for c in 1..3 {
            let (dr, dt) = rel_error(
                out.poses[0].as_ref().unwrap(),
                out.poses[c].as_ref().unwrap(),
                &poses[0],
                &poses[c],
            );
            assert!(dt < 0.01, "cam {c} baseline error {dt} m");
            assert!(dr < 0.005, "cam {c} rotation error {dr} rad");
        }
    }

    #[test]
    fn three_cameras_two_tags_recover_all_poses() {
        let ks = [
            pinhole(400.0, 400.0, 320.0, 240.0),
            pinhole(450.0, 450.0, 320.0, 240.0),
            pinhole(500.0, 500.0, 320.0, 240.0),
        ];
        let poses = [
            Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.5)),
            yaw_pose(0.10, Vec3F64::new(-0.40, 0.0, -2.4)),
            yaw_pose(-0.08, Vec3F64::new(0.35, -0.05, -2.6)),
        ];

        let s = 0.08;
        let sq = square(s);
        let off = |c: Vec3F64, d: [f64; 3]| Vec3F64::new(c.x + d[0], c.y + d[1], c.z + d[2]);
        let d = [0.4, 0.1, 0.15];
        let b_world = [off(sq[0], d), off(sq[1], d), off(sq[2], d), off(sq[3], d)];

        let mk = |w: &[Vec3F64; 4], id: u16| TagObservation {
            tag_id: id,
            per_camera: (0..3)
                .map(|c| (c, corners_px(w, &poses[c], &ks[c])))
                .collect(),
        };
        let tags = [mk(&sq, 0), mk(&b_world, 1)];

        let cfg = CalibConfig::new(2.0 * s);
        let out = calibrate_apriltag(&ks, &tags, &[], &cfg).unwrap();
        assert!(
            out.reproj_rmse_px >= 0.0 && out.reproj_rmse_px < 1.0,
            "rms {}",
            out.reproj_rmse_px
        );
        assert!(out.reference_tag_id == 0 || out.reference_tag_id == 1);
        for c in 1..3 {
            let (dr, dt) = rel_error(
                out.poses[0].as_ref().unwrap(),
                out.poses[c].as_ref().unwrap(),
                &poses[0],
                &poses[c],
            );
            assert!(dt < 0.03, "cam {c} baseline error {dt} m");
            assert!(dr < 0.03, "cam {c} rotation error {dr} rad");
        }
    }
}
