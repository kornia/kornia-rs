//! Multi-shot rig calibration: refine focal + strengthen extrinsics from N snapshots of a moving board.
//!
//! Single-snapshot calibration cannot refine intrinsics: with one planar view per camera, focal is
//! absorbed by the pose (board→image is a homography). Capturing the board at **several poses** and
//! feeding every shot into calibration breaks that coupling — the classic reason COLMAP/Kalibr can
//! self-calibrate. This module implements the tractable, well-conditioned decomposition:
//!
//! 1. **Focal** — for each camera, `estimate_focal` (Zhang's homography-orthogonality) is run on every
//!    shot's tag view; the **median** across shots is a robust multi-view focal. One view is noisy;
//!    a dozen views' median is not. (Principal point + distortion are left fixed — refining them well
//!    needs the full joint Zhang LM; a future extension.)
//! 2. **Extrinsics** — the proven single-shot [`crate::calibrate_apriltag`] auto-board solver runs on
//!    each shot with the refined focal, and the per-shot rig poses are **robustly averaged** (rotation
//!    Karcher mean + translation mean) into a common anchor frame. The **spread across shots** is a
//!    real empirical covariance — repeatability, not a linearized guess.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64, SO3F64};

use crate::board::BoardGeometry;
use crate::error::CalibError;
use crate::intrinsics::estimate_focal;
use crate::types::{CalibConfig, CameraStats, TagObservation};

/// One captured shot: the tags each camera saw with the board at a single (unknown) pose. Move the
/// board between shots so each camera observes it from several angles.
pub struct Shot {
    /// Tags observed in this shot (each listing the cameras that saw it), same shape as a single-shot
    /// solve's `tags`.
    pub tags: Vec<TagObservation>,
}

/// Configuration for [`calibrate_multishot`].
pub struct MultiShotConfig {
    /// Per-shot solver configuration (tag size, robust params, …).
    pub base: CalibConfig,
    /// Refine each camera's focal from the multi-shot homographies. When off, intrinsics pass through.
    pub refine_focal: bool,
    /// Minimum per-camera focal samples (shots seeing a tag) required to trust the refined focal.
    pub min_focal_samples: usize,
    /// Known rigid board layout. `Some` → each shot solves with [`crate::calibrate_board`] (fixed grid
    /// anchors); `None` → [`crate::calibrate_apriltag`] auto-measures the tag arrangement per shot.
    pub board: Option<BoardGeometry>,
}

impl MultiShotConfig {
    /// Defaults for the given tag size (metres): focal refinement on, ≥3 samples required.
    pub fn new(tag_size_m: f64) -> Self {
        Self {
            base: CalibConfig::new(tag_size_m),
            refine_focal: true,
            min_focal_samples: 3,
            board: None,
        }
    }
}

/// Result of a multi-shot calibration.
pub struct MultiShotCalibration {
    /// Per-camera pose `T_anchor_cam` in the anchor camera's frame (anchor = identity). `None` for a
    /// camera that registered in no shot.
    pub poses: Vec<Option<Pose3d>>,
    /// Camera whose frame the poses are expressed in (the one registered in the most shots).
    pub anchor_cam: usize,
    /// Refined focal (`fx = fy`, pixels) per camera, or `None` where it was not refined (too few
    /// samples / refinement off) and the input focal is kept.
    pub focal: Vec<Option<f64>>,
    /// Per-camera **empirical** covariance across shots (rotation/translation sample spread).
    pub per_camera: Vec<CameraStats>,
    /// Mean per-shot reprojection RMS in pixels.
    pub reproj_rmse_px: f64,
    /// Number of shots that produced a usable single-shot solve.
    pub num_shots: usize,
}

/// Calibrate a rig from multiple shots of a moving board: refine per-camera focal and average the
/// extrinsics across shots. See the module docs for the method.
///
/// `cameras[i]` holds each camera's initial intrinsics (the focal is a starting point when
/// `refine_focal` is on). Returns poses in the anchor camera's frame — re-gauge as needed.
pub fn calibrate_multishot(
    cameras: &[PinholeCamera],
    shots: &[Shot],
    config: &MultiShotConfig,
) -> Result<MultiShotCalibration, CalibError> {
    let n_cams = cameras.len();
    if shots.is_empty() {
        return Err(CalibError::NoTags);
    }

    // 1. Refine focal per camera: median of the per-shot Zhang focal estimates.
    let s = config.base.tag_size_m;
    let square = [
        Vec2F64::new(0.0, 0.0),
        Vec2F64::new(s, 0.0),
        Vec2F64::new(s, s),
        Vec2F64::new(0.0, s),
    ];
    let mut focal: Vec<Option<f64>> = vec![None; n_cams];
    if config.refine_focal {
        for c in 0..n_cams {
            let (cx, cy) = (cameras[c].cx, cameras[c].cy);
            let mut fs: Vec<f64> = Vec::new();
            for shot in shots {
                for tag in &shot.tags {
                    if let Some((_, corners)) = tag.per_camera.iter().find(|(ci, _)| *ci == c) {
                        if let Some(f) = estimate_focal(&square, corners, cx, cy) {
                            fs.push(f);
                        }
                    }
                }
            }
            if fs.len() >= config.min_focal_samples {
                fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                focal[c] = Some(fs[fs.len() / 2]);
            }
        }
    }

    // Refined cameras: replace focal where estimated, keep principal point + distortion.
    let refined: Vec<PinholeCamera> = (0..n_cams)
        .map(|c| {
            let mut k = cameras[c].clone();
            if let Some(f) = focal[c] {
                k.fx = f;
                k.fy = f;
            }
            k
        })
        .collect();

    // 2. Per-shot single-shot solve, rebased into each shot's own reference-camera frame. Collect the
    // per-camera pose relative to a common anchor across shots.
    // Anchor = the camera registered in the most shots (so the most poses are comparable).
    let mut per_shot_poses: Vec<Vec<Option<Pose3d>>> = Vec::new(); // [shot][cam] = T_world_cam
    let mut rms_sum = 0.0;
    let mut rms_n = 0usize;
    for shot in shots {
        let solved = match &config.board {
            Some(board) => crate::calibrate_board(&refined, &shot.tags, &[], board, &config.base),
            None => crate::calibrate_apriltag(&refined, &shot.tags, &[], &config.base),
        };
        match solved {
            Ok(cal) => {
                per_shot_poses.push(cal.poses);
                if cal.reproj_rmse_px >= 0.0 {
                    rms_sum += cal.reproj_rmse_px;
                    rms_n += 1;
                }
            }
            Err(_) => continue, // a shot with no usable tag view is skipped, not fatal
        }
    }
    if per_shot_poses.is_empty() {
        return Err(CalibError::NoReferenceTagView);
    }

    // Count registrations per camera to pick the anchor.
    let mut reg_count = vec![0usize; n_cams];
    for ps in &per_shot_poses {
        for (c, p) in ps.iter().enumerate() {
            if p.is_some() {
                reg_count[c] += 1;
            }
        }
    }
    let anchor_cam = (0..n_cams).max_by_key(|&c| reg_count[c]).unwrap_or(0);

    // 3. For each shot where the anchor registered, rebase every camera into the anchor frame:
    //    T_anchor_cam = T_world_anchor⁻¹ ∘ T_world_cam. Accumulate per camera across shots.
    let mut samples: Vec<Vec<Pose3d>> = vec![Vec::new(); n_cams];
    for ps in &per_shot_poses {
        let Some(anchor_pose) = ps[anchor_cam] else {
            continue; // anchor not in this shot → poses not comparable, skip
        };
        let anchor_inv = anchor_pose.inverse();
        for (c, p) in ps.iter().enumerate() {
            if let Some(pc) = p {
                samples[c].push(anchor_inv.compose(pc));
            }
        }
    }

    // 4. Robustly combine per-camera pose samples; empirical covariance from the spread.
    let mut poses: Vec<Option<Pose3d>> = vec![None; n_cams];
    let mut per_camera: Vec<CameraStats> = Vec::with_capacity(n_cams);
    for c in 0..n_cams {
        let sm = &samples[c];
        if sm.is_empty() {
            per_camera.push(CameraStats::unconstrained(c, false, 0));
            continue;
        }
        let mean_rot = rotation_mean(sm.iter().map(|p| p.rotation));
        let mut t = [0.0f64; 3];
        for p in sm {
            t[0] += p.translation.x;
            t[1] += p.translation.y;
            t[2] += p.translation.z;
        }
        let inv = 1.0 / sm.len() as f64;
        let mean_t = Vec3F64::new(t[0] * inv, t[1] * inv, t[2] * inv);
        poses[c] = Some(Pose3d::new(mean_rot, mean_t));

        // Empirical spread: rotation = RMS geodesic angle from the mean; translation = RMS distance.
        let mean_inv_rot = mean_rot.transpose();
        let (mut rot_ss, mut tr_ss) = (0.0f64, 0.0f64);
        for p in sm {
            let rel = mean_inv_rot * p.rotation;
            let ang = SO3F64::from_matrix(&rel).log();
            rot_ss += ang.x * ang.x + ang.y * ang.y + ang.z * ang.z;
            let dt = Vec3F64::new(
                p.translation.x - mean_t.x,
                p.translation.y - mean_t.y,
                p.translation.z - mean_t.z,
            );
            tr_ss += dt.x * dt.x + dt.y * dt.y + dt.z * dt.z;
        }
        let n = sm.len() as f64;
        let rot_sigma_deg = (rot_ss / n).sqrt().to_degrees();
        let trans_sigma_m = (tr_ss / n).sqrt();
        per_camera.push(CameraStats {
            camera: c,
            registered: true,
            num_obs: sm.len(),
            reproj_rmse_px: -1.0, // per-shot RMS is reported globally; empirical stats are the point
            rot_sigma_deg,
            trans_sigma_m,
            // Empirical spread, not a linearized Hessian — there is no eigenvalue/weak-DOF to report.
            // Consumers key on the repeatability σ above (a `None` here is not "unobservable").
            min_eigenvalue: None,
            weakest_dof: None,
        });
    }

    let reproj_rmse_px = if rms_n > 0 {
        rms_sum / rms_n as f64
    } else {
        -1.0
    };
    Ok(MultiShotCalibration {
        poses,
        anchor_cam,
        focal,
        per_camera,
        reproj_rmse_px,
        num_shots: per_shot_poses.len(),
    })
}

/// Karcher (geodesic L2) mean of rotation matrices: iterate `R ← R·exp(mean log(Rᵀ·Rᵢ))`. Converges
/// in a few steps for clustered rotations (the multi-shot case). Falls back to the first sample on an
/// empty/degenerate set.
fn rotation_mean(rots: impl Iterator<Item = Mat3F64>) -> Mat3F64 {
    let list: Vec<Mat3F64> = rots.collect();
    if list.is_empty() {
        return Mat3F64::IDENTITY;
    }
    let mut mean = list[0];
    for _ in 0..20 {
        let mean_inv = mean.transpose();
        let mut acc = [0.0f64; 3];
        for r in &list {
            let rel = mean_inv * *r;
            let w = SO3F64::from_matrix(&rel).log();
            acc[0] += w.x;
            acc[1] += w.y;
            acc[2] += w.z;
        }
        let inv = 1.0 / list.len() as f64;
        let step = Vec3F64::new(acc[0] * inv, acc[1] * inv, acc[2] * inv);
        let norm = (step.x * step.x + step.y * step.y + step.z * step.z).sqrt();
        mean *= SO3F64::exp(step).matrix();
        if norm < 1e-10 {
            break;
        }
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinhole(f: f64, cx: f64, cy: f64) -> PinholeCamera {
        PinholeCamera {
            fx: f,
            fy: f,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn proj(pw: Vec3F64, twc: &Pose3d, k: &PinholeCamera) -> Vec2F64 {
        let pc = twc.transform_point(&pw);
        Vec2F64::new(k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy)
    }

    // Rotation from yaw (about Y) then pitch (about X), by columns.
    fn rot(yaw: f64, pitch: f64) -> Mat3F64 {
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(sy * sp, cp, cy * sp),
            Vec3F64::new(sy * cp, -sp, cy * cp),
        )
    }

    #[test]
    fn multishot_recovers_wrong_focal() {
        // Two cameras with TRUE focals 500 / 480, seeded WRONG (420 / 560). A single planar view can't
        // recover focal (pose absorbs it) — but N shots of the board at different poses can.
        let ks_true = [pinhole(500.0, 320.0, 240.0), pinhole(480.0, 320.0, 240.0)];
        // Rig (world→cam): cam0 = identity, cam1 yawed + translated.
        let rig = [
            Pose3d::IDENTITY,
            Pose3d::new(rot(0.20, 0.0), Vec3F64::new(-0.30, 0.0, 0.02)),
        ];
        let tag = 0.10; // 10 cm
        let h = tag / 2.0;
        let local = [
            Vec3F64::new(-h, h, 0.0), // TL
            Vec3F64::new(h, h, 0.0),  // TR
            Vec3F64::new(h, -h, 0.0), // BR
            Vec3F64::new(-h, -h, 0.0),
        ];
        // 10 shots: board at varied oblique poses in front of both cameras (z ≈ 2 m).
        let mut shots: Vec<Shot> = Vec::new();
        for i in 0..10 {
            let a = i as f64;
            let tag_pose = Pose3d::new(
                rot(0.35 * (a * 0.7).sin(), 0.30 * (a * 1.1).cos()),
                Vec3F64::new(0.15 * (a * 0.9).cos(), 0.10 * (a).sin(), 2.0),
            );
            let world: Vec<Vec3F64> = local.iter().map(|p| tag_pose.transform_point(p)).collect();
            let c0: [Vec2F64; 4] = std::array::from_fn(|k| proj(world[k], &rig[0], &ks_true[0]));
            let c1: [Vec2F64; 4] = std::array::from_fn(|k| proj(world[k], &rig[1], &ks_true[1]));
            shots.push(Shot {
                tags: vec![TagObservation {
                    tag_id: 0,
                    per_camera: vec![(0, c0), (1, c1)],
                }],
            });
        }

        let ks_wrong = [pinhole(420.0, 320.0, 240.0), pinhole(560.0, 320.0, 240.0)];
        let cfg = MultiShotConfig::new(tag);
        let out = calibrate_multishot(&ks_wrong, &shots, &cfg).unwrap();

        let f0 = out.focal[0].expect("cam0 focal refined");
        let f1 = out.focal[1].expect("cam1 focal refined");
        assert!(
            (f0 - 500.0).abs() / 500.0 < 0.03,
            "cam0 focal {f0}, want 500"
        );
        assert!(
            (f1 - 480.0).abs() / 480.0 < 0.03,
            "cam1 focal {f1}, want 480"
        );
        // Extrinsics: the anchor camera maps to identity; the other camera's translation is the
        // rig baseline (‖rig[1].translation‖ ≈ 0.3007 m, invariant to which camera is the anchor).
        let other = 1 - out.anchor_cam;
        let p = out.poses[other].expect("other cam pose");
        let baseline = p.translation.length();
        assert!(
            (baseline - 0.30067).abs() < 0.02,
            "baseline {baseline}, want 0.30067"
        );
    }
}
