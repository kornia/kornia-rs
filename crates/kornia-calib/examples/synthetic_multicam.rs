//! Minimal synthetic-scene example for `kornia-calib`.
//!
//! Builds a ground-truth 3-camera rig looking at a planar grid of AprilTags, projects every tag
//! corner into each camera, then recovers the rig extrinsics with [`calibrate_apriltag`] — WITHOUT
//! telling the solver the board layout (the tags' rigid arrangement is auto-measured). Prints the
//! reprojection RMS, the recovered camera baselines vs ground truth, and the per-camera observability.
//!
//! Run: `cargo run -p kornia-calib --example synthetic_multicam`

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_calib::{calibrate_apriltag, BoardGeometry, CalibConfig, TagObservation};

/// Square pixels, principal point at image centre (640x480), no distortion.
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

/// Rotation from yaw (about Y) then pitch (about X) — gives each camera an oblique view of the board
/// (the perspective foreshortening is what makes the geometry well-conditioned).
fn rot(yaw: f64, pitch: f64) -> Mat3F64 {
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    Mat3F64::from_cols(
        Vec3F64::new(cy, 0.0, -sy),
        Vec3F64::new(sy * sp, cp, cy * sp),
        Vec3F64::new(sy * cp, -sp, cy * cp),
    )
}

/// Pinhole projection of a world point through a `T_world_cam` (world→camera) pose.
fn project(pw: Vec3F64, t_world_cam: &Pose3d, k: &PinholeCamera) -> Vec2F64 {
    let pc = t_world_cam.transform_point(&pw);
    Vec2F64::new(k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy)
}

fn main() {
    // --- Ground-truth scene ------------------------------------------------------------------
    // 3 cameras (different focals) viewing a 3x3, 8 cm AprilGrid board ~2.5 m away from oblique angles.
    let cams = [pinhole(500.0), pinhole(480.0), pinhole(520.0)];
    let gt_pose = [
        Pose3d::new(rot(0.20, 0.12), Vec3F64::new(0.0, 0.0, 2.5)),
        Pose3d::new(rot(-0.25, 0.15), Vec3F64::new(-0.35, 0.0, 2.4)),
        Pose3d::new(rot(0.30, -0.10), Vec3F64::new(0.35, -0.05, 2.6)),
    ];
    let tag_size = 0.08;
    let board = BoardGeometry::april_grid(3, 3, tag_size, 0.02); // used only to GENERATE the scene

    // Project every board-tag corner into every camera → one TagObservation per tag id. Note the
    // solver is NOT given the board — it recovers the rigid layout from these observations.
    let mut tags = Vec::new();
    for id in 0..9u16 {
        let op = board.object_points(id).unwrap();
        let per_camera: Vec<(usize, [Vec2F64; 4])> = (0..3)
            .map(|c| {
                (
                    c,
                    [
                        project(op[0], &gt_pose[c], &cams[c]),
                        project(op[1], &gt_pose[c], &cams[c]),
                        project(op[2], &gt_pose[c], &cams[c]),
                        project(op[3], &gt_pose[c], &cams[c]),
                    ],
                )
            })
            .collect();
        tags.push(TagObservation {
            tag_id: id,
            per_camera,
        });
    }

    // --- Calibrate ---------------------------------------------------------------------------
    let cfg = CalibConfig::new(tag_size);
    let cal = calibrate_apriltag(&cams, &tags, &[], &cfg).expect("calibration failed");

    println!("reference tag id : {}", cal.reference_tag_id);
    println!("reprojection RMS : {:.4} px", cal.reproj_rmse_px);

    // Metric check: camera-to-camera baselines (translation is gauge-free once we compare distances).
    // Recovered poses are T_world_cam (camera optical → world), so their translation IS the camera
    // centre in the world; ground-truth poses are world→cam, so invert for the centre.
    let rec_c: Vec<Vec3F64> = cal
        .poses
        .iter()
        .map(|p| p.as_ref().map(|q| q.translation).unwrap_or(Vec3F64::ZERO))
        .collect();
    // Recovered poses are T_world_cam (their translation is the camera centre in world); ground-truth
    // poses are world→cam, so invert for the centre.
    let gt_c: Vec<Vec3F64> = gt_pose.iter().map(|p| p.inverse().translation).collect();
    let dist = |a: &Vec3F64, b: &Vec3F64| (*a - *b).length();
    println!("\nbaseline (m)      recovered   ground-truth   error");
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let r = dist(&rec_c[i], &rec_c[j]);
        let g = dist(&gt_c[i], &gt_c[j]);
        println!(
            "  cam{i}-cam{j}         {r:.4}      {g:.4}       {:.1} mm",
            (r - g).abs() * 1000.0
        );
    }

    // Per-camera observability (from the fixed tag-corner reprojections): rotation/translation σ and
    // the min-eigenvalue that would flag a weak (e.g. single-planar-tag) geometry.
    println!("\nper-camera observability");
    for s in &cal.per_camera {
        if !s.registered {
            println!("  cam{}: not registered", s.camera);
            continue;
        }
        println!(
            "  cam{}: obs={:>2}  reproj={:.3}px  rotσ={:.3}°  transσ={:.1}mm  min_eig={:.2e}",
            s.camera,
            s.num_obs,
            s.reproj_rmse_px,
            s.rot_sigma_deg,
            s.trans_sigma_m * 1000.0,
            s.min_eigenvalue.unwrap_or(f64::NAN),
        );
    }
}
