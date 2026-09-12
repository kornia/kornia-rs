//! Shared helpers for tests across sfm modules.
//!
//! Only compiled in test builds.

#![cfg(test)]

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_calib::FeatureTrack;

use crate::reconstruction::make_camera;

/// Intrinsics used by the synthetic scene.
pub const FX: f64 = 500.0;
pub const FY: f64 = 500.0;
pub const CX: f64 = 320.0;
pub const CY: f64 = 240.0;
/// Number of cameras in the synthetic scene.
pub const N_FRAMES: usize = 3;

/// A yaw/pitch rotation matrix (matches kornia-calib's own test helper).
pub fn rot(yaw: f64, pitch: f64) -> Mat3F64 {
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    Mat3F64::from_cols(
        Vec3F64::new(cy, 0.0, -sy),
        Vec3F64::new(sy * sp, cp, cy * sp),
        Vec3F64::new(sy * cp, -sp, cy * cp),
    )
}

/// Project a world point through a world→cam pose using the pinhole model.
pub fn project(pw: Vec3F64, pose_w2c: &Pose3d, cam: &kornia_3d::camera::PinholeCamera) -> Vec2F64 {
    let pc = pose_w2c.transform_point(&pw);
    Vec2F64::new(cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy)
}

/// Build a synthetic scene: a grid of 3D points ~2 m away viewed by three
/// cameras that converge on it (as a real overlapping-FOV rig would). One
/// track per point; a camera only contributes an observation when the point is
/// in front of it and inside the 640x480 image.
pub fn synthetic_tracks() -> Vec<FeatureTrack> {
    let cam = || make_camera(FX, FY, CX, CY);
    let cams = [cam(), cam(), cam()];
    let gt = [
        Pose3d::new(rot(0.0, 0.05), Vec3F64::new(0.0, 0.0, 0.0)),
        Pose3d::new(rot(0.40, 0.05), Vec3F64::new(-0.6, 0.0, 0.10)),
        Pose3d::new(rot(-0.40, 0.05), Vec3F64::new(0.6, 0.0, 0.15)),
    ];
    let (w, h) = (640.0, 480.0);
    let visible = |p: Vec3F64, c: usize| -> Option<Vec2F64> {
        let pc = gt[c].transform_point(&p);
        if pc.z <= 0.1 {
            return None;
        }
        let uv = project(p, &gt[c], &cams[c]);
        (uv.x >= 0.0 && uv.x < w && uv.y >= 0.0 && uv.y < h).then_some(uv)
    };

    let mut tracks = Vec::new();
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
    tracks
}
