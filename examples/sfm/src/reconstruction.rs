//! Structure-from-Motion reconstruction wrapper.
//!
//! Builds a [`PinholeCamera`] per video frame from user-supplied intrinsics
//! and drives `kornia_calib`'s incremental SfM (`reconstruct`) over the
//! feature tracks. Because no AprilTag anchors the scale, the recovered map is
//! [`ScaleSource::UpToScale`]: correct shape, arbitrary units.

use std::error::Error;

use kornia_3d::camera::PinholeCamera;
use kornia_calib::{reconstruct, FeatureTrack, Reconstruction, ReconstructionConfig};

/// Build a pinhole camera from intrinsics, with zero distortion.
pub fn make_camera(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        ..PinholeCamera::IDENTITY
    }
}

/// Run incremental SfM over the given feature tracks.
///
/// Each video frame is treated as one camera with the given intrinsics; the
/// `tracks[i].obs` camera indices index into that per-frame camera list. No
/// tag is supplied, so the reconstruction is up to scale.
///
/// # Arguments
///
/// * `tracks` - Multi-view feature tracks produced by `kornia_calib::build_tracks`.
/// * `fx` / `fy` - Focal lengths in pixels.
/// * `cx` / `cy` - Principal point in pixels.
/// * `n_frames` - Number of video frames (one camera per frame).
///
/// # Returns
///
/// The [`Reconstruction`] with per-frame poses and reconstructed 3D points.
///
/// # Errors
///
/// Returns an error if the reconstruction cannot bootstrap (e.g. no usable
/// tracks, or a degenerate camera configuration).
pub fn run_sfm(
    tracks: &[FeatureTrack],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    n_frames: usize,
) -> Result<Reconstruction, Box<dyn Error>> {
    let cameras = vec![make_camera(fx, fy, cx, cy); n_frames];
    // `.sequential()` tunes the config for video walkthroughs (smaller
    // parallax threshold, more BA iterations).
    let config = ReconstructionConfig::new(0.0).sequential();
    Ok(reconstruct(&cameras, &[], tracks, &config, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
    use kornia_calib::ScaleSource;

    #[test]
    fn make_camera_constructs_pinhole_camera() {
        let cam = make_camera(500.0, 520.0, 320.0, 240.0);
        assert_eq!(cam.fx, 500.0);
        assert_eq!(cam.fy, 520.0);
        assert_eq!(cam.cx, 320.0);
        assert_eq!(cam.cy, 240.0);
        assert_eq!(cam.k1, 0.0);
        assert_eq!(cam.k2, 0.0);
        assert_eq!(cam.p1, 0.0);
        assert_eq!(cam.p2, 0.0);
    }

    /// Project a world point through a world→cam pose using the pinhole model.
    fn project(pw: Vec3F64, pose_w2c: &Pose3d, cam: &PinholeCamera) -> Vec2F64 {
        let pc = pose_w2c.transform_point(&pw);
        Vec2F64::new(cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy)
    }

    /// A yaw/pitch rotation matrix (matches kornia-calib's own test helper).
    fn rot(yaw: f64, pitch: f64) -> Mat3F64 {
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(sy * sp, cp, cy * sp),
            Vec3F64::new(sy * cp, -sp, cy * cp),
        )
    }

    /// Build a synthetic scene: a grid of 3D points ~2 m away viewed by three
    /// cameras that converge on it (as a real overlapping-FOV rig would). One
    /// track per point; a camera only contributes an observation when the
    /// point is in front of it and inside the 640x480 image.
    fn synthetic_tracks() -> Vec<FeatureTrack> {
        let cam = || make_camera(500.0, 500.0, 320.0, 240.0);
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

    #[test]
    fn run_sfm_recovers_points_and_poses_from_synthetic_scene() {
        let tracks = synthetic_tracks();

        let recon = run_sfm(&tracks, 500.0, 500.0, 320.0, 240.0, 3)
            .expect("synthetic scene must reconstruct");

        assert!(
            !recon.points.is_empty(),
            "reconstruction should recover 3D points"
        );
        let registered = recon.views.iter().filter(|v| v.is_some()).count();
        assert!(
            registered >= 2,
            "at least two of three cameras should register (got {registered})"
        );
    }

    #[test]
    fn run_sfm_is_up_to_scale_without_tag() {
        let tracks = synthetic_tracks();

        let recon = run_sfm(&tracks, 500.0, 500.0, 320.0, 240.0, 3)
            .expect("synthetic scene must reconstruct");
        assert_eq!(recon.scale, ScaleSource::UpToScale);
    }
}
