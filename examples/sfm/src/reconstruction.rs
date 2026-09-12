//! Structure-from-Motion reconstruction wrapper.
//!
//! Builds a [`PinholeCamera`] per video frame from user-supplied intrinsics
//! and drives `kornia_calib`'s incremental SfM (`reconstruct`) over the
//! feature tracks. Because no AprilTag anchors the scale, the recovered map is
//! [`ScaleSource::UpToScale`]: correct shape, arbitrary units.

use std::error::Error;
use std::sync::Arc;

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
    progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
) -> Result<Reconstruction, Box<dyn Error>> {
    let cameras = vec![make_camera(fx, fy, cx, cy); n_frames];
    // `.sequential()` tunes the config for video walkthroughs (smaller
    // parallax threshold, more BA iterations).
    let config = ReconstructionConfig::new(0.0).sequential();
    let mut config = config;
    config.progress = progress;
    Ok(reconstruct(&cameras, &[], tracks, &config, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;
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

    #[test]
    fn run_sfm_recovers_points_and_poses_from_synthetic_scene() {
        let tracks = test_util::synthetic_tracks();

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            None,
        )
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
        let tracks = test_util::synthetic_tracks();

        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            None,
        )
        .expect("synthetic scene must reconstruct");
        assert_eq!(recon.scale, ScaleSource::UpToScale);
    }
}
