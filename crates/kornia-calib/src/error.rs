//! Error type for calibration.

use kornia_apriltag::pose::AprilTagPoseError;

/// Errors that can occur during multi-camera calibration.
#[derive(Debug, thiserror::Error)]
pub enum CalibError {
    /// No tag observations were provided.
    #[error("no tag observations provided")]
    NoTags,

    /// No camera observed the selected reference tag, so no gauge can be anchored.
    #[error("no camera observed the reference tag")]
    NoReferenceTagView,

    /// A per-camera planar tag pose estimate failed.
    #[error("apriltag pose estimation failed: {0}")]
    TagPose(#[from] AprilTagPoseError),

    /// The bundle adjustment solve failed.
    #[error("bundle adjustment failed: {0}")]
    BundleAdjust(String),

    /// A rebasing was requested against a reference camera that was not solved.
    #[error("reference camera {0} was not solved")]
    ReferenceNotSolved(usize),
}
