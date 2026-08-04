//! Tag pose estimation — a thin naming layer over [`kornia_3d::pose::planar`].
//!
//! The solver moved to `kornia-3d`: it estimates the pose of any planar quad from 4 coplanar
//! correspondences and has no AprilTag-specific content, so keeping it here forced every consumer
//! of planar pose to depend on the tag decoder. A tag is one caller among many.
//!
//! The names below are kept so existing code continues to compile. New code should prefer the
//! `kornia_3d::pose` names, which say what the solver actually does.

pub use kornia_3d::camera::PinholeCamera;
pub use kornia_3d::pose::{
    estimate_planar_pose as estimate_tag_pose, PlanarPose as TagPose,
    PlanarPoseError as AprilTagPoseError, PlanarPosePair as TagPosePair,
};
