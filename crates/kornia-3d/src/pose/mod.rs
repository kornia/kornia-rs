//! # Pose estimation
//!
//! Two-view geometry: recovering camera poses and 3D structure from 2D correspondences.
//!
//! - [`fundamental`] — fundamental matrix (epipolar geometry in pixel space, 7 DOF)
//! - [`essential`] — essential matrix (epipolar geometry in metric space, 5 DOF, quotient of SE(3))
//! - [`homography`] — homography (planar scenes or pure rotation, 8 DOF)
//! - [`twoview`] — full pipeline: RANSAC model estimation, model selection, triangulation

mod affine;
pub use affine::*;

mod homography;
pub use homography::*;

mod fundamental;
pub use fundamental::*;

mod essential;
pub use essential::*;

mod twoview;
pub use twoview::*;
