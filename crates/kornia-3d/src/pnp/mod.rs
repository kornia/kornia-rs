//! Perspective-n-Point (PnP) solvers and robust pose estimation.

/// EPnP solver implementation.
pub mod epnp;

/// RANSAC for robust PnP pose estimation.
pub mod ransac;

/// LM-based pose refinement.
pub mod refine;

mod ops;

pub use epnp::{EPnP, EPnPParams};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
pub use ransac::{solve_pnp_ransac, PnPRansacError, PnPRansacResult, RansacParams};
pub use refine::{refine_pose_lm, LMRefineParams};
use thiserror::Error;

/// Error types for PnP solvers.
#[derive(Debug, Error)]
pub enum PnPError {
    /// Invalid input data - insufficient correspondences for the specific solver.
    #[error("PnP solver requires at least {required} 2D-3D correspondences, got {actual}")]
    InsufficientCorrespondences {
        /// Minimum number of correspondences required by the solver.
        required: usize,
        /// Actual number of correspondences provided.
        actual: usize,
    },

    /// Invalid input data - mismatched array lengths with descriptive labels.
    #[error("Mismatched array lengths: {left_name} ({left_len}) != {right_name} ({right_len})")]
    MismatchedArrayLengths {
        /// Label for the left-hand slice.
        left_name: &'static str,
        /// Length of the left-hand slice.
        left_len: usize,
        /// Label for the right-hand slice.
        right_name: &'static str,
        /// Length of the right-hand slice.
        right_len: usize,
    },

    /// Singular value decomposition failed.
    #[error("SVD computation failed: {0}")]
    SvdFailed(String),
}

/// Numeric tolerances used by linear algebra routines throughout the PnP pipeline.
#[derive(Debug, Clone)]
pub struct NumericTol {
    /// Tolerance for singular-value decomposition.
    pub svd: f32,
    /// Epsilon threshold for determinant / singular-value checks when deciding whether to fall back to a pseudo-inverse.
    pub eps: f32,
}

impl Default for NumericTol {
    fn default() -> Self {
        Self {
            svd: 1e-12,
            eps: 1e-12,
        }
    }
}

/// Result returned by any PnP solver.
///
/// The rotation matrix maps coordinates from the **world** frame to the
/// **camera** frame.
#[derive(Debug, Clone)]
pub struct PnPResult {
    /// Estimated rotation matrix.
    pub rotation: Mat3AF32,
    /// Estimated translation vector.
    pub translation: Vec3AF32,
    /// Rodrigues axis-angle representation of the rotation.
    pub rvec: Vec3AF32,
    /// Root-mean-square reprojection error in pixels (if computed).
    pub reproj_rmse: Option<f32>,
    /// Number of iterations taken (if applicable).
    pub num_iterations: Option<usize>,
    /// Whether the solver converged (if applicable).
    pub converged: Option<bool>,
}

/// Trait for PnP solvers.
pub trait PnPSolver {
    /// Solver-specific parameters.
    type Param;

    /// Solve for camera pose given 2D-3D correspondences.
    ///
    /// # Arguments
    /// - `world` – 3-D coordinates in the world frame.
    /// - `image` – Corresponding pixel coordinates.
    /// - `k` – Camera intrinsics matrix.
    /// - `distortion` – Optional camera distortion model (image-space).
    /// - `params` – Solver-specific parameters.
    fn solve(
        world: &[Vec3AF32],
        image: &[Vec2F32],
        k: &Mat3AF32,
        distortion: Option<&PolynomialDistortion>,
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError>;
}

/// Enumeration of the Perspective-n-Point algorithms available in this module.
#[derive(Debug, Clone)]
pub enum PnPMethod {
    /// Efficient PnP solver with a user-supplied parameter object.
    EPnP(EPnPParams),
    /// Efficient PnP solver with the module's default parameters.
    EPnPDefault,
    // Placeholder for future solvers such as P3P, DLS, etc.
}

/// Dispatch function that routes to the chosen PnP solver.
pub fn solve_pnp(
    world: &[Vec3AF32],
    image: &[Vec2F32],
    k: &Mat3AF32,
    distortion: Option<&PolynomialDistortion>,
    method: PnPMethod,
) -> Result<PnPResult, PnPError> {
    match method {
        PnPMethod::EPnP(params) => EPnP::solve(world, image, k, distortion, &params),
        PnPMethod::EPnPDefault => EPnP::solve(world, image, k, distortion, &EPnPParams::default()),
    }
}
