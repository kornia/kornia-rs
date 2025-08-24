//! Common data types shared across Perspective-n-Point (PnP) solvers.

use crate::camera::CameraModel;
use thiserror::Error;

/// Error types for PnP solvers.
#[derive(Debug, Error)]
pub enum PnPError {
    /// Invalid input data - insufficient correspondences for the specific solver
    #[error("PnP solver requires at least {required} 2D-3D correspondences, got {actual}")]
    InsufficientCorrespondences {
        /// Minimum number of correspondences required by the solver
        required: usize,
        /// Actual number of correspondences provided
        actual: usize,
    },

    /// Invalid input data - mismatched array lengths with descriptive labels.
    #[error("Mismatched array lengths: {left_name} ({left_len}) != {right_name} ({right_len})")]
    MismatchedArrayLengths {
        /// Label for the left-hand slice
        left_name: &'static str,
        /// Length of the left-hand slice
        left_len: usize,
        /// Label for the right-hand slice
        right_name: &'static str,
        /// Length of the right-hand slice
        right_len: usize,
    },

    /// Singular value decomposition failed
    #[error("SVD computation failed: {0}")]
    SvdFailed(String),

    /// Camera model error
    #[error("Camera model error: {0}")]
    CameraError(String),
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
    pub rotation: [[f32; 3]; 3],
    /// Estimated translation vector.
    pub translation: [f32; 3],
    /// Rodrigues axis-angle representation of the rotation.
    pub rvec: [f32; 3],
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
    /// * `world` – 3-D coordinates in the world frame.
    /// * `image` – Corresponding pixel coordinates.
    /// * `k` – Camera intrinsics matrix.
    /// * `params` – Solver-specific parameters.
    fn solve(
        world: &[[f32; 3]],
        image: &[[f32; 2]],
        k: &[[f32; 3]; 3],
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError>;
}

/// Trait for PnP solvers that support camera models with distortion.
pub trait PnPSolverWithCamera {
    /// Solver-specific parameters.
    type Param;

    /// Solve for camera pose given 2D-3D correspondences with camera model support.
    ///
    /// # Arguments
    /// * `world` – 3-D coordinates in the world frame.
    /// * `image` – Corresponding pixel coordinates (may be distorted).
    /// * `camera` – Camera model with intrinsics and optional distortion.
    /// * `params` – Solver-specific parameters.
    fn solve_with_camera(
        world: &[[f32; 3]],
        image: &[[f32; 2]],
        camera: &CameraModel,
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError>;
}

/// Default implementation for PnPSolverWithCamera that undistorts points and calls the original solver.
impl<T: PnPSolver> PnPSolverWithCamera for T {
    type Param = T::Param;

    fn solve_with_camera(
        world: &[[f32; 3]],
        image: &[[f32; 2]],
        camera: &CameraModel,
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError> {
        // If camera has distortion, undistort the image points first
        let undistorted_image = if camera.has_distortion() {
            camera
                .undistort_points(image)
                .map_err(|e| PnPError::CameraError(format!("Failed to undistort points: {}", e)))?
        } else {
            image.to_vec()
        };

        // Get intrinsics matrix for the solver
        let k = camera.intrinsics_matrix();

        // Call the original solver with undistorted points
        T::solve(world, &undistorted_image, &k, params)
    }
}
