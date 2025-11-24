#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia PnP (Perspective-n-Point)
//!
//! Efficient camera pose estimation from 2D-3D point correspondences.
//!
//! ## Key Features
//!
//! - **EPnP Algorithm**: Efficient Perspective-n-Point solver for camera pose estimation
//! - **RANSAC Support**: Robust estimation with outlier rejection
//! - **Distortion Handling**: Support for lens distortion correction
//! - **Multiple Solvers**: Extensible framework for different PnP methods
//!
//! ## Example: Basic EPnP
//!
//! ```rust
//! use kornia_pnp::{solve_pnp, PnPMethod, EPnPParams};
//!
//! // 3D world points
//! let world_points = vec![
//!     [0.0, 0.0, 0.0],
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0],
//! ];
//!
//! // Corresponding 2D image points
//! let image_points = vec![
//!     [320.0, 240.0],
//!     [420.0, 240.0],
//!     [320.0, 340.0],
//!     [320.0, 140.0],
//! ];
//!
//! // Camera intrinsics (3x3 matrix)
//! let k = [
//!     [800.0, 0.0, 320.0],
//!     [0.0, 800.0, 240.0],
//!     [0.0, 0.0, 1.0],
//! ];
//!
//! // Solve for camera pose
//! let result = solve_pnp(
//!     &world_points,
//!     &image_points,
//!     &k,
//!     None,  // No distortion
//!     PnPMethod::EPnPDefault,
//! )?;
//!
//! println!("Rotation: {:?}", result.rotation);
//! println!("Translation: {:?}", result.translation);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: Robust PnP with RANSAC
//!
//! ```rust
//! use kornia_pnp::{solve_pnp_ransac, PnPMethod, RansacParams};
//!
//! # let world_points = vec![[0.0, 0.0, 0.0]; 10];
//! # let image_points = vec![[320.0, 240.0]; 10];
//! # let k = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]];
//! // Use RANSAC to handle outliers
//! let ransac_params = RansacParams {
//!     max_iterations: 1000,
//!     inlier_threshold: 5.0,
//!     confidence: 0.99,
//!     min_inliers: 4,
//! };
//!
//! let result = solve_pnp_ransac(
//!     &world_points,
//!     &image_points,
//!     &k,
//!     None,
//!     PnPMethod::EPnPDefault,
//!     ransac_params,
//! )?;
//!
//! println!("Inliers: {}/{}", result.inliers.len(), world_points.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Efficient Perspective-n-Point (EPnP) solver implementation.
///
/// A fast and accurate method for computing camera pose from 2D-3D correspondences.
pub mod epnp;

/// Common data types and traits for PnP solvers.
///
/// Defines the interface and result types shared across different PnP algorithms.
pub mod pnp;

/// RANSAC-based robust PnP pose estimation.
///
/// Handles outliers in point correspondences through random sampling consensus.
pub mod ransac;

pub use epnp::{EPnP, EPnPParams};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
pub use pnp::{PnPError, PnPResult, PnPSolver};
pub use ransac::{solve_pnp_ransac, PnPRansacError, PnPRansacResult, RansacParams};

mod ops;

/// Enumeration of the Perspective-n-Point algorithms available in this crate.
#[derive(Debug, Clone)]
pub enum PnPMethod {
    /// Efficient PnP solver with a user-supplied parameter object.
    EPnP(EPnPParams),
    /// Efficient PnP solver with the crate's default parameters.
    EPnPDefault,
    // Placeholder for future solvers such as P3P, DLS, etc.
}

/// Dispatch function that routes to the chosen PnP solver.
pub fn solve_pnp(
    world: &[[f32; 3]],
    image: &[[f32; 2]],
    k: &[[f32; 3]; 3],
    distortion: Option<&PolynomialDistortion>,
    method: PnPMethod,
) -> Result<PnPResult, PnPError> {
    match method {
        PnPMethod::EPnP(params) => EPnP::solve(world, image, k, distortion, &params),
        PnPMethod::EPnPDefault => EPnP::solve(world, image, k, distortion, &EPnPParams::default()),
    }
}
