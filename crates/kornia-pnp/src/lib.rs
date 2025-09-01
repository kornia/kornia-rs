#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Camera models and distortion handling
pub mod camera;

/// EPnP solver implementation
pub mod epnp;

/// Common data types shared across PnP solvers.
pub mod pnp;

pub use camera::{CameraError, CameraIntrinsics, CameraModel, CameraResult};
pub use epnp::{EPnP, EPnPParams};
pub use pnp::{PnPError, PnPResult, PnPSolver};

pub use kornia_imgproc::calibration::distortion::PolynomialDistortion;

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
    distortion: &PolynomialDistortion,
    method: PnPMethod,
) -> Result<PnPResult, PnPError> {
    match method {
        PnPMethod::EPnP(params) => EPnP::solve(world, image, k, distortion, &params),
        PnPMethod::EPnPDefault => EPnP::solve(world, image, k, distortion, &EPnPParams::default()),
    }

    // match PolynomialDistortion
}

// /// Dispatch function that routes to the chosen PnP solver with camera model support.
// pub fn solve_pnp_with_camera(
//     world: &[[f32; 3]],
//     image: &[[f32; 2]],
//     camera: &CameraModel,
//     method: PnPMethod,
// ) -> Result<PnPResult, PnPError> {
//     // If camera has distortion, undistort the image points first
//     let undistorted_image = if camera.has_distortion() {
//         camera
//             .undistort_points(image)
//             .map_err(|e| PnPError::SvdFailed(format!("Failed to undistort points: {}", e)))?
//     } else {
//         image.to_vec()
//     };

//     // Get intrinsics matrix for the solver
//     let k = camera.intrinsics_matrix();

//     // Call the original solver with undistorted points
//     match method {
//         PnPMethod::EPnP(params) => EPnP::solve(world, &undistorted_image, &k, &params),
//         PnPMethod::EPnPDefault => {
//             EPnP::solve(world, &undistorted_image, &k, &EPnPParams::default())
//         }
//     }
// }

// /// Convenience function to solve PnP with distortion parameters.
// pub fn solve_pnp_with_distortion(
//     world: &[[f32; 3]],
//     image: &[[f32; 2]],
//     k: &[[f32; 3]; 3],
//     distortion: &PolynomialDistortion,
//     method: PnPMethod,
// ) -> Result<PnPResult, PnPError> {
//     let intrinsics = CameraIntrinsics::from_matrix(k)
//         .map_err(|e| PnPError::SvdFailed(format!("Invalid intrinsics matrix: {}", e)))?;
//     let camera = CameraModel::with_distortion(intrinsics, distortion.clone());
//     solve_pnp_with_camera(world, image, &camera, method)
// }
