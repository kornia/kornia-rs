//! Common data types shared across Perspective-n-Point (PnP) solvers.

/// Numeric tolerances used by linear algebra routines throughout the PnP pipeline.
#[derive(Debug, Clone)]
pub struct NumericTol {
    /// Tolerance for singular-value decomposition.
    pub svd: f64,
    /// Epsilon threshold for determinant / singular-value checks when deciding whether to fall back to a pseudo-inverse.
    pub eps: f64,
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
    pub rotation: [[f64; 3]; 3],
    /// Estimated translation vector.
    pub translation: [f64; 3],
    /// Rodrigues axis-angle representation (log-map) of `rotation`.
    pub rvec: [f64; 3],
    /// Optional root-mean-square reprojection error in pixels.
    pub reproj_rmse: Option<f64>,
    /// Optional number of iterations taken by an iterative solver.
    pub num_iterations: Option<usize>,
    /// Indicates whether an iterative solver reported convergence.
    pub converged: Option<bool>,
}

/// Trait implemented by every PnP solver available in this crate.
pub trait PnPSolver {
    /// Parameter object specific to the solver.
    type Param;

    /// Runs the solver.
    fn solve(
        world: &[[f64; 3]],
        image: &[[f64; 2]],
        k: &[[f64; 3]; 3],
        params: &Self::Param,
    ) -> Result<PnPResult, &'static str>;
}
