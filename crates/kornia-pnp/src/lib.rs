#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// EPnP solver implementation
pub mod epnp;

/// Common data types shared across PnP solvers.
pub mod pnp;

    /// Optional refinement routines (Levenberg–Marquardt, etc.)
mod refine;

pub use epnp::{EPnP, EPnPParams};
pub use pnp::{PnPError, PnPResult, PnPSolver};
pub use refine::LMParams;

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
    method: PnPMethod,
) -> Result<PnPResult, PnPError> {
    match method {
        PnPMethod::EPnP(params) => EPnP::solve(world, image, k, &params),
        PnPMethod::EPnPDefault => EPnP::solve(world, image, k, &EPnPParams::default()),
    }
}
