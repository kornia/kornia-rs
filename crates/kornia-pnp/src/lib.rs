#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// EPnP solver implementation
pub mod epnp;

/// Common data types shared across PnP solvers.
pub mod types;

pub use epnp::{EPNPParams, EPnP};
use types::{PnPResult, PnPSolver};

mod ops;

/// Enumeration of the Perspective-n-Point algorithms available in this crate.
#[derive(Debug, Clone)]
pub enum Method {
    /// Efficient PnP solver with a user-supplied parameter object.
    EPnP(EPNPParams),
    /// Efficient PnP solver with the crate’s default parameters.
    EPnPDefault,
    // Placeholder for future solvers such as P3P, DLS, etc.
}

/// Dispatch function that routes to the chosen PnP solver.
pub fn solve(
    world: &[[f64; 3]],
    image: &[[f64; 2]],
    k: &[[f64; 3]; 3],
    method: Method,
) -> Result<PnPResult, &'static str> {
    match method {
        Method::EPnP(params) => EPnP::solve(world, image, k, &params),
        Method::EPnPDefault => EPnP::solve(world, image, k, &EPNPParams::default()),
    }
}
