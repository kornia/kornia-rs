#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// EPnP solver implementation
pub mod epnp;

/// Common data types shared across PnP solvers.
pub mod types;

pub use epnp::{EPNPParams, EPnP};
pub use types::PnPSolver;

mod ops;
