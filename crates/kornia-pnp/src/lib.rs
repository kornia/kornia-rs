#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// EPnP solver implementation
pub mod epnp; 

/// Common data types shared across PnP solvers.
pub mod types;

pub use types::PnPSolver;
pub use epnp::{EPnP, EPNPParams};

mod ops;