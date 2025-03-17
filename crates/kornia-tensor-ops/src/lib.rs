#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Error types for the core-ops module.
pub mod error;

/// module containing ops implementations.
pub mod ops;

pub use crate::error::TensorOpsError;
pub use crate::ops::TensorOps;
