#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Deep neural network kernels.
pub mod dnn;

/// Error types for the core-ops module.
pub mod error;

/// module containing the kernels for the tensor operations.
pub mod kernels;

/// module containing ops implementations.
pub mod ops;

pub use error::TensorOpsError;
pub use ops::TensorOps;
