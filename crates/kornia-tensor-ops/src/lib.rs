#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Error types for the tensor-ops module.
pub mod error;

/// module containing the kernels for the tensor operations. (e.g. `kornia_tensor::ops::add`)
pub mod kernels;

/// module containing ops implementations.
pub mod ops;

pub use error::TensorOpsError;
pub use ops::TensorOps;
