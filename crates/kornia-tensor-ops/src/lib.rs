#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Error types for tensor operations.
///
/// Defines [`TensorOpsError`] for handling failures during tensor computations.
pub mod error;

/// Low-level computational kernels for tensor operations.
///
/// Optimized implementations of basic operations like addition, multiplication,
/// and other element-wise operations. These kernels may use SIMD instructions
/// for better performance.
pub mod kernels;

/// High-level tensor operations and traits.
///
/// Provides the [`TensorOps`] trait with implementations for common operations
/// including arithmetic, comparison, reduction, and transformation operations.
pub mod ops;

pub use error::TensorOpsError;
pub use ops::TensorOps;
