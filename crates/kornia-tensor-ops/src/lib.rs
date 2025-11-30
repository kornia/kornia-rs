#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Tensor Operations
//!
//! High-performance tensor operations for machine learning and computer vision.
//!
//! ## Key Features
//!
//! - **Element-wise Operations**: Add, subtract, multiply, divide tensors
//! - **Matrix Operations**: Matrix multiplication, transposition
//! - **Reduction Operations**: Sum, mean, max, min along dimensions
//! - **Broadcasting**: Automatic shape broadcasting for compatible operations
//!
//! ## Example: Basic Tensor Operations
//!
//! ```rust
//! use kornia_tensor::{Tensor, CpuAllocator};
//! use kornia_tensor_ops::TensorOps;
//!
//! // Create two tensors
//! let a = Tensor::<f32, 2, _>::from_shape_vec(
//!     [2, 3],
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//!     CpuAllocator
//! )?;
//!
//! let b = Tensor::<f32, 2, _>::from_shape_vec(
//!     [2, 3],
//!     vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
//!     CpuAllocator
//! )?;
//!
//! // Element-wise addition
//! let c = a.add(&b)?;
//!
//! // Verify result
//! assert_eq!(c.as_slice()[0], 3.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: Matrix Operations
//!
//! ```rust
//! use kornia_tensor::{Tensor, CpuAllocator};
//! use kornia_tensor_ops::TensorOps;
//!
//! // Create a 2x3 matrix
//! let matrix = Tensor::<f32, 2, _>::from_shape_vec(
//!     [2, 3],
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//!     CpuAllocator
//! )?;
//!
//! // Scale by a constant
//! let scaled = matrix.mul_scalar(2.0)?;
//!
//! // Sum all elements
//! let total: f32 = matrix.as_slice().iter().sum();
//! assert_eq!(total, 21.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Error types for tensor operations.
///
/// Defines errors that can occur during tensor computations, such as shape mismatches.
pub mod error;

/// Low-level computational kernels for tensor operations.
///
/// Optimized implementations of basic operations like addition, multiplication, etc.
pub mod kernels;

/// High-level tensor operation implementations.
///
/// Provides the [`TensorOps`] trait with methods for common tensor computations.
pub mod ops;

pub use error::TensorOpsError;
pub use ops::TensorOps;
