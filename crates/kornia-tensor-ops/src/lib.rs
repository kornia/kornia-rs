#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Architecture
//!
//! `kornia-tensor-ops` is designed as an extension layer on top of
//! [`kornia-tensor`](https://docs.rs/kornia-tensor). While `kornia-tensor`
//! provides the minimal, core tensor abstractions (storage, views, allocators,
//! and memory layout), this crate houses the higher-level and composite
//! operations that build on those primitives.
//!
//! This separation keeps the core tensor crate lean and composable, while
//! providing a clear home for operations that involve more complex logic,
//! additional trait bounds, or domain-specific computation.
//!
//! # Provided Operations
//!
//! * **Element-wise arithmetic** – add, subtract, multiply, divide
//! * **Scalar operations** – multiply by scalar, power (float / integer)
//! * **Reductions** – sum along a dimension, mean, min
//! * **Absolute value**
//! * **Similarity metrics** – dot product, cosine similarity, cosine distance
//!
//! # Low-level Kernels
//!
//! The [`kernels`] module exposes standalone functions (e.g.
//! [`kernels::dot_product1_kernel`], [`kernels::cosine_similarity_float_kernel`])
//! that operate directly on slices, useful when working outside the `Tensor`
//! abstraction.
//!
//! # When to Add Code Here
//!
//! New tensor operations that go beyond basic storage or view manipulation
//! should be added to this crate rather than to `kornia-tensor`. A good rule
//! of thumb: if the operation requires additional trait bounds (e.g.
//! `num_traits::Float`) or combines multiple lower-level steps, it belongs
//! here.

/// Error types for the tensor-ops module.
pub mod error;

/// module containing the kernels for the tensor operations. (e.g. `kornia_tensor::ops::add`)
pub mod kernels;

/// module containing ops implementations.
pub mod ops;

pub use error::TensorOpsError;
pub use ops::TensorOps;
