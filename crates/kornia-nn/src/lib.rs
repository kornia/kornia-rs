#![deny(missing_docs)]
//! # Kornia Neural Network Operations
//!
//! Efficient neural network primitives and layer implementations for
//! computer vision models.
//!
//! This crate provides optimized implementations of common neural network
//! operations, with optional hardware acceleration via Intel MKL.
//!
//! # Features
//!
//! - **mkl**: Enable Intel MKL acceleration for linear algebra operations

/// Linear (fully-connected) layer operations (feature-gated).
///
/// Matrix-vector and matrix-matrix multiplication for dense neural network layers.
/// Requires the `mkl` feature flag for hardware-accelerated computation.
#[cfg(feature = "mkl")]
pub mod linear;
