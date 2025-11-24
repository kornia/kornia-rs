#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Neural Network Operations
//!
//! Efficient neural network layer implementations optimized for computer vision tasks.
//!
//! ## Key Features
//!
//! - **Linear Layers**: Fully-connected layers with optional MKL acceleration
//! - **Hardware Acceleration**: Intel MKL support for high-performance computation
//! - **Vision-Focused**: Optimized for common CV network architectures
//!
//! ## Example: Linear Layer
//!
//! ```rust,ignore
//! use kornia_nn::linear::Linear;
//!
//! // Create a linear layer: input_dim=512, output_dim=256
//! let layer = Linear::new(512, 256)?;
//!
//! // Forward pass on a batch of features
//! let input = vec![0.1; 512];  // Single feature vector
//! let output = layer.forward(&input)?;
//!
//! assert_eq!(output.len(), 256);
//! ```
//!
//! ## Requirements
//!
//! The `mkl` feature must be enabled to use neural network operations:
//!
//! ```toml
//! [dependencies]
//! kornia-nn = { version = "0.1", features = ["mkl"] }
//! ```

/// Linear (fully-connected) layer operations.
///
/// Implements matrix multiplication-based linear transformations with optional bias.
/// Requires the `mkl` feature for hardware-accelerated computation.
#[cfg(feature = "mkl")]
pub mod linear;
