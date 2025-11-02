//! Image quality and similarity metrics.
//!
//! This module provides functions for quantitatively comparing images,
//! useful for:
//!
//! - Evaluating image processing algorithms
//! - Measuring reconstruction quality
//! - Loss functions in optimization
//! - Image compression evaluation
//!
//! # Available Metrics
//!
//! - **MSE** (Mean Squared Error): Average squared difference between pixels
//! - **PSNR** (Peak Signal-to-Noise Ratio): Quality metric in dB scale
//! - **L1 Loss**: Mean absolute difference (Manhattan distance)
//! - **Huber Loss**: Robust loss combining L1 and L2
//!
//! # Examples
//!
//! Comparing two images using MSE and PSNR metrics for quality assessment.

mod huber;
mod l1;
mod mse;

pub use huber::huber;
pub use l1::l1_loss;
pub use mse::{mse, psnr};
