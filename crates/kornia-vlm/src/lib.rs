#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Vision-Language Models
//!
//! This crate provides implementations of Vision-Language Models (VLMs) for multimodal AI tasks
//! that combine computer vision and natural language processing.
//!
//! ## Supported Models
//!
//! - **PaliGemma**: Google's vision-language model for image understanding and captioning
//! - **SmolVLM**: Compact vision-language model optimized for efficiency
//! - **SmolVLM2**: Enhanced version of SmolVLM with improved performance
//!
//! ## Features
//!
//! - Unified interface for multiple VLM architectures
//! - Image and video processing capabilities
//! - Efficient inference with optional GPU acceleration (via `cuda` feature)
//! - Integration with Hugging Face model hub
//!
//! ## Example
//!
//! ```rust,no_run
//! use kornia_vlm::paligemma::PaliGemma;
//!
//! // Load a pre-trained model
//! let model = PaliGemma::builder()
//!     .with_model_id("google/paligemma-3b-pt-224")
//!     .build()?;
//!
//! // Process an image with a text prompt
//! let image_path = "path/to/image.jpg";
//! let prompt = "Describe this image";
//! let response = model.generate(image_path, prompt)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// PaliGemma vision-language model implementation.
pub mod paligemma;

/// SmolVLM vision-language model implementation.
pub mod smolvlm;

/// SmolVLM2 vision-language model implementation.
pub mod smolvlm2;

/// Video processing utilities for VLMs.
pub mod video;

/// Internal context management for model execution.
mod context;
