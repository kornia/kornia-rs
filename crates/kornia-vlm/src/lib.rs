//! # Kornia Vision-Language Models (VLM)
//!
//! High-level interfaces for running vision-language models for image understanding,
//! visual question answering, and multimodal tasks.
//!
//! This crate provides integrations with state-of-the-art VLMs including:
//!
//! - **PaliGemma**: Google's vision-language model for image captioning
//! - **SmolVLM/SmolVLM2**: Efficient small vision-language models
//!
//! # Features
//!
//! - Efficient model inference with quantization support
//! - Streaming token generation
//! - Video understanding capabilities
//! - Flexible tokenizer interfaces

/// PaliGemma vision-language model integration.
///
/// Google's PaliGemma model for image understanding and visual question answering.
pub mod paligemma;

/// SmolVLM vision-language model integration.
///
/// Efficient small vision-language model for resource-constrained environments.
pub mod smolvlm;

/// SmolVLM2 vision-language model integration.
///
/// Improved version of SmolVLM with enhanced performance and capabilities.
pub mod smolvlm2;

/// Video understanding utilities.
///
/// Tools for processing and understanding video streams with VLMs.
pub mod video;

/// Internal context management for model execution.
mod context;
