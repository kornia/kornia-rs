//! Common types and functions for SmolVLM

use std::path::Path;
use thiserror::Error;

/// Error type for SmolVLM operations
#[derive(Error, Debug)]
pub enum SmolVLMError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Failed to process image: {0}")]
    ImageProcessingError(String),

    #[error("Failed to tokenize text: {0}")]
    TokenizationError(String),

    #[error("Failed to generate text: {0}")]
    GenerationError(String),

    #[error("No backend available")]
    NoBackendAvailable,

    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image error: {0}")]
    ImageError(String),

    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[cfg(feature = "onnx")]
    #[error("ONNX error: {0}")]
    OnnxError(String),
}

/// Model configuration for SmolVLM
#[derive(Debug, Clone)]
pub struct SmolVLMConfig {
    /// Model size (small, medium, large)
    pub model_size: ModelSize,

    /// Image input dimensions (height, width)
    pub image_size: (usize, usize),

    /// Number of image patches in each dimension (height, width)
    pub patch_size: (usize, usize),

    /// Number of vision encoder layers
    pub vision_layers: usize,

    /// Vision encoder hidden dimension
    pub vision_hidden_dim: usize,

    /// Number of language model layers
    pub lm_layers: usize,

    /// Language model hidden dimension
    pub lm_hidden_dim: usize,

    /// Language model embedding dimension
    pub lm_embedding_dim: usize,

    /// Language model vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Tokenizer directory path
    pub tokenizer_path: String,
}

/// Model size options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

/// Backend options for SmolVLM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmolVLMBackend {
    Candle,
    Onnx,
}

impl SmolVLMConfig {
    /// Create a configuration for the small model
    pub fn small() -> Self {
        Self {
            model_size: ModelSize::Small,
            image_size: (224, 224),
            patch_size: (16, 16),
            vision_layers: 12,
            vision_hidden_dim: 768,
            lm_layers: 12,
            lm_hidden_dim: 768,
            lm_embedding_dim: 768,
            vocab_size: 32000,
            max_seq_len: 512,
            tokenizer_path: "models/smolvlm/tokenizer-small".to_string(),
        }
    }

    /// Create a configuration for the medium model
    pub fn medium() -> Self {
        Self {
            model_size: ModelSize::Medium,
            image_size: (224, 224),
            patch_size: (16, 16),
            vision_layers: 24,
            vision_hidden_dim: 1024,
            lm_layers: 16,
            lm_hidden_dim: 1024,
            lm_embedding_dim: 1024,
            vocab_size: 32000,
            max_seq_len: 768,
            tokenizer_path: "models/smolvlm/tokenizer-medium".to_string(),
        }
    }

    /// Create a configuration for the large model
    pub fn large() -> Self {
        Self {
            model_size: ModelSize::Large,
            image_size: (224, 224),
            patch_size: (16, 16),
            vision_layers: 32,
            vision_hidden_dim: 1280,
            lm_layers: 24,
            lm_hidden_dim: 1280,
            lm_embedding_dim: 1280,
            vocab_size: 32000,
            max_seq_len: 1024,
            tokenizer_path: "models/smolvlm/tokenizer-large".to_string(),
        }
    }

    /// Check if the configuration is valid
    pub fn validate(&self) -> Result<(), SmolVLMError> {
        // Check if tokenizer path exists
        if !Path::new(&self.tokenizer_path).exists() {
            return Err(SmolVLMError::ConfigurationError(format!(
                "Tokenizer path does not exist: {}",
                self.tokenizer_path
            )));
        }

        // Check other configuration parameters
        if self.image_size.0 == 0 || self.image_size.1 == 0 {
            return Err(SmolVLMError::ConfigurationError(
                "Image dimensions must be greater than zero".to_string(),
            ));
        }

        if self.patch_size.0 == 0 || self.patch_size.1 == 0 {
            return Err(SmolVLMError::ConfigurationError(
                "Patch dimensions must be greater than zero".to_string(),
            ));
        }

        if self.vision_layers == 0 || self.lm_layers == 0 {
            return Err(SmolVLMError::ConfigurationError(
                "Number of layers must be greater than zero".to_string(),
            ));
        }

        if self.vision_hidden_dim == 0 || self.lm_hidden_dim == 0 || self.lm_embedding_dim == 0 {
            return Err(SmolVLMError::ConfigurationError(
                "Hidden dimensions must be greater than zero".to_string(),
            ));
        }

        if self.vocab_size == 0 {
            return Err(SmolVLMError::ConfigurationError(
                "Vocabulary size must be greater than zero".to_string(),
            ));
        }

        if self.max_seq_len == 0 {
            return Err(SmolVLMError::ConfigurationError(
                "Maximum sequence length must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

/// ProcessedImage contains the preprocessed image data ready for model inference
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// The processed image data as a flattened vector
    pub data: Vec<f32>,

    /// The shape of the processed image (batch, channels, height, width)
    pub shape: (usize, usize, usize, usize),
}

impl ProcessedImage {
    /// Create a new processed image
    pub fn new(data: Vec<f32>, shape: (usize, usize, usize, usize)) -> Self {
        Self { data, shape }
    }

    /// Get the total number of elements in the processed image
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the processed image is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
