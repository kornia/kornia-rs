//! SmolVLM implementation for Kornia
//!
//! SmolVLM (Small Vision Language Model) is a lightweight multimodal model
//! capable of understanding images and generating text based on prompts.

mod common;
mod processor;
mod tokenizer;

#[cfg(feature = "candle")]
mod candle;

#[cfg(feature = "onnx")]
mod onnx;

pub use common::*;
pub use processor::*;
pub use tokenizer::*;

#[cfg(feature = "candle")]
pub use self::candle::*;

#[cfg(feature = "onnx")]
pub use self::onnx::*;

/// Main SmolVLM model that provides the public API
#[derive(Debug)]
pub struct SmolVLM {
    config: SmolVLMConfig,
    backend: SmolVLMBackendImpl,
    processor: ImageProcessor,
    tokenizer: Tokenizer,
}

impl SmolVLM {
    /// Create a new SmolVLM instance with the Candle backend
    #[cfg(feature = "candle")]
    pub fn with_candle(model_path: &str, config: SmolVLMConfig) -> Result<Self, SmolVLMError> {
        let processor = ImageProcessor::new(&config)?;
        let tokenizer = Tokenizer::new(&config)?;
        let backend =
            SmolVLMBackendImpl::Candle(self::candle::CandleBackend::new(model_path, &config)?);

        Ok(Self {
            config,
            backend,
            processor,
            tokenizer,
        })
    }

    /// Create a new SmolVLM instance with the ONNX backend
    #[cfg(feature = "onnx")]
    pub fn with_onnx(model_path: &str, config: SmolVLMConfig) -> Result<Self, SmolVLMError> {
        let processor = ImageProcessor::new(&config)?;
        let tokenizer = Tokenizer::new(&config)?;
        let backend = SmolVLMBackendImpl::Onnx(self::onnx::OnnxBackend::new(model_path, &config)?);

        Ok(Self {
            config,
            backend,
            processor,
            tokenizer,
        })
    }

    /// Generate text for an image with a given prompt
    pub fn generate(&self, image_path: &str, prompt: &str) -> Result<String, SmolVLMError> {
        // Load and preprocess the image
        let processed_image = self.processor.process_image_from_path(image_path)?;

        // Generate the text based on the processed image and prompt
        let token_ids = match &self.backend {
            #[cfg(feature = "candle")]
            SmolVLMBackendImpl::Candle(backend) => backend.generate(&processed_image, prompt)?,
            #[cfg(feature = "onnx")]
            SmolVLMBackendImpl::Onnx(backend) => backend.generate(&processed_image, prompt)?,
            #[allow(unreachable_patterns)]
            _ => return Err(SmolVLMError::NoBackendAvailable),
        };

        // Decode the generated token IDs to text
        let text = self.tokenizer.decode(&token_ids)?;

        Ok(text)
    }

    /// Generate text for an image from raw bytes with a given prompt
    pub fn generate_from_bytes(
        &self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<String, SmolVLMError> {
        // Load and preprocess the image from bytes
        let processed_image = self.processor.process_image_from_bytes(image_bytes)?;

        // Generate the text based on the processed image and prompt
        let token_ids = match &self.backend {
            #[cfg(feature = "candle")]
            SmolVLMBackendImpl::Candle(backend) => backend.generate(&processed_image, prompt)?,
            #[cfg(feature = "onnx")]
            SmolVLMBackendImpl::Onnx(backend) => backend.generate(&processed_image, prompt)?,
            #[allow(unreachable_patterns)]
            _ => return Err(SmolVLMError::NoBackendAvailable),
        };

        // Decode the generated token IDs to text
        let text = self.tokenizer.decode(&token_ids)?;

        Ok(text)
    }

    /// Get the configuration of this model
    pub fn config(&self) -> &SmolVLMConfig {
        &self.config
    }
}

/// The available backends for SmolVLM
#[derive(Debug)]
enum SmolVLMBackendImpl {
    #[cfg(feature = "candle")]
    Candle(self::candle::CandleBackend),
    #[cfg(feature = "onnx")]
    Onnx(self::onnx::OnnxBackend),
}
