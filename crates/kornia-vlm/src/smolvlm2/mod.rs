//! SmolVLM2 Vision Language Model
//!
//! This module provides a Rust implementation of SmolVLM2, a vision-language model
//! that can understand both text and images. The model supports both loading from
//! HuggingFace Hub and custom safetensor files.
//!
//! # Examples
//!
//! ## Basic usage with HuggingFace Hub weights:
//!
//! ```no_run
//! use kornia_vlm::smolvlm2::{SmolVlm2, SmolVlm2Config};
//! use kornia_tensor::CpuAllocator;
//!
//! let config = SmolVlm2Config::default();
//! let mut model = SmolVlm2::new(config).unwrap();
//!
//! let response = model.inference(
//!     "What do you see in this image?",
//!     None, // Add image here
//!     100,  // max tokens
//!     CpuAllocator,
//!     false // debug
//! ).unwrap();
//! ```
//!
//! ## Loading from custom safetensor files:
//!
//! ```no_run
//! use kornia_vlm::smolvlm2::{SmolVlm2, SmolVlm2Config};
//!
//! // Method 1: Direct constructor
//! let weights = vec![
//!     "/path/to/model-00001-of-00002.safetensors",
//!     "/path/to/model-00002-of-00002.safetensors",
//! ];
//! let model = SmolVlm2::from_safetensors(weights, SmolVlm2Config::default()).unwrap();
//!
//! // Method 2: Using config
//! let config = SmolVlm2Config::with_custom_weights(vec![
//!     "/path/to/model.safetensors"
//! ]);
//! let model = SmolVlm2::new(config).unwrap();
//! ```

mod model;
pub(super) mod text_processor;

use log::debug;
use std::io::Write;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};

use crate::smolvlm2::text_processor::{Line, Message, Role, TextProcessor};

/// Utilities for the SmolVLM2 module.
#[derive(thiserror::Error, Debug)]
pub enum SmolVlm2Error {
    #[error(transparent)]
    FailedToLoadModel(#[from] hf_hub::api::sync::ApiError),

    #[error(transparent)]
    CandleError(#[from] candle_core::Error),

    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),

    #[error(transparent)]
    TokenizerError(#[from] tokenizers::Error),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    JinjaError(#[from] minijinja::Error),

    #[error(transparent)]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid logits detected: {0}")]
    InvalidLogits(String),

    #[error("Invalid encoding detected: {0}")]
    InvalidEncoding(String),

    #[error("Missing chat template: {0}")]
    MissingChatTemplate(String),

    #[error("Message history mistmatch: {0}")]
    MessageHistoryMismatch(String),

    #[error("Cannot find the <end_of_utterance> token")]
    EosTokenNotFound,

    #[error("Mismatched image count: tags = {tags}, images = {images}")]
    MismatchedImageCount { tags: usize, images: usize },
}

/// Configuration for the SmolVLM2 model

#[derive(Clone)]
pub struct SmolVlm2Config {
    pub seed: u64,
    pub temp: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub do_sample: bool,
    pub repeat_last_n: usize,
    /// Optional path to custom safetensor weights files. If provided, these will be used
    /// instead of downloading from HuggingFace Hub. Can be a single file or multiple files.
    pub custom_weights_paths: Option<Vec<std::path::PathBuf>>,
}

impl Default for SmolVlm2Config {
    fn default() -> Self {
        Self {
            seed: 42,
            temp: 1.0,
            top_p: 0.8,
            repeat_penalty: 1.0,
            do_sample: true,
            repeat_last_n: 64,
            custom_weights_paths: None,
        }
    }
}

pub struct SmolVlm2 {
    model: model::Model,
    _config: SmolVlm2Config,
    device: Device,
    index_pos: usize, // index of the next token to be processed

    txt_processor: TextProcessor,
    response: String,
}

impl SmolVlm2 {
    const MODEL_IDENTIFIER: &'static str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct";

    /// Create a new SmolVLM2 instance with the default configuration.
    /// This will download weights from HuggingFace Hub.
    pub fn new(config: SmolVlm2Config) -> Result<Self, SmolVlm2Error> {
        #[cfg(feature = "cuda")]
        let (device, dtype) = match Device::cuda_if_available(0) {
            Ok(device) => (device, DType::F32),
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {e:?}");
                (Device::Cpu, DType::F32)
            }
        };

        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let (model, txt_processor) = Self::load_model(&config, dtype, &device)?;

        Ok(Self {
            model,
            txt_processor,
            _config: config,
            device,
            index_pos: 0,

            response: String::new(),
        })
    }

    /// Create a new SmolVLM2 instance with custom safetensor weights files.
    ///
    /// # Arguments
    ///
    /// * `weights_paths` - Vector of paths to safetensor files to load
    /// * `config` - Configuration for the model (custom_weights_paths will be overridden)
    ///
    /// # Returns
    ///
    /// A new SmolVLM2 instance loaded with the specified weights
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kornia_vlm::smolvlm2::{SmolVlm2, SmolVlm2Config};
    ///
    /// let weights = vec![
    ///     "/path/to/model-00001-of-00002.safetensors".into(),
    ///     "/path/to/model-00002-of-00002.safetensors".into(),
    /// ];
    /// let model = SmolVlm2::from_safetensors(weights, SmolVlm2Config::default()).unwrap();
    /// ```
    pub fn from_safetensors<P: Into<std::path::PathBuf>>(
        weights_paths: Vec<P>,
        mut config: SmolVlm2Config,
    ) -> Result<Self, SmolVlm2Error> {
        config.custom_weights_paths = Some(weights_paths.into_iter().map(|p| p.into()).collect());
        Self::new(config)
    }

    /// Create a new SmolVLM2 instance from a single safetensor file.
    ///
    /// # Arguments
    ///
    /// * `weights_path` - Path to a single safetensor file
    /// * `config` - Configuration for the model
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kornia_vlm::smolvlm2::{SmolVlm2, SmolVlm2Config};
    ///
    /// let model = SmolVlm2::from_single_safetensor(
    ///     "/path/to/model.safetensors",
    ///     SmolVlm2Config::default()
    /// ).unwrap();
    /// ```
    pub fn from_single_safetensor<P: Into<std::path::PathBuf>>(
        weights_path: P,
        config: SmolVlm2Config,
    ) -> Result<Self, SmolVlm2Error> {
        Self::from_safetensors(vec![weights_path], config)
    }

    pub fn clear_context(&mut self) -> Result<(), SmolVlm2Error> {
        self.model.clear_context()?;
        self.txt_processor.clear_history();

        self.index_pos = 0;
        Ok(())
    }

    /// Run the inference of the SmolVLM2 model with previous context added.
    /// Run the inference of the SmolVLM2 model with default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user prompt string to generate a caption for
    /// * `image` - Optional RGB image (`Image<u8, 3, A>`) to use for captioning
    /// * `sample_len` - The maximum number of tokens to generate
    /// * `alloc` - The image allocator to use
    /// * `debug` - Whether to enable debug output
    ///
    /// # Returns
    ///
    /// * `Result<String, SmolVlm2Error>` - The generated caption or error
    pub fn inference<A: ImageAllocator>(
        &mut self,
        prompt: &str, // TODO: should it be structured?
        image: Option<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        alloc: A,
        debug: bool,
    ) -> Result<String, SmolVlm2Error> {
        let full_prompt = self.txt_processor.reformat_with_additional_prompts(
            vec![Message {
                role: Role::User,
                content: vec![Line::Text {
                    text: prompt.to_string(),
                }],
            }],
            true,
        )?;

        let images = if let Some(img) = image {
            vec![img]
        } else {
            vec![]
        };

        let response = self.inference_raw(&full_prompt, images, sample_len, alloc, debug)?;

        Ok(response)
    }

    /// Run the inference of the SmolVLM2 model without the default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `full_prompt` - The fully formatted prompt string (should include <image> tags if images are provided)
    /// * `images` - Vector of RGB images (`Vec<Image<u8, 3, A>>`) to use for captioning
    /// * `sample_len` - The maximum number of tokens to generate
    /// * `_alloc` - The image allocator to use (unused)
    /// * `debug` - Whether to enable debug output
    ///
    /// # Returns
    ///
    /// * `Result<String, SmolVlm2Error>` - The generated caption or error
    pub fn inference_raw<A: ImageAllocator>(
        &mut self,
        full_prompt: &str,
        images: Vec<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        _alloc: A,
        debug: bool,
    ) -> Result<String, SmolVlm2Error> {
        if debug {
            std::io::stdout().flush()?;
        }

        self.response.clear();

        let image_tags_pos: Vec<_> = full_prompt.match_indices("<image>").collect();

        if image_tags_pos.len() != images.len() {
            return Err(SmolVlm2Error::MismatchedImageCount {
                tags: image_tags_pos.len(),
                images: images.len(),
            });
        }

        let mut delta_token = self.txt_processor.encode_all(full_prompt)?;

        if debug {
            debug!("Initial tokens: {delta_token:?}");
        }
        let start_gen = if debug { Some(Instant::now()) } else { None };
        let mut generated_tokens = 0usize;

        for _i in 0..sample_len {
            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;
            let logits = self.model.forward(&input, self.index_pos)?;
            let out_token = self.txt_processor.sample_logits(&logits)?;

            self.index_pos += delta_token.len();
            delta_token.clear();
            delta_token.push(out_token);

            let token_output = self.txt_processor.decode(out_token)?;
            self.txt_processor
                .update_last_textual_response(token_output.clone())?;

            if !self.txt_processor.is_eos(token_output.as_str()) {
                self.response += &token_output;

                if debug {
                    generated_tokens += 1;
                    print!("{token_output}");
                    std::io::stdout().flush()?;
                }
            } else {
                if debug {
                    println!();
                    std::io::stdout().flush()?;
                }
                break;
            }
        }

        if debug {
            if let Some(start_gen) = start_gen {
                let dt = start_gen.elapsed();
                println!(
                    "\n{generated_tokens} tokens generated ({:.2} token/s)",
                    generated_tokens as f64 / dt.as_secs_f64(),
                );

                debug!("Generated text: {:?}", self.response);
            }
        }

        self.txt_processor.add_textual_response(&self.response)?;

        Ok(self.response.clone())
    }

    fn load_model(
        config: &SmolVlm2Config,
        dtype: DType,
        device: &Device,
    ) -> Result<(model::Model, TextProcessor), SmolVlm2Error> {
        let txt_processor = TextProcessor::new(Self::MODEL_IDENTIFIER.into(), config.clone())?;

        // Check if custom weights paths are provided
        let vb = if let Some(ref weights_paths) = config.custom_weights_paths {
            debug!("Loading model from custom weights: {:?}", weights_paths);

            // Convert PathBuf to actual paths for mmap loading
            let paths: Vec<_> = weights_paths.iter().map(|p| p.as_path()).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? }
        } else {
            debug!(
                "Loading model from HuggingFace Hub: {}",
                Self::MODEL_IDENTIFIER
            );

            let api = Api::new()?;
            let repo = api.model(Self::MODEL_IDENTIFIER.to_string());

            let w1 = repo.get("model-00001-of-00002.safetensors")?;
            let w2 = repo.get("model-00002-of-00002.safetensors")?;

            unsafe { VarBuilder::from_mmaped_safetensors(&[w1, w2], dtype, device)? }
        };

        model::Model::load(vb, dtype, device)
            .map_err(SmolVlm2Error::CandleError)
            .map(|m| (m, txt_processor))
    }
}

#[cfg(test)]
mod tests {
    use kornia_tensor::CpuAllocator;

    use super::*;

    // cargo test -p kornia-vlm test_smolvlm2_text_inference --features cuda -- --nocapture --ignored
    #[test]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_text_inference() {
        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            ..Default::default()
        };
        let mut model = SmolVlm2::new(config).unwrap();

        let prompt = "What is life?";
        let sample_len = 500;

        let _response = model
            .inference(prompt, None, sample_len, CpuAllocator, true)
            .expect("Inference failed");
    }

    // Example test for custom weights loading (commented out as it requires actual files)
    #[test]
    #[ignore = "Requires custom safetensor files"]
    fn test_smolvlm2_custom_weights() {
        // Example of loading from custom safetensor files
        let weights_paths = vec![
            "/path/to/model-00001-of-00002.safetensors",
            "/path/to/model-00002-of-00002.safetensors",
        ];

        // Method 1: Using from_safetensors
        let _model1 = SmolVlm2::from_safetensors(weights_paths.clone(), SmolVlm2Config::default());

        // Method 2: Using from_single_safetensor (if you have a single file)
        let _model2 = SmolVlm2::from_single_safetensor(
            "/path/to/single-model.safetensors",
            SmolVlm2Config::default(),
        );

        // Method 3: Using config with custom weights
        let config = SmolVlm2Config::with_custom_weights(weights_paths);
        let _model3 = SmolVlm2::new(config);

        // Method 4: Setting custom weights on existing config
        let mut config = SmolVlm2Config::default();
        config.set_custom_weights(vec!["/path/to/model.safetensors"]);
        let _model4 = SmolVlm2::new(config);
    }
}
