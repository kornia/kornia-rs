//! ONNX Runtime backend for SmolVLM

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "onnx")]
use ort::{
    tensor::{InputTensor, OrtOwnedTensor, TensorElementDataType},
    Environment, GraphOptimizationLevel, OrtError, Session, SessionBuilder, Value,
};

use crate::smolvlm::common::{ModelSize, ProcessedImage, SmolVLMConfig, SmolVLMError};
use crate::smolvlm::tokenizer::Tokenizer;

/// OnnxBackend implements the SmolVLM model using ONNX Runtime
#[derive(Debug)]
pub struct OnnxBackend {
    config: SmolVLMConfig,
    model_path: PathBuf,
    #[cfg(feature = "onnx")]
    vision_session: Session,
    #[cfg(feature = "onnx")]
    llm_session: Session,
    #[cfg(feature = "onnx")]
    tokenizer: Arc<Tokenizer>,
}

impl OnnxBackend {
    /// Create a new ONNX Runtime backend from the given model path and configuration
    pub fn new(model_path: &str, config: &SmolVLMConfig) -> Result<Self, SmolVLMError> {
        // Validate configuration
        config.validate()?;

        // Check if model path exists
        let model_path = Path::new(model_path);
        if !model_path.exists() {
            return Err(SmolVLMError::ModelLoadError(format!(
                "Model path does not exist: {}",
                model_path.display()
            )));
        }

        #[cfg(feature = "onnx")]
        {
            // Create ONNX Runtime environment
            let environment = Environment::builder()
                .with_name("SmolVLM")
                .build()
                .map_err(|e| {
                    SmolVLMError::OnnxError(format!("Failed to create ONNX environment: {}", e))
                })?;

            // Create ONNX Runtime session options
            let mut session_options = SessionBuilder::new(&environment)
                .map_err(|e| {
                    SmolVLMError::OnnxError(format!("Failed to create ONNX session builder: {}", e))
                })?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| {
                    SmolVLMError::OnnxError(format!("Failed to set optimization level: {}", e))
                })?;

            // Adjust thread settings based on hardware platform
            if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
                // Specific optimizations for NVIDIA Jetson (ARM)
                session_options = session_options
                    .with_inter_threads(1)
                    .map_err(|e| {
                        SmolVLMError::OnnxError(format!("Failed to set inter threads: {}", e))
                    })?
                    .with_intra_threads(4)
                    .map_err(|e| {
                        SmolVLMError::OnnxError(format!("Failed to set intra threads: {}", e))
                    })?;
            }

            // Initialize tokenizer
            let tokenizer = Arc::new(Tokenizer::new(&config)?);

            // Find the model files
            let vision_model_path = model_path.join("vision_encoder.onnx");
            let llm_model_path = model_path.join("llm.onnx");

            if !vision_model_path.exists() || !llm_model_path.exists() {
                return Err(SmolVLMError::ModelLoadError(format!(
                    "Required model files not found in {}",
                    model_path.display()
                )));
            }

            // Load the vision encoder model
            log::info!(
                "Loading vision encoder model from {}",
                vision_model_path.display()
            );
            let vision_session = session_options
                .with_model_from_file(&vision_model_path)
                .map_err(|e| {
                    SmolVLMError::OnnxError(format!("Failed to load vision encoder model: {}", e))
                })?;

            // Load the language model
            log::info!("Loading language model from {}", llm_model_path.display());
            let llm_session = session_options
                .with_model_from_file(&llm_model_path)
                .map_err(|e| {
                    SmolVLMError::OnnxError(format!("Failed to load language model: {}", e))
                })?;

            Ok(Self {
                config: config.clone(),
                model_path: model_path.to_path_buf(),
                vision_session,
                llm_session,
                tokenizer,
            })
        }

        #[cfg(not(feature = "onnx"))]
        {
            Err(SmolVLMError::ModelLoadError(
                "ONNX Runtime support is not enabled".to_string(),
            ))
        }
    }

    /// Get the metadata about the model
    #[cfg(feature = "onnx")]
    pub fn get_metadata(&self) -> Result<HashMap<String, String>, SmolVLMError> {
        let vision_metadata = self.vision_session.metadata().map_err(|e| {
            SmolVLMError::OnnxError(format!("Failed to get vision model metadata: {}", e))
        })?;

        let llm_metadata = self
            .llm_session
            .metadata()
            .map_err(|e| SmolVLMError::OnnxError(format!("Failed to get LLM metadata: {}", e)))?;

        let mut result = HashMap::new();

        // Vision model metadata
        if let Some(producer_name) = vision_metadata.producer_name() {
            result.insert("vision_producer".to_string(), producer_name.to_string());
        }

        if let Some(domain) = vision_metadata.domain() {
            result.insert("vision_domain".to_string(), domain.to_string());
        }

        result.insert(
            "vision_version".to_string(),
            vision_metadata.version().to_string(),
        );

        // LLM metadata
        if let Some(producer_name) = llm_metadata.producer_name() {
            result.insert("llm_producer".to_string(), producer_name.to_string());
        }

        if let Some(domain) = llm_metadata.domain() {
            result.insert("llm_domain".to_string(), domain.to_string());
        }

        result.insert(
            "llm_version".to_string(),
            llm_metadata.version().to_string(),
        );

        // Model configuration
        result.insert(
            "model_size".to_string(),
            format!("{:?}", self.config.model_size),
        );
        result.insert(
            "image_size".to_string(),
            format!("{:?}", self.config.image_size),
        );

        Ok(result)
    }

    /// Print information about the model
    #[cfg(feature = "onnx")]
    pub fn print_model_info(&self) -> Result<(), SmolVLMError> {
        // Get vision model inputs
        let vision_inputs = self
            .vision_session
            .inputs
            .iter()
            .map(|input| {
                format!(
                    "Vision Input: '{}' with dimensions {:?} and type {:?}",
                    input.name,
                    input.dimensions(),
                    input.input_type
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Get vision model outputs
        let vision_outputs = self
            .vision_session
            .outputs
            .iter()
            .map(|output| {
                format!(
                    "Vision Output: '{}' with dimensions {:?} and type {:?}",
                    output.name,
                    output.dimensions(),
                    output.output_type
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Get LLM inputs
        let llm_inputs = self
            .llm_session
            .inputs
            .iter()
            .map(|input| {
                format!(
                    "LLM Input: '{}' with dimensions {:?} and type {:?}",
                    input.name,
                    input.dimensions(),
                    input.input_type
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Get LLM outputs
        let llm_outputs = self
            .llm_session
            .outputs
            .iter()
            .map(|output| {
                format!(
                    "LLM Output: '{}' with dimensions {:?} and type {:?}",
                    output.name,
                    output.dimensions(),
                    output.output_type
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        log::info!("Vision Model:");
        log::info!("Inputs:\n{}", vision_inputs);
        log::info!("Outputs:\n{}", vision_outputs);

        log::info!("Language Model:");
        log::info!("Inputs:\n{}", llm_inputs);
        log::info!("Outputs:\n{}", llm_outputs);

        // Get model metadata
        let metadata = self.get_metadata()?;
        log::info!("Model metadata: {:?}", metadata);

        Ok(())
    }

    /// Get input names of the vision model
    #[cfg(feature = "onnx")]
    fn get_vision_input_names(&self) -> Vec<String> {
        self.vision_session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    /// Get output names of the vision model
    #[cfg(feature = "onnx")]
    fn get_vision_output_names(&self) -> Vec<String> {
        self.vision_session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    /// Get input names of the LLM
    #[cfg(feature = "onnx")]
    fn get_llm_input_names(&self) -> Vec<String> {
        self.llm_session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    /// Get output names of the LLM
    #[cfg(feature = "onnx")]
    fn get_llm_output_names(&self) -> Vec<String> {
        self.llm_session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    /// Run the vision encoder to extract image features
    #[cfg(feature = "onnx")]
    fn encode_vision(&self, image: &ProcessedImage) -> Result<Vec<f32>, SmolVLMError> {
        // Create input tensor from processed image
        let input_shape = vec![
            image.shape.0 as i64, // batch size
            image.shape.1 as i64, // channels
            image.shape.2 as i64, // height
            image.shape.3 as i64, // width
        ];

        let input_tensor = InputTensor::FloatTensor(&image.data, &input_shape);

        // Get input names
        let input_names = self.get_vision_input_names();
        if input_names.is_empty() {
            return Err(SmolVLMError::OnnxError(
                "No inputs found in vision model".to_string(),
            ));
        }

        // Run vision model inference
        let inputs = vec![(input_names[0].as_str(), input_tensor)];
        let outputs = self
            .vision_session
            .run(inputs)
            .map_err(|e| SmolVLMError::OnnxError(format!("Failed to run vision model: {}", e)))?;

        // Process outputs
        if outputs.is_empty() {
            return Err(SmolVLMError::OnnxError(
                "No outputs from vision model".to_string(),
            ));
        }

        // Extract features from the output tensor
        let output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().map_err(|e| {
            SmolVLMError::OnnxError(format!("Failed to extract vision output: {}", e))
        })?;

        // Convert to Vec<f32>
        let features = output
            .view()
            .as_slice()
            .map_err(|e| SmolVLMError::OnnxError(format!("Failed to get vision features: {}", e)))?
            .to_vec();

        Ok(features)
    }

    /// Generate text using the LLM based on image features and input tokens
    #[cfg(feature = "onnx")]
    fn generate_text(
        &self,
        image_features: &[f32],
        input_ids: &[i64],
        max_length: i64,
    ) -> Result<Vec<i64>, SmolVLMError> {
        // Prepare input tensors
        let batch_size = 1;
        let feature_dim = image_features.len() as i64 / batch_size;
        let image_feature_shape = vec![batch_size, feature_dim];
        let image_feature_tensor = InputTensor::FloatTensor(image_features, &image_feature_shape);

        let input_ids_shape = vec![batch_size, input_ids.len() as i64];
        let input_ids_tensor = InputTensor::Int64Tensor(input_ids, &input_ids_shape);

        let max_length_tensor = InputTensor::Int64Tensor(&[max_length], &[1]);

        // Get input names from the LLM model
        let input_names = self.get_llm_input_names();
        if input_names.len() < 3 {
            return Err(SmolVLMError::OnnxError(format!(
                "Expected at least 3 inputs for LLM model, got {}",
                input_names.len()
            )));
        }

        // Prepare inputs for the model
        let mut inputs = Vec::new();

        // Add inputs based on expected names in the model
        // Note: The actual names would depend on the specific ONNX model
        inputs.push((input_names[0].as_str(), input_ids_tensor)); // Input token IDs
        inputs.push((input_names[1].as_str(), image_feature_tensor)); // Image features
        inputs.push((input_names[2].as_str(), max_length_tensor)); // Max generation length

        // Run inference
        let outputs = self
            .llm_session
            .run(inputs)
            .map_err(|e| SmolVLMError::OnnxError(format!("Failed to run LLM: {}", e)))?;

        // Process outputs
        if outputs.is_empty() {
            return Err(SmolVLMError::OnnxError("No outputs from LLM".to_string()));
        }

        // Extract token IDs from the output tensor
        let output: OrtOwnedTensor<i64, _> = outputs[0]
            .try_extract()
            .map_err(|e| SmolVLMError::OnnxError(format!("Failed to extract LLM output: {}", e)))?;

        let token_ids = output
            .view()
            .as_slice()
            .map_err(|e| SmolVLMError::OnnxError(format!("Failed to get token IDs: {}", e)))?
            .to_vec();

        Ok(token_ids)
    }

    /// Generate text based on the given processed image and prompt
    pub fn generate(&self, image: &ProcessedImage, prompt: &str) -> Result<Vec<u32>, SmolVLMError> {
        #[cfg(feature = "onnx")]
        {
            // Log start of generation
            log::info!("Generating text using ONNX backend");
            log::info!("Image shape: {:?}", image.shape);
            log::info!("Prompt: {}", prompt);

            // Encode the prompt to token IDs
            let input_token_ids = self.tokenizer.encode(prompt)?;

            // Add special tokens (BOS)
            let input_token_ids = self.tokenizer.prepare_input(&input_token_ids);
            log::info!("Input tokens: {:?}", input_token_ids);

            // Step 1: Run the vision encoder to extract image features
            log::info!("Running vision encoder...");
            let image_features = self.encode_vision(image)?;
            log::info!("Extracted {} image features", image_features.len());

            // Step 2: Convert token IDs from u32 to i64 (required by ONNX)
            let input_ids: Vec<i64> = input_token_ids.iter().map(|&id| id as i64).collect();

            // Step 3: Run the LLM to generate text from image features and tokens
            log::info!("Running language model for text generation...");
            let max_length = 256;
            let output_ids = self.generate_text(&image_features, &input_ids, max_length as i64)?;

            // Step 4: Convert generated token IDs back to u32
            let token_ids: Vec<u32> = output_ids.iter().map(|&id| id as u32).collect();
            log::info!("Generated {} tokens", token_ids.len());

            Ok(token_ids)
        }

        #[cfg(not(feature = "onnx"))]
        {
            Err(SmolVLMError::GenerationError(
                "Generation not available: onnx feature is disabled".to_string(),
            ))
        }
    }

    /// Perform greedy decoding to generate text tokens
    #[cfg(feature = "onnx")]
    fn greedy_decode(
        &self,
        image_features: &[f32],
        input_ids: &[i64],
        max_length: i64,
        eos_token_id: i64,
    ) -> Result<Vec<i64>, SmolVLMError> {
        let feature_dim = image_features.len();
        let batch_size = 1;

        // Create image feature shape
        let image_feature_shape = vec![batch_size, feature_dim as i64];

        // Initialize sequence with input IDs
        let mut generated_ids = input_ids.to_vec();

        // Get input and output names
        let input_names = self.get_llm_input_names();
        let output_names = self.get_llm_output_names();

        if input_names.len() < 2 || output_names.is_empty() {
            return Err(SmolVLMError::OnnxError(format!(
                "Invalid model structure: inputs={}, outputs={}",
                input_names.len(),
                output_names.len()
            )));
        }

        // Main generation loop
        for _ in 0..max_length {
            // Prepare input shapes
            let seq_len = generated_ids.len() as i64;
            let input_shape = vec![batch_size, seq_len];

            // Create input tensors
            let image_tensor = InputTensor::FloatTensor(image_features, &image_feature_shape);
            let input_ids_tensor = InputTensor::Int64Tensor(&generated_ids, &input_shape);

            // Run a single step of the model
            let inputs = vec![
                (input_names[0].as_str(), input_ids_tensor),
                (input_names[1].as_str(), image_tensor),
            ];

            let outputs = self
                .llm_session
                .run(inputs)
                .map_err(|e| SmolVLMError::OnnxError(format!("Failed to run LLM step: {}", e)))?;

            // Extract logits from output
            let logits: OrtOwnedTensor<f32, _> = outputs[0]
                .try_extract()
                .map_err(|e| SmolVLMError::OnnxError(format!("Failed to extract logits: {}", e)))?;

            // Get logits for the last token (shape: [batch_size, vocab_size])
            let logits_view = logits.view();
            let last_token_idx = seq_len - 1;

            // Find the token with highest probability (greedy decoding)
            let mut max_token_id = 0;
            let mut max_prob = f32::NEG_INFINITY;

            let vocab_size = logits_view.dims()[logits_view.dims().len() - 1];

            for token_id in 0..vocab_size {
                let prob = logits_view[[0, last_token_idx as usize, token_id as usize]];
                if prob > max_prob {
                    max_prob = prob;
                    max_token_id = token_id as i64;
                }
            }

            // Add the predicted token to the sequence
            generated_ids.push(max_token_id);

            // If EOS token is generated, stop
            if max_token_id == eos_token_id {
                break;
            }
        }

        Ok(generated_ids)
    }
}
