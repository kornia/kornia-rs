//! SmolVLM library wrapping the kornia-models implementation
//!
//! This crate provides a convenient API for using the SmolVLM model
//! from kornia-models. SmolVLM is a lightweight vision-language model
//! designed for efficient image understanding and captioning.

// Re-export key types from kornia-models
pub use kornia_models::{
    SmolVLMModel,
    SmolVLMConfig,
    SmolVLMVariant,
    Error,
};

/// Helper function to create a SmolVLM model with the appropriate backend
///
/// # Arguments
///
/// * `variant` - The model variant to use
/// * `model_dir` - Path to the directory containing model files
/// * `use_cpu` - Whether to use CPU (true) or GPU (false)
///
/// # Returns
///
/// A new SmolVLM model instance
///
/// # Errors
///
/// Returns an error if model initialization fails
pub fn create_model(
    variant: SmolVLMVariant,
    model_dir: impl AsRef<std::path::Path>,
    use_cpu: bool,
) -> Result<SmolVLMModel, Error> {
    let config = SmolVLMConfig::new(variant, model_dir, use_cpu);
    
    #[cfg(feature = "candle")]
    return SmolVLMModel::new_candle(config);
    
    #[cfg(all(feature = "onnx", not(feature = "candle")))]
    return SmolVLMModel::new_onnx(config);
    
    #[cfg(not(any(feature = "candle", feature = "onnx")))]
    return Err(Error::UnsupportedBackend(
        "No backend enabled. Enable either 'candle' or 'onnx' feature.".to_string()
    ));
}

/// Process an image with a prompt
///
/// # Arguments
///
/// * `image_path` - Path to the input image
/// * `prompt` - Text prompt for the model
/// * `variant` - The model variant to use
/// * `model_dir` - Path to the directory containing model files
/// * `use_cpu` - Whether to use CPU (true) or GPU (false)
///
/// # Returns
///
/// The generated text description
///
/// # Errors
///
/// Returns an error if processing fails
pub fn process_image(
    image_path: impl AsRef<std::path::Path>,
    prompt: &str,
    variant: SmolVLMVariant,
    model_dir: impl AsRef<std::path::Path>,
    use_cpu: bool,
) -> Result<String, Error> {
    let model = create_model(variant, model_dir, use_cpu)?;
    model.generate_from_image(image_path, prompt)
}