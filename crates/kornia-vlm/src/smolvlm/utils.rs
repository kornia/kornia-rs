/// Configuration for the SmolVLM model
#[derive(Clone, Copy)]
pub struct SmolVlmConfig {
    pub seed: u64,
    pub temp: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub do_sample: bool,
    pub debug: bool,
}

impl Default for SmolVlmConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            temp: 1.0,
            top_p: 0.8,
            repeat_penalty: 1.0,
            do_sample: true,
            debug: false,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SmolVlmError {
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

    #[error("Mismatched image count: found {tags} <image> tags but {images} images provided")]
    MismatchedImageCount { tags: usize, images: usize },

    #[error("Invalid logits detected: {0}")]
    InvalidLogits(String),
}
