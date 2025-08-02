/// Configuration for the SmolVLM model
#[derive(Clone, Copy)]
pub struct SmolVlmConfig {
    pub seed: u64,
    pub temp: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub do_sample: bool,

    // TODO: check if SmolVLM needs this
    pub repeat_last_n: usize,
}

impl Default for SmolVlmConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            temp: 1.0,
            top_p: 0.8,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            do_sample: true,
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

    // TODO: not used right now (currently, the end token is handled via if/else, which might be preferred)
    #[error("Cannot find the <end_of_utterance> token")]
    EosTokenNotFound,
}
