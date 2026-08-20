/// Configuration for the Moondream model
#[derive(Clone, Copy)]
pub struct MoondreamConfig {
    /// Seed for the sampling RNG.
    pub seed: u64,
    /// Sampling temperature. `None` (or `Some(0.0)`) selects greedy decoding.
    pub temp: Option<f64>,
    /// Nucleus sampling probability. Ignored under greedy decoding.
    pub top_p: Option<f64>,
    /// Penalty applied to recently generated tokens. `1.0` disables it.
    pub repeat_penalty: f32,
    /// How many trailing tokens the repeat penalty looks at.
    pub repeat_last_n: usize,
}

impl Default for MoondreamConfig {
    fn default() -> Self {
        Self {
            // Greedy by default: a perception node feeding downstream consumers
            // should give the same answer for the same frame.
            seed: 299792458,
            temp: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MoondreamError {
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

    #[error("Cannot find the <|endoftext|> token in the tokenizer vocabulary")]
    SpecialTokenNotFound,

    #[error("Empty prompts are not supported by the Moondream model")]
    EmptyPrompt,
}
