use kornia_image::{allocator::ImageAllocator, Image};

use crate::video::{self, Video};

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

    #[error(transparent)]
    VideoError(#[from] video::VideoError),

    #[error("Invalid logits detected: {0}")]
    InvalidLogits(String),

    #[error("Invalid encoding detected: {0}")]
    InvalidEncoding(String),

    #[error("Missing chat template: {0}")]
    MissingChatTemplate(String),

    #[error("Message history mistmatch: {0}")]
    MessageHistoryMismatch(String),

    #[error("Missing tokenizer")]
    MissingTokenizer,

    #[error("Cannot find the <end_of_utterance> token")]
    EosTokenNotFound,

    #[error("Mismatched image count: tags = {tags}, images = {images}")]
    MismatchedImageCount { tags: usize, images: usize },

    #[error("Mismatched video count: tags = {tags}, videos = {videos}")]
    MismatchedVideoCount { tags: usize, videos: usize },

    #[error("Video processing error")]
    VideoProcessingError,
}

/// Configuration for the SmolVLM2 model

#[derive(Clone, Copy)]
pub struct SmolVlm2Config {
    pub seed: u64,
    pub temp: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub do_sample: bool,
    pub repeat_last_n: usize,
    pub debug: bool,
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
            debug: false,
        }
    }
}

pub enum InputMedia<A: ImageAllocator> {
    Images(Vec<Image<u8, 3, A>>),
    Video(Vec<Video<A>>),
    None,
}
