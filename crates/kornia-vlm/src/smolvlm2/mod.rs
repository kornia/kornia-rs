pub(super) mod image_processor;
mod model;
pub(super) mod text_processor;
pub(super) mod video_processor;

use log::debug;
use std::io::Write;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use kornia_image::allocator::ImageAllocator;

use crate::context::InferenceContext;
use crate::smolvlm2::image_processor::{ImageProcessor, ImageProcessorConfig};
use crate::smolvlm2::text_processor::{Message, TextProcessor};
use crate::smolvlm2::video_processor::{VideoProcessor, VideoProcessorConfig};

use kornia_image::Image;

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

pub struct SmolVlm2<A: ImageAllocator> {
    model: model::Model,
    config: SmolVlm2Config,
    device: Device,
    index_pos: usize, // index of the next token to be processed

    txt_processor: TextProcessor,
    img_processor: ImageProcessor<A>,
    vid_processor: VideoProcessor,
    response: String,
}

impl<A: ImageAllocator> SmolVlm2<A> {
    const MODEL_IDENTIFIER: &'static str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct";
    const IMG_PROCESSOR_CONFIG: ImageProcessorConfig = ImageProcessorConfig {
        size_longest_edge: 1536,
        max_image_size_longest_edge: 384,
        image_mean: [0.5, 0.5, 0.5],
        image_std: [0.5, 0.5, 0.5],
        rescale_factor: 1.0 / 255.0,
        image_token: "<image>",
    };
    const VID_PROCESSOR_CONFIG: VideoProcessorConfig = VideoProcessorConfig {
        max_frames: 64,
        video_size_longest_edge: 384,
        image_token: "<image>",
        video_token: "<video>",
        frame_mean: [0.5, 0.5, 0.5],
        frame_std: [0.5, 0.5, 0.5],
        rescale_factor: 1.0 / 255.0,
    };
    // https://github.com/huggingface/transformers/blob/3e975acc8bf6d029ec0a54b1c5d0691489dfb051/src/transformers/models/smolvlm/processing_smolvlm.py#L57C26-L57C479
    const UPDATED_VIDEO_CHAT_TEMPLATE: &'static str = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

    pub fn new(config: SmolVlm2Config) -> Result<Self, SmolVlm2Error> {
        #[cfg(feature = "cuda")]
        let (device, dtype) = match Device::cuda_if_available(0) {
            Ok(device) => (device, DType::F32),
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {e:?}");
                (Device::Cpu, DType::BF16)
            }
        };

        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let (model, txt_processor, img_processor, vid_processor) =
            Self::load_model(config, dtype, &device)?;

        Ok(Self {
            model,
            txt_processor,
            img_processor,
            vid_processor,
            config,
            device,
            index_pos: 0,

            response: String::new(),
        })
    }

    pub fn clear_context(&mut self) -> Result<(), SmolVlm2Error> {
        self.model.clear_context()?;
        self.txt_processor.clear_history();

        self.index_pos = 0;
        Ok(())
    }

    /// Run inference with prompt formatting and media input.
    /// # Arguments
    /// * `prompt` - Vector of `Message` representing the conversation history and user prompt.
    /// * `media` - Input media (images, video, or none) as `InputMedia<A>`.
    /// * `sample_len` - Maximum number of tokens to generate.
    /// * `alloc` - Image allocator for image/video processing.
    /// # Returns
    /// * `Result<String, SmolVlm2Error>` - The generated caption or error.
    pub fn inference(
        &mut self,
        prompt: Vec<Message>,
        media: InputMedia<A>,
        sample_len: usize,
        alloc: A,
    ) -> Result<String, SmolVlm2Error> {
        let full_prompt = self
            .txt_processor
            .reformat_with_additional_prompts(prompt, true)?;
        let response = self.inference_raw(&full_prompt, media, sample_len, alloc)?;
        Ok(response)
    }

    /// Run inference with a pre-formatted prompt and media input.
    /// # Arguments
    /// * `full_prompt` - The fully formatted prompt string (should include any required tags for media).
    /// * `media` - Input media (images, video, or none) as `InputMedia<A>`.
    /// * `sample_len` - Maximum number of tokens to generate.
    /// * `alloc` - Image allocator for image/video processing.
    /// # Returns
    /// * `Result<String, SmolVlm2Error>` - The generated caption or error.
    pub fn inference_raw(
        &mut self,
        full_prompt: &str,
        media: InputMedia<A>,
        sample_len: usize,
        alloc: A,
    ) -> Result<String, SmolVlm2Error> {
        if self.config.debug {
            std::io::stdout().flush()?;
        }
        let mut ctx = InferenceContext::new(self.config.debug, self.config.debug);

        self.response.clear();

        let mut converted_prompt = String::from(full_prompt);

        let mut use_video = false;

        match media {
            InputMedia::Images(images) => {
                if images.is_empty() {
                    return Err(SmolVlm2Error::MissingChatTemplate(
                        "No images provided".to_string(),
                    ));
                }
                if images.len() > 1 && self.config.debug {
                    debug!("Multiple images provided: {}", images.len());
                }
                self.img_processor.binding_images_to_prompt(
                    &mut converted_prompt,
                    images,
                    &self.device,
                    alloc,
                )?;
            }
            InputMedia::Video(videos) => {
                if videos.is_empty() {
                    return Err(SmolVlm2Error::MissingChatTemplate(
                        "No video provided".to_string(),
                    ));
                }
                if videos.len() > 1 && self.config.debug {
                    debug!("Multiple video provided: {}", videos.len());
                }
                self.vid_processor.binding_videos_to_prompt(
                    &mut converted_prompt,
                    videos,
                    &self.device,
                    alloc,
                )?;

                use_video = true;
            }
            InputMedia::None => {
                // No media to process
            }
        }

        debug!("Initial converted prompt: {}", converted_prompt);

        let mut delta_token = self.txt_processor.encode_all(&converted_prompt)?;

        if self.config.debug {
            debug!("Initial tokens: {delta_token:?}");
        }
        let start_gen = if self.config.debug {
            Some(Instant::now())
        } else {
            None
        };
        let mut generated_tokens = 0usize;

        for _i in 0..sample_len {
            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;
            let img_token_mask = if use_video {
                self.vid_processor.get_video_token_mask(&input)?
            } else {
                self.img_processor.get_image_token_mask(&input)?
            };
            let img_data = if use_video {
                self.vid_processor.get_processed_videos()
            } else {
                self.img_processor.get_processed_images()
            };

            let logits =
                self.model
                    .forward(&input, self.index_pos, &img_token_mask, img_data, &mut ctx)?;
            self.img_processor.clear_processed_images();
            self.vid_processor.clear_processed_videos();
            let out_token = self.txt_processor.sample_logits(&logits)?;

            self.index_pos += delta_token.len();
            delta_token.clear();
            delta_token.push(out_token);

            let token_output = self.txt_processor.decode(out_token)?;
            self.txt_processor
                .update_last_textual_response(token_output.clone())?;

            if !self.txt_processor.is_eos(token_output.as_str()) {
                self.response += &token_output;

                if self.config.debug {
                    generated_tokens += 1;
                    print!("{token_output}");
                    std::io::stdout().flush()?;
                }
            } else {
                if self.config.debug {
                    println!();
                    std::io::stdout().flush()?;
                }
                break;
            }
        }

        if self.config.debug {
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
        config: SmolVlm2Config,
        dtype: DType,
        device: &Device,
    ) -> Result<
        (
            model::Model,
            TextProcessor,
            ImageProcessor<A>,
            VideoProcessor,
        ),
        SmolVlm2Error,
    > {
        let txt_processor = TextProcessor::new(Self::MODEL_IDENTIFIER.into(), config)?
            .with_template_string(Self::UPDATED_VIDEO_CHAT_TEMPLATE.into())?;
        let img_processor =
            ImageProcessor::new(Self::IMG_PROCESSOR_CONFIG, device, &txt_processor)?;
        let vid_processor =
            VideoProcessor::new(Self::VID_PROCESSOR_CONFIG, device, &txt_processor)?;

        let api = Api::new()?;
        let repo = api.model(Self::MODEL_IDENTIFIER.to_string());

        let w1 = repo.get("model-00001-of-00002.safetensors")?;
        let w2 = repo.get("model-00002-of-00002.safetensors")?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w1, w2], dtype, device)? };

        model::Model::load(vb, dtype, device)
            .map_err(SmolVlm2Error::CandleError)
            .map(|m| (m, txt_processor, img_processor, vid_processor))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use kornia_io::{jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8};
    use kornia_tensor::CpuAllocator;

    use crate::smolvlm2::text_processor::{Line, Role};
    #[cfg(feature = "gstreamer")]
    use crate::video::{Video, VideoSamplingMethod};

    use super::*;

    // cargo test -p kornia-vlm test_smolvlm2_text_inference --features cuda -- --nocapture --ignored
    // RUST_LOG=debug cargo test -p kornia-vlm test_smolvlm2_text_inference --features cuda -- --nocapture --ignored
    #[test]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_text_inference() {
        env_logger::init();

        let path = Path::new("../../100462016.jpeg"); // or .png

        let image = match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(path).ok(),
            Some("png") => read_image_png_rgb8(path).ok(),
            _ => None,
        };
        let image = image.unwrap();

        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            debug: true,
            ..Default::default()
        };
        let mut model = SmolVlm2::new(config).unwrap();

        let prompt = "Describe the image.";
        let sample_len = 500;

        let _response = model
            .inference(
                vec![Message {
                    role: Role::User,
                    content: vec![
                        Line::Image,
                        Line::Text {
                            text: prompt.to_string(),
                        },
                    ],
                }],
                InputMedia::Images(vec![image]),
                sample_len,
                CpuAllocator,
            )
            .expect("Inference failed");
    }

    // cargo test -p kornia-vlm test_smolvlm2_video_inference --features cuda -- --nocapture --ignored
    // RUST_LOG=debug cargo test -p kornia-vlm test_smolvlm2_video_inference --features cuda -- --nocapture --ignored
    #[test]
    #[cfg(feature = "gstreamer")]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_video_inference() {
        env_logger::init();

        let path = Path::new("../../example_video.mp4"); // or .png

        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            debug: true,
            ..Default::default()
        };
        let mut model = SmolVlm2::new(config).unwrap();

        let prompt = "Describe the video. Is the Earth rotating?";
        let sample_len = 500;

        let _response = model
            .inference(
                vec![Message {
                    role: Role::User,
                    content: vec![
                        Line::Video,
                        Line::Text {
                            text: prompt.to_string(),
                        },
                    ],
                }],
                InputMedia::Video(vec![Video::from_video_path(
                    path,
                    VideoSamplingMethod::Fps(1),
                    60,
                    CpuAllocator,
                )
                .unwrap()]),
                sample_len,
                CpuAllocator,
            )
            .expect("Inference failed");
    }
}
