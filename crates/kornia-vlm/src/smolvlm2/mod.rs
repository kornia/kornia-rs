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
use crate::smolvlm2::text_processor::TextProcessor;
use crate::smolvlm2::video_processor::{VideoProcessor, VideoProcessorConfig};

use kornia_image::Image;

use crate::video::{self, VideoSample};

// Re-export public types for external use
pub use crate::smolvlm2::text_processor::{Line, Message, Role};

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

    #[error("Empty media provided: {0}")]
    EmptyMedia(String),

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

#[derive(Clone)]
pub struct SmolVlm2Config {
    pub seed: u64,
    pub temp: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub do_sample: bool,
    pub repeat_last_n: usize,
    pub debug: bool,
    /// Optional path to custom safetensor weights files. If provided, these will be used
    /// instead of downloading from HuggingFace Hub. Can be a single file or multiple files.
    pub weights_path: Option<Vec<std::path::PathBuf>>,
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
            weights_path: None,
        }
    }
}

pub enum InputMedia<'v, const N: usize, A: ImageAllocator> {
    Images(Vec<Image<u8, 3, A>>),
    Video(Vec<&'v mut VideoSample<N, A>>),
}

pub struct SmolVlm2<const N: usize, A: ImageAllocator> {
    model: model::Model,
    config: SmolVlm2Config,
    device: Device,
    dtype: DType,
    index_pos: usize, // index of the next token to be processed

    txt_processor: TextProcessor,
    img_processor: ImageProcessor<A>,
    vid_processor: VideoProcessor<N>,
    response: String,

    buf_single_zero_tensor: Tensor, // buffer for a single zero tensor
}

impl<const N: usize, A: ImageAllocator> SmolVlm2<N, A> {
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

    /// Create a new SmolVLM2 instance with the default configuration.
    /// This will download weights from HuggingFace Hub.
    pub fn new(config: SmolVlm2Config) -> Result<Self, SmolVlm2Error> {
        #[cfg(feature = "cuda")]
        let (device, dtype) = match Device::cuda_if_available(0) {
            Ok(device) => (device, DType::BF16),
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {e:?}");
                (Device::Cpu, DType::F32)
            }
        };

        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let (model, txt_processor, img_processor, vid_processor) =
            Self::load_model(&config, dtype, &device)?;

        Ok(Self {
            model,
            txt_processor,
            img_processor,
            vid_processor,
            config,
            dtype,
            index_pos: 0,

            response: String::new(),
            buf_single_zero_tensor: Tensor::zeros(&[1], DType::U8, &device)?,

            device,
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
    /// let weights: Vec<&str> = vec![
    ///     "/path/to/model-00001-of-00002.safetensors",
    ///     "/path/to/model-00002-of-00002.safetensors",
    /// ];
    /// let model = SmolVlm2::from_safetensors(weights, SmolVlm2Config::default()).unwrap();
    /// ```
    pub fn from_safetensors<P: Into<std::path::PathBuf>>(
        weights_paths: Vec<P>,
        mut config: SmolVlm2Config,
    ) -> Result<Self, SmolVlm2Error> {
        config.weights_path = Some(weights_paths.into_iter().map(|p| p.into()).collect());
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
        prompt: Vec<text_processor::Message>,
        media: Option<InputMedia<N, A>>,
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
        media: Option<InputMedia<N, A>>,
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

        if let Some(media) = media {
            match media {
                InputMedia::Images(images) => {
                    if images.is_empty() {
                        return Err(SmolVlm2Error::EmptyMedia(
                            "No images provided in Images variant".to_string(),
                        ));
                    }
                    if images.len() > 1 && self.config.debug {
                        debug!("Multiple images provided: {}", images.len());
                    }
                    self.img_processor.binding_images_to_prompt(
                        &mut converted_prompt,
                        images,
                        self.dtype,
                        &self.device,
                        alloc,
                    )?;
                }
                InputMedia::Video(videos) => {
                    if videos.is_empty() {
                        return Err(SmolVlm2Error::EmptyMedia(
                            "No videos provided in Video variant".to_string(),
                        ));
                    }
                    if videos.len() > 1 && self.config.debug {
                        debug!("Multiple video provided: {}", videos.len());
                    }
                    self.vid_processor.binding_videos_to_prompt(
                        &mut converted_prompt,
                        videos,
                        self.dtype,
                        &self.device,
                        alloc,
                    )?;

                    use_video = true;
                }
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
            let img_token_mask = if _i > 0 {
                &self.buf_single_zero_tensor
            } else if use_video {
                &self.vid_processor.get_video_token_mask(&input)?
            } else {
                &self.img_processor.get_image_token_mask(&input)?
            };
            let img_data = if _i > 0 {
                Vec::new()
            } else if use_video {
                self.vid_processor.get_processed_videos()
            } else {
                self.img_processor.get_processed_images()
            };

            let logits =
                self.model
                    .forward(&input, self.index_pos, img_token_mask, img_data, &mut ctx)?;

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
        config: &SmolVlm2Config,
        dtype: DType,
        device: &Device,
    ) -> Result<
        (
            model::Model,
            TextProcessor,
            ImageProcessor<A>,
            VideoProcessor<N>,
        ),
        SmolVlm2Error,
    > {
        let txt_processor = TextProcessor::new(Self::MODEL_IDENTIFIER.into(), config.clone())?
            .with_template_string(Self::UPDATED_VIDEO_CHAT_TEMPLATE.into())?;
        let img_processor =
            ImageProcessor::new(Self::IMG_PROCESSOR_CONFIG, dtype, device, &txt_processor)?;
        let vid_processor =
            VideoProcessor::new(Self::VID_PROCESSOR_CONFIG, device, dtype, &txt_processor)?;

        let vb = if let Some(weights_paths) = &config.weights_path {
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
            .map(|m| (m, txt_processor, img_processor, vid_processor))
    }
}

#[cfg(test)]
mod tests {
    use log::info;
    use std::path::Path;

    use kornia_io::{jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8};
    use kornia_tensor::CpuAllocator;

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
        let mut model = SmolVlm2::<32, _>::new(config).unwrap();

        let prompt = "Describe the image.";
        let sample_len = 500;

        let _response = model
            .inference(
                vec![text_processor::Message {
                    role: text_processor::Role::User,
                    content: vec![
                        text_processor::Line::Image,
                        text_processor::Line::Text {
                            text: prompt.to_string(),
                        },
                    ],
                }],
                Some(InputMedia::Images(vec![image])),
                sample_len,
                CpuAllocator,
            )
            .expect("Inference failed");
    }

    // cargo test -p kornia-vlm test_smolvlm2_speed_comparison --features "cuda,gstreamer,flash-attn" -- --nocapture --ignored
    // RUST_LOG=debug cargo test -p kornia-vlm test_smolvlm2_speed_comparison --features "cuda,gstreamer,flash-attn" -- --nocapture --ignored
    #[test]
    #[ignore = "Speed comparison test - requires CUDA"]
    fn test_smolvlm2_speed_comparison() {
        env_logger::init();

        info!("============================================================");
        info!("SMOLVLM2 RUST SPEED TEST RESULTS");
        info!("============================================================");

        // Create a test blue image (224x224)
        let test_image = create_test_blue_image();

        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            debug: false, // Turn off debug for clean timing
            ..Default::default()
        };
        let mut model = SmolVlm2::<32, _>::new(config).expect("Failed to load model");

        info!("\n==================== Single Image ====================\n");

        let test_prompts = [
            "Describe this image.",
            "What color is this image?",
            "What do you see?",
        ];

        let mut total_time = 0.0;
        let mut total_tokens = 0;
        let mut test_count = 0;

        for (i, prompt) in test_prompts.iter().enumerate() {
            info!("Test {}: '{}'", i + 1, prompt);

            // Run each test twice to match Python behavior
            let mut test_times = Vec::new();
            let mut test_tokens = Vec::new();
            let mut test_outputs = Vec::new();

            for run in 1..=2 {
                // Clear model context before each run
                model.clear_context().expect("Failed to clear context");

                let start_time = std::time::Instant::now();

                let response = model
                    .inference(
                        vec![text_processor::Message {
                            role: text_processor::Role::User,
                            content: vec![
                                text_processor::Line::Image,
                                text_processor::Line::Text {
                                    text: prompt.to_string(),
                                },
                            ],
                        }],
                        Some(InputMedia::Images(vec![test_image.clone()])),
                        200, // Match Python's max_new_tokens=200
                        CpuAllocator,
                    )
                    .expect("Inference failed");

                let duration = start_time.elapsed();
                let time_secs = duration.as_secs_f64();

                // Count tokens (rough approximation by splitting on whitespace)
                let token_count = response.split_whitespace().count();
                let tokens_per_sec = token_count as f64 / time_secs;

                test_times.push(time_secs);
                test_tokens.push(token_count);
                test_outputs.push(response.clone());

                // Truncate output for display
                let truncated_output = if response.len() > 60 {
                    format!("{}...", &response[..57])
                } else {
                    response.clone()
                };

                info!(
                    "  Run {}: {:.2}s, {} tokens, {:.1} tok/s",
                    run, time_secs, token_count, tokens_per_sec
                );
                info!("         Output: {}", truncated_output);

                total_time += time_secs;
                total_tokens += token_count;
                test_count += 1;
            }

            // Calculate averages for this test
            let avg_time = test_times.iter().sum::<f64>() / test_times.len() as f64;
            let avg_tokens = test_tokens.iter().sum::<usize>() as f64 / test_tokens.len() as f64;
            let avg_speed = avg_tokens / avg_time;

            info!(
                "  Average: {:.2}s, {:.0} tokens, {:.1} tok/s\n",
                avg_time, avg_tokens, avg_speed
            );
        }

        // Overall performance summary
        let overall_avg_time = total_time / test_count as f64;
        let overall_avg_speed = total_tokens as f64 / total_time;

        info!("🏁 Single Image Section Performance:");
        info!("   Average Time: {:.2}s", overall_avg_time);
        info!("   Average Speed: {:.1} tokens/second", overall_avg_speed);
        info!("   Total Tests: {}", test_count);

        // // Video test section (if video processing is available)
        // #[cfg(feature = "gstreamer")]
        // {
        //     info!("\n==================== Video 32 frames ====================\n");

        //     let video_path = Path::new("../../car-detection.mp4");
        //     if video_path.exists() {
        //         let video_prompts = [
        //             "Describe what happens in this video.",
        //             "What do you see in this video?",
        //             "Summarize the video content.",
        //         ];

        //         let mut video_total_time = 0.0;
        //         let mut video_total_tokens = 0;
        //         let mut video_test_count = 0;

        //         for (i, prompt) in video_prompts.iter().enumerate() {
        //             info!("Test {}: '{}'", i + 1, prompt);

        //             // Run each test twice to match Python behavior
        //             let mut test_times = Vec::new();
        //             let mut test_tokens = Vec::new();

        //             for run in 1..=2 {
        //                 model.clear_context().expect("Failed to clear context");

        //                 // Using 32 frames to match Python test for fair comparison
        //                 let video_result = VideoBuffer::from_video_path(
        //                     video_path,
        //                     VideoSamplingMethod::Fps(1),
        //                     32, // Match Python's 32 frames for fair comparison
        //                     CpuAllocator,
        //                 );

        //                 match video_result {
        //                     Ok(mut video) => {
        //                         let start_time = std::time::Instant::now();
        //                         let inference_result = model.inference(
        //                             vec![text_processor::Message {
        //                                 role: text_processor::Role::User,
        //                                 content: vec![
        //                                     text_processor::Line::Video,
        //                                     text_processor::Line::Text {
        //                                         text: prompt.to_string(),
        //                                     },
        //                                 ],
        //                             }],
        //                             InputMedia::Video(vec![&mut video]),
        //                             200, // Match Python's max_new_tokens=200
        //                             CpuAllocator,
        //                         );

        //                         match inference_result {
        //                             Ok(response) => {
        //                                 let duration = start_time.elapsed();
        //                                 let time_secs = duration.as_secs_f64();
        //                                 let token_count = response.split_whitespace().count();
        //                                 let tokens_per_sec = token_count as f64 / time_secs;

        //                                 test_times.push(time_secs);
        //                                 test_tokens.push(token_count);

        //                                 let truncated_output = if response.len() > 60 {
        //                                     format!("{}...", &response[..57])
        //                                 } else {
        //                                     response.clone()
        //                                 };

        //                                 info!(
        //                                     "  Run {}: {:.2}s, {} tokens, {:.1} tok/s",
        //                                     run, time_secs, token_count, tokens_per_sec
        //                                 );
        //                                 info!("         Output: {}", truncated_output);

        //                                 video_total_time += time_secs;
        //                                 video_total_tokens += token_count;
        //                                 video_test_count += 1;
        //                             }
        //                             Err(e) => {
        //                                 info!("  Video inference failed: {}", e);
        //                                 break;
        //                             }
        //                         }
        //                     }
        //                     Err(e) => {
        //                         info!("  Failed to load video: {}", e);
        //                         break;
        //                     }
        //                 }
        //             }

        //             // Calculate averages for this test
        //             if !test_times.is_empty() {
        //                 let avg_time = test_times.iter().sum::<f64>() / test_times.len() as f64;
        //                 let avg_tokens =
        //                     test_tokens.iter().sum::<usize>() as f64 / test_tokens.len() as f64;
        //                 let avg_speed = avg_tokens / avg_time;

        //                 info!(
        //                     "  Average: {:.2}s, {:.0} tokens, {:.1} tok/s\n",
        //                     avg_time, avg_tokens, avg_speed
        //                 );
        //             }
        //         }

        //         // Overall video performance summary
        //         if video_test_count > 0 {
        //             let video_avg_time = video_total_time / video_test_count as f64;
        //             let video_avg_speed = video_total_tokens as f64 / video_total_time;

        //             info!("🏁 Video 32 frames Section Performance:");
        //             info!("   Average Time: {:.2}s", video_avg_time);
        //             info!("   Average Speed: {:.1} tokens/second", video_avg_speed);
        //             info!("   Total Tests: {}", video_test_count);
        //         }
        //     } else {
        //         info!("  Video file not found: {:?}", video_path);
        //     }
        // }

        // #[cfg(not(feature = "gstreamer"))]
        // {
        //     info!("\n==================== Video Test ====================\n");
        //     info!("  Video testing requires 'gstreamer' feature to be enabled");
        //     info!("  Run with: cargo test --features \"gstreamer,cuda\"");
        // }

        // Overall summary to match Python format
        info!("\n============================================================");
        info!("📊 SECTION-BY-SECTION PERFORMANCE:");
        info!("============================================================");
        info!("  Single Image:");
        info!("    Average Time: {:.2}s", overall_avg_time);
        info!("    Average Speed: {:.1} tokens/second", overall_avg_speed);
        info!("    Tests Run: {}", test_count);
        info!("");

        info!("============================================================");
        info!("🎯 OVERALL PERFORMANCE:");
        info!("  Average Time: {:.2}s", overall_avg_time);
        info!("  Average Speed: {:.1} tokens/second", overall_avg_speed);
        info!(
            "  Device: {}",
            if cfg!(feature = "cuda") {
                "cuda"
            } else {
                "cpu"
            }
        );
        info!("  Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct");
        // #[cfg(feature = "gstreamer")]
        // {
        //     let video_path = Path::new("../../example_video.mp4");
        //     if video_path.exists() {
        //         info!("  Video Tested: {:?}", video_path);
        //     }
        // }
        info!("============================================================");

        info!("\n============================================================");
        info!("RUST SPEED TEST COMPLETED");
        info!("============================================================");
    }

    /// Create a 224x224 blue RGB image for testing
    fn create_test_blue_image() -> Image<u8, 3, CpuAllocator> {
        let width = 224;
        let height = 224;
        let channels = 3;

        // Create blue color data (RGB: 0, 0, 255)
        let mut data = vec![0u8; width * height * channels];
        for pixel in data.chunks_mut(3) {
            pixel[0] = 0; // R
            pixel[1] = 0; // G
            pixel[2] = 255; // B
        }

        let image_size = kornia_image::ImageSize { width, height };
        Image::new(image_size, data, CpuAllocator).expect("Failed to create test image")
    }
}
