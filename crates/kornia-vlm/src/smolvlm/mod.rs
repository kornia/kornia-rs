mod custom_rmsnorm;
mod model;
mod preprocessor;
mod text_model;
pub mod utils;
mod vision_model;

use std::io::Write;

use crate::{
    context::InferenceContext,
    smolvlm::{
        model::SmolModel,
        preprocessor::SmolVlmImagePreprocessor,
        utils::{SmolVlmConfig, SmolVlmError},
    },
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};
use log::debug;
use preprocessor::get_prompt_split_image;
use tokenizers::Tokenizer;

pub struct SmolVlm<A: ImageAllocator> {
    model: SmolModel,
    tokenizer: Tokenizer,
    image_token_tensor: Tensor,
    config: SmolVlmConfig,
    logits_processor: LogitsProcessor,
    device: Device,
    image_history: Vec<(Tensor, Tensor)>,
    index_pos: usize,        // index of the next token to be processed
    first_prompt: bool,      // whether this is the first prompt
    token_history: Vec<u32>, // stores the history of generated tokens
    preprocessor: SmolVlmImagePreprocessor<A>,
    response: String,
}

impl<A: ImageAllocator> SmolVlm<A> {
    /// Create a new SmolVlm model
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the SmolVlm model
    ///
    /// # Returns
    pub fn new(config: SmolVlmConfig) -> Result<Self, SmolVlmError> {
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

        // TODO: find a way to use FP32 if cuda is not available

        let (model, tokenizer) = Self::load_model(dtype, &device)?;
        let image_token = tokenizer.encode("<image>", false)?;
        let image_token_tensor = Tensor::from_slice(image_token.get_ids(), &[1], &device)?;

        Ok(Self {
            model,
            tokenizer,
            image_token_tensor,
            config,
            logits_processor: if config.do_sample {
                LogitsProcessor::new(config.seed, Some(config.temp), Some(config.top_p))
            } else {
                LogitsProcessor::from_sampling(config.seed, Sampling::ArgMax)
            },
            device: device.clone(),
            image_history: Vec::new(),
            index_pos: 0,
            first_prompt: true,
            token_history: Vec::new(),
            preprocessor: SmolVlmImagePreprocessor::new(1536, 384, &device)?,
            response: String::new(),
        })
    }

    /// Update the configuration of the SmolVLM model
    pub fn update_config(&mut self, config: SmolVlmConfig) -> Result<(), SmolVlmError> {
        self.config = config;
        self.logits_processor = if config.do_sample {
            LogitsProcessor::new(config.seed, Some(config.temp), Some(config.top_p))
        } else {
            LogitsProcessor::from_sampling(config.seed, Sampling::ArgMax)
        };
        Ok(())
    }

    /// Run the inference of the SmolVLM model with previous context added.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8    image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The length of the generated caption
    /// * `alloc` - The image allocator to use
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
    pub fn inference(
        &mut self,
        prompt: &str, // TODO: make it structured
        image: Option<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        alloc: A,
    ) -> Result<String, SmolVlmError> {
        let mut full_prompt = if self.first_prompt {
            self.first_prompt = false;
            String::from("<|im_start|>")
        } else {
            String::new()
        };

        if image.is_some() {
            full_prompt += "User:<image>";
        } else {
            full_prompt += "User: ";
        }

        full_prompt += prompt;
        full_prompt += "<end_of_utterance>\nAssistant:";

        let images = if let Some(img) = image {
            vec![img]
        } else {
            vec![]
        };

        let response = self.inference_raw(&full_prompt, images, sample_len, alloc, false)?;

        Ok(response)
    }

    /// Run the inference of the SmolVLM model without the default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8    image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The length of the generated caption
    /// * `alloc` - The image allocator to use
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
    pub fn inference_raw(
        &mut self,
        full_prompt: &str,
        images: Vec<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        alloc: A,
        debug: bool,
    ) -> Result<String, SmolVlmError> {
        if debug {
            std::io::stdout().flush()?;
        }

        let mut ctx = InferenceContext::new(debug, debug);

        self.response.clear();

        let mut converted_prompt = String::from(full_prompt);
        let image_tags_pos: Vec<_> = full_prompt.match_indices("<image>").collect();
        let image_tag_len = "<image>".len();
        let mut offset = 0;

        if image_tags_pos.len() != images.len() {
            return Err(SmolVlmError::MismatchedImageCount {
                tags: image_tags_pos.len(),
                images: images.len(),
            });
        }

        let mut processed_images = vec![];
        for ((start, _), image) in image_tags_pos.iter().zip(images.into_iter()) {
            let (img_patches, mask_patches, size) =
                self.preprocessor
                    .preprocess(&image, &self.device, alloc.clone())?;
            processed_images.push((img_patches, mask_patches));

            let img_token = get_prompt_split_image(81, size);
            converted_prompt.replace_range(
                &(start + offset)..&(start + offset + image_tag_len),
                &img_token,
            );
            offset += img_token.len() - image_tag_len;
        }

        let full_token = self.tokenizer.encode(converted_prompt, false)?;

        let mut delta_token = full_token.get_ids().to_vec();

        if debug {
            debug!("Initial tokens: {delta_token:?}");
        }
        let start_gen = if debug {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut generated_tokens = 0usize;

        for _i in 0..sample_len {
            self.token_history.extend(&delta_token);
            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;

            let image_token_mask = input.broadcast_eq(&self.image_token_tensor)?;
            let logits = self.model.forward(
                &input,
                self.index_pos,
                &image_token_mask,
                processed_images.iter().map(|(a, b)| (a, b)).collect(),
                &mut ctx,
            )?;
            processed_images.clear();

            let (s, _embed_dim) = logits.dims2()?;
            let last_logit = logits.i((s - 1, ..))?;

            let output_logit = if self.config.do_sample {
                candle_transformers::utils::apply_repeat_penalty(
                    &last_logit,
                    self.config.repeat_penalty,
                    &delta_token,
                )?
            } else {
                last_logit.clone()
            };

            let last_logit = output_logit;

            let out_token = if self.config.do_sample {
                self.logits_processor.sample(&last_logit)?
            } else {
                // Use deterministic sampling for reproducible results
                self.sample_deterministic(&last_logit)?
            };

            ctx.text_introspector.insert("logits", &last_logit);
            ctx.text_introspector.increment_batch_pos();

            self.index_pos += delta_token.len();
            delta_token.clear();
            delta_token.push(out_token);

            let token_output = self.tokenizer.decode(&[out_token], false)?;

            if token_output != "<end_of_utterance>" {
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

        ctx.text_introspector
            .save_as("examples/smol_vlm/validation_data/rust_isp_decoder.safetensors");
        ctx.vis_introspector
            .save_as("examples/smol_vlm/validation_data/rust_isp_encoder.safetensors");

        Ok(self.response.clone())
    }

    /// Clear the context of the model (reset image and token history)
    pub fn clear_context(&mut self) {
        self.model.reset_cache();

        self.image_history.clear();
        self.index_pos = 0;
        self.first_prompt = true;
        self.token_history.clear();
    }

    /// Deterministic sampling that always selects the token with the lowest index for ties
    fn sample_deterministic(&self, logits: &Tensor) -> Result<u32, SmolVlmError> {
        // Convert to f32 for consistent precision
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        // Filter out NaNs
        let filtered: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
            .collect();
        if filtered.is_empty() {
            return Err(SmolVlmError::InvalidLogits(
                "No valid logits found - all values may be NaN or invalid".to_string(),
            ));
        }

        // Find the maximum value among valid logits
        let max_value = filtered
            .iter()
            .map(|&(_, v)| v)
            .fold(f32::NEG_INFINITY, f32::max);

        // Find all indices with the maximum value (exact equality)
        let max_indices: Vec<usize> = filtered
            .iter()
            .filter(|&&(_, v)| v == max_value)
            .map(|&(i, _)| i)
            .collect();

        // Always select the first index (deterministic tiebreaker) in order of the token
        let best_token = max_indices[0] as u32;

        Ok(best_token)
    }

    #[inline]
    pub fn image_history_count(&self) -> usize {
        self.image_history.len()
    }

    // utility function to load the model
    fn load_model(dtype: DType, device: &Device) -> Result<(SmolModel, Tokenizer), SmolVlmError> {
        let tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None)?;
        let api = Api::new()?;
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors")?;
        let mut weights = candle_core::safetensors::load(weights, device)?;

        for value in weights.values_mut() {
            if value.dtype() != dtype {
                *value = value.to_dtype(dtype)?;
            }
        }

        let model = SmolModel::load(&weights)?;

        Ok((model, tokenizer))
    }
}
