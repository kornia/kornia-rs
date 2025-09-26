mod model;
mod text_postprocessor;
mod text_preprocessor;
pub mod utils;

use log::debug;
use std::io::Write;

use candle_core::IndexOp;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};
use tokenizers::Tokenizer;

use crate::smolvlm2::utils::{SmolVlm2Config, SmolVlm2Error};

pub struct SmolVlm2 {
    model: model::Model,
    config: SmolVlm2Config,
    device: Device,
    logits_processor: LogitsProcessor,
    index_pos: usize,        // index of the next token to be processed
    token_history: Vec<u32>, // stores the history of generated tokens

    first_prompt: bool,
    tokenizer: Tokenizer,
    response: String,
}

impl SmolVlm2 {
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

        let (model, tokenizer) = Self::load_model(dtype, &device)?;

        let logits_processor = if config.do_sample {
            LogitsProcessor::new(config.seed, Some(config.temp), Some(config.top_p))
        } else {
            LogitsProcessor::from_sampling(config.seed, Sampling::ArgMax)
        };

        Ok(Self {
            model,
            tokenizer,
            first_prompt: true,
            config,
            device,
            logits_processor,
            index_pos: 0,
            token_history: Vec::new(),
            response: String::new(),
        })
    }

    pub fn clear_context(&mut self) -> Result<(), SmolVlm2Error> {
        self.model.clear_context();

        self.first_prompt = true;
        self.index_pos = 0;
        self.token_history.clear();
        Ok(())
    }

    /// Run the inference of the SmolVLM2 model with previous context added.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8 image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The length of the generated caption
    /// * `alloc` - The image allocator to use
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
    pub fn inference<A: ImageAllocator>(
        &mut self,
        prompt: &str, // TODO: make it structured
        image: Option<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        alloc: A,
    ) -> Result<String, SmolVlm2Error> {
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

    /// Run the inference of the SmolVLM2 model without the default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8 image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The maximum number of tokens to generate
    /// * `alloc` - The image allocator to use
    /// * `debug` - Debug mode
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
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

        let converted_prompt = String::from(full_prompt);
        let image_tags_pos: Vec<_> = full_prompt.match_indices("<image>").collect();

        if image_tags_pos.len() != images.len() {
            return Err(SmolVlm2Error::MismatchedImageCount {
                tags: image_tags_pos.len(),
                images: images.len(),
            });
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

            let logits = self.model.forward(&input, self.index_pos)?;

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

        Ok(self.response.clone())
    }

    /// Deterministic sampling that always selects the token with the lowest index for ties
    fn sample_deterministic(&self, logits: &Tensor) -> Result<u32, SmolVlm2Error> {
        // Convert to f32 for consistent precision
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        // Filter out NaNs
        let filtered: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
            .collect();
        if filtered.is_empty() {
            return Err(SmolVlm2Error::InvalidLogits(
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

    fn load_model(
        dtype: DType,
        device: &Device,
    ) -> Result<(model::Model, Tokenizer), SmolVlm2Error> {
        let tokenizer =
            Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", None).unwrap();
        let api = Api::new()?;
        let repo = api.model("HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string());

        let w1 = repo.get("model-00001-of-00002.safetensors")?;
        let w2 = repo.get("model-00002-of-00002.safetensors")?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w1, w2], dtype, device)? };

        model::Model::load(vb, dtype, device)
            .map_err(|e| SmolVlm2Error::CandleError(e))
            .map(|m| (m, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use kornia_tensor::CpuAllocator;

    use super::*;

    #[test]
    #[ignore = "Testing for the output prompt, requiring CUDA"]
    fn test_smolvlm2_text_inference() {
        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            ..Default::default()
        };
        let mut model = SmolVlm2::new(config).unwrap();

        let prompt = "What is life?";
        let sample_len = 500;

        let _response = model.inference(prompt, None, sample_len, CpuAllocator);
    }
}
