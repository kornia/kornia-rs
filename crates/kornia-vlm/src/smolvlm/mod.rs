mod model;
mod preprocessor;
mod text_model;
pub mod utils;
mod vision_model;

use std::io;
use std::io::Write;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use kornia_image::allocator::ImageAllocator;
use kornia_image::Image;
use tokenizers::Tokenizer;

use preprocessor::{get_prompt_split_image, preprocess_image};

use crate::smolvlm::model::SmolModel;
use crate::smolvlm::utils::{SmolVlmConfig, SmolVlmError};

pub struct SmolVlm {
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
}

impl SmolVlm {
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
            logits_processor: LogitsProcessor::new(
                config.seed,
                Some(config.temp),
                Some(config.top_p),
            ),
            device,
            image_history: Vec::new(),
            index_pos: 0,
            first_prompt: true,
            token_history: Vec::new(),
        })
    }

    /// Run the inference of the SmolVLM model with previous context added.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8    image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The length of the generated caption
    /// * `stdout_debug` - Whether to print the debug information to the stdout
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
    pub fn inference<A: ImageAllocator>(
        &mut self,
        image: Option<Image<u8, 3, A>>,
        prompt: &str,
        sample_len: usize, // per prompt
        stdout_debug: bool,
    ) -> Result<String, SmolVlmError> {
        if stdout_debug {
            std::io::stdout().flush()?;
        }

        let mut response = String::new(); // collection of tokens

        let mut full_prompt = if self.first_prompt {
            self.first_prompt = false;
            String::from("<|im_start|>")
        } else {
            String::new()
        };

        if let Some(raw_img) = image {
            let (img_patches, mask_patches, size) =
                preprocess_image(raw_img, 1920, 384, &self.device);

            let img_token = get_prompt_split_image(81, size);
            full_prompt += "\nUser:<image>";
            full_prompt = full_prompt.replace("<image>", &img_token);

            self.image_history.push((img_patches, mask_patches));
        } else {
            full_prompt += "\nUser: ";
        }

        full_prompt += prompt;
        full_prompt += "<end_of_utterance>\nAssistant:";

        let full_token = self.tokenizer.encode(full_prompt, false)?;

        let mut delta_token = full_token.get_ids().to_vec();

        let start_gen = std::time::Instant::now();
        let mut generated_tokens = 0usize;

        for _i in 0..sample_len {
            self.token_history.extend(&delta_token);

            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;
            let vision_data = if !self.image_history.is_empty() {
                let image_token_mask = input.broadcast_eq(&self.image_token_tensor)?;
                Some((
                    image_token_mask,
                    &self.image_history[0].0,
                    &self.image_history[0].1,
                ))
            } else {
                None
            };

            let logits = self.model.forward(&input, self.index_pos, vision_data)?;
            let (s, _embed_dim) = logits.dims2()?;
            let last_logit = logits.i((s - 1, ..))?;

            let last_logit = candle_transformers::utils::apply_repeat_penalty(
                &last_logit,
                self.config.repeat_penalty,
                &delta_token,
            )?;
            let out_token = self.logits_processor.sample(&last_logit)?;

            self.index_pos += delta_token.len();
            delta_token.clear();
            delta_token.push(out_token);

            let token_output = self.tokenizer.decode(&[out_token], false)?;

            if token_output != "<end_of_utterance>" {
                response += &token_output;
                generated_tokens += 1;

                if stdout_debug {
                    print!("{token_output}");
                    io::stdout().flush().unwrap();
                }
            } else {
                if stdout_debug {
                    println!();
                    io::stdout().flush().unwrap();
                }

                break;
            }
        }

        if stdout_debug {
            let dt = start_gen.elapsed();
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }

        Ok(response)
    }

    #[inline]
    pub fn image_history_count(&self) -> usize {
        self.image_history.len()
    }

    // utility function to load the model
    fn load_model(_dtype: DType, device: &Device) -> Result<(SmolModel, Tokenizer), SmolVlmError> {
        let tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None)?;
        let api = Api::new()?;
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors")?;
        let weights = candle_core::safetensors::load(weights, device)?;
        let model = SmolModel::load(&weights)?;

        Ok((model, tokenizer))
    }
}

#[cfg(test)]
mod tests {}
