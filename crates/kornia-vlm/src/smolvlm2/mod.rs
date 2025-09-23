mod model;
mod utils;

use std::io::{self, Write};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};
use tokenizers::Tokenizer;
use candle_core::IndexOp;

use crate::smolvlm2::utils::{SmolVlm2Error, SmolVlm2Config};

pub struct SmolVlm2 {
    model: model::Model,
    tokenizer: Tokenizer,
    first_prompt: bool,
    config: SmolVlm2Config,
    device: Device,
    logits_processor: LogitsProcessor,
    image_history: Vec<(Tensor, Tensor)>,
    index_pos: usize,        // index of the next token to be processed
    token_history: Vec<u32>, // stores the history of generated tokens
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

        Ok(Self {
            model,
            tokenizer,
            first_prompt: true,
            config,
            device,
            logits_processor: LogitsProcessor::new(
                config.seed,
                Some(config.temp),
                Some(config.top_p),
            ),
            image_history: Vec::new(),
            index_pos: 0,
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
    ) -> Result<String, SmolVlm2Error> {
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

        // if let Some(raw_img) = image {
        //     let (img_patches, mask_patches, size) =
        //         preprocess_image(raw_img, 1920, 384, &self.device);

        //     let img_token = get_prompt_split_image(81, size);
        //     full_prompt += "\nUser:<image>";
        //     full_prompt = full_prompt.replace("<image>", &img_token);

        //     self.image_history.push((img_patches, mask_patches));
        // } else {
        full_prompt += "\nUser: ";
        // }

        full_prompt += prompt;
        full_prompt += "<end_of_utterance>\nAssistant:";

        let full_token = self.tokenizer.encode(full_prompt, false)?;

        let mut delta_token = full_token.get_ids().to_vec();

        let start_gen = std::time::Instant::now();
        let mut generated_tokens = 0usize;

        for _i in 0..sample_len {
            self.token_history.extend(&delta_token);

            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;
            let vision_data = 
            // if !self.image_history.is_empty() {
            //     let image_token_mask = input.broadcast_eq(&self.image_token_tensor)?;
            //     Some((
            //         image_token_mask,
            //         &self.image_history[0].0,
            //         &self.image_history[0].1,
            //     ))
            // } else {
                None;
            // };

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
        
        
        // let w1 = repo.get("model-00001-of-00002.safetensors")?;
        // let w2 = repo.get("model-00002-of-00002.safetensors")?;
        // let weights = candle_core::safetensors::load(w1, device)?;
        // let weights2 = candle_core::safetensors::load(w2, device)?;

        // println!("Variables loaded: {:?}", weights.keys());
        // println!("--------------------------------------");
        // println!("Variables2 loaded: {:?}", weights2.keys());

        model::Model::load(vb, dtype, device)
            .map_err(|e| SmolVlm2Error::CandleError(e))
            .map(|m| (m, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::VarBuilder;
    use hf_hub::api::sync::Api;
    use kornia_tensor::CpuAllocator;
    use tokenizers::Tokenizer;

    use super::*;

    #[test]
    fn test_smolvlm2_inference() {
        let config = SmolVlm2Config::default();
        let mut model = SmolVlm2::new(config).unwrap();

        let prompt = "What is life?";
        let sample_len = 50;
        let stdout_debug = true;

        let response = model.inference::<CpuAllocator>(None, prompt, sample_len, stdout_debug);
    }
}
