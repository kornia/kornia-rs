mod model;
pub(super) mod text_processor;
pub mod utils;

use log::debug;
use std::io::Write;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};

use crate::smolvlm2::text_processor::{Line, Message, Role, TextProcessor};
use crate::smolvlm2::utils::{SmolVlm2Config, SmolVlm2Error};

pub struct SmolVlm2 {
    model: model::Model,
    _config: SmolVlm2Config,
    device: Device,
    index_pos: usize, // index of the next token to be processed

    txt_processor: TextProcessor,
    response: String,
}

impl SmolVlm2 {
    const MODEL_IDENTIFIER: &'static str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct";

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

        let (model, txt_processor) = Self::load_model(config, dtype, &device)?;

        Ok(Self {
            model,
            txt_processor,
            _config: config,
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
        prompt: &str, // TODO: should it be structured?
        image: Option<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        alloc: A,
        debug: bool,
    ) -> Result<String, SmolVlm2Error> {
        let full_prompt = self.txt_processor.reformat_with_additional_prompts(
            vec![Message {
                role: Role::User,
                content: vec![Line::Text {
                    text: prompt.to_string(),
                }],
            }],
            true,
        )?;

        let images = if let Some(img) = image {
            vec![img]
        } else {
            vec![]
        };

        let response = self.inference_raw(&full_prompt, images, sample_len, alloc, debug)?;

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

        let image_tags_pos: Vec<_> = full_prompt.match_indices("<image>").collect();

        if image_tags_pos.len() != images.len() {
            return Err(SmolVlm2Error::MismatchedImageCount {
                tags: image_tags_pos.len(),
                images: images.len(),
            });
        }

        let mut delta_token = self.txt_processor.encode_all(full_prompt)?;

        if debug {
            debug!("Initial tokens: {delta_token:?}");
        }
        let start_gen = if debug { Some(Instant::now()) } else { None };
        let mut generated_tokens = 0usize;

        for _i in 0..sample_len {
            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;
            let logits = self.model.forward(&input, self.index_pos)?;
            let out_token = self.txt_processor.sample_logits(&logits)?;

            self.index_pos += delta_token.len();
            delta_token.clear();
            delta_token.push(out_token);

            let token_output = self.txt_processor.decode(out_token)?;
            self.txt_processor
                .update_last_textual_response(token_output.clone())?;

            if !self.txt_processor.is_eos(token_output.as_str()) {
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

        self.txt_processor.add_textual_response(&self.response)?;

        Ok(self.response.clone())
    }

    fn load_model(
        config: SmolVlm2Config,
        dtype: DType,
        device: &Device,
    ) -> Result<(model::Model, TextProcessor), SmolVlm2Error> {
        let txt_processor = TextProcessor::new(Self::MODEL_IDENTIFIER.into(), config)?;

        let api = Api::new()?;
        let repo = api.model(Self::MODEL_IDENTIFIER.to_string());

        let w1 = repo.get("model-00001-of-00002.safetensors")?;
        let w2 = repo.get("model-00002-of-00002.safetensors")?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w1, w2], dtype, device)? };

        model::Model::load(vb, dtype, device)
            .map_err(SmolVlm2Error::CandleError)
            .map(|m| (m, txt_processor))
    }
}

#[cfg(test)]
mod tests {
    use kornia_tensor::CpuAllocator;

    use super::*;

    // cargo test -p kornia-vlm test_smolvlm2_text_inference --features cuda -- --nocapture --ignored
    #[test]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_text_inference() {
        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            ..Default::default()
        };
        let mut model = SmolVlm2::new(config).unwrap();

        let prompt = "What is life?";
        let sample_len = 500;

        let _response = model
            .inference(prompt, None, sample_len, CpuAllocator, true)
            .expect("Inference failed");
    }
}
