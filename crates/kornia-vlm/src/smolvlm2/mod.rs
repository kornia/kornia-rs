pub(super) mod image_processor;
mod model;
pub(super) mod text_processor;

use log::debug;
use std::io::Write;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};

use crate::context::InferenceContext;
use crate::smolvlm2::image_processor::{ImageProcessor, ImageProcessorConfig};
use crate::smolvlm2::text_processor::{Message, TextProcessor};
use crate::smolvlm2::utils::{SmolVlm2Config, SmolVlm2Error};

pub struct SmolVlm2<A: ImageAllocator> {
    model: model::Model,
    config: SmolVlm2Config,
    device: Device,
    index_pos: usize, // index of the next token to be processed

    txt_processor: TextProcessor,
    img_processor: ImageProcessor<A>,
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

        let (model, txt_processor, img_processor) = Self::load_model(config, dtype, &device)?;

        Ok(Self {
            model,
            txt_processor,
            img_processor,
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

    /// Run the inference of the SmolVLM2 model with previous context added.
    /// Run the inference of the SmolVLM2 model with default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user prompt string to generate a caption for
    /// * `image` - Optional RGB image (`Image<u8, 3, A>`) to use for captioning
    /// * `sample_len` - The maximum number of tokens to generate
    /// * `alloc` - The image allocator to use
    /// * `debug` - Whether to enable debug output
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
    pub fn inference(
        &mut self,
        prompt: Vec<Message>,
        image: Option<Image<u8, 3, A>>,
        sample_len: usize, // per prompt
        alloc: A,
    ) -> Result<String, SmolVlm2Error> {
        let full_prompt = self
            .txt_processor
            .reformat_with_additional_prompts(prompt, true)?;

        let images = if let Some(img) = image {
            vec![img]
        } else {
            vec![]
        };

        let response = self.inference_raw(&full_prompt, images, sample_len, alloc)?;

        Ok(response)
    }

    /// Run the inference of the SmolVLM2 model without the default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `full_prompt` - The fully formatted prompt string (should include <image> tags if images are provided)
    /// * `images` - Vector of RGB images (`Vec<Image<u8, 3, A>>`) to use for captioning
    /// * `sample_len` - The maximum number of tokens to generate
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
    ) -> Result<String, SmolVlm2Error> {
        if self.config.debug {
            std::io::stdout().flush()?;
        }
        let mut ctx = InferenceContext::new(self.config.debug, self.config.debug);

        self.response.clear();

        let mut converted_prompt = String::from(full_prompt);

        self.img_processor.binding_images_to_prompt(
            &mut converted_prompt,
            images,
            &self.device,
            alloc,
        )?;

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

            let logits = self.model.forward(
                &input,
                self.index_pos,
                &self.img_processor.get_image_token_mask(&input)?,
                self.img_processor.get_processed_images(),
                &mut ctx,
            )?;
            self.img_processor.clear_processed_images();
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
    ) -> Result<(model::Model, TextProcessor, ImageProcessor<A>), SmolVlm2Error> {
        let txt_processor = TextProcessor::new(Self::MODEL_IDENTIFIER.into(), config)?;
        let img_processor =
            ImageProcessor::new(Self::IMG_PROCESSOR_CONFIG, device, &txt_processor)?;

        let api = Api::new()?;
        let repo = api.model(Self::MODEL_IDENTIFIER.to_string());

        let w1 = repo.get("model-00001-of-00002.safetensors")?;
        let w2 = repo.get("model-00002-of-00002.safetensors")?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w1, w2], dtype, device)? };

        model::Model::load(vb, dtype, device)
            .map_err(SmolVlm2Error::CandleError)
            .map(|m| (m, txt_processor, img_processor))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use kornia_io::{jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8};
    use kornia_tensor::CpuAllocator;

    use crate::smolvlm2::text_processor::{Line, Role};

    use super::*;

    // cargo test -p kornia-vlm test_smolvlm2_text_inference --features cuda -- --nocapture --ignored
    // RUST_LOG=debug cargo test -p kornia-vlm test_smolvlm2_text_inference --features cuda -- --nocapture --ignored
    #[test]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_text_inference() {
        env_logger::init();

        let path = Path::new("../../100462016.jpeg"); // or .png

        let image = match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(&path).ok(),
            Some("png") => read_image_png_rgb8(&path).ok(),
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
                Some(image),
                sample_len,
                CpuAllocator,
            )
            .expect("Inference failed");
    }
}
