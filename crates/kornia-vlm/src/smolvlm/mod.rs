mod generator;
mod model;
mod preprocessor;
mod text_model;
mod utils;
mod vision_model;

use std::io;
use std::io::{Error, Write};

use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tokenizers::Tokenizer;

use preprocessor::{get_prompt_split_image, load_image_url, preprocess_image};

use crate::smolvlm::model::SmolModel;
use crate::smolvlm::utils::{SmolVlmConfig, SmolVlmError};

fn read_input(cli_prompt: &str) -> String {
    let mut input = String::new();
    print!("{}", cli_prompt);
    io::stdout().flush().unwrap();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    input.trim().to_owned()
}

pub struct SmolVlm {
    model: SmolModel,
    tokenizer: Tokenizer,
    image_token_tensor: Tensor,
    config: SmolVlmConfig,
    logits_processor: LogitsProcessor,
    device: Device,
    token_history: String,
    image_history: Vec<(Tensor, Tensor)>,
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
                log::warn!("CUDA not available, defaulting to CPU: {}", e);
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
            token_history: String::from("<|im_start|>"),
            image_history: Vec::new(),
        })
    }

    /// Run the inference of the SmolVLM model
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
    pub fn inference(
        &mut self,
        // image: &Image<u8, 3, CpuAllocator>,
        // prompt: &str,
        sample_len: usize, // per prompt
        stdout_debug: bool,
    ) -> Result<String, SmolVlmError> {
        if stdout_debug {
            std::io::stdout().flush()?;
        }

        // let image_token = self.image_token_enc.get_ids(); // STRUCT-WIDE
        // let mut token_history = String::from("<|im_start|>"); // STRUCT-WIDE
        // let mut image: Vec<(Tensor, Tensor)> = Vec::new(); // STRUCT-WIDE
        let mut response = String::new(); // collection of tokens
        let mut token_output = String::new(); // single token

        for i in 0..sample_len {
            // OUTSIDE

            // OUTSIDE (i == 0 --> initially, EOU --> end of inference)
            if i == 0 || token_output == "<end_of_utterance>" {
                let img_url = read_input("img> ");
                let raw_img = load_image_url(&img_url)
                    .and_then(|v| {
                        if self.image_history.len() > 1 {
                            println!("One image max. Cannot add another image. (Restart)");
                            Err(Box::new(Error::new(io::ErrorKind::Other, "One image max")))
                        } else {
                            Ok(v)
                        }
                    })
                    .map_or_else(
                        |err| {
                            println!("Invalid or empty URL (no image)");
                            println!("Error: {:?}", err);

                            Err(err)
                        },
                        |ok| Ok(ok),
                    )
                    .ok();

                let raw_prompt = read_input("txt> ");

                //  ^^^ MOVE OUTSIDE

                let mut txt_prompt = String::new();
                if let Some(raw_img) = raw_img {
                    let (img_patches, mask_patches, rows, cols) =
                        preprocess_image(raw_img, 1920, 384, &self.device);

                    let img_token = get_prompt_split_image(81, rows, cols);
                    txt_prompt += "\nUser:<image>";
                    txt_prompt = txt_prompt.replace("<image>", &img_token);

                    self.image_history.push((img_patches, mask_patches));
                } else {
                    txt_prompt += "\nUser: ";
                }

                txt_prompt += &raw_prompt;
                txt_prompt += "<end_of_utterance>\nAssistant:";
                response.clear();
                self.token_history += &txt_prompt;
            }
            let encoding = self.tokenizer.encode(self.token_history.clone(), false)?;
            let tokens = encoding.get_ids();
            let input =
                Tensor::from_slice(tokens, Shape::from_dims(&[tokens.len()]), &self.device)?;
            let vision_data = if self.image_history.len() > 0 {
                // let image_token_mask = Tensor::from_slice(image_token, &[1], &self.device)?;
                let image_token_mask = input.broadcast_eq(&self.image_token_tensor)?;
                Some((
                    image_token_mask,
                    &self.image_history[0].0,
                    &self.image_history[0].1,
                ))
            } else {
                None
            };
            let logits = self.model.forward(&input, i, vision_data)?;
            let (s, _embed_dim) = logits.dims2()?;
            let last_logit = logits.i((s - 1, ..))?;
            let out_token = self.logits_processor.sample(&last_logit)?;
            token_output = self.tokenizer.decode(&[out_token], false)?;

            //  vvv MOVE OUTSIDE
            if stdout_debug {
                print!("{}", token_output);
                io::stdout().flush().unwrap();
            }
            //  ^^^ MOVE OUTSIDE

            self.token_history += &token_output;
            if token_output != "<end_of_utterance>" {
                response += &token_output;
            }
        } // OUTSIDE

        Ok(response) // OUTSIDE
    }

    // utility function to load the model
    fn load_model(_dtype: DType, device: &Device) -> Result<(SmolModel, Tokenizer), SmolVlmError> {
        let tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None).unwrap();
        let api = Api::new()?;
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &device)?;
        let model = SmolModel::load(&weights)?;

        Ok((model, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use candle_core::Device;

    #[test]
    fn test_smolvlm_inference() {
        let mut model = SmolVlm::new(SmolVlmConfig::default()).unwrap();

        // cargo test -p kornia-vlm test_smolvlm_inference --features "cuda" -- --nocapture
        println!("Running inference on SmolVlm model...");
        model.inference(10_000, true).unwrap();
    }
}
