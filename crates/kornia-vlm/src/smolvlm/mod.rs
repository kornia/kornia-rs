mod generator;
mod model;
mod preprocessor;
mod text_model;
pub mod utils;
mod vision_model;

use std::io;
use std::io::{Error, Write};

use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
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
    prompt_history: String,
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
            prompt_history: String::from("<|im_start|>"),
            image_history: Vec::new(),
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
    pub fn inference(
        &mut self,
        image: Option<Image<u8, 3, CpuAllocator>>,
        prompt: &str,
        sample_len: usize, // per prompt
        stdout_debug: bool,
    ) -> Result<String, SmolVlmError> {
        if stdout_debug {
            std::io::stdout().flush()?;
        }

        let mut response = String::new(); // collection of tokens

        let mut full_prompt = String::new();
        if let Some(raw_img) = image {
            let (img_patches, mask_patches, rows, cols) =
                preprocess_image(raw_img, 1920, 384, &self.device);

            let img_token = get_prompt_split_image(81, rows, cols);
            full_prompt += "\nUser:<image>";
            full_prompt = full_prompt.replace("<image>", &img_token);

            self.image_history.push((img_patches, mask_patches));
        } else {
            full_prompt += "\nUser: ";
        }

        full_prompt += prompt;
        full_prompt += "<end_of_utterance>\nAssistant:";
        response.clear();
        self.prompt_history += &full_prompt;

        for i in 0..sample_len {
            let encoding = self.tokenizer.encode(self.prompt_history.clone(), false)?;
            let tokens = encoding.get_ids();
            let input =
                Tensor::from_slice(tokens, Shape::from_dims(&[tokens.len()]), &self.device)?;
            let vision_data = if self.image_history.len() > 0 {
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

            let token_output = self.tokenizer.decode(&[out_token], false)?;

            self.prompt_history += &token_output;
            if token_output != "<end_of_utterance>" {
                response += &token_output;

                if stdout_debug {
                    print!("{}", token_output);
                    io::stdout().flush().unwrap();
                }
            } else {
                if stdout_debug {
                    print!("\n");
                    io::stdout().flush().unwrap();
                }

                break;
            }
        }

        Ok(response)
    }

    pub fn image_history_count(&self) -> usize {
        self.image_history.len()
    }

    // utility function to load the model
    fn load_model(_dtype: DType, device: &Device) -> Result<(SmolModel, Tokenizer), SmolVlmError> {
        let tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None)?;
        let api = Api::new()?;
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors")?;
        let weights = candle_core::safetensors::load(weights, &device)?;
        let model = SmolModel::load(&weights)?;

        Ok((model, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use kornia_io::jpeg::read_image_jpeg_rgb8;
    use kornia_io::png::{write_image_png_gray8, write_image_png_rgb8};
    use reqwest;
    use std::error::Error;
    use std::fs;
    use std::path::Path;

    pub fn load_image_url(
        url: &str,
    ) -> std::result::Result<Image<u8, 3, CpuAllocator>, Box<dyn Error>> {
        let dir = Path::new(".vscode");

        let file_path = {
            let parsed_url = reqwest::Url::parse(url)?;
            let path = parsed_url.path();
            dir.join(
                Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown_file.png") // Use PNG as default
                    .to_string(),
            )
        };

        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        }

        // Check if the file exists locally
        if file_path.exists() {
            // Use kornia_io to read the JPEG file
            let img = read_image_jpeg_rgb8(&file_path)?;
            println!("Loaded image from local cache.");
            return Ok(img);
        }

        // If the file does not exist, download it and save it
        println!("Downloading image from URL...");

        // Fetch the image as bytes
        let response = reqwest::blocking::get(url)?.bytes()?;
        fs::write(&file_path, &response)?;

        // Use kornia_io to read the JPEG file
        let img = read_image_jpeg_rgb8(&file_path)?;
        println!("Saved image locally as {}", file_path.to_str().unwrap());
        Ok(img)
    }

    fn read_input(cli_prompt: &str) -> String {
        let mut input = String::new();
        print!("{}", cli_prompt);
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        input.trim().to_owned()
    }

    #[test]
    fn test_smolvlm_inference() {
        let mut model = SmolVlm::new(SmolVlmConfig::default()).unwrap();

        // cargo test -p kornia-vlm test_smolvlm_inference --features "cuda" -- --nocapture
        for _ in 0..10 {
            let img_url = read_input("img> ");
            let image = load_image_url(&img_url)
                .and_then(|v| {
                    if model.image_history.len() > 1 {
                        println!("One image max. Cannot add another image. (Restart)");
                        Err(Box::new(io::Error::new(
                            io::ErrorKind::Other,
                            "One image max",
                        )))
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

            let prompt = read_input("txt> ");

            model.inference(image, &prompt, 10_000, true).unwrap();
        }
    }
}
