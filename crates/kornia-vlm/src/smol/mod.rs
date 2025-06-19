mod text_model;
mod vision_model;
mod utils;


use std::io::{Error, Write};
use std::io;

use candle_core::{Device, DType, IndexOp, Shape, Tensor};
use hf_hub::api::sync::Api;
use tokenizers::{Encoding, Tokenizer};
use candle_nn::ops;
use rand::rng;
use rand::prelude::IndexedRandom;
use std::cmp::Ordering;
use terminal_size::{terminal_size, Width};

use utils::{preprocess_image, load_image_url, get_prompt_split_image};

use crate::smol::text_model::SmolVLM;



fn count_lines(text: &str) -> usize {
    if let Some((Width(w), _)) = terminal_size() {
        (text.len() + w as usize + 1) / w as usize
    } else {
        1
    }
}

fn clear_lines(n: usize) {
    for _ in 0..n {
        print!("\x1B[1A\x1B[2K");
    }
    io::stdout().flush().unwrap();
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


#[derive(thiserror::Error, Debug)]
pub enum SmolError {
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

    #[error("Cannot find the <end_of_utterance> token")]
    EosTokenNotFound,
}


pub struct Smol {
    model: text_model::SmolVLM,
    tokenizer: Tokenizer,
    image_token_enc: Encoding,
    device: Device,
}

impl Smol {
    /// Create a new Smol model
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the Smol model
    ///
    /// # Returns
    pub fn new() -> Result<Self, SmolError> {
        // #[cfg(feature = "cuda")]
        // let (device, dtype) = match Device::cuda_if_available(0) {
        //     Ok(device) => (device, DType::BF16),
        //     Err(e) => {
        //         log::warn!("CUDA not available, defaulting to CPU: {}", e);
        //         (Device::Cpu, DType::F32)
        //     }
        // };

        // #[cfg(not(feature = "cuda"))]
        let (device, _dtype) = (Device::new_cuda(0).unwrap(), DType::F32);
        let (model, tokenizer) = Self::load_model(&device)?;
        let image_token_enc = tokenizer.encode("<image>", false).unwrap();

        Ok(Self {
            model,
            tokenizer,
            image_token_enc,
            device,
        })
    }

    /// Run the inference of the Paligemma model
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
        // sample_len: usize,
        // stdout_debug: bool,
    ) -> Result<String, SmolError> {

        let image_token = self.image_token_enc.get_ids();
        let mut message = String::from("<|im_start|>");
        let mut image: Vec<(Tensor, Tensor)> = Vec::new();
        let mut response = String::new();
        let mut output = String::new();
        let mut lines_printed = 0;
        for i in 0..10_000 {
            if i == 0 || output == "<end_of_utterance>" {
                let img_url = read_input("img> ");
                let img = load_image_url(&img_url)
                    .and_then(
                        |v| if image.len() > 1 {
                            println!("One image max. Cannot add another image. (Restart)");
                            Err(Box::new(Error::new(io::ErrorKind::Other, "One image max")))
                        } else {Ok(v)}
                    )
                    .map(|img| preprocess_image(img, 1920, 384, &self.device));
                let mut txt_prompt = String::new();
                if let Ok((_,_,cols,rows)) = img {
                    let img_token = get_prompt_split_image(81, rows, cols);
                    txt_prompt += "\nUser:<image>";
                    txt_prompt = txt_prompt.replace("<image>", &img_token);
                } else {
                    println!("Invalid or empty URL (no image)");
                    println!("Error: {:?}", img);
                    txt_prompt += "\nUser: ";
                }
                txt_prompt += &read_input("txt> ");
                txt_prompt += "<end_of_utterance>\nAssistant:";
                response.clear();
                message += &txt_prompt;
                if let Ok((img_patches, mask_patches,_,_)) = img {
                    image.push((img_patches, mask_patches));
                }
            }
            let encoding = self.tokenizer.encode(message.clone(), false).unwrap();
            let tokens = encoding.get_ids();
            let input = Tensor::from_slice(tokens, Shape::from_dims(&[tokens.len()]), &self.device)?;
            let vision_data = if image.len() > 0 {
                let image_token_mask = Tensor::from_slice(image_token, &[1], &self.device)?;
                let image_token_mask = input.broadcast_eq(&image_token_mask)?;
                Some((image_token_mask, &image[0].0, &image[0].1))
            } else {
                None
            };
            let logits = self.model.forward(&input, i, vision_data, &self.device)?;
            let (s, _embed_dim) = logits.dims2()?;
            let last_logit = logits.i((s-1, ..))?;
            let out_token = {
                let temperature = Tensor::from_slice(&[
                    0.2f32
                ], (1,), &self.device)?;
                let k = 50;
                let scaled = last_logit.broadcast_div(&temperature)?;
                let probs = ops::softmax(&scaled, 0)?;
                let probs_vec: Vec<f32> = probs.to_vec1()?;
                let mut indices: Vec<usize> = (0..probs_vec.len()).collect(); 
                indices.sort_by(|&i, &j| probs_vec[j].partial_cmp(&probs_vec[i]).unwrap_or(Ordering::Equal));
                let top_k_indices = &indices[..k];
                let top_k_probs: Vec<f32> = top_k_indices.iter().map(|&i| probs_vec[i]).collect();
                let sum_probs: f32 = top_k_probs.iter().sum();
                let normalized_probs: Vec<f32> = top_k_probs.iter().map(|p| p / sum_probs).collect();
                let mut rng = rng();
                let sampled_index = top_k_indices
                    .choose_weighted(&mut rng, |&idx| normalized_probs[top_k_indices.iter().position(|&x| x == idx).unwrap()])
                    .expect("Sampling failed");
                [*sampled_index as u32]
            };
            output = self.tokenizer.decode(&out_token.as_slice(), false).unwrap();
            if !response.is_empty() {
                clear_lines(lines_printed);
            }
            println!("{:?}", response);
            io::stdout().flush().unwrap();
            lines_printed = count_lines(&response);
            message += &output;
            if output != "<end_of_utterance>" {
                response += &output;
            }
        }

        Ok(response)
    }

    // utility function to load the model
    fn load_model(device: &Device) -> Result<(SmolVLM, Tokenizer), SmolError> {
        let tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None).unwrap();
        let api = Api::new().unwrap();
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &device)?;
        let model = text_model::SmolVLM::load(&weights, &device)?;

        Ok((model, tokenizer))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_smol_inference() {
        let mut model = Smol::new().unwrap();

        // cargo test -p kornia-vlm test_smol_inference --features "cuda" -- --nocapture
        println!("Running inference on SmolVLM model...");
        model.inference();
    }
}