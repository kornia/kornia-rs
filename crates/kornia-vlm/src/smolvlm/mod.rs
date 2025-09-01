mod custom_layernorm;
mod custom_rmsnorm;
mod introspector;
mod model;
mod preprocessor;
mod text_model;
pub mod utils;
mod vision_model;

use core::alloc;
#[cfg(feature = "debug")]
use std::io;
#[cfg(feature = "debug")]
use std::io::Write;

use crate::smolvlm::{
    model::SmolModel, preprocessor::SmolVlmImagePreprocessor, utils::SmolVlmConfig,
    utils::SmolVlmError,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::api::sync::Api;
use kornia_image::{allocator::ImageAllocator, Image};
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
            preprocessor: SmolVlmImagePreprocessor::new(1536, 384, &device),
            response: String::new(),
        })
    }

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

        if let Some(_) = &image {
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

        let response = self.inference_raw(&full_prompt, images, sample_len, alloc)?;

        Ok(response)
    }

    /// Run the inference of the SmolVLM model without the default prompt formatting.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8    image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The length of the generated caption
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
    ) -> Result<String, SmolVlmError> {
        #[cfg(feature = "debug")]
        std::io::stdout().flush()?;

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
            // preprocess_image(image, 1536, 384, &self.device);
            processed_images.push((img_patches, mask_patches));

            let img_token = get_prompt_split_image(81, size);
            converted_prompt.replace_range(
                &(start + offset)..&(start + offset + image_tag_len),
                &img_token,
            );
            offset += img_token.len() - image_tag_len;
        }

        // println!("[SmolVLM] Full prompt: {converted_prompt}");

        let full_token = self.tokenizer.encode(converted_prompt, false)?;

        let mut delta_token = full_token.get_ids().to_vec();

        #[cfg(feature = "debug")]
        println!("Initial tokens: {delta_token:?}");
        #[cfg(feature = "debug")]
        let start_gen = std::time::Instant::now();
        #[cfg(feature = "debug")]
        let mut generated_tokens = 0usize;
        let mut introspector = introspector::ActivationIntrospector::new();
        let mut vis_introspector = crate::smolvlm::introspector::ActivationIntrospector::new();

        for _i in 0..sample_len {
            self.token_history.extend(&delta_token);
            let input = Tensor::from_slice(&delta_token, &[delta_token.len()], &self.device)?;

            let image_token_mask = input.broadcast_eq(&self.image_token_tensor)?;
            let logits = self.model.forward(
                &input,
                self.index_pos,
                &image_token_mask,
                processed_images.iter().map(|(a, b)| (a, b)).collect(),
                &mut introspector,
                &mut vis_introspector,
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
            // println!("#>:{last_logit}");

            #[cfg(feature = "debug")]
            {
                introspector.insert("logits", &last_logit);
                introspector.increment_batch_pos();
            }

            self.index_pos += delta_token.len();
            delta_token.clear();
            delta_token.push(out_token);

            let token_output = self.tokenizer.decode(&[out_token], false)?;

            if token_output != "<end_of_utterance>" {
                self.response += &token_output;

                #[cfg(feature = "debug")]
                {
                    generated_tokens += 1;

                    print!("{token_output}");
                    io::stdout().flush().unwrap();
                }
            } else {
                #[cfg(feature = "debug")]
                {
                    println!();
                    io::stdout().flush().unwrap();
                }

                break;
            }
        }

        #[cfg(feature = "debug")]
        {
            let dt = start_gen.elapsed();
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );

            introspector
                .save_as("examples/smol_vlm/validation_data/rust_isp_decoder.safetensors")?;
            vis_introspector
                .save_as("examples/smol_vlm/validation_data/rust_isp_encoder.safetensors")?;
            // println!("Token history: {:?}", self.token_history);
        }

        Ok(self.response.clone())
    }

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
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let logits_vec = logits_f32.to_vec1::<f32>()?;

        // Find the maximum value
        let max_value = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Use a small epsilon for floating-point comparison to handle precision issues
        let epsilon = 1e-8;

        // println!("Logits: {:?}", logits_vec);

        // Find all indices with the maximum value (for debugging ties)
        let max_indices: Vec<usize> = logits_vec
            .iter()
            .enumerate()
            .filter(|(_, &logit)| 0.0 <= (logit - max_value) && (logit - max_value) < epsilon)
            .map(|(i, _)| i)
            .collect();

        // Log if there are ties (for debugging)
        // if max_indices.len() > 1 {
        //     println!("DEBUG: Found {} tokens with max logit {}: {:?}",
        //              max_indices.len(), max_value, max_indices);
        // }

        // println!(
        //     "DEBUG: Max logit value: {}, indices: {:?}",
        //     max_value, max_indices
        // );

        // Always select the first index (deterministic tiebreaker)
        if max_indices.is_empty() {
            return Err(SmolVlmError::InvalidLogits(
                "No valid logits found - all values may be NaN or invalid".to_string(),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Linear, Module};
    use candle_onnx;
    use std::collections::HashMap;

    #[test]
    fn test_linear_layer_validation() -> Result<(), Box<dyn std::error::Error>> {
        let test_data = candle_core::safetensors::load(
            "../../tests/data/linear_test_data_float32.safetensors",
            // &Device::cuda_if_available(0)?,
            &Device::Cpu,
        )?;
        let num_tests = test_data.get("num_tests").unwrap().to_scalar::<i64>()? as usize;
        let mut failures = Vec::new();

        for i in 0..num_tests {
            let test_name = format!("test_{}", i);
            let input = test_data.get(&format!("{}_input", test_name)).unwrap();
            let weight = test_data.get(&format!("{}_weight", test_name)).unwrap();
            let expected_output = test_data.get(&format!("{}_output", test_name)).unwrap();
            let input_dim = test_data
                .get(&format!("{}_input_dim", test_name))
                .unwrap()
                .to_scalar::<i64>()? as usize;
            let output_dim = test_data
                .get(&format!("{}_output_dim", test_name))
                .unwrap()
                .to_scalar::<i64>()? as usize;

            // let linear = Linear::new(weight.clone(), None);
            // let actual_output = linear.forward(input)?;
            // let actual_output = input.matmul(&weight.t()?)?;
            let actual_output = input.broadcast_matmul(&weight.t()?)?;

            let diff = (&actual_output - expected_output)?;
            let mse = diff
                .powf(2.0)?
                .mean_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            let mae = diff
                .abs()?
                .mean_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;

            if mse <= 0.00 {
                println!(
                    "✅ {}: MSE={:.16}, MAE={:.8} (dims: {}x{}, dtype: {:?}), device: {:?})",
                    test_name,
                    mse,
                    mae,
                    input_dim,
                    output_dim,
                    actual_output.dtype(),
                    actual_output.device()
                );
            } else {
                println!(
                    "❌ {}: MSE={:.16}, MAE={:.8} (dims: {}x{}, dtype: {:?}, device: {:?})",
                    test_name,
                    mse,
                    mae,
                    input_dim,
                    output_dim,
                    actual_output.dtype(),
                    actual_output.device()
                );
                failures.push(format!("{} MSE={:.8} > 0.00", test_name, mse));
            }
        }

        if !failures.is_empty() {
            panic!("Test failures:\n{}", failures.join("\n"));
        }

        Ok(())
    }

    #[test]
    fn test_layernorm_validation() -> Result<(), Box<dyn std::error::Error>> {
        use crate::smolvlm::custom_layernorm::CustomLayerNorm;

        let test_data = candle_core::safetensors::load(
            "../../examples/smol_vlm/validation_data/layernorm_test_data_float32.safetensors",
            &Device::Cpu,
        )?;
        let num_tests = test_data.get("num_tests").unwrap().to_scalar::<i64>()? as usize;
        let eps = test_data.get("eps").unwrap().to_scalar::<f32>()? as f64;
        let mut failures = Vec::new();

        for i in 0..num_tests {
            let test_name = format!("test_{}", i);
            let input = test_data.get(&format!("{}_input", test_name)).unwrap();
            let weight = test_data.get(&format!("{}_weight", test_name)).unwrap();
            let bias = test_data.get(&format!("{}_bias", test_name)).unwrap();
            let expected_output = test_data.get(&format!("{}_output", test_name)).unwrap();
            let dim = test_data
                .get(&format!("{}_dim", test_name))
                .unwrap()
                .to_scalar::<i64>()? as usize;
            let batch_size = test_data
                .get(&format!("{}_batch_size", test_name))
                .unwrap()
                .to_scalar::<i64>()? as usize;

            // Create custom layer norm with the test parameters
            let layer_norm = CustomLayerNorm::new(weight.clone(), bias.clone(), eps);
            let actual_output = layer_norm.forward(input)?;

            let diff = (&actual_output - expected_output)?;
            let mse = diff
                .powf(2.0)?
                .mean_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            let mae = diff
                .abs()?
                .mean_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;

            // Layer norm should be quite accurate, but allow some floating point tolerance
            let tolerance = 1e-5;
            if mse <= tolerance {
                println!(
                    "✅ {}: MSE={:.16}, MAE={:.8} (dims: {}x{}, dtype: {:?}, device: {:?})",
                    test_name,
                    mse,
                    mae,
                    batch_size,
                    dim,
                    actual_output.dtype(),
                    actual_output.device()
                );
            } else {
                println!(
                    "❌ {}: MSE={:.16}, MAE={:.8} (dims: {}x{}, dtype: {:?}, device: {:?})",
                    test_name,
                    mse,
                    mae,
                    batch_size,
                    dim,
                    actual_output.dtype(),
                    actual_output.device()
                );
                failures.push(format!("{} MSE={:.8} > {}", test_name, mse, tolerance));
            }
        }

        if !failures.is_empty() {
            panic!("Test failures:\n{}", failures.join("\n"));
        }

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_via_onnx() -> Result<(), Box<dyn std::error::Error>> {
        // Load ONNX model
        let model: candle_onnx::onnx::ModelProto =
            candle_onnx::read_file("../../tests/data/onnx/decoder_model_merged.onnx")?;
        let embedder: candle_onnx::onnx::ModelProto =
            candle_onnx::read_file("../../tests/data/onnx/embed_tokens.onnx")?;
        let vision: candle_onnx::onnx::ModelProto =
            candle_onnx::read_file("../../tests/data/onnx/vision_encoder.onnx")?;

        // Example: Convert a prompt string to input embeddings using a tokenizer
        let prompt = "A photo of a cat sitting on a mat.";
        // Load a HuggingFace tokenizer (adjust as needed)
        let tokenizer = tokenizers::Tokenizer::from_pretrained("bert-base-uncased", None)
            .map_err(|e| format!("Tokenizer load error: {e}"))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenizer encode error: {e}"))?;
        let input_ids = encoding.get_ids();
        println!("Prompt: {}", prompt);
        println!("Token IDs: {:?}", input_ids);

        let seq_len = input_ids.len();

        // Convert token IDs to a tensor (embeddings would require an embedding layer, which should be part of the ONNX model)
        // Here we just show how to prepare the input tensor
        let input_tensor = Tensor::from_slice(input_ids, &[1, input_ids.len()], &Device::Cpu)?;
        let input_tensor = input_tensor.to_dtype(DType::I64)?; //.squeeze(0)?;
        println!(
            "Input tensor: {:?} {:?}",
            input_tensor.shape(),
            input_tensor.dtype()
        );

        let attention_mask = Tensor::ones(&[1, seq_len], DType::I64, &Device::Cpu)?;
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();
        let position_ids_tensor = Tensor::from_slice(&position_ids, &[1, seq_len], &Device::Cpu)?;

        // At this point, you would feed `input_tensor` (and vision features if needed) into the ONNX model
        // The ONNX model should contain the embed_tokens, vision, and decoder layers merged

        if let Some(graph) = &model.graph {
            for input in &graph.input {
                println!("### Input: {}", input.name);
                if let candle_onnx::onnx::type_proto::Value::TensorType(t) =
                    input.r#type.as_ref().unwrap().value.as_ref().unwrap()
                {
                    println!(
                        "{:?} {:?}",
                        t.elem_type,
                        t.shape
                            .as_ref()
                            .unwrap()
                            .dim
                            .iter()
                            .map(|d| d.value.as_ref().unwrap())
                            .collect::<Vec<_>>()
                    );
                } else {
                    println!("##################################");
                }
            }
        }
        if let Some(graph) = &embedder.graph {
            for input in &graph.input {
                println!("### Input (Embedder): {}", input.name);
            }
        }

        // Run the ONNX model
        let embedder_out = candle_onnx::simple_eval(&embedder, {
            // Prepare input map for ONNX model
            let mut inputs = HashMap::new();
            inputs.insert("input_ids".to_string(), input_tensor.clone());

            inputs
        })?;

        for (name, tensor) in &embedder_out {
            println!("### Output (Embedder): {}", name);
        }

        // Run the ONNX model
        let outputs = candle_onnx::simple_eval(&model, {
            // Prepare input map for ONNX model
            let mut inputs = HashMap::new();
            // Replace "input" with the actual input name expected by your ONNX model
            inputs.insert(
                "inputs_embeds".to_string(),
                embedder_out.get("inputs_embeds").unwrap().clone(),
            );
            inputs.insert("attention_mask".to_string(), attention_mask);
            inputs.insert("position_ids".to_string(), position_ids_tensor);
            for i in 0..30 {
                inputs.insert(
                    format!("past_key_values.{}.key", i),
                    Tensor::zeros(&[1, 3, 1, 64], DType::F32, &Device::Cpu)?,
                );
                inputs.insert(
                    format!("past_key_values.{}.value", i),
                    Tensor::zeros(&[1, 3, 1, 64], DType::F32, &Device::Cpu)?,
                );
            }

            inputs
        })?;

        // Print all output names and shapes
        for (name, tensor) in &outputs {
            println!("Output: {} shape: {:?}", name, tensor.shape());
        }

        Ok(())
    }

    use ndarray::Array2;
    use ort::{Environment, SessionBuilder, Value};
    use std::sync::Arc;

    #[ignore]
    #[test]
    fn test_via_onnx_ort() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize the ONNX Runtime environment
        let environment = Arc::new(Environment::builder().with_name("test").build()?);

        // Load the ONNX model
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file("../../tests/data/onnx/decoder_model_merged.onnx")?;

        // Prepare input (example: 2D array of f32)
        let input_array = Array2::<f32>::zeros((1, 10)).into_dyn(); // shape as required by your model
        let arr = ndarray::CowArray::from(&input_array);

        // Wrap input in ORT Value
        let input_tensor = Value::from_array(session.allocator(), &arr)?;

        // Run inference
        let outputs = session.run(vec![input_tensor])?;

        // Get output as OrtOwnedTensor
        let output_tensor = outputs[0].try_extract::<f32>()?;
        println!("Output: {:?}", output_tensor);

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_via_onnx_specific_layer() -> Result<(), Box<dyn std::error::Error>> {
        // Load the extracted layer ONNX model
        let layer_model: candle_onnx::onnx::ModelProto =
            candle_onnx::read_file("../../tests/data/extracted_layer.onnx")?;

        // Load the safetensor file containing input and output activations
        let safetensor_path = "../../tests/data/extracted_layer_activations.safetensors";
        let activations = candle_core::safetensors::load(safetensor_path, &Device::Cpu)?;
        let input_tensor = activations.get("input").expect("Missing input tensor");
        let expected_output = activations.get("output").expect("Missing output tensor");

        // Print input tensor shape and dtype for debug
        println!(
            "Input tensor shape: {:?}, dtype: {:?}",
            input_tensor.shape(),
            input_tensor.dtype()
        );

        // Print ONNX model input names and shapes for debug
        if let Some(graph) = &layer_model.graph {
            for input in &graph.input {
                println!("### Input: {}", input.name);
                if let candle_onnx::onnx::type_proto::Value::TensorType(t) =
                    input.r#type.as_ref().unwrap().value.as_ref().unwrap()
                {
                    println!(
                        "{:?} {:?}",
                        t.elem_type,
                        t.shape
                            .as_ref()
                            .unwrap()
                            .dim
                            .iter()
                            .map(|d| d.value.as_ref().unwrap())
                            .collect::<Vec<_>>()
                    );
                } else {
                    println!("##################################");
                }
            }
        }

        // Run the ONNX model for the extracted layer using the actual input
        let outputs = candle_onnx::simple_eval(&layer_model, {
            let mut inputs = HashMap::new();
            // You may need to adjust the input name below to match your ONNX model
            inputs.insert(
                "/model/layers.0/input_layernorm/output_0".to_string(),
                input_tensor.clone(),
            );
            inputs
        })?;

        // Print all output names and shapes
        for (name, tensor) in &outputs {
            println!("### Output: {} shape: {:?}", name, tensor.shape());
        }

        // Get the ONNX output tensor (adjust the key as needed)
        let onnx_output = outputs
            .get("/model/layers.0/attn/q_proj/MatMul/output_0")
            .expect("Missing ONNX output tensor");

        // Compare the ONNX output tensor with the expected output tensor
        let diff = (onnx_output - expected_output)?;
        let mse = diff
            .powf(2.0)?
            .mean_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let mae = diff
            .abs()?
            .mean_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;

        println!("MSE: {:.16}, MAE: {:.16}", mse, mae);
        assert!(mse < 1e-4, "MSE too high: {}", mse);

        Ok(())
    }
}
