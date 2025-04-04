use crate::paligemma::PaligemmaError;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::paligemma::Model;
use tokenizers::Tokenizer;

pub struct TextGenerationConfig {
    pub seed: u64,
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    config: TextGenerationConfig,
}

impl TextGeneration {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        device: Device,
        config: TextGenerationConfig,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(config.seed, config.temp, config.top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            config,
            device,
        }
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn run(
        &mut self,
        image: &Tensor,
        prompt: &str,
        sample_len: usize,
    ) -> Result<String, PaligemmaError> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(PaligemmaError::TokenizerError)?
            .get_ids()
            .to_vec();
        let mut response = String::new();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}");
                response.push_str(&t);
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => return Err(PaligemmaError::EosTokenNotFound),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = if index > 0 {
                self.model.forward(&input)?
            } else {
                self.model.setup(image, &input)?
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.config.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.config.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
                response.push_str(&t);
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest()? {
            print!("{rest}");
            response.push_str(&rest);
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        // postprocess the response by removing the prompt
        let response = response[prompt.len()..].to_string();

        Ok(response)
    }
}
