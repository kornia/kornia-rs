/*
    To convert the raw logits output into a usable tokens through custom configurations and aggregation.
*/

use candle_core::{DType, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

use crate::smolvlm2::text_preprocessor::Message;
use crate::smolvlm2::utils::{SmolVlm2Config, SmolVlm2Error};

struct TextPostprocessor {
    config: SmolVlm2Config,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl TextPostprocessor {
    pub fn new(identifier: String, config: SmolVlm2Config) -> Self {
        let logits_processor = if config.do_sample {
            LogitsProcessor::new(config.seed, Some(config.temp), Some(config.top_p))
        } else {
            LogitsProcessor::from_sampling(config.seed, Sampling::ArgMax)
        };

        Self {
            config,
            tokenizer: Tokenizer::from_pretrained(identifier, None).unwrap(),
            logits_processor,
        }
    }

    pub fn get_response(
        &mut self,
        raw_logits: &candle_core::Tensor,
        delta_token: &[u32],
    ) -> Result<String, SmolVlm2Error> {
        let (s, _embed_dim) = raw_logits.dims2()?;
        let last_logit = raw_logits.i((s - 1, ..))?;

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

        let token_output = self.tokenizer.decode(&[out_token], false)?;

        Ok(token_output)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_postprocessing() {
        let postprocessor = TextPostprocessor::new();
    }
}
