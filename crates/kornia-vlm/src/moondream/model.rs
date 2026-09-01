use std::io::Write;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::moondream::Model;
use tokenizers::Tokenizer;

use crate::moondream::utils::{MoondreamConfig, MoondreamError};
use crate::token_output_stream::TokenOutputStream;

/// Token sequence for the literal `<END>` marker Moondream emits when it is done
/// answering. It is not a single special token, so it has to be matched as a suffix.
/// Mirrors the check in candle's own moondream example.
///
/// This alone is not sufficient: the tokenizer merges `<` differently depending
/// on the preceding text, so the first marker of a response can arrive as a
/// different id sequence and slip through. The decoded text is checked too.
const END_MARKER_TOKENS: [u32; 3] = [27, 10619, 29];

/// The same marker as text, which is what actually terminates generation.
const END_MARKER: &str = "<END>";

/// Length of the prefix of `text` that cannot be the beginning of `marker`.
///
/// Streaming output must hold back a trailing `<`, `<E`, `<EN`, ... until we
/// know whether it starts the end marker, otherwise the marker is printed
/// before it can be recognised.
fn safe_prefix_len(text: &str, marker: &str) -> usize {
    let max_partial = marker.len().min(text.len());
    for n in (1..=max_partial).rev() {
        let start = text.len() - n;
        if text.is_char_boundary(start) && marker.starts_with(&text[start..]) {
            return start;
        }
    }
    text.len()
}

pub struct TextGenerationConfig {
    pub seed: u64,
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl From<MoondreamConfig> for TextGenerationConfig {
    fn from(config: MoondreamConfig) -> Self {
        Self {
            seed: config.seed,
            temp: config.temp,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
        }
    }
}

/// Statistics for a single [`TextGeneration::run`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationStats {
    /// Time spent encoding the image and prefilling the prompt.
    pub prefill: std::time::Duration,
    /// Time spent generating every token after the first forward pass.
    pub decode: std::time::Duration,
    /// Number of tokens sampled, including the terminating one.
    pub generated_tokens: usize,
}

impl GenerationStats {
    /// Decode throughput in tokens per second, or `None` if nothing was decoded.
    pub fn tokens_per_second(&self) -> Option<f64> {
        // The first sampled token comes out of the prefill pass, so it is not
        // attributable to the decode loop.
        let decoded = self.generated_tokens.saturating_sub(1);
        let secs = self.decode.as_secs_f64();
        if decoded == 0 || secs <= 0.0 {
            return None;
        }
        Some(decoded as f64 / secs)
    }
}

pub struct TextGeneration {
    model: Model,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    config: TextGenerationConfig,
    device: Device,
    special_token: u32,
    stats: GenerationStats,
}

impl TextGeneration {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        device: Device,
        config: TextGenerationConfig,
    ) -> Result<Self, MoondreamError> {
        // Moondream uses "<|endoftext|>" as both the BOS and the EOS token.
        // https://huggingface.co/vikhyatk/moondream1/blob/main/special_tokens_map.json
        let special_token = tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .copied()
            .ok_or(MoondreamError::SpecialTokenNotFound)?;

        Ok(Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor: LogitsProcessor::new(config.seed, config.temp, config.top_p),
            config,
            device,
            special_token,
            stats: GenerationStats::default(),
        })
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Statistics for the most recent [`Self::run`] call.
    #[inline]
    pub fn stats(&self) -> GenerationStats {
        self.stats
    }

    /// Encode `image` and generate an answer to `prompt`.
    ///
    /// # Arguments
    ///
    /// * `image` - normalized image tensor with shape [1, 3, 378, 378]
    /// * `prompt` - the fully formatted prompt, including the Question/Answer scaffold
    /// * `sample_len` - maximum number of tokens to generate
    /// * `stdout_debug` - stream the generated tokens to stdout as they are sampled
    pub fn run(
        &mut self,
        image: &Tensor,
        prompt: &str,
        sample_len: usize,
        stdout_debug: bool,
    ) -> Result<String, MoondreamError> {
        // Each call is a fresh conversation: drop the detokenizer state and the
        // attention cache left behind by the previous frame.
        self.tokenizer.clear();
        self.model.text_model.clear_kv_cache();
        self.stats = GenerationStats::default();

        let start_prefill = std::time::Instant::now();

        // The 729 patch embeddings are computed once and reused for every token.
        let image_embeds = image.apply(self.model.vision_encoder())?;

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Err(MoondreamError::EmptyPrompt);
        }

        let mut response = String::new();
        // Bytes of `response` already streamed to stdout.
        let mut printed = 0usize;
        for index in 0..sample_len {
            // After the prefill pass the text model keeps its own KV cache, so
            // only the newest token has to be fed back in.
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = if index > 0 {
                self.model.text_model.forward(&input)?
            } else {
                // Moondream splices the sequence as
                // <bos embedding><image embedding><prompt embedding>.
                let bos_token = Tensor::new(&[self.special_token], &self.device)?.unsqueeze(0)?;
                let logits =
                    self.model
                        .text_model
                        .forward_with_img(&bos_token, &input, &image_embeds)?;
                // Metal (and CUDA) enqueue work asynchronously, so without an
                // explicit barrier this timestamp measures enqueue time, not
                // execution — reporting an impossible ~2ms prefill and pushing
                // the real cost into the decode loop.
                self.device.synchronize()?;
                self.stats.prefill = start_prefill.elapsed();
                logits
            };

            // Always sample in F32: the model itself may be running in F16 on
            // Metal or CUDA, where softmax over the vocabulary is not safe.
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.config.repeat_penalty == 1.0 {
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
            self.stats.generated_tokens += 1;

            if next_token == self.special_token || tokens.ends_with(&END_MARKER_TOKENS) {
                break;
            }

            if let Some(piece) = self.tokenizer.next_token(next_token)? {
                response.push_str(&piece);

                if let Some(idx) = response.find(END_MARKER) {
                    response.truncate(idx);
                    if stdout_debug && response.len() > printed {
                        print!("{}", &response[printed..]);
                        std::io::stdout().flush()?;
                    }
                    break;
                }

                if stdout_debug {
                    // Hold back a trailing partial "<END>" so the marker is
                    // never streamed to the user.
                    let safe = safe_prefix_len(&response, END_MARKER);
                    if safe > printed {
                        print!("{}", &response[printed..safe]);
                        printed = safe;
                        std::io::stdout().flush()?;
                    }
                }
            }
        }

        // Flush any bytes still buffered by the incremental detokenizer.
        if let Some(rest) = self.tokenizer.decode_rest()? {
            response.push_str(&rest);
            if let Some(idx) = response.find(END_MARKER) {
                response.truncate(idx);
            }
        }
        if stdout_debug && response.len() > printed {
            print!("{}", &response[printed..]);
        }
        if stdout_debug {
            println!();
            std::io::stdout().flush()?;
        }

        self.device.synchronize()?;
        self.stats.decode = start_prefill.elapsed().saturating_sub(self.stats.prefill);

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::{safe_prefix_len, END_MARKER};

    #[test]
    fn holds_back_a_trailing_partial_marker() {
        assert_eq!(safe_prefix_len("a dog<", END_MARKER), 5);
        assert_eq!(safe_prefix_len("a dog<E", END_MARKER), 5);
        assert_eq!(safe_prefix_len("a dog<EN", END_MARKER), 5);
        assert_eq!(safe_prefix_len("a dog<END", END_MARKER), 5);
    }

    #[test]
    fn releases_text_that_cannot_start_the_marker() {
        assert_eq!(safe_prefix_len("a dog", END_MARKER), 5);
        assert_eq!(safe_prefix_len("", END_MARKER), 0);
        // A '<' that turned out not to be the marker is still held until the
        // next piece disambiguates it — that is the price of never leaking it.
        assert_eq!(safe_prefix_len("2 < 3", END_MARKER), 5);
    }

    #[test]
    fn does_not_split_a_multibyte_character() {
        // A trailing multi-byte char must not be mistaken for a marker prefix
        // nor sliced mid-character.
        let text = "café";
        assert_eq!(safe_prefix_len(text, END_MARKER), text.len());
        assert!(text.is_char_boundary(safe_prefix_len(text, END_MARKER)));
    }
}
