//! Tokenizer for SmolVLM

use std::path::Path;
#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer as HuggingFaceTokenizer;

use crate::smolvlm::common::{SmolVLMConfig, SmolVLMError};

/// Tokenizer handles text tokenization and decoding for SmolVLM
#[derive(Debug)]
pub struct Tokenizer {
    config: SmolVLMConfig,
    #[cfg(feature = "tokenizers")]
    tokenizer: HuggingFaceTokenizer,

    // Special token IDs
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
}

impl Tokenizer {
    /// Create a new tokenizer from the given configuration
    pub fn new(config: &SmolVLMConfig) -> Result<Self, SmolVLMError> {
        // Validate configuration
        config.validate()?;

        // Path to tokenizer files
        let tokenizer_path = Path::new(&config.tokenizer_path);
        let tokenizer_json_path = tokenizer_path.join("tokenizer.json");

        // Check if tokenizer files exist
        if !tokenizer_json_path.exists() {
            return Err(SmolVLMError::TokenizationError(format!(
                "Tokenizer file not found: {}",
                tokenizer_json_path.display()
            )));
        }

        #[cfg(feature = "tokenizers")]
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_json_path).map_err(|e| {
            SmolVLMError::TokenizationError(format!("Failed to load tokenizer: {}", e))
        })?;

        // Read special tokens
        // In practice, we would read these from the tokenizer
        // For now, we use common defaults
        let bos_token_id = 1;
        let eos_token_id = 2;
        let pad_token_id = 0;

        Ok(Self {
            config: config.clone(),
            #[cfg(feature = "tokenizers")]
            tokenizer,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, SmolVLMError> {
        #[cfg(feature = "tokenizers")]
        {
            let encoding = self.tokenizer.encode(text, false).map_err(|e| {
                SmolVLMError::TokenizationError(format!("Failed to encode text: {}", e))
            })?;

            let token_ids = encoding
                .get_ids()
                .iter()
                .copied()
                .map(|id| id as u32)
                .collect();

            Ok(token_ids)
        }

        #[cfg(not(feature = "tokenizers"))]
        {
            // Fallback for when tokenizers feature is disabled
            Err(SmolVLMError::TokenizationError(
                "Tokenization not available: tokenizers feature is disabled".to_string(),
            ))
        }
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, SmolVLMError> {
        #[cfg(feature = "tokenizers")]
        {
            // Convert to tokenizers::Encoding format
            let token_ids: Vec<_> = token_ids.iter().map(|&id| id as u32).collect();

            // Filter out special tokens
            let filtered_ids: Vec<_> = token_ids
                .into_iter()
                .filter(|&id| id != self.pad_token_id && id != self.eos_token_id)
                .collect();

            // Decode the token IDs
            let text = self.tokenizer.decode(&filtered_ids, true).map_err(|e| {
                SmolVLMError::TokenizationError(format!("Failed to decode tokens: {}", e))
            })?;

            Ok(text)
        }

        #[cfg(not(feature = "tokenizers"))]
        {
            // Fallback for when tokenizers feature is disabled
            Err(SmolVLMError::TokenizationError(
                "Decoding not available: tokenizers feature is disabled".to_string(),
            ))
        }
    }

    /// Add special tokens to input token IDs
    pub fn prepare_input(&self, token_ids: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(token_ids.len() + 2);
        result.push(self.bos_token_id);
        result.extend_from_slice(token_ids);
        result.push(self.eos_token_id);
        result
    }

    /// Get the BOS (beginning of sequence) token ID
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get the EOS (end of sequence) token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the PAD (padding) token ID
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
