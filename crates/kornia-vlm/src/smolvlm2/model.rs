use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::{
    llama::{self, Cache},
    siglip,
};
use log::debug;

use crate::context::InferenceContext;

/// A connector module that bridges vision and language modalities in SmolVLM2.
///
/// The Connector transforms visual features from the vision encoder into a format
/// compatible with the language model through pixel shuffling and linear projection.
pub struct Connector {
    modality_proj: Linear,
}

impl Connector {
    const SCALE_FACTOR: usize = 3;
    const HEIGHT: usize = 27;
    const WIDTH: usize = 27;

    /// Perform pixel shuffling operation to rearrange tensor dimensions.
    ///
    /// This operation reorganizes image patches by applying a pixel shuffle transformation
    /// that increases spatial resolution while reducing the number of channels. The operation
    /// reshapes the input from `[batch, patches, embed_dim]` to a format suitable for
    /// multi-modal processing.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor with shape `[batch, patches, embed_dim]` where patches = HEIGHT*WIDTH
    ///
    /// # Returns
    ///
    /// A `Result` containing the pixel-shuffled tensor with reduced patch count and increased embedding dimension
    ///
    /// # Tensor Transformations
    ///
    /// 1. `B,P,E` → `B,H,W,E` (reshape patches to spatial dimensions)
    /// 2. `B,H,W,E` → `B,H/S,S,W/S,S,E` (split spatial dims by scale factor)
    /// 3. `B,H/S,S,W/S,S,E` → `B,H/S,W/S,S,S,E` (permute dimensions)
    /// 4. `B,H/S,W/S,S,S,E` → `B,P/S²,S²*E` (reshape to final format)
    fn pixel_shuffle(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, patches, embed_dim) = x.dims3()?; // patches == HEIGHT*WIDTH

        // B,P,E => B,H,W,E => B,H/S,S,W/S,S,E => B,H/S,W/S,S,S,E => B,P/S^2,S^2*E
        x.reshape(&[batch, Self::HEIGHT, Self::WIDTH, embed_dim])?
            .reshape(&[
                batch,
                Self::HEIGHT / Self::SCALE_FACTOR,
                Self::SCALE_FACTOR,
                Self::WIDTH / Self::SCALE_FACTOR,
                Self::SCALE_FACTOR,
                embed_dim,
            ])?
            .permute([0, 1, 3, 2, 4, 5])?
            .reshape(&[
                batch,
                patches / (Self::SCALE_FACTOR * Self::SCALE_FACTOR),
                embed_dim * Self::SCALE_FACTOR * Self::SCALE_FACTOR,
            ])
    }

    /// Forward pass through the connector module.
    ///
    /// Processes image hidden states through pixel shuffling and linear projection
    /// to transform visual features into a format compatible with the language model.
    ///
    /// # Arguments
    ///
    /// * `image_hidden_states` - Tensor containing encoded image features from the vision model
    ///
    /// # Returns
    ///
    /// A `Result` containing the projected image features ready for fusion with text embeddings
    ///
    /// # Process
    ///
    /// 1. Apply pixel shuffling to reorganize spatial dimensions
    /// 2. Project through linear layer to match language model embedding dimension
    pub fn forward(&self, image_hidden_states: &Tensor) -> Result<Tensor> {
        let image_hidden_states = self.pixel_shuffle(image_hidden_states)?;
        self.modality_proj.forward(&image_hidden_states)
    }
}

/// The main SmolVLM2 model combining vision and language understanding.
///
/// This model integrates a vision encoder (SigLIP), a language model (Llama), and a connector
/// module to enable multi-modal understanding. It processes both text and images together,
/// allowing for vision-language tasks like image captioning, visual question answering, and more.
///
/// # Architecture
///
/// - **Vision Model**: SigLIP vision encoder for processing images
/// - **Text Model**: Llama language model for text understanding and generation  
/// - **Connector**: Bridge module that aligns vision and text representations
/// - **Cache**: Maintains attention cache for efficient sequential generation
///
/// # Model Configuration
///
/// - Hidden size: 2048 dimensions
/// - Vocabulary size: 49,280 tokens
/// - Max sequence length: 16,384 tokens
/// - Attention heads: 32
/// - Hidden layers: 24
pub struct Model {
    text_model: candle_transformers::models::llama::Llama,
    connector: Connector,
    vision_model: candle_transformers::models::siglip::VisionModel,

    // TODO: move these caching into inference context
    cache_dtype: DType,
    cache_device: Device,
    cache: Cache,
}

impl Model {
    const HIDDEN_SIZE: usize = 2048;
    const VOCAB_SIZE: usize = 49280;
    const TEXT_CONFIG: llama::Config = llama::Config {
        vocab_size: Self::VOCAB_SIZE,
        max_position_embeddings: 16384,
        num_attention_heads: 32,
        num_hidden_layers: 24,
        intermediate_size: 8192,
        hidden_size: Self::HIDDEN_SIZE,
        num_key_value_heads: 32,
        use_flash_attn: true,
        rms_norm_eps: 1e-5,
        rope_theta: 273768.0,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        tie_word_embeddings: false,
    };
    const VISION_CONFIG: siglip::VisionConfig = siglip::VisionConfig {
        hidden_size: 1152,
        intermediate_size: 4304,
        num_hidden_layers: 27,
        num_attention_heads: 16,
        num_channels: 3,
        image_size: 384,
        patch_size: 14,
        hidden_act: candle_nn::Activation::GeluPytorchTanh,
        layer_norm_eps: 1e-6,
    };

    /// Load a SmolVLM2 model from pretrained weights.
    ///
    /// Creates a new SmolVLM2 model instance by loading pretrained weights for the vision model,
    /// text model, and connector components. The method handles weight name mapping to ensure
    /// compatibility with the expected model architecture.
    ///
    /// # Arguments
    ///
    /// * `vb` - Variable builder containing the model weights
    /// * `dtype` - Data type for model parameters (typically F16 or F32)
    /// * `device` - Device where the model should be loaded (CPU or CUDA)
    ///
    /// # Returns
    ///
    /// A `Result` containing either:
    /// - `Ok(Model)` - Successfully loaded SmolVLM2 model ready for inference
    /// - `Err(candle_core::Error)` - If weight loading or model initialization fails
    ///
    /// # Model Components Loaded
    ///
    /// - **Text Model**: Llama language model with 24 layers and 2048 hidden size
    /// - **Vision Model**: SigLIP vision encoder with 27 layers and 1152 hidden size  
    /// - **Connector**: Modality projection layer (10368 → 2048 dimensions)
    /// - **Cache**: Attention cache initialized for the specified device and dtype
    ///
    /// # Weight Mapping
    ///
    /// The method applies weight name transformations to handle model checkpoints:
    /// - `model.text_model.model` → `model.text_model`
    /// - `model.text_model.lm_head.weight` → `lm_head.weight`
    pub fn load(vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let vb = vb.rename_f(|f: &str| {
            if let Some(rest) = f.strip_prefix("model.text_model.model") {
                // exact `model.text_model.model` -> `model.text_model`
                if rest.is_empty() {
                    return "model.text_model".to_string();
                }
                // keep the leading dot from rest (e.g. ".layers.0...")
                if rest.starts_with('.') {
                    return format!("model.text_model{}", rest);
                }
            }

            if f == "model.text_model.lm_head.weight" {
                return "lm_head.weight".to_string();
            }

            f.to_string()
        });

        Ok(Self {
            text_model: candle_transformers::models::llama::Llama::load(
                vb.pp("model.text_model"),
                &Self::TEXT_CONFIG,
            )?,
            connector: Connector {
                modality_proj: candle_nn::linear_no_bias(
                    10368,
                    Self::HIDDEN_SIZE,
                    vb.pp("model.connector.modality_projection.proj"),
                )?,
            },
            vision_model: candle_transformers::models::siglip::VisionModel::new(
                &Self::VISION_CONFIG,
                false,
                vb.pp("model.vision_model"),
            )?,
            cache_device: device.clone(),
            cache_dtype: dtype,
            cache: Cache::new(true, dtype, &Self::TEXT_CONFIG, device)?,
        })
    }

    /// Merge image and text embeddings based on image token positions.
    ///
    /// This method combines visual and textual embeddings into a single sequence by inserting
    /// image features at positions marked by the image token mask. This allows the language
    /// model to process both modalities in a unified manner.
    ///
    /// # Arguments
    ///
    /// * `image_token_mask` - Binary tensor indicating where image tokens should be placed
    /// * `image_hidden_states` - Encoded image features from the vision model and connector
    /// * `inputs_embeds` - Text embeddings from the language model's embedding layer
    ///
    /// # Returns
    ///
    /// A `Result` containing the merged embedding sequence ready for language model processing
    ///
    /// # Process
    ///
    /// 1. Clear and prepare the merged embeddings cache
    /// 2. Iterate through each position in the sequence
    /// 3. Insert image features where mask is non-zero, text embeddings otherwise
    /// 4. Stack all embeddings into a single tensor
    ///
    /// # Performance Notes
    ///
    /// Currently uses a sequential approach for merging. Future optimizations could include
    /// scatter assignment or CUDA kernels for better performance.
    fn inputs_merger(
        &mut self,
        image_token_mask: &Tensor,
        image_hidden_states: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (seq_len, hidden_size) = inputs_embeds.dims2()?;
        let (img_seq_len, _hidden_size) = image_hidden_states.dims2()?;

        // Convert mask to float for computations
        let mask_f32 = image_token_mask.to_dtype(candle_core::DType::F32)?;

        // Count number of image tokens to validate input
        let num_image_tokens = mask_f32.sum_all()?.to_scalar::<f32>()? as usize;

        // Verify we have the expected number of image hidden states
        if img_seq_len != num_image_tokens {
            return Err(candle_core::Error::Msg(format!(
                "Expected {} image hidden states, got {}",
                num_image_tokens,
                image_hidden_states.dims1()?
            )));
        }

        if num_image_tokens == 0 {
            return Ok(inputs_embeds.clone());
        }

        // Create cumulative sum to get indices for image positions
        // This creates a mapping from sequence positions to image indices
        // Subtract 1 from cumsum where mask is 1 to get 0-based indices
        let image_indices = mask_f32
            .cumsum(0)?
            .affine(1.0, -1.0)?
            .to_dtype(candle_core::DType::I64)?;

        // For positions where mask is 0, we need dummy indices (we'll mask them out anyway)
        // Clamp indices to valid range [0, num_image_tokens-1]
        let max_idx = (num_image_tokens - 1) as i64;
        let clamped_indices = image_indices.clamp(0i64, max_idx)?;

        // Gather image embeddings using the computed indices
        let gathered_image_embeds = image_hidden_states.index_select(&clamped_indices, 0)?;

        // Select image embeddings where mask is true, input embeddings where false
        image_token_mask
            .unsqueeze(1)?
            .expand(&[seq_len, hidden_size])?
            .where_cond(&gathered_image_embeds, inputs_embeds)
    }

    /// Perform a forward pass through the multi-modal SmolVLM2 model.
    ///
    /// This method processes both text tokens and image data through their respective encoders,
    /// merges the representations, and generates the next token logits through the language model.
    /// It handles multiple images by concatenating their features and supports both text-only
    /// and multi-modal inputs.
    ///
    /// # Arguments
    ///
    /// * `xs` - Input token IDs tensor with shape `[sequence_length]`
    /// * `index_pos` - Current position in the sequence for attention cache management
    /// * `image_token_mask` - Binary mask indicating positions where image tokens should be inserted
    /// * `image_data` - Vector of (pixel_values, pixel_attention_masks) tuples for each image
    /// * `ctx` - Inference context containing debug settings and visual introspection
    ///
    /// # Returns
    ///
    /// A `Result` containing logits tensor for next token prediction with shape `[vocab_size]`
    ///
    /// # Process Flow
    ///
    /// 1. **Text Embedding**: Convert input tokens to embeddings via language model
    /// 2. **Vision Processing**: For each image:
    ///    - Encode through SigLIP vision model
    ///    - Transform via connector module (pixel shuffle + projection)
    ///    - Flatten spatial dimensions
    /// 3. **Multi-modal Fusion**: Merge image and text embeddings based on token mask
    /// 4. **Language Generation**: Process fused embeddings through language model
    /// 5. **Output Processing**: Return logits, handling batch dimension appropriately
    ///
    /// # Multi-Image Support
    ///
    /// Multiple images are processed independently and their features are concatenated
    /// before merging with text embeddings, enabling complex multi-image reasoning.
    pub fn forward(
        &mut self,
        xs: &Tensor,
        index_pos: usize,
        image_token_mask: &Tensor,
        image_data: Vec<(&Tensor, &Tensor)>,
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        let input_embeds = self.text_model.embed(xs)?;

        let mut agg_image_hidden_states = vec![];
        for (pixel_values, _pixel_attention_masks) in image_data {
            // TODO: masking
            let image_hidden_states = self.vision_model.forward(pixel_values)?;

            let image_hidden_states = self.connector.forward(&image_hidden_states)?;
            let image_hidden_states = image_hidden_states.flatten(0, 1)?;
            agg_image_hidden_states.push(image_hidden_states);

            if ctx.debug {
                debug!(
                    "[Sub-image] image_hidden_states length: {}",
                    if let Some(el) = agg_image_hidden_states.last() {
                        format!("{}", el.dims2()?.0)
                    } else {
                        "<No hidden image states>".to_string()
                    }
                );
            }

            ctx.vis_introspector.increment_batch_pos();
        }

        let input_embeds = if !agg_image_hidden_states.is_empty() {
            let image_hidden = Tensor::cat(&agg_image_hidden_states, 0)?;
            self.inputs_merger(image_token_mask, &image_hidden, &input_embeds)?
                .unsqueeze(0)?
        } else {
            // No images to process, return original embeddings
            input_embeds.unsqueeze(0)?
        };

        let out = self
            .text_model
            .forward_input_embed(&input_embeds, index_pos, &mut self.cache)?;

        if out.dims().len() == 3 {
            Ok(out.squeeze(0)?)
        } else {
            Ok(out)
        }
    }

    /// Clear the model's attention cache and reset context.
    ///
    /// This method reinitializes the attention cache, effectively clearing any previous
    /// conversation or context history. This is useful when starting a new conversation
    /// or when you want to ensure the model doesn't retain information from previous
    /// inference sessions.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of cache reinitialization
    ///
    /// # Use Cases
    ///
    /// - Starting a new conversation or inference session
    /// - Clearing previous context to prevent interference
    /// - Resetting model state after processing multiple independent inputs
    /// - Memory management in long-running applications
    ///
    /// # Performance Impact
    ///
    /// Cache clearing is a lightweight operation that only reinitializes the attention
    /// cache structure without affecting the model weights or other components.
    pub fn clear_context(&mut self) -> Result<()> {
        self.cache = Cache::new(
            true,
            self.cache_dtype,
            &Self::TEXT_CONFIG,
            &self.cache_device,
        )?;
        Ok(())
    }
}
