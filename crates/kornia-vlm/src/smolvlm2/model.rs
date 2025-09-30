use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::{
    llama::{self, Cache},
    siglip,
};
use log::debug;

use crate::context::InferenceContext;

pub struct Connector {
    modality_proj: Linear,
}

impl Connector {
    const SCALE_FACTOR: usize = 3;
    const HEIGHT: usize = 27;
    const WIDTH: usize = 27;

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

    pub fn forward(&self, image_hidden_states: &Tensor) -> Result<Tensor> {
        let image_hidden_states = self.pixel_shuffle(image_hidden_states)?;
        self.modality_proj.forward(&image_hidden_states)
    }
}

pub struct Model {
    text_model: candle_transformers::models::llama::Llama,
    connector: Connector,
    vision_model: candle_transformers::models::siglip::VisionModel,

    // TODO: move these caching into inference context
    cache_dtype: DType,
    cache_device: Device,
    cache: Cache,

    merged_embeds: Vec<Tensor>, // cache results
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
        use_flash_attn: false,
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

            merged_embeds: Vec::new(),
        })
    }

    fn inputs_merger(
        &mut self,
        image_token_mask: &Tensor,
        image_hidden_states: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let total_length = image_token_mask.dims1()?;

        self.merged_embeds.clear();
        if self.merged_embeds.capacity() < total_length {
            self.merged_embeds
                .reserve(total_length - self.merged_embeds.capacity());
        }

        let mut c = 0;
        // TODO: is there a better way to do this? (scatter assignment? cuda kernel?)
        for (i, mask) in image_token_mask.to_vec1::<u8>()?.into_iter().enumerate() {
            self.merged_embeds.push(if mask != 0 {
                c += 1;
                image_hidden_states.i(c - 1)?
            } else {
                inputs_embeds.i(i)?
            });
        }

        let merged_embeds = Tensor::stack(&self.merged_embeds, 0)?;

        Ok(merged_embeds)
    }

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
                    agg_image_hidden_states
                        .last()
                        .expect("No image hidden states")
                        .dims2()?
                        .0
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
