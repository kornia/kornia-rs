use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, Module};
use log::debug;

use crate::{context::InferenceContext, smolvlm::text_model::SmolText};

use super::vision_model::SmolVision;

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

pub struct SmolModel {
    embed: Embedding,

    pub vision: SmolVision,

    connector: Connector,

    pub text: SmolText,
}

impl SmolModel {
    pub fn load(c: &HashMap<String, Tensor>) -> Result<Self> {
        Ok(Self {
            embed: Embedding::new(c["model.text_model.embed_tokens.weight"].clone(), 2048),

            vision: SmolVision::load(c)?,

            connector: Connector {
                modality_proj: Linear::new(
                    c["model.connector.modality_projection.proj.weight"].clone(),
                    None,
                ),
            },

            text: SmolText::load(c)?,
        })
    }

    fn inputs_merger(
        &mut self,
        image_token_mask: &Tensor,
        image_hidden_states: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        Self::merge_tensors_impl(image_token_mask, image_hidden_states, inputs_embeds)
    }

    // Static and for tests
    fn merge_tensors_impl(
        image_token_mask: &Tensor,
        image_hidden_states: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        if image_hidden_states.dim(0)? == 0 {
            return Ok(inputs_embeds.clone());
        }
        let device = image_token_mask.device();

        // Type conversion for GPU memory savings, cumsum() handling and - 1 handling
        let mask_f32 = image_token_mask.to_dtype(candle_core::DType::F32)?;
        let cumsum = mask_f32.cumsum(0)?;

        // clamp() protects from -1 index underflow and max index overflows
        let one = Tensor::new(&[1.0f32], device)?;
        let max_index = (image_hidden_states.dim(0)? as f64) - 1.0;

        let image_indices = cumsum
            .broadcast_sub(&one)?
            .clamp(0.0, max_index)?
            .to_dtype(candle_core::DType::U32)?;

        // Stretching the image embeddings
        let image_stretched = image_hidden_states.index_select(&image_indices, 0)?;

        // Merging
        let mask_bool = image_token_mask.ne(0.0)?;
        let mask_broadcast = mask_bool
            .unsqueeze(1)?
            .broadcast_as(inputs_embeds.shape())?;

        let merged_embeds = mask_broadcast.where_cond(&image_stretched, inputs_embeds)?;

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
        let inputs_embeds = self.embed.forward(xs)?;

        ctx.text_introspector
            .insert("input_embeddings", &inputs_embeds);

        let mut agg_image_hidden_states = vec![];
        for (pixel_values, pixel_attention_masks) in image_data {
            let image_hidden_states =
                self.vision
                    .forward(pixel_values, pixel_attention_masks, ctx)?;
            ctx.vis_introspector
                .insert("post_layernorm", &image_hidden_states);

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

        let inputs_embeds = if !agg_image_hidden_states.is_empty() {
            let image_hidden = Tensor::cat(&agg_image_hidden_states, 0)?;
            self.inputs_merger(image_token_mask, &image_hidden, &inputs_embeds)?
        } else {
            // No images to process, return original embeddings
            inputs_embeds
        };

        self.text.forward(inputs_embeds, index_pos, ctx)
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        for b in self.text.blocks.iter_mut() {
            b.attn.reset_cache()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::smolvlm::model::SmolModel;

    // Creates text_embeds of 1s and img_embeds of 2s
    fn create_data(
        mask_pattern: Vec<u8>,
        img_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let len = mask_pattern.len();

        let mask = Tensor::from_vec(mask_pattern, (len,), device)?;

        let text_embeds = Tensor::ones((len, 1), DType::F32, device)?;

        let img_embeds = Tensor::full(2.0_f32, (img_len, 1), device)?;

        Ok((mask, text_embeds, img_embeds))
    }

    #[test]
    fn test_sandwich_merge() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![0, 1, 1, 0], 2, &Device::Cpu)?;

        let result = SmolModel::merge_tensors_impl(&mask, &img_embeds, &text_embeds)?;

        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;

        assert_eq!(result_vec, vec![1.0, 2.0, 2.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_img_only_merge() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![1, 1, 1, 1], 4, &Device::Cpu)?;

        let result = SmolModel::merge_tensors_impl(&mask, &img_embeds, &text_embeds)?;

        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;

        assert_eq!(result_vec, vec![2.0, 2.0, 2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_text_only_merge() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![0, 0, 0, 0], 4, &Device::Cpu)?;

        let result = SmolModel::merge_tensors_impl(&mask, &img_embeds, &text_embeds)?;

        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(result_vec, vec![1.0, 1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_no_image() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![0, 0, 0, 0], 0, &Device::Cpu)?;

        let result = SmolModel::merge_tensors_impl(&mask, &img_embeds, &text_embeds)?;

        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(result_vec, vec![1.0, 1.0, 1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_all_images() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![1, 1, 1, 1], 4, &Device::Cpu)?;

        let result = SmolModel::merge_tensors_impl(&mask, &img_embeds, &text_embeds)?;

        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(result_vec, vec![2.0, 2.0, 2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_alternate_embedding() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![1, 0, 1, 0], 2, &Device::Cpu)?;

        let result = SmolModel::merge_tensors_impl(&mask, &img_embeds, &text_embeds)?;

        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(result_vec, vec![2.0, 1.0, 2.0, 1.0]);

        Ok(())
    }
}
