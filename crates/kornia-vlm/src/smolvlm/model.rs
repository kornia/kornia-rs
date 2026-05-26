use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

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

    /// Interleaves image features into text embeddings at positions marked by the boolean mask.
    ///
    /// Image indices are derived using a clamped cumulative sum of the mask. To maintain a
    /// constant memory footprint, the sequence is processed and reassembled in 1024-token chunks
    /// using zero-copy views (`.narrow()`) and vectorized conditional selection (`.where_cond()`).
    fn inputs_merger(
        image_token_mask: &Tensor,
        image_hidden_states: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        const BATCH_OR_SEQ_DIM: usize = 0;
        const CHANNEL_OR_EMBED_DIM: usize = 1;
        let (seq_len, _) = inputs_embeds.dims2()?;

        let chunk_size = 1024;
        let mut chunks = Vec::new();

        if image_hidden_states.dim(BATCH_OR_SEQ_DIM)? == 0 {
            return Ok(inputs_embeds.clone());
        }
        let device = image_token_mask.device();

        // Cumsum from binary mask
        let mask_f32 = image_token_mask.to_dtype(candle_core::DType::F32)?;
        let cumsum = mask_f32.cumsum(BATCH_OR_SEQ_DIM)?;

        let one = Tensor::new(&[1.0f32], device)?;

        let image_indices = cumsum
            .broadcast_sub(&one)?
            .clamp(0.0, f64::MAX)? // Protects from -1.0 but allows out-of-bounds to fail downstream.
            .to_dtype(candle_core::DType::U32)?;

        // Chunk-Based Merging
        for start in (0..seq_len).step_by(chunk_size) {
            let len = usize::min(chunk_size, seq_len - start);

            let mask_chunk = image_token_mask.narrow(0, start, len)?;
            let text_chunk = inputs_embeds.narrow(0, start, len)?;
            let image_indices_chunk = image_indices.narrow(0, start, len)?;

            let image_chunk_stretched =
                image_hidden_states.index_select(&image_indices_chunk, BATCH_OR_SEQ_DIM)?;

            // Broadcast the 1D mask to match the embedding dimension and blend the tensors.
            let mask_chunk_bool = mask_chunk.ne(0.0)?;
            let mask_chunk_broad = mask_chunk_bool
                .unsqueeze(CHANNEL_OR_EMBED_DIM)?
                .broadcast_as(text_chunk.shape())?;

            // Select from image_features if mask == true, else text_features.
            let merged_chunk = mask_chunk_broad.where_cond(&image_chunk_stretched, &text_chunk)?;

            chunks.push(merged_chunk);
        }

        // Recombine the chunks
        Tensor::cat(&chunks, 0)
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
                if let Some(last) = agg_image_hidden_states.last() {
                    log::debug!(
                        "[Sub-image] image_hidden_states length: {}",
                        last.dims2()?.0
                    );
                }
            }

            ctx.vis_introspector.increment_batch_pos();
        }

        let inputs_embeds = if !agg_image_hidden_states.is_empty() {
            let image_hidden = Tensor::cat(&agg_image_hidden_states, 0)?;
            Self::inputs_merger(image_token_mask, &image_hidden, &inputs_embeds)?
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
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;
        assert_eq!(result_vec, vec![1.0, 2.0, 2.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_img_only_merge() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![1, 1, 1, 1], 4, &Device::Cpu)?;
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;
        assert_eq!(result_vec, vec![2.0, 2.0, 2.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_text_only_merge() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![0, 0, 0, 0], 4, &Device::Cpu)?;
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(result_vec, vec![1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_no_image() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![0, 0, 0, 0], 0, &Device::Cpu)?;
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(result_vec, vec![1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_alternate_embedding() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![1, 0, 1, 0], 2, &Device::Cpu)?;
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(result_vec, vec![2.0, 1.0, 2.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_alternate_embeddings_multichunk_2049() -> Result<()> {
        let seq_len = 2049;
        let mask_vec: Vec<u8> = (0..seq_len)
            .map(|i| if i % 2 == 0 { 1 } else { 0 })
            .collect();
        let img_count = mask_vec.iter().map(|&x| x as usize).sum::<usize>();

        let (mask, text_embeds, img_embeds) = create_data(mask_vec, img_count, &Device::Cpu)?;

        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;
        let expected_vec: Vec<f32> = (0..seq_len)
            .map(|i| if i % 2 == 0 { 2.0 } else { 1.0 })
            .collect();

        assert_eq!(result_vec, expected_vec);

        Ok(())
    }

    #[test]
    fn test_contiguous_spanning_boundary_2049() -> Result<()> {
        let seq_len = 2049;
        let mut mask_vec: Vec<u8> = vec![0; seq_len]; // Start with all text (0)

        mask_vec[1000..1050].fill(1);
        mask_vec[2040..2049].fill(1);

        let img_count = mask_vec.iter().map(|&x| x as usize).sum::<usize>();
        let (mask, text_embeds, img_embeds) =
            create_data(mask_vec, img_count, &candle_core::Device::Cpu)?;

        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        let result_vec = result.flatten_all()?.to_vec1::<f32>()?;

        let expected_vec: Vec<f32> = (0..seq_len)
            .map(|i| {
                if (1000..1050).contains(&i) || (2040..2049).contains(&i) {
                    2.0
                } else {
                    1.0
                }
            })
            .collect();

        assert_eq!(result_vec, expected_vec);

        Ok(())
    }

    #[test]
    fn test_out_of_bounds_image_tokens() -> Result<()> {
        let mask_pattern = vec![1, 1, 1, 0];
        let (mask, text_embeds, img_embeds) = create_data(mask_pattern, 2, &Device::Cpu)?;
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds);
        assert!(result.is_err(),);
        Ok(())
    }

    #[test]
    fn test_empty_sequence() -> Result<()> {
        let (mask, text_embeds, img_embeds) = create_data(vec![], 0, &Device::Cpu)?;
        let result = SmolModel::inputs_merger(&mask, &img_embeds, &text_embeds)?;
        assert_eq!(result.dims2()?, (0, 1));
        Ok(())
    }
}
