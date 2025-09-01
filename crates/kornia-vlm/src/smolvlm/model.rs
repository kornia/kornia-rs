use std::collections::HashMap;

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

use crate::smolvlm::text_model::SmolText;

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
    merged_embeds: Vec<Tensor>, // cache results

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
            merged_embeds: Vec::new(),

            text: SmolText::load(c)?,
        })
    }

    fn inputs_merger(
        &mut self,
        image_token_mask: &Tensor,
        image_hidden_states: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let total_length = image_token_mask.dims1()?;

        // let image_hidden_states = self.image_hidden_states.as_ref().unwrap().flatten(0, 1)?;

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
        introspector: &mut super::introspector::ActivationIntrospector,
        vis_introspector: &mut super::introspector::ActivationIntrospector,
    ) -> Result<Tensor> {
        let inputs_embeds = self.embed.forward(xs)?;

        #[cfg(feature = "debug")]
        introspector.insert("input_embeddings", &inputs_embeds);

        let mut agg_image_hidden_states = vec![];
        for (pixel_values, pixel_attention_masks) in image_data {
            let image_hidden_states =
                self.vision
                    .forward(pixel_values, pixel_attention_masks, vis_introspector)?;
            #[cfg(feature = "debug")]
            vis_introspector.insert("post_layernorm", &image_hidden_states);

            let image_hidden_states = self.connector.forward(&image_hidden_states)?;
            let image_hidden_states = image_hidden_states.flatten(0, 1)?;
            agg_image_hidden_states.push(image_hidden_states);

            #[cfg(feature = "debug")]
            println!(
                "[Sub-image] image_hidden_states length: {}",
                agg_image_hidden_states.last().unwrap().dims2()?.0
            );

            vis_introspector.increment_batch_pos();
        }

        let inputs_embeds = if !agg_image_hidden_states.is_empty() {
            let image_hidden = Tensor::cat(&agg_image_hidden_states, 0)?;
            self.inputs_merger(&image_token_mask, &image_hidden, &inputs_embeds)?
        } else {
            // No images to process, return original embeddings
            inputs_embeds
        };

        self.text.forward(inputs_embeds, index_pos, introspector)
    }

    pub fn reset_cache(&mut self) {
        self.text
            .blocks
            .iter_mut()
            .for_each(|b| b.attn.reset_cache());
    }
}
