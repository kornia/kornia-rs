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

    vision: SmolVision,

    connector: Connector,
    image_hidden_states: Option<Tensor>, // TODO: to be used for caching previous image hidden states
    merged_embeds: Vec<Tensor>,          // cache results

    text: SmolText,

    pub DEBUG_embeds: Option<Tensor>, // for debugging purposes, to be removed later
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
            image_hidden_states: None,
            merged_embeds: Vec::new(),

            text: SmolText::load(c)?,

            DEBUG_embeds: None,
        })
    }

    fn inputs_merger(
        &mut self,
        image_token_mask: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let total_length = image_token_mask.dims1()?;

        let image_hidden_states = self.image_hidden_states.as_ref().unwrap().flatten(0, 1)?;

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
        vision_data: Option<(Tensor, &Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.embed.forward(xs)?;
        self.DEBUG_embeds = Some(inputs_embeds.clone());

        if let Some((image_token_mask, pixel_values, pixel_attention_masks)) = vision_data {
            // println!(
            //     "image_token_mask: {:?}, pixel_values: {:?}, pixel_attention_masks: {:?}",
            //     image_token_mask.dims(),
            //     pixel_values.dims(),
            //     pixel_attention_masks.dims()
            // );

            // TODO: this assumes there will be at most one new images added
            inputs_embeds = if self.image_hidden_states.is_some() {
                self.inputs_merger(&image_token_mask, &inputs_embeds)?
            } else {
                let image_hidden_states =
                    self.vision.forward(pixel_values, pixel_attention_masks)?;
                let image_hidden_states = self.connector.forward(&image_hidden_states)?;
                self.image_hidden_states = Some(image_hidden_states);

                self.inputs_merger(&image_token_mask, &inputs_embeds)?
            };
        }

        self.text.forward(inputs_embeds, index_pos)
    }
}
