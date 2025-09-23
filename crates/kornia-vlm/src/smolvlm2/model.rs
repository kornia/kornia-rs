use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use candle_transformers::models::llama::{self, Cache};

// pub struct Connector {
//     modality_proj: Linear,
// }

// impl Connector {
//     const SCALE_FACTOR: usize = 3;
//     const HEIGHT: usize = 27;
//     const WIDTH: usize = 27;

//     fn load(vb: VarBuilder) -> Result<Self> {
//         Ok(Self {
//             modality_proj: Linear::new(vb.get("modality_projection.proj.weight")?, None),
//         })
//     }

//     fn pixel_shuffle(&self, x: &Tensor) -> Result<Tensor> {
//         let (batch, patches, embed_dim) = x.dims3()?; // patches == HEIGHT*WIDTH

//         // B,P,E => B,H,W,E => B,H/S,S,W/S,S,E => B,H/S,W/S,S,S,E => B,P/S^2,S^2*E
//         x.reshape(&[batch, Self::HEIGHT, Self::WIDTH, embed_dim])?
//             .reshape(&[
//                 batch,
//                 Self::HEIGHT / Self::SCALE_FACTOR,
//                 Self::SCALE_FACTOR,
//                 Self::WIDTH / Self::SCALE_FACTOR,
//                 Self::SCALE_FACTOR,
//                 embed_dim,
//             ])?
//             .permute([0, 1, 3, 2, 4, 5])?
//             .reshape(&[
//                 batch,
//                 patches / (Self::SCALE_FACTOR * Self::SCALE_FACTOR),
//                 embed_dim * Self::SCALE_FACTOR * Self::SCALE_FACTOR,
//             ])
//     }

//     pub fn forward(&self, image_hidden_states: &Tensor) -> Result<Tensor> {
//         let image_hidden_states = self.pixel_shuffle(image_hidden_states)?;
//         self.modality_proj.forward(&image_hidden_states)
//     }
// }

pub struct Model {
    vision_model: (),
    connector: (),
    text_model: candle_transformers::models::llama::Llama,

    // TODO: move these caching into inference context
    image_hidden_states: Option<Tensor>,
    merged_embeds: Vec<Tensor>,
    cache: Cache,
}

impl Model {
    const HIDDEN_SIZE: usize = 2048;
    const VOCAB_SIZE: usize = 49280;
    const CONFIG: llama::Config = llama::Config {
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

    pub fn load(vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let vb = vb.rename_f(|f: &str| {
            // If variables were namespaced as `model.text_model.model.*`, remove
            // the extra `.model` so we look up `model.text_model.*` which is
            // what the checkpoint uses in this repository.
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

            // Map bare lm_head key to the text model lm_head namespace.
            if f == "model.text_model.lm_head.weight" {
                return "lm_head.weight".to_string();
            }

            f.to_string()
        });

        Ok(Self {
            vision_model: (),
            connector: (), // Connector::load(vb.pp("model.connector"))?,
            text_model: candle_transformers::models::llama::Llama::load(
                vb.pp("model.text_model"),
                &Self::CONFIG,
            )?,

            image_hidden_states: None,
            merged_embeds: Vec::new(),
            cache: Cache::new(true, dtype, &Self::CONFIG, device)?,
        })
    }

    // fn inputs_merger(
    //     &mut self,
    //     image_token_mask: &Tensor,
    //     inputs_embeds: &Tensor,
    // ) -> Result<Tensor> {
    //     let total_length = image_token_mask.dims1()?;

    //     let image_hidden_states = self.image_hidden_states.as_ref().unwrap().flatten(0, 1)?;

    //     self.merged_embeds.clear();
    //     if self.merged_embeds.capacity() < total_length {
    //         self.merged_embeds
    //             .reserve(total_length - self.merged_embeds.capacity());
    //     }

    //     let mut c = 0;
    //     // TODO: is there a better way to do this? (scatter assignment? cuda kernel?)
    //     for (i, mask) in image_token_mask.to_vec1::<u8>()?.into_iter().enumerate() {
    //         self.merged_embeds.push(if mask != 0 {
    //             c += 1;
    //             image_hidden_states.i(c - 1)?
    //         } else {
    //             inputs_embeds.i(i)?
    //         });
    //     }

    //     let merged_embeds = Tensor::stack(&self.merged_embeds, 0)?;

    //     Ok(merged_embeds)
    // }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        index_pos: usize,
        vision_data: Option<(Tensor, &Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        // let mut inputs_embeds = self.embed.forward(xs)?;

        let mut input_embeds = self.text_model.embed(xs)?;

        // if let Some((image_token_mask, pixel_values, pixel_attention_masks)) = vision_data {
        //     // TODO: this assumes there will be at most one new images added
        //     inputs_embeds = if self.image_hidden_states.is_some() {
        //         self.inputs_merger(&image_token_mask, &inputs_embeds)?
        //     } else {
        //         let image_hidden_states =
        //             self.vision.forward(pixel_values, pixel_attention_masks)?;
        //         let image_hidden_states = self.connector.forward(&image_hidden_states)?;
        //         self.image_hidden_states = Some(image_hidden_states);

        //         self.inputs_merger(&image_token_mask, &inputs_embeds)?
        //     };
        // }

        input_embeds = input_embeds.unsqueeze(0)?;

        let out = self
            .text_model
            .forward_input_embed(&input_embeds, index_pos, &mut self.cache)?;

        if out.dims().len() == 3 {
            Ok(out.squeeze(0)?)
        } else {
            Ok(out)
        }
    }
}
