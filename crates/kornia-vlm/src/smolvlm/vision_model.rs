#![allow(unused_variables)]
#![allow(unused_attributes)]

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module};
use std::collections::HashMap;

const NUM_OF_HEADS: usize = 16;
const HEAD_DIM: usize = 72;

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl Attention {
    // fn new(q: Tensor, k: Tensor, v: Tensor, o: Tensor, device: &Device) -> Result<Self> {
    //     Ok(Self {
    //         q_proj: Linear::new(q, None),
    //         k_proj: Linear::new(k, None),
    //         v_proj: Linear::new(v, None),
    //         o_proj: Linear::new(o, None),
    //     })
    // }

    fn forward(&self, x: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batches, patches, hidden_size) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batches, patches, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batches, patches, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batches, patches, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;

        let y =
        // if false {
        //     let q = q.transpose(1, 2)?;
        //     let k = k.transpose(1, 2)?;
        //     let v = v.transpose(1, 2)?;
        //     let softmax_scale = 1f32 / (HEAD_DIM as f32).sqrt();
        //     flash_attn(&q, &k, &v, softmax_scale, batches > 1)?.transpose(1, 2)?.into()
        // } else
        {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;

            let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
            let att = att.broadcast_add(attention_mask)?;

            // println!("{:?}", att.shape());

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v)?.contiguous()?.to_dtype(in_dtype)?
        };
        let y = y
            .transpose(1, 2)?
            .reshape(&[batches, patches, hidden_size])?;
        self.o_proj.forward(&y)
    }
}

struct MLP {
    fc1: Linear,
    fc2: Linear,
    gelu_coeff: Tensor,
    sqrt_2_over_pi: Tensor,
    one: Tensor,
    half: Tensor,
}

impl MLP {
    pub fn new(fc1: Linear, fc2: Linear, device: &Device) -> Result<Self> {
        Ok(Self {
            fc1,
            fc2,
            gelu_coeff: Tensor::new(0.044715, device)?.to_dtype(DType::BF16)?,
            sqrt_2_over_pi: Tensor::new(0.7978845608028654, device)?.to_dtype(DType::BF16)?,
            one: Tensor::new(1.0, device)?.to_dtype(DType::BF16)?,
            half: Tensor::new(0.5, device)?.to_dtype(DType::BF16)?,
        })
    }

    /// PyTorch-like GELU activation with `tanh` approximation.
    pub fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        // Compute: 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
        let x_cubed = input.powf(3.0)?;
        let inner = (input + x_cubed.broadcast_mul(&self.gelu_coeff)?)?;
        let tanh_arg = inner.broadcast_mul(&self.sqrt_2_over_pi)?;
        let tanh = tanh_arg.tanh()?;
        self.half
            .broadcast_mul(&input.broadcast_mul(&(tanh.broadcast_add(&self.one)?))?)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.gelu_tanh(&self.fc1.forward(xs)?)?; // python impl. uses gelu approximated with tanh
        self.fc2.forward(&x)
    }
}

struct Block {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: MLP,
    layer_norm2: LayerNorm,
}

impl Block {
    pub fn new(c: &HashMap<String, Tensor>, id: u8, device: &Device) -> Result<Self> {
        let w = |k| {
            c[&("model.vision_model.encoder.layers.".to_owned()
                + &id.to_string()
                + "."
                + k
                + ".weight")]
                .clone()
        };
        let b = |k| {
            c[&("model.vision_model.encoder.layers.".to_owned()
                + &id.to_string()
                + "."
                + k
                + ".bias")]
                .clone()
        };

        println!("Loaded layer (VT): {:?}", id);

        Ok(Self {
            self_attn: Attention {
                q_proj: Linear::new(w("self_attn.q_proj"), Some(b("self_attn.q_proj"))),
                k_proj: Linear::new(w("self_attn.k_proj"), Some(b("self_attn.k_proj"))),
                v_proj: Linear::new(w("self_attn.v_proj"), Some(b("self_attn.v_proj"))),
                o_proj: Linear::new(w("self_attn.out_proj"), Some(b("self_attn.out_proj"))),
            },
            layer_norm1: LayerNorm::new(w("layer_norm1"), b("layer_norm1"), 1e-6),
            mlp: MLP::new(
                Linear::new(w("mlp.fc1"), Some(b("mlp.fc1"))),
                Linear::new(w("mlp.fc2"), Some(b("mlp.fc2"))),
                device,
            )?,
            layer_norm2: LayerNorm::new(w("layer_norm2"), b("layer_norm2"), 1e-6),
        })
    }

    pub fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let x = self.layer_norm1.forward(xs)?;
        let x = self.self_attn.forward(&x, attention_mask)?;
        let x = (residual + x)?;
        let residual = &x;
        let x = self.layer_norm2.forward(&x)?;
        let x = self.mlp.forward(&x);
        residual + x
    }
}

pub struct SmolVision {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    blocks: Vec<Block>,
    post_layernorm: LayerNorm,
}

impl SmolVision {
    const SUB_PATCH_SIZE: usize = 14;

    pub fn load(c: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        Ok(Self {
            patch_embedding: Conv2d::new(
                c["model.vision_model.embeddings.patch_embedding.weight"].clone(),
                Some(c["model.vision_model.embeddings.patch_embedding.bias"].clone()),
                Conv2dConfig {
                    // kernel/patch size are intrinsically defined in the weights
                    padding: 0,
                    stride: 14,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: Some(candle_core::conv::CudnnFwdAlgo::Direct),
                },
            ),
            position_embedding: Embedding::new(
                c["model.vision_model.embeddings.position_embedding.weight"].clone(),
                1152,
            ),
            blocks: (0u8..=26)
                .into_iter()
                .map(|id| Block::new(c, id, device).unwrap())
                .collect(),
            post_layernorm: LayerNorm::new(
                c["model.vision_model.post_layernorm.weight"].clone(),
                c["model.vision_model.post_layernorm.bias"].clone(),
                1e-6,
            ),
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_masks: &Tensor,
        device: &Device,
    ) -> Result<Tensor> {
        // B = patch rows x patch cols (x number of images)
        // pixel_values: B x 3 x PatchHeight x PatchWidth
        // pixel_attention_masks: B x PatchHeight x PatchWidth
        let (batch, patch_h, patch_w) = pixel_attention_masks.dims3()?;

        // the unfold operation (splitting patches into 27x27 subpatches of 14x14 pixels)
        // 384x384 -> 378x378 where 378=14*27 (divisible by 14)
        // truncate around the middle
        let truncated = pixel_attention_masks
            .narrow(1, 3, patch_h - 3 * 2)?
            .narrow(2, 3, patch_w - 3 * 2)?;
        let patch_attention_masks = truncated
            .unsqueeze(2)?
            .unsqueeze(4)?
            .reshape(&[
                batch,
                patch_h / Self::SUB_PATCH_SIZE,
                Self::SUB_PATCH_SIZE,
                patch_w / Self::SUB_PATCH_SIZE,
                Self::SUB_PATCH_SIZE,
            ])?
            .permute([0, 1, 3, 2, 4])?
            .sum_keepdim([3, 4])?
            .squeeze(4)?
            .squeeze(3)?
            .gt(0.0)?
            .reshape(&[batch, 27 * 27])?
            .contiguous()?
            .to_dtype(DType::U32)?;
        // patch_attention_masks: B x PatchRows x PatchCols x 196

        // println!("{:?}", truncated.shape());
        // println!("{:?}", patch_attention_masks);
        // println!("{:?}", patch_attention_masks.to_vec3::<u8>());

        let mut hidden_states = {
            let patch_embeddings = self
                .patch_embedding
                .forward(&pixel_values.to_dtype(DType::BF16)?)?;
            // println!("{:?}", patch_embeddings.shape());
            let patch_embeddings = patch_embeddings.flatten_from(2)?.transpose(1, 2)?;
            // println!("{:?}", patch_embeddings.shape());

            let position_ids = {
                let raw_ids = Tensor::arange(0u32, 27 * 27, device)?.expand(&[batch, 27 * 27])?;
                (raw_ids * &patch_attention_masks)?
            };
            let position_embeddings = self.position_embedding.forward(&position_ids)?;
            // println!("{:?}", patch_embeddings);
            // println!("{:?}", position_embeddings);
            // println!("{:?}", patch_embeddings.to_dtype(DType::F32)?.to_vec3::<f32>());
            // println!("{:?}", position_embeddings.to_vec3::<u32>());
            patch_embeddings + position_embeddings
        }?;
        // println!(">> {:?}", hidden_states);

        let patch_attention_masks =
            {
                let expanded_masks = patch_attention_masks
                    .unsqueeze(1)?
                    .unsqueeze(1)?
                    .expand(&[batch, 1, 27 * 27, 27 * 27])?; // batch, head_dim, subpatches, subpatches
                let inverted_mask = Tensor::ones_like(&expanded_masks)?.sub(&expanded_masks)?;
                let neg_infs = Tensor::full(f32::NEG_INFINITY, inverted_mask.shape(), device)?;
                inverted_mask.where_cond(&neg_infs, &inverted_mask.to_dtype(DType::F32)?)?
            };
        // println!(">> {:?}", patch_attention_masks);

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &patch_attention_masks)?;
        }
        self.post_layernorm.forward(&hidden_states)
    }
}
