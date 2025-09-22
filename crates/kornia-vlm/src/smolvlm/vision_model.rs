use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module};
use std::collections::HashMap;

use crate::context::InferenceContext;

const NUM_OF_HEADS: usize = 16;
const HEAD_DIM: usize = 72;

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl Attention {
    fn new(q: Linear, k: Linear, v: Linear, o: Linear) -> Result<Self> {
        Ok(Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
        })
    }

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
            .transpose(1, 2)?
            .contiguous()?;

        let y = {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;

            let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
            let att = att.broadcast_add(attention_mask)?;

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v)?.contiguous()?.to_dtype(in_dtype)?
        };
        let y = y
            .transpose(1, 2)?
            .reshape(&[batches, patches, hidden_size])?;
        self.o_proj.forward(&y)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
    gelu_coeff: Tensor,
    sqrt_2_over_pi: Tensor,
    one: Tensor,
    half: Tensor,
}

impl Mlp {
    pub fn new(fc1: Linear, fc2: Linear) -> Result<Self> {
        let device = &fc1.weight().device().clone();
        let dtype = fc1.weight().dtype();

        Ok(Self {
            fc1,
            fc2,
            gelu_coeff: Tensor::new(0.044715, device)?.to_dtype(dtype)?,
            sqrt_2_over_pi: Tensor::new(0.7978845608028654, device)?.to_dtype(dtype)?,
            one: Tensor::new(1.0, device)?.to_dtype(dtype)?,
            half: Tensor::new(0.5, device)?.to_dtype(dtype)?,
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

    pub fn forward(&self, xs: &Tensor, ctx: &mut InferenceContext) -> Result<Tensor> {
        let i = &self.fc1.forward(xs)?;
        ctx.vis_introspector.insert("fc1", i);

        let x = self.gelu_tanh(i)?; // python impl. uses gelu approximated with tanh
        ctx.vis_introspector.insert("mlp_act_fn", &x);

        let o = self.fc2.forward(&x)?;
        ctx.vis_introspector.insert("fc2", &o);

        Ok(o)
    }
}

struct Block {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl Block {
    pub fn new(c: &HashMap<String, Tensor>, id: u8) -> Result<Self> {
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

        Ok(Self {
            self_attn: Attention::new(
                Linear::new(w("self_attn.q_proj"), Some(b("self_attn.q_proj"))),
                Linear::new(w("self_attn.k_proj"), Some(b("self_attn.k_proj"))),
                Linear::new(w("self_attn.v_proj"), Some(b("self_attn.v_proj"))),
                Linear::new(w("self_attn.out_proj"), Some(b("self_attn.out_proj"))),
            )?,
            layer_norm1: LayerNorm::new(w("layer_norm1"), b("layer_norm1"), 1e-6),
            mlp: Mlp::new(
                Linear::new(w("mlp.fc1"), Some(b("mlp.fc1"))),
                Linear::new(w("mlp.fc2"), Some(b("mlp.fc2"))),
            )?,
            layer_norm2: LayerNorm::new(w("layer_norm2"), b("layer_norm2"), 1e-6),
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &Tensor,
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        let residual = xs;

        let x = self.layer_norm1.forward(xs)?;
        ctx.vis_introspector.insert("input_layernorm", &x);

        let x = self.self_attn.forward(&x, attention_mask)?;
        ctx.vis_introspector.insert("self_attn", &x);

        let x = (residual + x)?;
        let residual = &x;

        let x = self.layer_norm2.forward(&x)?;
        ctx.vis_introspector.insert("post_layernorm", &x);

        let x = self.mlp.forward(&x, ctx)?;

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

    pub fn load(c: &HashMap<String, Tensor>) -> Result<Self> {
        Ok(Self {
            patch_embedding: Conv2d::new(
                c["model.vision_model.embeddings.patch_embedding.weight"].clone(),
                Some(c["model.vision_model.embeddings.patch_embedding.bias"].clone()),
                Conv2dConfig {
                    // kernel/patch size are intrinsically defined in the weights
                    padding: 0, // "valid" padding
                    stride: 14, // stride equals patch size (14)
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            position_embedding: Embedding::new(
                c["model.vision_model.embeddings.position_embedding.weight"].clone(),
                1152,
            ),
            blocks: (0u8..=26).map(|id| Block::new(c, id).unwrap()).collect(),
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
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        let device = pixel_values.device();
        let dtype = self.patch_embedding.weight().dtype();

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

        let mut hidden_states = {
            let patch_embeddings = self
                .patch_embedding
                .forward(&pixel_values.to_dtype(dtype)?)?;

            ctx.vis_introspector
                .insert("patch_embeddings", &patch_embeddings);

            let patch_embeddings = patch_embeddings.flatten_from(2)?.transpose(1, 2)?;

            let position_ids = {
                let raw_ids = Tensor::arange(0u32, 27 * 27, device)?.expand(&[batch, 27 * 27])?;
                (raw_ids * &patch_attention_masks)?
            };
            let position_embeddings = self.position_embedding.forward(&position_ids)?;

            ctx.vis_introspector
                .insert("position_embeddings", &position_embeddings);

            patch_embeddings + position_embeddings
        }?;

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

        ctx.vis_introspector.start_tracking_depth();
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &patch_attention_masks, ctx)?;
            ctx.vis_introspector.increment_depth();
        }
        ctx.vis_introspector.stop_tracking_depth();

        self.post_layernorm.forward(&hidden_states)
    }
}
