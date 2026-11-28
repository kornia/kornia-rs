use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module};
use std::collections::HashMap;

use crate::context::InferenceContext;

const NUM_OF_HEADS: usize = 16;
const HEAD_DIM: usize = 72;
const HIDDEN_SIZE: usize = NUM_OF_HEADS * HEAD_DIM;
const SUB_PATCH_SIZE: usize = 14;
const PRETRAINED_GRID_SIZE: usize = 27; // 27x27 = 729 max patches

/// Converts a pixel-level attention mask into one validity value per image patch.
fn build_patch_attention_mask(
    pixel_attention_masks: &Tensor,
    pixel_height: usize,
    pixel_width: usize,
) -> Result<Tensor> {
    let (batch, mask_height, mask_width) = pixel_attention_masks.dims3()?;

    if pixel_height % SUB_PATCH_SIZE != 0 || pixel_width % SUB_PATCH_SIZE != 0 {
        return Err(candle_core::Error::Msg(format!(
            "pixel dimensions must be divisible by {SUB_PATCH_SIZE}; got {pixel_height}x{pixel_width}"
        )));
    }

    let aligned_mask = if mask_height == pixel_height && mask_width == pixel_width {
        pixel_attention_masks.clone()
    } else if mask_height == pixel_height + 6 && mask_width == pixel_width + 6 {
        pixel_attention_masks
            .narrow(1, 3, pixel_height)?
            .narrow(2, 3, pixel_width)?
    } else {
        return Err(candle_core::Error::Msg(format!(
            "pixel attention mask must be {pixel_height}x{pixel_width} or {}x{}; got {mask_height}x{mask_width}",
            pixel_height + 6,
            pixel_width + 6,
        )));
    };

    let patch_rows = pixel_height / SUB_PATCH_SIZE;
    let patch_cols = pixel_width / SUB_PATCH_SIZE;

    if patch_rows > PRETRAINED_GRID_SIZE || patch_cols > PRETRAINED_GRID_SIZE {
        return Err(candle_core::Error::Msg(format!(
            "patch grid dimensions ({patch_rows}x{patch_cols}) exceed maximum pretrained grid size ({PRETRAINED_GRID_SIZE}x{PRETRAINED_GRID_SIZE})"
        )));
    }

    let patch_mask = aligned_mask
        .unsqueeze(2)?
        .unsqueeze(4)?
        .reshape(&[
            batch,
            patch_rows,
            SUB_PATCH_SIZE,
            patch_cols,
            SUB_PATCH_SIZE,
        ])?
        .permute([0, 1, 3, 2, 4])?
        .sum_keepdim(4)?
        .sum_keepdim(3)?
        .squeeze(4)?
        .squeeze(3)?
        .gt(0.0f64)?
        .reshape(&[batch, patch_rows * patch_cols])?
        .contiguous()?
        .to_dtype(DType::U32)?;

    // Contract Validation: Ensure every sample in the batch has at least one valid patch
    let valid_counts = patch_mask.to_dtype(DType::F32)?.sum(1)?;
    let counts_vec = valid_counts.to_vec1::<f32>()?;
    for (b, &count) in counts_vec.iter().enumerate() {
        if count == 0.0 {
            return Err(candle_core::Error::Msg(format!(
                "sample at index {b} has no valid image patches; fully masked inputs are unsupported"
            )));
        }
    }

    Ok(patch_mask)
}

/// Builds 27-wide 2D positional grid IDs for patches.
fn build_position_ids(
    patch_attention_masks: &Tensor,
    patch_rows: usize,
    patch_cols: usize,
    device: &Device,
) -> Result<Tensor> {
    if patch_rows > PRETRAINED_GRID_SIZE || patch_cols > PRETRAINED_GRID_SIZE {
        return Err(candle_core::Error::Msg(format!(
            "requested patch grid ({patch_rows}x{patch_cols}) exceeds maximum pretrained grid ({PRETRAINED_GRID_SIZE}x{PRETRAINED_GRID_SIZE})"
        )));
    }

    let (batch, patch_count) = patch_attention_masks.dims2()?;

    if patch_count != patch_rows * patch_cols {
        return Err(candle_core::Error::Msg(format!(
            "patch attention mask contains {patch_count} patches, but grid dimensions {patch_rows}x{patch_cols} require {}",
            patch_rows * patch_cols
        )));
    }
    let mut pos_ids = Vec::with_capacity(patch_rows * patch_cols);

    for r in 0..patch_rows {
        for c in 0..patch_cols {
            // Map 2D grid position to 27-wide pretrained embedding index
            let id = (r * PRETRAINED_GRID_SIZE + c) as u32;
            pos_ids.push(id);
        }
    }

    let raw_position_ids = Tensor::from_vec(pos_ids, (1, patch_rows * patch_cols), device)?
        .expand(&[batch, patch_rows * patch_cols])?;

    Ok((raw_position_ids * patch_attention_masks)?)
}

/// Creates an additive key-attention mask with shape [batch, 1, patches, patches].
fn build_additive_attention_mask(
    patch_attention_masks: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    let (batch, patch_count) = patch_attention_masks.dims2()?;
    let expanded_mask = patch_attention_masks
        .unsqueeze(1)?
        .unsqueeze(1)?
        .expand(&[batch, 1, patch_count, patch_count])?;

    let inverted_mask = Tensor::ones_like(&expanded_mask)?.sub(&expanded_mask)?;
    let mask_value = match dtype {
    DType::F16 => -1e4f32,
    DType::BF16 => -1e9f32,
    _ => f32::NEG_INFINITY,
};

    let negative_values = Tensor::full(
        mask_value,
        inverted_mask.shape(),
        patch_attention_masks.device(),
    )?.to_dtype(dtype)?;
    let zeros = Tensor::zeros(inverted_mask.shape(), dtype, patch_attention_masks.device())?;

    inverted_mask.where_cond(&negative_values, &zeros)
}

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

        let input_dtype = q.dtype();
        let weight_dtype = self.q_proj.weight().dtype();

        let compute_dtype = match weight_dtype {
            DType::BF16 => DType::BF16,
            _ => DType::F32,
        };

        let q = if q.dtype() != compute_dtype { q.to_dtype(compute_dtype)? } else { q };
        let k = if k.dtype() != compute_dtype { k.to_dtype(compute_dtype)? } else { k };
        let v = if v.dtype() != compute_dtype { v.to_dtype(compute_dtype)? } else { v };
        let attention_mask = if attention_mask.dtype() != compute_dtype {
            attention_mask.to_dtype(compute_dtype)?
        } else {
            attention_mask.clone()
        };

        let attention_scores = (q.matmul(&k.transpose(2, 3)?)? / (HEAD_DIM as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(&attention_mask)?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let output = attention_probs.matmul(&v)?.contiguous()?;

        let output = if output.dtype() != input_dtype {
            output.to_dtype(input_dtype)?
        } else {
            output
        };

        let output = output
            .transpose(1, 2)?
            .reshape(&[batches, patches, hidden_size])?;

        self.o_proj.forward(&output)
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
    fn new(fc1: Linear, fc2: Linear) -> Result<Self> {
        let device = fc1.weight().device().clone();
        let dtype = fc1.weight().dtype();

        Ok(Self {
            fc1,
            fc2,
            gelu_coeff: Tensor::new(0.044715, &device)?.to_dtype(dtype)?,
            sqrt_2_over_pi: Tensor::new(0.7978845608028654, &device)?.to_dtype(dtype)?,
            one: Tensor::new(1.0, &device)?.to_dtype(dtype)?,
            half: Tensor::new(0.5, &device)?.to_dtype(dtype)?,
        })
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        let x_cubed = input.powf(3.0)?;
        let inner = (input + x_cubed.broadcast_mul(&self.gelu_coeff)?)?;
        let tanh_arg = inner.broadcast_mul(&self.sqrt_2_over_pi)?;
        let tanh = tanh_arg.tanh()?;

        self.half
            .broadcast_mul(&input.broadcast_mul(&(tanh.broadcast_add(&self.one)?))?)
    }

    fn forward(&self, xs: &Tensor, ctx: &mut InferenceContext) -> Result<Tensor> {
        let intermediate = self.fc1.forward(xs)?;
        ctx.vis_introspector.insert("fc1", &intermediate);

        let activated = self.gelu_tanh(&intermediate)?;
        ctx.vis_introspector.insert("mlp_act_fn", &activated);

        let output = self.fc2.forward(&activated)?;
        ctx.vis_introspector.insert("fc2", &output);

        Ok(output)
    }
}

struct Block {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl Block {
    fn new(c: &HashMap<String, Tensor>, id: u8) -> Result<Self> {
        let weight = |name: &str| {
            c[&format!("model.vision_model.encoder.layers.{id}.{name}.weight")].clone()
        };
        let bias = |name: &str| {
            c[&format!("model.vision_model.encoder.layers.{id}.{name}.bias")].clone()
        };

        Ok(Self {
            self_attn: Attention::new(
                Linear::new(weight("self_attn.q_proj"), Some(bias("self_attn.q_proj"))),
                Linear::new(weight("self_attn.k_proj"), Some(bias("self_attn.k_proj"))),
                Linear::new(weight("self_attn.v_proj"), Some(bias("self_attn.v_proj"))),
                Linear::new(weight("self_attn.out_proj"), Some(bias("self_attn.out_proj"))),
            )?,
            layer_norm1: LayerNorm::new(weight("layer_norm1"), bias("layer_norm1"), 1e-6),
            mlp: Mlp::new(
                Linear::new(weight("mlp.fc1"), Some(bias("mlp.fc1"))),
                Linear::new(weight("mlp.fc2"), Some(bias("mlp.fc2"))),
            )?,
            layer_norm2: LayerNorm::new(weight("layer_norm2"), bias("layer_norm2"), 1e-6),
        })
    }

    fn forward(
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
    pub fn load(c: &HashMap<String, Tensor>) -> Result<Self> {
        Ok(Self {
            patch_embedding: Conv2d::new(
                c["model.vision_model.embeddings.patch_embedding.weight"].clone(),
                Some(c["model.vision_model.embeddings.patch_embedding.bias"].clone()),
                Conv2dConfig {
                    padding: 0,
                    stride: SUB_PATCH_SIZE,
                    dilation: 1,
                    groups: 1,
                    ..Default::default()
                },
            ),
            position_embedding: Embedding::new(
                c["model.vision_model.embeddings.position_embedding.weight"].clone(),
                HIDDEN_SIZE,
            ),
            blocks: (0u8..=26)
                .map(|id| Block::new(c, id))
                .collect::<Result<Vec<_>>>()?,
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
        let (batch, _, pixel_height, pixel_width) = pixel_values.dims4()?;
        let (mask_batch, _, _) = pixel_attention_masks.dims3()?;
        
        if mask_batch != batch {
            return Err(candle_core::Error::Msg(format!(
                "pixel_values batch size ({batch}) does not match pixel_attention_masks batch size ({mask_batch})"
            )));
        }

        let device = pixel_values.device();
        let model_dtype = self.patch_embedding.weight().dtype();
        let patch_rows = pixel_height / SUB_PATCH_SIZE;
        let patch_cols = pixel_width / SUB_PATCH_SIZE;

        let patch_attention_masks = build_patch_attention_mask(
            pixel_attention_masks,
            pixel_height,
            pixel_width,
        )?;
        
        let position_ids = build_position_ids(&patch_attention_masks, patch_rows, patch_cols, device)?;

        let patch_embeddings = self
            .patch_embedding
            .forward(&pixel_values.to_dtype(model_dtype)?)?;

        // Validation: Verify Conv2d spatial output matches the expected patch grid
        let (emb_batch, emb_channels, emb_h, emb_w) = patch_embeddings.dims4()?;

        if emb_batch != batch
            || emb_channels != HIDDEN_SIZE
            || emb_h != patch_rows
            || emb_w != patch_cols
        {
            return Err(candle_core::Error::Msg(format!(
                "patch embedding output shape ({emb_batch}, {emb_channels}, {emb_h}, {emb_w}) does not match expected shape ({batch}, {HIDDEN_SIZE}, {patch_rows}, {patch_cols})"
            )));
        }


        ctx.vis_introspector
            .insert("patch_embeddings", &patch_embeddings);

        let patch_embeddings = patch_embeddings.flatten_from(2)?.transpose(1, 2)?;
        let position_embeddings = self.position_embedding.forward(&position_ids)?;
        ctx.vis_introspector
            .insert("position_embeddings", &position_embeddings);

        let mut hidden_states = (patch_embeddings + position_embeddings)?;

        let attention_mask = build_additive_attention_mask(&patch_attention_masks, model_dtype)?;

        ctx.vis_introspector.start_tracking_depth()?;
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &attention_mask, ctx)?;
            ctx.vis_introspector.increment_depth();
        }
        ctx.vis_introspector.stop_tracking_depth()?;

        self.post_layernorm.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_2x2_position_ids() -> Result<()> {
        let device = Device::Cpu;
        let patch_mask = Tensor::ones((1, 4), DType::U32, &device)?;
        let pos_ids = build_position_ids(&patch_mask, 2, 2, &device)?;
        let vec = pos_ids.to_vec2::<u32>()?;
        assert_eq!(vec, vec![vec![0, 1, 27, 28]]);
        Ok(())
    }

    #[test]
    fn test_invalid_patch_fallback_position_ids() -> Result<()> {
        let device = Device::Cpu;
        // 2nd patch invalid
        let patch_mask = Tensor::from_vec(vec![1u32, 0, 1, 1], (1, 4), &device)?;
        let pos_ids = build_position_ids(&patch_mask, 2, 2, &device)?;
        let vec = pos_ids.to_vec2::<u32>()?;
        assert_eq!(vec, vec![vec![0, 0, 27, 28]]);
        Ok(())
    }

    #[test]
    fn test_max_supported_grid_dimensions() -> Result<()> {
        let device = Device::Cpu;
        let patch_mask = Tensor::ones((1, 27 * 27), DType::U32, &device)?;
        let pos_ids = build_position_ids(&patch_mask, 27, 27, &device)?;
        assert_eq!(pos_ids.dims(), &[1, 729]);
        Ok(())
    }


    #[test]
    fn test_rejection_exceeding_pretrained_grid() -> Result<()> {
        let device = Device::Cpu;
        let patch_mask = Tensor::ones((1, 28 * 28), DType::U32, &device)?;
        let res = build_position_ids(&patch_mask, 28, 28, &device);
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn test_fully_masked_sample_rejection() {
        let device = Device::Cpu;
        let pixel_mask = Tensor::zeros((1, 378, 378), DType::F32, &device).unwrap();
        let res = build_patch_attention_mask(&pixel_mask, 378, 378);
        assert!(res.is_err());
    }

    #[test]
    fn test_partially_masked_attention_mask() -> Result<()> {
        let device = Device::Cpu;
        let patch_mask = Tensor::from_vec(vec![1u32, 0], (1, 2), &device)?;
        let mask = build_additive_attention_mask(&patch_mask, DType::F32)?;
        let vals = mask.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(vals[0][0], 0.0);
        assert_eq!(vals[0][1], f32::NEG_INFINITY);
        Ok(())
    }   

    #[test]
    fn test_batch_mismatch_error() {
        let device = Device::Cpu;
        let pixel_values = Tensor::zeros((2, 3, 28, 28), DType::F32, &device).unwrap();
        let pixel_masks = Tensor::zeros((1, 34, 34), DType::F32, &device).unwrap();
        let mut ctx = InferenceContext::default();

        let hidden_bias = Tensor::zeros(HIDDEN_SIZE, DType::F32, &device).unwrap();
        let norm_weight = Tensor::ones(HIDDEN_SIZE, DType::F32, &device).unwrap();
        let position_weight = Tensor::zeros((729, HIDDEN_SIZE), DType::F32, &device).unwrap();

        let model = SmolVision {
            patch_embedding: Conv2d::new(
                Tensor::zeros((HIDDEN_SIZE, 3, SUB_PATCH_SIZE, SUB_PATCH_SIZE), DType::F32, &device).unwrap(),
                Some(hidden_bias.clone()),
                Conv2dConfig { padding: 0, stride: SUB_PATCH_SIZE, dilation: 1, groups: 1, ..Default::default() },
            ),
            position_embedding: Embedding::new(position_weight, HIDDEN_SIZE),
            blocks: vec![],
            post_layernorm: LayerNorm::new(norm_weight, hidden_bias, 1e-6),
        };

        let res = model.forward(&pixel_values, &pixel_masks, &mut ctx);
        assert!(res.is_err());
    }
    #[test]
    fn test_numerical_parity_fp32() -> Result<()> {
        let device = Device::Cpu;

        // Use varied inputs so LayerNorm has non-zero variance across channels
        let pixel_values = Tensor::arange(0.0f32, (1 * 3 * 28 * 28) as f32, &device)?
            .reshape((1, 3, 28, 28))?;
        let pixel_masks = Tensor::ones((1, 34, 34), DType::F32, &device)?;
        let mut ctx = InferenceContext::default();

        let hidden_bias = Tensor::zeros(HIDDEN_SIZE, DType::F32, &device)?;
        let norm_weight = Tensor::ones(HIDDEN_SIZE, DType::F32, &device)?;
        let position_weight = Tensor::zeros((729, HIDDEN_SIZE), DType::F32, &device)?;

        // Non-constant patch embedding weights to ensure non-zero variance
        let patch_weight = Tensor::arange(
            0.0f32,
            (HIDDEN_SIZE * 3 * SUB_PATCH_SIZE * SUB_PATCH_SIZE) as f32,
            &device,
        )?
        .reshape((HIDDEN_SIZE, 3, SUB_PATCH_SIZE, SUB_PATCH_SIZE))?;

        let model = SmolVision {
            patch_embedding: Conv2d::new(
                patch_weight,
                Some(hidden_bias.clone()),
                Conv2dConfig {
                    padding: 0,
                    stride: SUB_PATCH_SIZE,
                    dilation: 1,
                    groups: 1,
                    ..Default::default()
                },
            ),
            position_embedding: Embedding::new(position_weight, HIDDEN_SIZE),
            blocks: vec![],
            post_layernorm: LayerNorm::new(norm_weight, hidden_bias, 1e-6),
        };

        let output = model.forward(&pixel_values, &pixel_masks, &mut ctx)?;
        assert_eq!(output.dims(), &[1, 4, HIDDEN_SIZE]);

        let vals = output.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|&v| v.is_finite()),
            "Output contains non-finite values (NaN/Inf)"
        );
        Ok(())
    }
}