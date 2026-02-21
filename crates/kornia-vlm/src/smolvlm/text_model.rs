use std::collections::HashMap;

use candle_core::DType;
use candle_core::{Device, Result, Tensor};
use candle_nn::{rotary_emb::rope, Linear, Module};

use crate::context::InferenceContext;
use crate::smolvlm::custom_rmsnorm::CustomRmsNorm;

const NUM_OF_HEADS: usize = 32;
const HEAD_DIM: usize = 64;

/// Custom SiLU (Sigmoid Linear Unit) activation function
/// SiLU(x) = x * sigmoid(x) = x * (1 / (1 + e^(-x)))
fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = (x.neg()?.exp()? + 1.0)?.recip()?;
    x * sigmoid
}

fn calculate_default_inv_freq() -> Vec<f32> {
    (0..HEAD_DIM)
        .step_by(2)
        //            1 / rope theta
        .map(|i| 1f32 / (273768f32).powf(i as f32 / HEAD_DIM as f32))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    // caching
    cos: Tensor,
    sin: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
}

impl Attention {
    fn new(q: Tensor, k: Tensor, v: Tensor, o: Tensor) -> Result<Self> {
        let device = q.device();
        let dtype = q.dtype();

        let theta = Tensor::new(calculate_default_inv_freq(), device)?;
        // 0 -> max position embedding
        let idx_theta = Tensor::arange(0, 16384u32, device)?
            .to_dtype(DType::F32)?
            .reshape((16384, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        Ok(Self {
            cos: idx_theta.cos()?.to_dtype(q.dtype())?,
            sin: idx_theta.sin()?.to_dtype(q.dtype())?,
            k_cache: Tensor::zeros((NUM_OF_HEADS, 0, HEAD_DIM), dtype, device)?,
            v_cache: Tensor::zeros((NUM_OF_HEADS, 0, HEAD_DIM), dtype, device)?,
            q_proj: Linear::new(q, None),
            k_proj: Linear::new(k, None),
            v_proj: Linear::new(v, None),
            o_proj: Linear::new(o, None),
        })
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        self.k_cache = Tensor::zeros(
            (NUM_OF_HEADS, 0, HEAD_DIM),
            self.k_cache.dtype(),
            self.k_cache.device(),
        )?;
        self.v_cache = Tensor::zeros(
            (NUM_OF_HEADS, 0, HEAD_DIM),
            self.v_cache.dtype(),
            self.v_cache.device(),
        )?;
        Ok(())
    }

    fn apply_rotary_embedding(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_head_sz, seq_len, _hidden_size) = x.dims3()?;

        rope(
            &x.unsqueeze(0)?,
            &self.cos.narrow(0, index_pos, seq_len)?,
            &self.sin.narrow(0, index_pos, seq_len)?,
        )?
        .squeeze(0)
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let device = x.device();

        let (seq_len, hidden_size) = x.dims2()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((seq_len, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((seq_len, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((seq_len, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(0, 1)?;

        let q = self.apply_rotary_embedding(&q, index_pos)?;
        let k = self.apply_rotary_embedding(&k, index_pos)?;

        // use cache (always assumes new tokens are an extension of the previous sequence)
        // TODO: handle context length
        self.k_cache = Tensor::cat(&[&self.k_cache, &k], 1)?;
        self.v_cache = Tensor::cat(&[&self.v_cache, &v], 1)?;

        let y = {
            // TODO: implement flash attention
            // TODO: just using BF16 is plausible

            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = self.k_cache.to_dtype(DType::F32)?;
            let v = self.v_cache.to_dtype(DType::F32)?;

            let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = Self::generate_causal_mask(seq_len, self.k_cache.dims()[1], device)?;
                att.broadcast_add(&mask)?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;

            att.matmul(&v)?.contiguous()?.to_dtype(in_dtype)?
        };

        let y = y.transpose(0, 1)?.reshape(&[seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }

    fn generate_causal_mask(seq_len: usize, total_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = ((total_len - seq_len)..total_len)
            .flat_map(|i| (0..total_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(mask, (seq_len, total_len), device)
    }
}

#[derive(Debug, Clone)]
pub struct MLPGates {
    down_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
}

impl MLPGates {
    fn new(d: Tensor, g: Tensor, u: Tensor) -> Self {
        Self {
            down_proj: Linear::new(d, None),
            gate_proj: Linear::new(g, None),
            up_proj: Linear::new(u, None),
        }
    }

    fn forward(&mut self, x: &Tensor, ctx: &mut InferenceContext) -> Result<Tensor> {
        let gate_proj_out = self.gate_proj.forward(x)?;
        ctx.text_introspector
            .insert("mlp_gate_proj", &gate_proj_out);

        let gate = silu(&gate_proj_out)?;
        ctx.text_introspector.insert("mlp_act_fn", &gate);

        let up = self.up_proj.forward(x)?;
        ctx.text_introspector.insert("mlp_up_proj", &up);

        let hidden = (gate * up)?;

        let x = self.down_proj.forward(&hidden)?;
        ctx.text_introspector.insert("mlp_down_proj", &x);

        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    input_layer_norm: CustomRmsNorm,
    pub attn: Attention,
    post_layer_norm: CustomRmsNorm,
    pub gates: MLPGates,
}

impl Block {
    /*
    model.text_model.layers.2.input_layernorm.weight
    model.text_model.layers.2.self_attn.q_proj.weight
    model.text_model.layers.2.self_attn.k_proj.weight
    model.text_model.layers.2.self_attn.v_proj.weight
    model.text_model.layers.2.self_attn.o_proj.weight
    model.text_model.layers.2.post_attention_layernorm.weight
    model.text_model.layers.2.mlp.up_proj.weight
    model.text_model.layers.2.mlp.gate_proj.weight
    model.text_model.layers.2.mlp.down_proj.weight
     */
    fn load(c: &HashMap<String, Tensor>, id: u8) -> Result<Self> {
        let val = |k| {
            c[&("model.text_model.layers.".to_owned() + &id.to_string() + "." + k + ".weight")]
                .clone()
        };

        Ok(Self {
            input_layer_norm: CustomRmsNorm::new(val("input_layernorm"), 1e-5),
            attn: Attention::new(
                val("self_attn.q_proj"),
                val("self_attn.k_proj"),
                val("self_attn.v_proj"),
                val("self_attn.o_proj"),
            )?,
            post_layer_norm: CustomRmsNorm::new(val("post_attention_layernorm"), 1e-5),
            gates: MLPGates::new(
                val("mlp.down_proj"),
                val("mlp.gate_proj"),
                val("mlp.up_proj"),
            ),
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        let residual = x;

        let x = self.input_layer_norm.forward(x)?;
        ctx.text_introspector.insert("input_layernorm", &x);

        let att = self.attn.forward(&x, index_pos)?;
        ctx.text_introspector.insert("self_attn", &att);

        let x = (residual + att)?;
        let residual = &x;

        let x = self.post_layer_norm.forward(&x)?;
        ctx.text_introspector.insert("post_layernorm", &x);

        let x = self.gates.forward(&x, ctx)?;
        ctx.text_introspector.insert("mlp", &x);

        let x = (residual + x)?;
        ctx.text_introspector.insert("layers", &x);

        Ok(x)
    }
}

pub struct SmolText {
    pub blocks: Vec<Block>,
    norm: CustomRmsNorm,
    lm_head: Linear,
}

impl SmolText {
    pub fn load(c: &HashMap<String, Tensor>) -> Result<Self> {
        Ok(Self {
            blocks: (0u8..=23)
                .map(|i| Block::load(c, i))
                .collect::<Result<Vec<_>>>()?,
            norm: CustomRmsNorm::new(c["model.text_model.norm.weight"].clone(), 1e-5),
            lm_head: Linear::new(c["lm_head.weight"].clone(), None),
        })
    }

    pub fn forward(
        &mut self,
        mut x: Tensor,
        index_pos: usize,
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        let _ = ctx.text_introspector.start_tracking_depth();
        for block in &mut self.blocks {
            x = block.forward(&x, index_pos, ctx)?;
            ctx.text_introspector.increment_depth();
        }
        let _ = ctx.text_introspector.stop_tracking_depth();

        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }
}
