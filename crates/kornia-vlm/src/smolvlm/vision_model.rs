use candle_core::{bail, DType, Result, Tensor};
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
            let weight_dtype = self.q_proj.weight().dtype();

            // BF16: sufficient dynamic range, keep in BF16 directly.
            // FP16: pre-Ampere matmul accumulation is unsafe in FP16, upcast to F32.
            // F32: no-op either way.
            let compute_dtype = match weight_dtype {
                DType::BF16 => DType::BF16,
                _ => DType::F32,
            };

            let q = if q.dtype() != compute_dtype {
                q.to_dtype(compute_dtype)?
            } else {
                q
            };
            let k = if k.dtype() != compute_dtype {
                k.to_dtype(compute_dtype)?
            } else {
                k
            };
            let v = if v.dtype() != compute_dtype {
                v.to_dtype(compute_dtype)?
            } else {
                v
            };

            // The scores live in `compute_dtype`, so the additive mask must too
            // (candle's broadcast_add rejects mixed dtypes, e.g. F32 scores + F16 mask).
            let attention_mask = if attention_mask.dtype() != compute_dtype {
                attention_mask.to_dtype(compute_dtype)?
            } else {
                attention_mask.clone()
            };

            let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
            let att = att.broadcast_add(&attention_mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let y = att.matmul(&v)?.contiguous()?;

            // cast back to original input dtype if we upcasted
            if y.dtype() != in_dtype {
                y.to_dtype(in_dtype)?
            } else {
                y
            }
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

/// Turns a pixel-level mask (`B x H x W`, non-zero = valid pixel) into a patch-level
/// mask (`B x (H/14 * W/14)`, `U32`, 1 = patch contains at least one valid pixel).
///
/// The 14x14 windows start at pixel 0 and drop the trailing remainder. That is exactly
/// what the stride-14 patch-embedding conv and HF's `unfold(size=14, step=14)` do, so
/// the mask stays aligned with the patches that were actually embedded.
///
/// Returns `(patch_mask, grid_h, grid_w)`.
fn patch_mask_from_pixel_mask(pixel_masks: &Tensor) -> Result<(Tensor, usize, usize)> {
    let (batch, height, width) = pixel_masks.dims3()?;
    let p = SmolVision::SUB_PATCH_SIZE;
    let (grid_h, grid_w) = (height / p, width / p);
    if grid_h == 0 || grid_w == 0 {
        bail!("pixel attention mask is {height}x{width}, smaller than one {p}x{p} patch");
    }

    let cropped = pixel_masks
        .narrow(1, 0, grid_h * p)?
        .narrow(2, 0, grid_w * p)?;
    let patch_mask = cropped
        .reshape((batch, grid_h, p, grid_w, p))?
        .permute([0, 1, 3, 2, 4])?
        .sum_keepdim([3, 4])?
        .squeeze(4)?
        .squeeze(3)?
        .gt(0.0)?
        .reshape((batch, grid_h * grid_w))?
        .contiguous()?
        .to_dtype(DType::U32)?;

    Ok((patch_mask, grid_h, grid_w))
}

/// Position ids into the `side x side` learned position table, following HF's
/// `Idefics3VisionEmbeddings`: for each image the valid region (`nb_h x nb_w` patches,
/// assumed to be a top-left rectangle) is stretched over the whole table using
/// fractional coordinates, so padded or non-square images get the same embeddings as
/// the reference. Padded patches get id 0.
///
/// `bucket(i, nb) = floor(i * side / nb)` is the integer form of
/// `bucketize(i / nb, [1/side, 2/side, ..], right=True)`.
///
/// For an all-valid `side x side` grid this is just `0..side*side`.
fn position_ids_from_patch_mask(
    patch_mask: &Tensor,
    grid_h: usize,
    grid_w: usize,
    side: usize,
) -> Result<Tensor> {
    let device = patch_mask.device();
    let rows = patch_mask.to_vec2::<u32>()?;
    let n = grid_h * grid_w;
    let mut ids = vec![0u32; rows.len() * n];

    for (b, row) in rows.iter().enumerate() {
        let nb_h = (0..grid_h).filter(|&r| row[r * grid_w] != 0).count();
        let nb_w = (0..grid_w).filter(|&c| row[c] != 0).count();
        if nb_h == 0 || nb_w == 0 {
            continue; // fully padded sample: keep id 0 everywhere
        }

        let bucket = |i: usize, nb: usize| (i * side / nb) as u32;
        let valid_slots = (0..n).filter(|&k| row[k] != 0);
        let coords = (0..nb_h).flat_map(|bh| (0..nb_w).map(move |bw| (bh, bw)));
        for (slot, (bh, bw)) in valid_slots.zip(coords) {
            ids[b * n + slot] = bucket(bh, nb_h) * side as u32 + bucket(bw, nb_w);
        }
    }

    Tensor::from_vec(ids, (rows.len(), n), device)
}

/// Additive attention mask of shape `B x 1 x 1 x N` (0 for valid keys, a large finite
/// negative for padded keys), in `dtype`. Broadcasts over heads and queries instead of
/// materialising `B x 1 x N x N`.
///
/// The masked value is finite on purpose: with `-inf`, a fully padded sample makes
/// every softmax row all `-inf` and the output NaN. With a finite value those rows
/// degrade to a uniform softmax instead.
fn additive_attention_mask(patch_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let device = patch_mask.device();
    let valid = patch_mask.unsqueeze(1)?.unsqueeze(1)?;
    let masked_value = match dtype {
        DType::F16 => -1e4f32,  // finite, safe in f16 range (max ~65504)
        DType::BF16 => -1e9f32, // finite; note f32::MIN would round to -inf in bf16
        _ => f32::MIN,
    };
    let neg = Tensor::full(masked_value, valid.shape(), device)?.to_dtype(dtype)?;
    let zeros = Tensor::zeros(valid.shape(), dtype, device)?;
    valid.where_cond(&zeros, &neg)
}

pub struct SmolVision {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    num_patches_per_side: usize,
    blocks: Vec<Block>,
    post_layernorm: LayerNorm,
}

impl SmolVision {
    const SUB_PATCH_SIZE: usize = 14;

    pub fn load(c: &HashMap<String, Tensor>) -> Result<Self> {
        let pos_weight = c["model.vision_model.embeddings.position_embedding.weight"].clone();
        let (num_positions, pos_hidden) = pos_weight.dims2()?;
        let num_patches_per_side = (num_positions as f64).sqrt().round() as usize;
        if num_patches_per_side * num_patches_per_side != num_positions {
            bail!("position embedding has {num_positions} rows, which is not a square grid");
        }

        Ok(Self {
            patch_embedding: Conv2d::new(
                c["model.vision_model.embeddings.patch_embedding.weight"].clone(),
                Some(c["model.vision_model.embeddings.patch_embedding.bias"].clone()),
                Conv2dConfig {
                    // kernel/patch size are intrinsically defined in the weights
                    padding: 0, // "valid" padding
                    stride: Self::SUB_PATCH_SIZE,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            position_embedding: Embedding::new(pos_weight, pos_hidden),
            num_patches_per_side,
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

    fn run_blocks(
        &self,
        mut hidden_states: Tensor,
        attention_mask: &Tensor,
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, attention_mask, ctx)?;
            ctx.vis_introspector.increment_depth();
        }
        Ok(hidden_states)
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_masks: &Tensor,
        ctx: &mut InferenceContext,
    ) -> Result<Tensor> {
        let dtype = self.patch_embedding.weight().dtype();

        // B = patch rows x patch cols (x number of images)
        // pixel_values: B x 3 x PixelHeight x PixelWidth
        // pixel_attention_masks: B x PixelHeight x PixelWidth
        // patch_mask: B x (PixelHeight/14 * PixelWidth/14), grid size derived from input
        let (patch_mask, grid_h, grid_w) = patch_mask_from_pixel_mask(pixel_attention_masks)?;

        let hidden_states = {
            let patch_embeddings = self
                .patch_embedding
                .forward(&pixel_values.to_dtype(dtype)?)?;

            let (_, _, conv_h, conv_w) = patch_embeddings.dims4()?;
            if (conv_h, conv_w) != (grid_h, grid_w) {
                bail!(
                    "pixel_values give a {conv_h}x{conv_w} patch grid but the attention mask \
                     gives {grid_h}x{grid_w}; they must have the same pixel size"
                );
            }

            ctx.vis_introspector
                .insert("patch_embeddings", &patch_embeddings);

            let patch_embeddings = patch_embeddings.flatten_from(2)?.transpose(1, 2)?;

            let position_ids = position_ids_from_patch_mask(
                &patch_mask,
                grid_h,
                grid_w,
                self.num_patches_per_side,
            )?;
            let position_embeddings = self.position_embedding.forward(&position_ids)?;

            ctx.vis_introspector
                .insert("position_embeddings", &position_embeddings);

            patch_embeddings + position_embeddings
        }?;

        let attention_mask = additive_attention_mask(&patch_mask, dtype)?;

        ctx.vis_introspector.start_tracking_depth()?;
        let blocks_result = self.run_blocks(hidden_states, &attention_mask, ctx);
        // Always leave tracking mode, even if a block failed.
        let stop_result = ctx.vis_introspector.stop_tracking_depth();
        let hidden_states = blocks_result?;
        stop_result?;

        self.post_layernorm.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const HIDDEN: usize = NUM_OF_HEADS * HEAD_DIM; // 1152
    const TEST_INTERMEDIATE: usize = 32; // tiny MLP width, keeps tests fast

    fn dev() -> Device {
        Device::Cpu
    }

    fn test_ctx() -> InferenceContext {
        InferenceContext::default()
    }

    // ---------------------------------------------------------------- helpers

    /// Deterministic pseudo-random F32 tensor, uniform in [-scale, scale].
    /// (candle can't seed the CPU RNG, so we roll a tiny LCG for reproducibility.)
    fn det(shape: &[usize], seed: u64, scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        Tensor::from_vec(data, shape.to_vec(), &dev()).unwrap()
    }

    fn linear(out: usize, inp: usize, seed: u64, scale: f32, dtype: DType) -> Linear {
        Linear::new(
            det(&[out, inp], seed, scale).to_dtype(dtype).unwrap(),
            Some(det(&[out], seed + 1000, scale).to_dtype(dtype).unwrap()),
        )
    }

    fn zero_linear(out: usize, inp: usize, dtype: DType) -> Linear {
        Linear::new(
            Tensor::zeros((out, inp), dtype, &dev()).unwrap(),
            Some(Tensor::zeros(out, dtype, &dev()).unwrap()),
        )
    }

    fn make_attention(dtype: DType) -> Attention {
        Attention::new(
            linear(HIDDEN, HIDDEN, 1, 0.02, dtype),
            linear(HIDDEN, HIDDEN, 2, 0.02, dtype),
            linear(HIDDEN, HIDDEN, 3, 0.02, dtype),
            linear(HIDDEN, HIDDEN, 4, 0.02, dtype),
        )
        .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn all_finite(t: &Tensor) -> bool {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    }

    fn has_nan(t: &Tensor) -> bool {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .any(|v| v.is_nan())
    }

    /// Additive mask of shape (b, 1, n, n): `neg` in the columns listed in
    /// `masked_keys`, 0 elsewhere.
    fn key_mask(b: usize, n: usize, masked_keys: &[usize], neg: f32, dtype: DType) -> Tensor {
        let mut data = vec![0f32; b * n * n];
        for i in 0..b * n {
            for &j in masked_keys {
                data[i * n + j] = neg;
            }
        }
        Tensor::from_vec(data, (b, 1, n, n), &dev())
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn gelu_tanh_ref(x: f64) -> f64 {
        let c = (2.0f64 / std::f64::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
    }

    // -------------------------------------------------------------- Attention

    #[test]
    fn smol_vision_attention_preserves_shape_and_dtype() {
        let attn = make_attention(DType::F32);
        let x = det(&[2, 5, HIDDEN], 10, 1.0);
        let mask = key_mask(2, 5, &[], 0.0, DType::F32);

        let y = attn.forward(&x, &mask).unwrap();

        assert_eq!(y.dims(), &[2, 5, HIDDEN]);
        assert_eq!(y.dtype(), DType::F32);
        assert!(all_finite(&y));
    }

    #[test]
    fn smol_vision_attention_masked_keys_do_not_leak() {
        let attn = make_attention(DType::F32);
        let x1 = det(&[1, 4, HIDDEN], 11, 1.0);
        // Replace token 3 with different content.
        let x2 = Tensor::cat(
            &[&x1.narrow(1, 0, 3).unwrap(), &det(&[1, 1, HIDDEN], 99, 1.0)],
            1,
        )
        .unwrap();
        // Token 3 is masked out as a key for every query.
        let mask = key_mask(1, 4, &[3], f32::MIN, DType::F32);

        let y1 = attn.forward(&x1, &mask).unwrap();
        let y2 = attn.forward(&x2, &mask).unwrap();

        // Queries 0..3 must be unaffected by the masked token...
        let d_visible = max_abs_diff(&y1.narrow(1, 0, 3).unwrap(), &y2.narrow(1, 0, 3).unwrap());
        assert!(d_visible < 1e-5, "masked key leaked: diff = {d_visible}");
        // ...while the query at position 3 itself must change (sanity check).
        let d_self = max_abs_diff(&y1.narrow(1, 3, 1).unwrap(), &y2.narrow(1, 3, 1).unwrap());
        assert!(d_self > 1e-3, "test is vacuous: diff = {d_self}");
    }

    #[test]
    fn smol_vision_attention_is_permutation_equivariant() {
        // No positional info inside Attention, so permuting tokens permutes outputs.
        let attn = make_attention(DType::F32);
        let x = det(&[1, 4, HIDDEN], 12, 1.0);
        let mask = key_mask(1, 4, &[], 0.0, DType::F32);
        let idx = Tensor::new(&[2u32, 0, 3, 1], &dev()).unwrap();

        let y = attn.forward(&x, &mask).unwrap();
        let y_perm = attn
            .forward(&x.index_select(&idx, 1).unwrap(), &mask)
            .unwrap();

        let d = max_abs_diff(&y.index_select(&idx, 1).unwrap(), &y_perm);
        assert!(d < 1e-4, "diff = {d}");
    }

    #[test]
    fn smol_vision_attention_finite_negative_mask_survives_full_masking() {
        // Every key masked with a finite value: softmax degenerates to uniform, no NaN.
        let attn = make_attention(DType::F32);
        let x = det(&[1, 4, HIDDEN], 13, 1.0);
        let mask = key_mask(1, 4, &[0, 1, 2, 3], f32::MIN, DType::F32);

        let y = attn.forward(&x, &mask).unwrap();
        assert!(all_finite(&y));
    }

    #[test]
    fn smol_vision_attention_neg_inf_full_mask_yields_nan() {
        // Attention does not guard against all -inf rows; this is why additive_attention_mask uses a finite value.
        let attn = make_attention(DType::F32);
        let x = det(&[1, 4, HIDDEN], 14, 1.0);
        let mask = key_mask(1, 4, &[0, 1, 2, 3], f32::NEG_INFINITY, DType::F32);

        let y = attn.forward(&x, &mask).unwrap();
        assert!(has_nan(&y));
    }

    #[test]
    fn smol_vision_attention_f16_with_f16_mask() {
        let attn = make_attention(DType::F16);
        let x = det(&[1, 4, HIDDEN], 15, 1.0).to_dtype(DType::F16).unwrap();
        let mask = key_mask(1, 4, &[], 0.0, DType::F16);

        let y = attn.forward(&x, &mask).expect("F16 forward should work");
        assert_eq!(y.dtype(), DType::F16);
        assert!(all_finite(&y));
    }

    // -------------------------------------------------------------------- Mlp

    fn make_mlp(dtype: DType) -> Mlp {
        Mlp::new(
            linear(TEST_INTERMEDIATE, HIDDEN, 20, 0.02, dtype),
            linear(HIDDEN, TEST_INTERMEDIATE, 21, 0.02, dtype),
        )
        .unwrap()
    }

    #[test]
    fn smol_vision_gelu_tanh_matches_f64_reference() {
        let mlp = make_mlp(DType::F32);
        let xs: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.1).collect(); // -6.0 ..= 6.0
        let input = Tensor::from_vec(xs.clone(), xs.len(), &dev()).unwrap();

        let out = mlp.gelu_tanh(&input).unwrap().to_vec1::<f32>().unwrap();

        for (x, y) in xs.iter().zip(out.iter()) {
            let expected = gelu_tanh_ref(*x as f64);
            assert!(
                (*y as f64 - expected).abs() < 1e-4,
                "x = {x}: got {y}, expected {expected}"
            );
        }
    }

    #[test]
    fn smol_vision_gelu_tanh_matches_candle_gelu() {
        // candle's Tensor::gelu is the tanh approximation; the hand-rolled one should agree.
        let mlp = make_mlp(DType::F32);
        let input = det(&[3, 7, 16], 30, 4.0);

        let ours = mlp.gelu_tanh(&input).unwrap();
        let theirs = input.gelu().unwrap();

        assert!(max_abs_diff(&ours, &theirs) < 1e-5);
    }

    #[test]
    fn smol_vision_gelu_tanh_f16_saturates_without_nan() {
        // x^3 overflows F16 for |x| > ~40; make sure the result is still finite and
        // saturates to x (positive side) / 0 (negative side).
        let mlp = make_mlp(DType::F16);
        let input = Tensor::new(&[-50f32, -45.0, -5.0, 0.0, 5.0, 45.0, 50.0], &dev())
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let out = mlp
            .gelu_tanh(&input)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        assert!(out[0].abs() < 1e-2 && out[1].abs() < 1e-2);
        assert!((out[5] - 45.0).abs() < 0.1 && (out[6] - 50.0).abs() < 0.1);
    }

    #[test]
    fn smol_vision_mlp_forward_shape() {
        let mlp = make_mlp(DType::F32);
        let x = det(&[1, 3, HIDDEN], 31, 1.0);
        let mut ctx = test_ctx();

        let y = mlp.forward(&x, &mut ctx).unwrap();

        assert_eq!(y.dims(), &[1, 3, HIDDEN]);
        assert!(all_finite(&y));
    }

    // ------------------------------------------------------------------ Block

    fn layer_key(id: u8, k: &str, suffix: &str) -> String {
        format!("model.vision_model.encoder.layers.{id}.{k}.{suffix}")
    }

    /// Inserts all weights for encoder layer `id`. With `zero_out`, the attention
    /// output projection and fc2 are zeroed, which makes the block an identity map.
    fn insert_layer(c: &mut HashMap<String, Tensor>, id: u8, zero_out: bool) {
        let f = DType::F32;
        let mut put = |k: &str, l: Linear| {
            c.insert(layer_key(id, k, "weight"), l.weight().clone());
            c.insert(layer_key(id, k, "bias"), l.bias().unwrap().clone());
        };
        put("self_attn.q_proj", linear(HIDDEN, HIDDEN, 41, 0.02, f));
        put("self_attn.k_proj", linear(HIDDEN, HIDDEN, 42, 0.02, f));
        put("self_attn.v_proj", linear(HIDDEN, HIDDEN, 43, 0.02, f));
        put(
            "self_attn.out_proj",
            if zero_out {
                zero_linear(HIDDEN, HIDDEN, f)
            } else {
                linear(HIDDEN, HIDDEN, 44, 0.02, f)
            },
        );
        put("mlp.fc1", linear(TEST_INTERMEDIATE, HIDDEN, 45, 0.02, f));
        put(
            "mlp.fc2",
            if zero_out {
                zero_linear(HIDDEN, TEST_INTERMEDIATE, f)
            } else {
                linear(HIDDEN, TEST_INTERMEDIATE, 46, 0.02, f)
            },
        );
        for ln in ["layer_norm1", "layer_norm2"] {
            c.insert(
                layer_key(id, ln, "weight"),
                Tensor::ones(HIDDEN, f, &dev()).unwrap(),
            );
            c.insert(
                layer_key(id, ln, "bias"),
                Tensor::zeros(HIDDEN, f, &dev()).unwrap(),
            );
        }
    }

    #[test]
    fn smol_vision_block_with_zeroed_output_projections_is_identity() {
        let mut c = HashMap::new();
        insert_layer(&mut c, 0, true);
        let block = Block::new(&c, 0).unwrap();
        let x = det(&[1, 4, HIDDEN], 50, 1.0);
        let mask = key_mask(1, 4, &[], 0.0, DType::F32);
        let mut ctx = test_ctx();

        let y = block.forward(&x, &mask, &mut ctx).unwrap();

        // Both residual branches contribute exactly 0 + bias(0) => y == x.
        assert!(max_abs_diff(&y, &x) < 1e-6);
    }

    #[test]
    fn smol_vision_block_forward_shape_and_finite() {
        let mut c = HashMap::new();
        insert_layer(&mut c, 3, false);
        let block = Block::new(&c, 3).unwrap();
        let x = det(&[2, 6, HIDDEN], 51, 1.0);
        let mask = key_mask(2, 6, &[5], f32::MIN, DType::F32);
        let mut ctx = test_ctx();

        let y = block.forward(&x, &mask, &mut ctx).unwrap();

        assert_eq!(y.dims(), x.dims());
        assert!(all_finite(&y));
    }

    #[test]
    #[should_panic]
    fn smol_vision_block_new_panics_on_missing_weight() {
        // Characterization test: Block::new indexes the map with `c[&key]`, so a missing
        // tensor panics instead of returning Err. If you fix that, flip this to
        // `assert!(Block::new(..).is_err())`.
        let c: HashMap<String, Tensor> = HashMap::new();
        let _ = Block::new(&c, 0);
    }

    // ------------------------------------------------------------- SmolVision
    //
    // These run all 27 encoder layers at hidden = 1152 on CPU, so they are slow.
    // Layer weights are shared via Tensor::clone (Arc), so memory stays small.

    fn full_weights(dtype: DType) -> HashMap<String, Tensor> {
        let mut layer0 = HashMap::new();
        insert_layer(&mut layer0, 0, false);

        let mut c = HashMap::new();
        for id in 0u8..=26 {
            for (k, v) in &layer0 {
                c.insert(k.replace("layers.0.", &format!("layers.{id}.")), v.clone());
            }
        }
        c.insert(
            "model.vision_model.embeddings.patch_embedding.weight".into(),
            det(&[HIDDEN, 3, 14, 14], 60, 0.02),
        );
        c.insert(
            "model.vision_model.embeddings.patch_embedding.bias".into(),
            Tensor::zeros(HIDDEN, DType::F32, &dev()).unwrap(),
        );
        c.insert(
            "model.vision_model.embeddings.position_embedding.weight".into(),
            det(&[27 * 27, HIDDEN], 61, 0.02),
        );
        c.insert(
            "model.vision_model.post_layernorm.weight".into(),
            Tensor::ones(HIDDEN, DType::F32, &dev()).unwrap(),
        );
        c.insert(
            "model.vision_model.post_layernorm.bias".into(),
            Tensor::zeros(HIDDEN, DType::F32, &dev()).unwrap(),
        );

        if dtype != DType::F32 {
            for v in c.values_mut() {
                *v = v.to_dtype(dtype).unwrap();
            }
        }
        c
    }

    #[test]
    #[ignore = "slow: runs the full 27-layer encoder on CPU"]
    fn smol_vision_forward_384_all_valid() {
        let vision = SmolVision::load(&full_weights(DType::F32)).unwrap();
        let pv = det(&[1, 3, 384, 384], 70, 1.0);
        let mask = Tensor::ones((1, 384, 384), DType::F32, &dev()).unwrap();
        let mut ctx = test_ctx();

        let y = vision.forward(&pv, &mask, &mut ctx).unwrap();

        assert_eq!(y.dims(), &[1, 27 * 27, HIDDEN]);
        assert!(all_finite(&y));
    }

    #[test]
    #[ignore = "slow: full 27-layer encoder"]
    fn smol_vision_forward_non_384_input() {
        let vision = SmolVision::load(&full_weights(DType::F32)).unwrap();
        let pv = det(&[1, 3, 378, 378], 71, 1.0);
        let mask = Tensor::ones((1, 378, 378), DType::F32, &dev()).unwrap();
        let mut ctx = test_ctx();

        let y = vision
            .forward(&pv, &mask, &mut ctx)
            .expect("378x378 should work");
        assert_eq!(y.dims(), &[1, 27 * 27, HIDDEN]);
    }

    #[test]
    #[ignore = "slow: full 27-layer encoder"]
    fn smol_vision_forward_fully_padded_sample_has_no_nan() {
        let vision = SmolVision::load(&full_weights(DType::F32)).unwrap();
        let pv = det(&[1, 3, 384, 384], 72, 1.0);
        let mask = Tensor::zeros((1, 384, 384), DType::F32, &dev()).unwrap();
        let mut ctx = test_ctx();

        let y = vision.forward(&pv, &mask, &mut ctx).unwrap();
        assert!(!has_nan(&y));
    }

    #[test]
    #[ignore = "slow: full 27-layer encoder"]
    fn smol_vision_forward_f16() {
        let vision = SmolVision::load(&full_weights(DType::F16)).unwrap();
        let pv = det(&[1, 3, 384, 384], 73, 1.0)
            .to_dtype(DType::F16)
            .unwrap();
        let mask = Tensor::ones((1, 384, 384), DType::F32, &dev()).unwrap();
        let mut ctx = test_ctx();

        let y = vision
            .forward(&pv, &mask, &mut ctx)
            .expect("F16 forward should work");
        assert!(all_finite(&y));
    }

    fn pixel_mask(h: usize, w: usize, valid: impl Fn(usize, usize) -> bool) -> Tensor {
        let data: Vec<f32> = (0..h * w)
            .map(|i| if valid(i / w, i % w) { 1.0 } else { 0.0 })
            .collect();
        Tensor::from_vec(data, (1, h, w), &dev()).unwrap()
    }

    fn u32_mask(ones_at: &[usize], n: usize) -> Tensor {
        let mut v = vec![0u32; n];
        for &i in ones_at {
            v[i] = 1;
        }
        Tensor::from_vec(v, (1, n), &dev()).unwrap()
    }

    #[test]
    fn smol_vision_patch_mask_grid_follows_input_size() {
        for (h, w, eh, ew) in [
            (384, 384, 27, 27),
            (378, 378, 27, 27),
            (448, 448, 32, 32),
            (392, 420, 28, 30),
        ] {
            let (m, gh, gw) = patch_mask_from_pixel_mask(&pixel_mask(h, w, |_, _| true)).unwrap();
            assert_eq!((gh, gw), (eh, ew), "{h}x{w}");
            assert_eq!(m.dims(), &[1, eh * ew]);
            assert!(m.to_vec2::<u32>().unwrap()[0].iter().all(|&v| v == 1));
        }
    }

    #[test]
    fn smol_vision_patch_mask_is_anchored_at_pixel_zero() {
        // Only the first 3 pixel rows are valid. The old centre crop (offset 3) dropped
        // them entirely; windows must start at pixel 0 like the conv does.
        let (m, _, gw) = patch_mask_from_pixel_mask(&pixel_mask(384, 384, |r, _| r < 3)).unwrap();
        let m = &m.to_vec2::<u32>().unwrap()[0];
        assert!(
            m[..gw].iter().all(|&v| v == 1),
            "first patch row must be valid"
        );
        assert!(
            m[gw..].iter().all(|&v| v == 0),
            "all other rows must be padding"
        );
    }

    #[test]
    fn smol_vision_patch_mask_rectangular_region() {
        // 28 valid pixel rows x 42 valid pixel cols -> 2 x 3 valid patches
        let (m, gh, gw) =
            patch_mask_from_pixel_mask(&pixel_mask(384, 384, |r, c| r < 28 && c < 42)).unwrap();
        let m = &m.to_vec2::<u32>().unwrap()[0];
        for r in 0..gh {
            for c in 0..gw {
                assert_eq!(m[r * gw + c] == 1, r < 2 && c < 3, "({r},{c})");
            }
        }
    }

    #[test]
    fn smol_vision_position_ids_full_grid_is_arange() {
        let ids =
            position_ids_from_patch_mask(&u32_mask(&(0..729).collect::<Vec<_>>(), 729), 27, 27, 27)
                .unwrap();
        let expected: Vec<u32> = (0..729).collect();
        assert_eq!(ids.to_vec2::<u32>().unwrap()[0], expected);
    }

    #[test]
    fn smol_vision_position_ids_padded_region_is_stretched_over_table() {
        // 4x4 grid, table side 4, valid region = top-left 2x2 (slots 0,1,4,5).
        // nb = 2 -> buckets (0, 2) -> ids 0, 2, 8, 10; padding stays 0.
        let ids = position_ids_from_patch_mask(&u32_mask(&[0, 1, 4, 5], 16), 4, 4, 4).unwrap();
        let expected = vec![0, 2, 0, 0, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(ids.to_vec2::<u32>().unwrap()[0], expected);
    }

    #[test]
    fn smol_vision_position_ids_fully_padded_is_all_zero() {
        let ids = position_ids_from_patch_mask(&u32_mask(&[], 16), 4, 4, 4).unwrap();
        assert!(ids.to_vec2::<u32>().unwrap()[0].iter().all(|&v| v == 0));
    }

    #[test]
    fn smol_vision_position_ids_larger_grid_stays_inside_table() {
        let all: Vec<usize> = (0..32 * 32).collect();
        let ids = position_ids_from_patch_mask(&u32_mask(&all, 32 * 32), 32, 32, 27).unwrap();
        assert!(ids.to_vec2::<u32>().unwrap()[0]
            .iter()
            .all(|&v| (v as usize) < 27 * 27));
    }

    #[test]
    fn smol_vision_additive_mask_shape_and_values_per_dtype() {
        let patch_mask = u32_mask(&[0, 1], 4); // slots 0,1 valid; 2,3 padding
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let m = additive_attention_mask(&patch_mask, dtype).unwrap();
            assert_eq!(m.dims(), &[1, 1, 1, 4]);
            assert_eq!(m.dtype(), dtype);
            let v = m
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(v.iter().all(|x| x.is_finite()), "{dtype:?}: {v:?}");
            assert_eq!(&v[..2], &[0.0, 0.0]);
            assert!(v[2] < -1e3 && v[3] < -1e3, "{dtype:?}: {v:?}");
        }
    }
}
