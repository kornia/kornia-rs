#![allow(unused_variables)]
#![allow(unused_attributes)]

use candle_core::{DType, Result, Tensor};
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

    pub fn forward(
        &self,
        xs: &Tensor,
        introspector: &mut super::introspector::ActivationIntrospector,
    ) -> Result<Tensor> {
        let i = &self.fc1.forward(xs)?;
        #[cfg(feature = "debug")]
        introspector.insert("fc1", i);

        let x = self.gelu_tanh(i)?; // python impl. uses gelu approximated with tanh
        #[cfg(feature = "debug")]
        introspector.insert("mlp_act_fn", &x);

        let o = self.fc2.forward(&x)?;
        #[cfg(feature = "debug")]
        introspector.insert("fc2", &o);

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
        introspector: &mut super::introspector::ActivationIntrospector,
    ) -> Result<Tensor> {
        let residual = xs;

        let x = self.layer_norm1.forward(xs)?;
        #[cfg(feature = "debug")]
        introspector.insert("input_layernorm", &x);

        let x = self.self_attn.forward(&x, attention_mask)?;
        #[cfg(feature = "debug")]
        introspector.insert("self_attn", &x);

        let x = (residual + x)?;
        let residual = &x;

        let x = self.layer_norm2.forward(&x)?;
        #[cfg(feature = "debug")]
        introspector.insert("post_layernorm", &x);

        let x = self.mlp.forward(&x, introspector)?;

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
        introspector: &mut super::introspector::ActivationIntrospector,
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
            // println!("Pixel values shape: {:?}", pixel_values.shape());
            let patch_embeddings = self
                .patch_embedding
                .forward(&pixel_values.to_dtype(dtype)?)?;

            #[cfg(feature = "debug")]
            introspector.insert("patch_embeddings", &patch_embeddings);

            let patch_embeddings = patch_embeddings.flatten_from(2)?.transpose(1, 2)?;

            let position_ids = {
                let raw_ids = Tensor::arange(0u32, 27 * 27, device)?.expand(&[batch, 27 * 27])?;
                (raw_ids * &patch_attention_masks)?
            };
            let position_embeddings = self.position_embedding.forward(&position_ids)?;

            #[cfg(feature = "debug")]
            introspector.insert("position_embeddings", &position_embeddings);

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

        introspector.start_tracking_depth();
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &patch_attention_masks, introspector)?;
            introspector.increment_depth();
        }
        introspector.stop_tracking_depth();

        self.post_layernorm.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::safetensors::save;
    use kornia_tensor::CpuAllocator;
    use std::collections::HashMap;

    use candle_nn::Module;

    #[ignore]
    #[test]
    fn test_validating_vision_enc() -> Result<(), Box<dyn std::error::Error>> {
        use candle_core::{DType, Device, Result, Tensor};

        let dtype = DType::F32;
        let device = &Device::Cpu;

        let zor = |l: &dyn Fn(&Tensor) -> Result<Tensor>,
                   s: &[usize]|
         -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
            let zeros = Tensor::zeros(s, dtype, device)?;
            let ones = Tensor::ones(s, dtype, device)?;
            let randn = Tensor::rand(0.0, 1.0, s, device)?.to_dtype(dtype)?;
            let z = l(&zeros)?;
            let o = l(&ones)?;
            let r = l(&randn)?;
            Ok((z, o, r, zeros, ones, randn))
        };

        let smol_vlm = super::super::SmolVlm::<CpuAllocator>::load_model(dtype, device)?.0;

        let mut layers = vec![
            zor(
                &|t: &Tensor| smol_vlm.vision.patch_embedding.forward(t),
                &[1, 3, 384, 384],
            )?,
            // zor(
            //     &|t: &Tensor| smol_vlm.vision.position_embedding.forward(t),
            //     &[1, 3, 384, 384],
            // )?,
            zor(
                &|t: &Tensor| smol_vlm.vision.post_layernorm.forward(t),
                &[729, 1152],
            )?,
        ];
        for i in 0..27 {
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].layer_norm1.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].self_attn.k_proj.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].self_attn.q_proj.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].self_attn.v_proj.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].self_attn.o_proj.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].layer_norm2.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].mlp.fc1.forward(t),
                &[729, 1152],
            )?);
            layers.push(zor(
                &|t: &Tensor| smol_vlm.vision.blocks[i].mlp.fc2.forward(t),
                &[729, 4304],
            )?);
        }

        // Save all activations and inputs to a safetensors file using candle_core

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (ind, layer) in layers.into_iter().enumerate() {
            tensors.insert(format!("{:?}_z", ind), layer.0);
            tensors.insert(format!("{:?}_o", ind), layer.1);
            tensors.insert(format!("{:?}_r", ind), layer.2);
            tensors.insert(format!("{:?}_zx", ind), layer.3);
            tensors.insert(format!("{:?}_ox", ind), layer.4);
            tensors.insert(format!("{:?}_rx", ind), layer.5);
        }

        save(
            &tensors,
            "../../examples/smol_vlm/validation_data/image_validations/vision_activations.safetensors",
        )
        .unwrap();
        println!("Activations saved to examples/smol_vlm/validation_data/image_validations/vision_activations.safetensors");

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_lanczos_resize_filtering() -> candle_core::Result<()> {
        use candle_core::Device;
        use kornia_imgproc::interpolation::InterpolationMode;
        use kornia_imgproc::resize::resize_fast_rgb;
        use kornia_io::png::read_image_png_mono8;

        let device = Device::Cpu;
        let test_image_path = "../../examples/smol_vlm/validation_data/Zoneplate.png";

        println!("Loading test image: {:?}", test_image_path);

        // Load as grayscale and convert to RGB tensor
        let gray_img = read_image_png_mono8(test_image_path)
            .map_err(|e| candle_core::Error::Msg(format!("PNG read error: {e}")))?;
        println!(
            "Successfully loaded as grayscale: {}x{}",
            gray_img.size().width,
            gray_img.size().height
        );

        // Convert grayscale image to RGB by duplicating channels
        let size = gray_img.size();
        let gray_data = gray_img.as_slice();
        let mut rgb_data = Vec::with_capacity(gray_data.len() * 3);
        for &g in gray_data {
            rgb_data.push(g); // R
            rgb_data.push(g); // G
            rgb_data.push(g); // B
        }
        let img_rgb = kornia_image::Image::<u8, 3, kornia_image::allocator::CpuAllocator>::new(
            size,
            rgb_data,
            kornia_image::allocator::CpuAllocator,
        )
        .map_err(|e| candle_core::Error::Msg(format!("RGB image creation error: {e}")))?;

        // Use u8 image for resize_fast
        let img_rgb_u8 = img_rgb;

        // Test various sizes with Lanczos filtering
        let test_sizes = [(256, 256), (512, 512), (128, 128)];

        use kornia_io::png::write_image_png_rgb8;
        for (target_w, target_h) in test_sizes {
            println!(
                "Resizing to {}x{} using Lanczos filtering...",
                target_w, target_h
            );
            let mut resized =
                kornia_image::Image::<u8, 3, kornia_image::allocator::CpuAllocator>::from_size_val(
                    kornia_image::ImageSize {
                        width: target_w,
                        height: target_h,
                    },
                    0,
                    kornia_image::allocator::CpuAllocator,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("resize image creation error: {e}"))
                })?;
            resize_fast_rgb(&img_rgb_u8, &mut resized, InterpolationMode::Lanczos)
                .map_err(|e| candle_core::Error::Msg(format!("resize_fast error: {e}")))?;
            println!("Successfully resized to shape: {}x{}", target_w, target_h);

            // Save the resized image
            let output_path = format!(
                "../../examples/smol_vlm/validation_data/image_validations/Zoneplate_{}x{}_lanczos.png",
                target_w, target_h
            );
            write_image_png_rgb8(&output_path, &resized)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to save image: {e}")))?;
            println!("Saved resized image: {}", output_path);
        }

        println!("\nTesting different interpolation modes on 256x256:");
        let target_size = (256, 256);
        let interpolation_modes = [
            ("Bilinear", InterpolationMode::Bilinear),
            ("Lanczos", InterpolationMode::Lanczos),
            ("Nearest", InterpolationMode::Nearest),
        ];

        for (mode_name, mode) in interpolation_modes {
            println!("Testing {} interpolation...", mode_name);
            let mut resized =
                kornia_image::Image::<u8, 3, kornia_image::allocator::CpuAllocator>::from_size_val(
                    kornia_image::ImageSize {
                        width: target_size.0,
                        height: target_size.1,
                    },
                    0,
                    kornia_image::allocator::CpuAllocator,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("resize image creation error: {e}"))
                })?;
            resize_fast_rgb(&img_rgb_u8, &mut resized, mode)
                .map_err(|e| candle_core::Error::Msg(format!("resize_fast error: {e}")))?;
            println!(
                "  {} result shape: {}x{}",
                mode_name, target_size.0, target_size.1
            );

            // Save the resized image for each interpolation mode
            let output_path = format!(
                "../../examples/smol_vlm/validation_data/image_validations/Zoneplate_256x256_{}.png",
                mode_name.to_lowercase()
            );
            write_image_png_rgb8(&output_path, &resized)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to save image: {e}")))?;
            println!("Saved {}: {}", mode_name, output_path);
        }

        println!("Lanczos filtering test completed successfully!");
        Ok(())
    }
}
