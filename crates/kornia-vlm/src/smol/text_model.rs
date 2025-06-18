use std::collections::HashMap;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{ops, kv_cache::Cache, rotary_emb::rope, Embedding, Linear, Module, RmsNorm};
use candle_core::DType;

use crate::vision_model::SmolVision;



const NUM_OF_HEADS: usize = 32;
const HEAD_DIM: usize = 64;


fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

fn calculate_default_inv_freq() -> Vec<f32> {
    (0..HEAD_DIM)
        .step_by(2)
        //            1 / rope theta
        .map(|i| 1f32 / (273768f32).powf(i as f32 / HEAD_DIM as f32))
        .collect()
}



#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    cos: Tensor,
    sin: Tensor,
}

impl Attention {
    fn new(q: Tensor, k: Tensor, v: Tensor, o: Tensor, device: &Device) -> Result<Self> {
        let theta = Tensor::new(calculate_default_inv_freq(), device)?;
        // 0 -> max position embedding
        let idx_theta = Tensor::arange(0, 16384u32, device)?
            .to_dtype(DType::F32)?
            .reshape((16384, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        Ok(Self {
            q_proj: Linear::new(q, None),
            k_proj: Linear::new(k, None),
            v_proj: Linear::new(v, None),
            o_proj: Linear::new(o, None),
            cos: idx_theta.cos()?.to_dtype(DType::BF16)?,
            sin: idx_theta.sin()?.to_dtype(DType::BF16)?,
        })
    }

    fn apply_rotary_embedding(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_head_sz, seq_len, _hidden_size) = x.dims3()?;

        rope(
            &x.unsqueeze(0)?,
            &self.cos.narrow(0, index_pos, seq_len).expect("Exceeded context limit"),
            &self.sin.narrow(0, index_pos, seq_len).expect("Exceeded context limit")
        )?.squeeze(0)
    }

    #[allow(unused_variables)]
    fn new_with_biases(q: Tensor, k: Tensor, v: Tensor, o: Tensor,
                         qb: Tensor, kb: Tensor, vb: Tensor, ob: Tensor) -> Self {
        todo!()
    }

    fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
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

        let y =
        // if false {
        //     let q = q.transpose(1, 2)?;
        //     let k = k.transpose(1, 2)?;
        //     let v = v.transpose(1, 2)?;
        //     let softmax_scale = 1f32 / (HEAD_DIM as f32).sqrt();
        //     flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?.into()
        // } else
        {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
    
            let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
            let mask = causal_mask(att.shape().dim(2)?, &Device::new_cuda(0)?)?;  // causal masking
    
            // println!("{:?}", att.shape());
    
            let att = candle_nn::ops::softmax_last_dim(&att.broadcast_add(&mask)?)?;
            att.matmul(&v)?.contiguous()?.to_dtype(in_dtype)?
        };
        let y = y.transpose(0, 1)?.reshape(&[seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }
}



#[derive(Debug, Clone)]
struct MLPGates {
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(&x)?.silu()?.to_dtype(DType::F32)?;
        let up = self.up_proj.forward(&x)?.to_dtype(DType::F32)?;
        let hidden = (gate * up)?.to_dtype(DType::BF16)?;
        let x = self.down_proj.forward(&hidden)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct Block {
    input_layer_norm: RmsNorm,
    attn: Attention,
    post_layer_norm: RmsNorm,
    gates: MLPGates,
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
    fn load(c: &HashMap<String, Tensor>, id: u8, device: &Device) -> Result<Self> {
        let val = |k| c[&("model.text_model.layers.".to_owned()+&id.to_string()+"."+k+".weight")].clone();

        println!("Loaded layer (LM): {:?}", id);

        Ok(Self {
            input_layer_norm: RmsNorm::new(val("input_layernorm"), 1e-5),
            attn: Attention::new(
                val("self_attn.q_proj"), val("self_attn.k_proj"), val("self_attn.v_proj"), val("self_attn.o_proj"),
                device,
            )?,
            post_layer_norm: RmsNorm::new(val("post_attention_layernorm"),1e-5),
            gates: MLPGates::new(
                val("mlp.down_proj"), val("mlp.gate_proj"), val("mlp.up_proj")
            )
        })
    }

    fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layer_norm.forward(x)?;
        let x = (residual + self.attn.forward(&x, index_pos)?)?;
        let residual = &x;
        let x = (residual + self.gates.forward(&self.post_layer_norm.forward(&x)?)?)?;
        Ok(x)
    }
}


pub struct Connector {
    // scale_factor = 3
    modality_proj: Linear,
}

impl Connector {
    const SCALE_FACTOR: usize = 3;
    const HEIGHT: usize = 27;
    const WIDTH: usize = 27;

    fn pixel_shuffle(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, patches, embed_dim) = x.dims3()?;  // patches == HEIGHT*WIDTH

        x   .reshape(&[batch, Self::HEIGHT, Self::WIDTH, embed_dim])?
            .reshape(&[batch, Self::HEIGHT, Self::WIDTH/Self::SCALE_FACTOR, embed_dim*Self::SCALE_FACTOR])?
            .permute([0, 2, 1, 3])?
            .reshape(&[batch, Self::WIDTH/Self::SCALE_FACTOR, Self::HEIGHT/Self::SCALE_FACTOR, embed_dim*Self::SCALE_FACTOR*Self::SCALE_FACTOR])?
            .permute([0, 2, 1, 3])?
            .reshape(&[batch, patches/(Self::SCALE_FACTOR*Self::SCALE_FACTOR), embed_dim*Self::SCALE_FACTOR*Self::SCALE_FACTOR])
    }

    pub fn forward(&self, image_hidden_states: &Tensor) -> Result<Tensor> {
        let image_hidden_states = self.pixel_shuffle(image_hidden_states)?;
        self.modality_proj.forward(&image_hidden_states)
    }
}



pub struct SmolVLM {
    vision: SmolVision,
    connector: Connector,
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,

    image_hidden_states: Option<Tensor>,
}


impl SmolVLM {
    const BLOCKS_PER_SAMPLE: u32 = 81;

    pub fn load(c: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        Ok(Self {
            vision: SmolVision::load(c, device)?,
            connector: Connector { modality_proj: Linear::new(c["model.connector.modality_projection.proj.weight"].clone(), None) },
            embed: Embedding::new(c["model.text_model.embed_tokens.weight"].clone(), 2048),
            blocks: (0u8..=23).into_iter().map(|id| Block::load(c, id, device).unwrap()).collect(),
            norm: RmsNorm::new(c["model.text_model.norm.weight"].clone(), 1e-5),
            lm_head: Linear::new(c["lm_head.weight"].clone(), None),

            image_hidden_states: None,
        })
    }

    fn inputs_merger(&self, image_token_mask: &Tensor, inputs_embeds: &Tensor, image_hidden_states: &Tensor, device: &Device) -> Result<Tensor> {
        let total_length = image_token_mask.dims1()?;
        let (patches, patch_size, hidden_dim) = image_hidden_states.dims3()?;

        // println!("Image tokens: {:?}", image_token_mask.to_dtype(DType::U32)?.sum_all()?);
        // println!("Patch sequences: {:?}", patches*81);
        // println!("Img tkn mask: {:?}", image_token_mask);
        // println!("Inp embeds: {:?}", inputs_embeds);
        // println!("Img hidden: {:?}", image_hidden_states);

        // let scatter_indices = {

        //     let indices = Tensor::arange(0u32, total_length as u32, device)?
        //         .mul(&image_token_mask.to_dtype(DType::U32)?)?;
        //     let scatter_indices = indices
        //         .to_vec1::<u32>()?
        //         .into_iter()
        //         .filter(|&x| x != 0)
        //         .collect::<Vec<u32>>();

        //     Tensor::from_vec(scatter_indices, patches*81, device)
        // }?;

        // println!("Img embed assign: {:?}", scatter_indices);
        // println!("Img embed assign: {:?}", scatter_indices.to_vec1::<u32>());

        // let image_embeds = inputs_embeds.zeros_like()?
        //     .scatter_add(&scatter_indices.unsqueeze(1)?, &image_hidden_states.flatten(0, 1)?, 0)?;
        
        // println!("Img embed assign: {:?}", image_embeds);

        let image_hidden_states = image_hidden_states.flatten(0, 1)?;

        let mut merged_embeds = Vec::with_capacity(total_length);
        let mut c = 0;
        for (i, mask) in image_token_mask.to_vec1::<u8>()?.into_iter().enumerate() {
            merged_embeds.push(if mask != 0 {
                // println!("{:?}", c);
                c += 1;
                image_hidden_states.i(c-1)?
            } else {
                inputs_embeds.i(i)?
            });
        }

        // let merged_embeds = Tensor::from_vec(merged_embeds, (total_length, hidden_dim), device)?;
        let merged_embeds = Tensor::stack(&merged_embeds, 0)?;

        Ok(merged_embeds)
    }

    pub fn forward(
        &mut self, xs: &Tensor, index_pos: usize, vision_data: Option<(Tensor, &Tensor, &Tensor)>, device: &Device
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.embed.forward(xs)?;

        if let Some((image_token_mask, pixel_values, pixel_attention_masks)) = vision_data {
            // println!("Vision...");
            // TODO: this assumes there will be at most one new images added
            let image_hidden_states = if let Some(ref image_hidden_states) = self.image_hidden_states {
                image_hidden_states
            } else {
                let image_hidden_states = self.vision.forward(pixel_values, pixel_attention_masks, device)?;
                let image_hidden_states = self.connector.forward(&image_hidden_states)?;
                self.image_hidden_states = Some(image_hidden_states);
                self.image_hidden_states.as_ref().unwrap()
            };
    
            inputs_embeds = self.inputs_merger(
                &image_token_mask, 
                &inputs_embeds,
                &image_hidden_states,
                device
            )?;
        }

        // println!("Language...");

        let mut x = inputs_embeds; //self.inputs_merger(image_token_mask, &inputs_embeds, &image_hidden_states)?;

        for block in &self.blocks {
            x = block.forward(&x, index_pos)?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }
}



#[cfg(test)]
mod tests {
    use crate::vision_model::{get_prompt_split_image, load_image_url, preprocess_image, SmolVision};

    use super::*; // Import functions from the outer scope
    use hf_hub::api::sync::Api;
    use image::GenericImageView;
    use tokenizers::Tokenizer;
    use candle_core::Shape;

    #[test]
    fn test_multimodal() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None).unwrap();
        let api = Api::new().unwrap();
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &device)?;
        
        let mut model = SmolVLM::load(&weights, &device)?;


        if let Ok(img) = load_image_url(
            "https://res.cloudinary.com/enchanting/q_70,f_auto,w_5472,h_3078,c_fit/exodus-web/2023/05/mont-blanc.jpg"
        ) {
            let (img, mask, cols, rows) = preprocess_image(img, 1920, 384, &device);
            let img_token = get_prompt_split_image(81, rows, cols);
    
            let sample_message = String::from("<|im_start|>
    User:<image>Where is this place?<end_of_utterance>
    Assistant:");
            let sample_message = sample_message.replace("<image>", &img_token);
    
            let tokens_enc = tokenizer.encode(sample_message.clone(), false).unwrap();
            let tokens = tokens_enc.get_ids();
            let image_token_enc = tokenizer.encode("<image>", false).unwrap();
            let image_token = image_token_enc.get_ids();
    
            let input = Tensor::from_slice(tokens, &[tokens.len()], &device)?;
            let image_token_mask = Tensor::from_slice(image_token, &[1], &device)?;
            let image_token_mask = input.broadcast_eq(&image_token_mask)?;
    
            // println!("{:?}", input);
            // println!("{:?}", image_token_mask);
            // println!("{:?}", image_token_mask.to_dtype(DType::U32)?.sum_all()?);
            let logits = model.forward(&input, 0, Some((&image_token_mask, &img, &mask)), &device)?;    
        } else {
            println!("Invalid or empty URL (no image)");

            let sample_message = String::from("<|im_start|>
    User:Where is this place?<end_of_utterance>
    Assistant:");
            let tokens_enc = tokenizer.encode(sample_message.clone(), false).unwrap();
            let tokens = tokens_enc.get_ids();
    
            let input = Tensor::from_slice(tokens, &[tokens.len()], &device)?;
    
            // println!("{:?}", input);
            // println!("{:?}", image_token_mask);
            // println!("{:?}", image_token_mask.to_dtype(DType::U32)?.sum_all()?);
            let logits = model.forward(&input, 0, None, &device)?;    
        }
        
        Ok(())
    }

    #[test]
    fn test_vision() -> Result<()> {
        let device = Device::new_cuda(0)?;
        
        let api = Api::new().unwrap();
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
    
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &device)?;
    
        let model = SmolVision::load(&weights, &device)?;

        /*
            Assume we are given an option of at most one image (conditional execution of vision llm) placed at the beginning.
         */

        let img = load_image_url(
            "https://res.cloudinary.com/enchanting/q_70,f_auto,w_5472,h_3078,c_fit/exodus-web/2023/05/mont-blanc.jpg"
        ).unwrap();
        let (img, mask, cols, rows) = preprocess_image(img, 1920, 384, &device);
        let img_token = get_prompt_split_image(81, rows, cols);

        model.forward(&img, &mask, &device)?;

        Ok(())
    }

    #[test]
    fn test_preprocessing_images() -> Result<()> {
        let device = Device::new_cuda(0)?;

        println!("Loading image...");
        let img = load_image_url(
            "https://res.cloudinary.com/enchanting/q_70,f_auto,w_5472,h_3078,c_fit/exodus-web/2023/05/mont-blanc.jpg"
        ).unwrap();

        println!("[PRE] DIM: {:?}  FORMAT: {:?}", img.dimensions(), img.color());
        let (img, mask, cols, rows) = preprocess_image(img, 1920, 384, &device);
        println!("[POST] SHAPE: {:?} MAX: {:?} MIN: {:?} ", img.shape(), img.max_all(), img.min_all());
        println!("[POST-MASK] SHAPE: {:?} MAX: {:?} MIN: {:?} ", mask.shape(), mask.max_all(), mask.min_all());

        let img_token = get_prompt_split_image(81, rows, cols);

        let sample_message = String::from("<|im_start|>
User:<image>Where is this place?<end_of_utterance>
Assistant:");

        let sample_message = sample_message.replace("<image>", &img_token);
        println!("Modified message: {:?} ", sample_message);
        
        /*
        assuming this image is RGB with channel dimension as the first dimension (C,W,H).
         - (padding) the produced image lists from splitting should have the same max_image_size
            - strategy: pad beforehand + produce padding attention mask
         - make sure its normalized (0-255 -> 0-1)
         - further normalized by ImageNet's mean and std
         - split

        size                = 1920 (longest edge)
        max_image_size      = 384 (longest edge)

                resample            = LANCZOS
            resize_for_vision_encoder (in multiple of max_image_size)
        do_image_splitting  = true
         */

        let mut tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None).unwrap();
        let encoding = tokenizer.encode(sample_message.clone(), false).unwrap();
        let tokens = encoding.get_tokens();

        println!("{:?}", tokens);


        Ok(())
    }
}
