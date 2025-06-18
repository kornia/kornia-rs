#![allow(unused_variables)]
#![allow(unused_attributes)] 


use std::fs;
use std::{collections::HashMap, error::Error};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module};
use candle_core::{DType, Device, Result, Shape, Tensor, D};
use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage, GenericImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage};
use reqwest;
use std::path::Path;
use std::io::Cursor;





// ImageNet mean and std for normalization
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];



pub fn load_image_url(url: &str) -> std::result::Result<DynamicImage, Box<dyn Error>> {
    let dir = Path::new("static/images");

    // Generate a file name based on the URL (you can customize this to your needs)
    let file_path = {
        let parsed_url = reqwest::Url::parse(url)?;
        let path = parsed_url.path();
        
        // Extract the last part of the URL (the file name)
        dir.join(
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown_file.jpg") // Default name if URL doesn't have a file name
                .to_string()
        )
    };

    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    }
    
    // Check if the file exists locally
    if file_path.exists() {
        // If the file exists, load it from the local directory
        let img = image::open(&file_path)?;
        println!("Loaded image from local cache.");
        return Ok(img);
    }

    // If the file does not exist, download it and save it
    println!("Downloading image from URL...");

    // Fetch the image as bytes
    let response = reqwest::blocking::get(url)?.bytes()?;

    // Decode the image
    let img = ImageReader::new(Cursor::new(response))
        .with_guessed_format()?
        .decode()?;

    // Save the image to the local directory
    img.save(&file_path)?;

    println!("Saved image locally as {}", file_path.to_str().unwrap());

    // Return the image
    Ok(img)
}

pub fn preprocess_image(
    img: DynamicImage, max_size: u32, outer_patch_size: u32, device: &Device
) -> (Tensor, Tensor, usize, usize) {

    // resizing image to match the max_size (on the longest edge)
    let img = {
        let (width, height) = img.dimensions();
        let longest_edge = width.max(height);

        if longest_edge <= max_size {
            img.clone()
        } else {
            let scale_factor = max_size as f32 / longest_edge as f32;
        
            let new_width = (width as f32 * scale_factor) as u32;
            let new_height = (height as f32 * scale_factor) as u32;
    
            img.resize(new_width, new_height, FilterType::Lanczos3)
        }
    };
    let global_img = {
        let (width, height) = img.dimensions();
        let longest_edge = width.max(height);

        if longest_edge <= outer_patch_size {
            img.clone()
        } else {
            let scale_factor = outer_patch_size as f32 / longest_edge as f32;
        
            let new_width = (width as f32 * scale_factor) as u32;
            let new_height = (height as f32 * scale_factor) as u32;
    
            img.resize(new_width, new_height, FilterType::Lanczos3)
        }
    };

    // padding image for all dimensions to be multiples of the outer_patch_size
    let (img, mask) = {
        let (width, height) = img.dimensions();
        let mask = GrayImage::from_pixel(width, height, Luma([255]));

        let new_width = u32::div_ceil(width, outer_patch_size)*outer_patch_size;
        let new_height = u32::div_ceil(height, outer_patch_size)*outer_patch_size;
    
        // Create a new blank image for padding
        let mut padded_img = RgbImage::from_pixel(new_width, new_height, Rgb([0, 0, 0]));
        padded_img.copy_from(&img.to_rgb8(), 0, 0).unwrap();
        let mut padded_mask = GrayImage::from_pixel(new_width, new_height, Luma([0]));
        padded_mask.copy_from(&mask, 0, 0).unwrap();

        (padded_img, padded_mask)
    };
    let (global_img, global_mask) = {
        let (width, height) = global_img.dimensions();
        let mask = GrayImage::from_pixel(width, height, Luma([255]));

        let new_width = u32::div_ceil(width, outer_patch_size)*outer_patch_size;
        let new_height = u32::div_ceil(height, outer_patch_size)*outer_patch_size;
    
        // Create a new blank image for padding
        let mut padded_img = RgbImage::from_pixel(new_width, new_height, Rgb([0, 0, 0]));
        padded_img.copy_from(&global_img.to_rgb8(), 0, 0).unwrap();
        let mut padded_mask = GrayImage::from_pixel(new_width, new_height, Luma([0]));
        padded_mask.copy_from(&mask, 0, 0).unwrap();

        (padded_img, padded_mask)
    };


    img.save("static/padded_img.png").unwrap();
    mask.save("static/mask.png").unwrap();
    global_img.save("static/global_padded_img.png").unwrap();
    global_mask.save("static/global_mask.png").unwrap();

    let img = {
        let (width, height) = img.dimensions();
        let img_data: Vec<u8> = img.pixels().flat_map(|p| p.0.iter().copied()).collect();

        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize, 3]), device
        ).unwrap()
            .permute(vec![2, 0, 1]).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let mask = {
        let (width, height) = mask.dimensions();
        let img_data: Vec<u8> = mask.pixels().flat_map(|p| p.0.iter().copied()).collect();

        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize]), device
        ).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let global_img = {
        let (width, height) = global_img.dimensions();
        let img_data: Vec<u8> = global_img.pixels().flat_map(|p| p.0.iter().copied()).collect();

        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize, 3]), device
        ).unwrap()
            .permute(vec![2, 0, 1]).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let global_mask = {
        let (width, height) = global_mask.dimensions();
        let img_data: Vec<u8> = global_mask.pixels().flat_map(|p| p.0.iter().copied()).collect();

        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize]), device
        ).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };


    // rescaling and normalizing
    let img = {
        let mut img = (img / 255.0).unwrap();
        let m = Tensor::from_slice(&MEAN, (3,1,1), device).unwrap();
        let s = Tensor::from_slice(&STD, (3,1,1), device).unwrap();

        img = img.broadcast_sub(&m).unwrap();
        img = img.broadcast_div(&s).unwrap();

        img
    };
    let global_img = {
        let mut global_img = (global_img / 255.0).unwrap();
        let m = Tensor::from_slice(&MEAN, (3,1,1), device).unwrap();
        let s = Tensor::from_slice(&STD, (3,1,1), device).unwrap();

        global_img = global_img.broadcast_sub(&m).unwrap();
        global_img = global_img.broadcast_div(&s).unwrap();

        global_img
    };

    let (c, h, w) = img.dims3().unwrap();
    let cols = w / outer_patch_size as usize;
    let rows = h / outer_patch_size as usize;

    // splitting
    let img_patches = {
        img
            .unsqueeze(2).unwrap()
            .unsqueeze(4).unwrap()
            .reshape(&[c, rows, outer_patch_size as usize, cols, outer_patch_size as usize]).unwrap()
            .permute([1, 3, 0, 2, 4]).unwrap()
            .reshape(&[rows*cols, c, outer_patch_size as usize, outer_patch_size as usize]).unwrap()
    };
    let mask_patches = {
        mask
            .unsqueeze(1).unwrap()
            .unsqueeze(3).unwrap()
            .reshape(&[rows, outer_patch_size as usize, cols, outer_patch_size as usize]).unwrap()
            .permute([0, 2, 1, 3]).unwrap()
            .reshape(&[rows*cols, outer_patch_size as usize, outer_patch_size as usize]).unwrap()
    };

    // concatenating global image
    let img_patches = Tensor::cat(&[&img_patches, &global_img.unsqueeze(0).unwrap()], 0).unwrap();
    let mask_patches = Tensor::cat(&[&mask_patches, &global_mask.unsqueeze(0).unwrap()], 0).unwrap();

    (img_patches, mask_patches, cols, rows)
}

pub fn get_prompt_split_image(
    img_seq_len: usize, img_rows: usize, img_cols: usize
) -> String {
    let mut s = String::new();

    for h in 0..img_rows {
        for w in 0..img_cols {
            s += &format!("<fake_token_around_image><row_{}_col_{}>{}", h+1, w+1, "<image>".repeat(img_seq_len));
        }
        s += "\n";
    }

    s += &format!("\n<fake_token_around_image><global-img>{}<fake_token_around_image>", "<image>".repeat(img_seq_len));

    s
}




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
        let y = y.transpose(1, 2)?.reshape(&[batches, patches, hidden_size])?;
        self.o_proj.forward(&y)
    }
}



struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    // pub fn new(c: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
    //     todo!()
    // }

    // Constants from PyTorch's GELU implementation
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;  // sqrt(2.0 / PI)
    const GELU_COEFF: f64 = 0.044715;

    /// PyTorch-like GELU activation with `tanh` approximation.
    pub fn gelu_tanh(input: &Tensor) -> Result<Tensor> {
        let gelu_coeff = Tensor::new(Self::GELU_COEFF, input.device())?.to_dtype(DType::BF16)?;
        let sqrt_2_over_pi = Tensor::new(Self::SQRT_2_OVER_PI, input.device())?.to_dtype(DType::BF16)?;
        let one = Tensor::new(1.0, input.device())?.to_dtype(DType::BF16)?;
        let half = Tensor::new(0.5, input.device())?.to_dtype(DType::BF16)?;

        // Compute: 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
        let x_cubed = input.powf(3.0)?;
        let inner = (input + x_cubed.broadcast_mul(&gelu_coeff)?)?;
        let tanh_arg = inner.broadcast_mul(&sqrt_2_over_pi)?;
        let tanh = tanh_arg.tanh()?;

        half.broadcast_mul(&input.broadcast_mul(&(tanh.broadcast_add(&one)?))?)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = Self::gelu_tanh(&self.fc1.forward(xs)?)?;  // python impl. uses gelo approximated with tanh
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
        let w = |k| c[&("model.vision_model.encoder.layers.".to_owned()+&id.to_string()+"."+k+".weight")].clone();
        let b = |k| c[&("model.vision_model.encoder.layers.".to_owned()+&id.to_string()+"."+k+".bias")].clone();

        println!("Loaded layer (VT): {:?}", id);

        Ok(Self {
            self_attn: Attention {
                q_proj: Linear::new(w("self_attn.q_proj"), Some(b("self_attn.q_proj"))),
                k_proj: Linear::new(w("self_attn.k_proj"), Some(b("self_attn.k_proj"))),
                v_proj: Linear::new(w("self_attn.v_proj"), Some(b("self_attn.v_proj"))),
                o_proj: Linear::new(w("self_attn.out_proj"), Some(b("self_attn.out_proj")))
            },
            layer_norm1: LayerNorm::new(w("layer_norm1"), b("layer_norm1"), 1e-6),
            mlp: MLP {
                fc1: Linear::new(w("mlp.fc1"), Some(b("mlp.fc1"))),
                fc2: Linear::new(w("mlp.fc2"), Some(b("mlp.fc2")))
            },
            layer_norm2: LayerNorm::new(w("layer_norm2"), b("layer_norm2"), 1e-6)
        })
    }

    pub fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let x = self.layer_norm1.forward(xs)?;
        let x = self.self_attn.forward(&x, attention_mask)?;
        let x = (residual+x)?;
        let residual = &x;
        let x = self.layer_norm2.forward(&x)?;
        let x = self.mlp.forward(&x);
        residual+x
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
                Conv2dConfig { // kernel/patch size are intrinsically defined in the weights 
                    padding: 0, stride: 14, dilation: 1, groups: 1, 
                }
            ),
            position_embedding: Embedding::new(c["model.vision_model.embeddings.position_embedding.weight"].clone(), 1152),
            blocks: (0u8..=26).into_iter().map(|id| Block::new(c, id, device).unwrap()).collect(),
            post_layernorm: LayerNorm::new(
                c["model.vision_model.post_layernorm.weight"].clone(), 
                c["model.vision_model.post_layernorm.bias"].clone(), 
                1e-6
            )
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, pixel_attention_masks: &Tensor, device: &Device) -> Result<Tensor> {
        // B = patch rows x patch cols (x number of images)
        // pixel_values: B x 3 x PatchHeight x PatchWidth
        // pixel_attention_masks: B x PatchHeight x PatchWidth
        let (batch, patch_h, patch_w) = pixel_attention_masks.dims3()?;

        // the unfold operation (splitting patches into 27x27 subpatches of 14x14 pixels)
        // 384x384 -> 378x378 where 378=14*27 (divisible by 14)
        // truncate around the middle
        let truncated = pixel_attention_masks.narrow(1, 3, patch_h-3*2)?.narrow(2, 3, patch_w-3*2)?;
        let patch_attention_masks = truncated
            .unsqueeze(2)?
            .unsqueeze(4)?
            .reshape(&[batch, patch_h / Self::SUB_PATCH_SIZE, Self::SUB_PATCH_SIZE, patch_w / Self::SUB_PATCH_SIZE, Self::SUB_PATCH_SIZE])?
            .permute([0, 1, 3, 2, 4])?
            .sum_keepdim([3,4])?
            .squeeze(4)?
            .squeeze(3)?
            .gt(0.0)?
            .reshape(&[batch, 27*27])?
            .contiguous()?
            .to_dtype(DType::U32)?;
        // patch_attention_masks: B x PatchRows x PatchCols x 196

        // println!("{:?}", truncated.shape());
        // println!("{:?}", patch_attention_masks);
        // println!("{:?}", patch_attention_masks.to_vec3::<u8>());

        let mut hidden_states = {
            let patch_embeddings = self.patch_embedding.forward(&pixel_values.to_dtype(DType::BF16)?)?;
            // println!("{:?}", patch_embeddings.shape());
            let patch_embeddings = patch_embeddings.flatten_from(2)?.transpose(1, 2)?;
            // println!("{:?}", patch_embeddings.shape());
    
    
            let position_ids = {
                let raw_ids = Tensor::arange(0u32, 27*27, device)?
                    .expand(&[batch, 27*27])?;
                (raw_ids * &patch_attention_masks)?
            };
            let position_embeddings = self.position_embedding.forward(&position_ids)?;
            // println!("{:?}", patch_embeddings);
            // println!("{:?}", position_embeddings);
            // println!("{:?}", patch_embeddings.to_dtype(DType::F32)?.to_vec3::<f32>());
            // println!("{:?}", position_embeddings.to_vec3::<u32>());
            patch_embeddings+position_embeddings
        }?;
        // println!(">> {:?}", hidden_states);

        let patch_attention_masks = {
            let expanded_masks = patch_attention_masks
                .unsqueeze(1)?
                .unsqueeze(1)?
                .expand(&[batch, 1, 27*27, 27*27])?;  // batch, head_dim, subpatches, subpatches
            let inverted_mask = Tensor::ones_like(&expanded_masks)?
                .sub(&expanded_masks)?;
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
