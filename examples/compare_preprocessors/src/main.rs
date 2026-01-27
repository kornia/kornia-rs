use anyhow::Result;
use candle_core::Device;
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
use kornia_io::{jpeg::read_image_jpeg_rgb8, png::write_image_png_rgb8};
use kornia_vlm::preprocessor::{Model, PreprocessConfig, Preprocessor};
use kornia_vlm::smolvlm::preprocessor::SmolVlmImagePreprocessor;
use std::path::Path;

fn main() -> Result<()> {
    // 1. Load Image
    let img_path = Path::new("tests/data/apriltags_tag36h11.jpg");
    if !img_path.exists() {
        println!("Image not found at {:?}", img_path);
        return Ok(());
    }
    println!("Loading image from {:?}", img_path);
    let image = read_image_jpeg_rgb8(img_path)?;

    let device = Device::Cpu;

    // --- SMOLVLM COMPARISON ---
    println!("\n--- SMOLVLM COMPARISON ---");

    // 2. Run Generic Preprocessor (SmolVLM Config)
    println!("Running Generic Preprocessor (SmolVLM Config)...");
    let config_smol = PreprocessConfig {
        model_type: Model::SmolVLM,
        target_size: 384,
        mean: [0.5, 0.5, 0.5], // SigLIP mean
        std: [0.5, 0.5, 0.5],  // SigLIP std
        interpolation: InterpolationMode::Lanczos,
    };
    let preprocessor_smol = Preprocessor::new(config_smol.clone());
    let out_tensors_smol = preprocessor_smol.process(&image, &device)?;

    println!(
        "Generic Preprocessor produced {} tensors",
        out_tensors_smol.len()
    );

    // Save generic outputs
    for (i, tensor) in out_tensors_smol.iter().enumerate() {
        let tensor = tensor.unsqueeze(0)?;
        let imgs = tensor_to_images(&tensor, config_smol.mean, config_smol.std)?;
        // Since tensor_to_images returns a Vec (handling batch dim), but here we process single images:
        if let Some(img) = imgs.first() {
            let path = format!("smol_generic_out_{}.png", i);
            write_image_png_rgb8(&path, img)?;
            println!("Saved {}", path);
        }
    }

    // 3. Run SmolVlmImagePreprocessor
    println!("Running SmolVlmImagePreprocessor...");
    let mut smol_pre = SmolVlmImagePreprocessor::new(2048, 384, &device)?;
    let (img_tensor, mask_tensor, size) = smol_pre.preprocess(&image, &device, CpuAllocator)?;

    println!("SmolVLM Output Tensor Shape: {:?}", img_tensor.dims());
    println!("SmolVLM Output Mask Shape: {:?}", mask_tensor.dims());
    println!("SmolVLM Output Size: {:?}", size);

    // Convert tensor back to images and save
    let patches = tensor_to_images(&img_tensor, config_smol.mean, config_smol.std)?;
    for (i, img) in patches.iter().enumerate() {
        let path = format!("smol_specific_out_{}.png", i);
        write_image_png_rgb8(&path, img)?;
        println!("Saved {}", path);
    }

    // --- PALIGEMMA COMPARISON ---
    println!("\n--- PALIGEMMA COMPARISON ---");

    // 4. Run Generic Preprocessor (PaliGemma Config)
    println!("Running Generic Preprocessor (PaliGemma Config)...");
    let config_pali = PreprocessConfig {
        model_type: Model::PaliGemma,
        target_size: 224,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        interpolation: InterpolationMode::Bicubic,
    };
    let preprocessor_pali = Preprocessor::new(config_pali.clone());
    let out_tensors_pali = preprocessor_pali.process(&image, &device)?;

    println!(
        "Generic Preprocessor produced {} tensors",
        out_tensors_pali.len()
    );

    // Save generic outputs
    for (i, tensor) in out_tensors_pali.iter().enumerate() {
        let tensor = tensor.unsqueeze(0)?;
        let imgs = tensor_to_images(&tensor, config_pali.mean, config_pali.std)?;
        if let Some(img) = imgs.first() {
            let path = format!("pali_generic_out_{}.png", i);
            write_image_png_rgb8(&path, img)?;
            println!("Saved {}", path);
        }
    }

    // 5. Run Specific PaliGemma Logic (Resize + Bilinear)
    println!("Running Specific PaliGemma Logic...");

    // Original implementation: resize_fast_rgb(image, &mut img_buf, InterpolationMode::Bilinear)?;
    let mut pali_specific_out = Image::from_size_val(
        ImageSize {
            width: 224,
            height: 224,
        },
        0u8,
        CpuAllocator,
    )?;

    resize_fast_rgb(&image, &mut pali_specific_out, InterpolationMode::Bicubic)?;

    let path = "pali_specific_out_0.png";
    write_image_png_rgb8(path, &pali_specific_out)?;
    println!("Saved {}", path);

    // --- CLIP COMPARISON ---
    println!("\n--- CLIP COMPARISON ---");

    // 6. Run Generic Preprocessor (Clip Config)
    println!("Running Generic Preprocessor (Clip Config)...");
    let config_clip = PreprocessConfig {
        model_type: Model::Clip,
        target_size: 224,
        mean: kornia_vlm::preprocessor::CLIP_MEAN,
        std: kornia_vlm::preprocessor::CLIP_STD,
        interpolation: InterpolationMode::Bicubic,
    };
    let preprocessor_clip = Preprocessor::new(config_clip.clone());
    let out_tensors_clip = preprocessor_clip.process(&image, &device)?;

    println!(
        "Generic Preprocessor produced {} tensors",
        out_tensors_clip.len()
    );

    // Save generic outputs
    for (i, tensor) in out_tensors_clip.iter().enumerate() {
        let tensor = tensor.unsqueeze(0)?;
        let imgs = tensor_to_images(&tensor, config_clip.mean, config_clip.std)?;
        if let Some(img) = imgs.first() {
            let path = format!("clip_generic_out_{}.png", i);
            write_image_png_rgb8(&path, img)?;
            println!("Saved {}", path);
        }
    }

    Ok(())
}

// Helper for tensor (which is [N, 3, H, W]) -> Vec<Image>
fn tensor_to_images(
    tensor: &candle_core::Tensor,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<Image<u8, 3, CpuAllocator>>> {
    let (n, c, h, w) = tensor.dims4()?;
    let mut images = Vec::new();

    // Convert to Vec<f32>
    let vec_data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let image_size = c * h * w;
    let plane_size = h * w;

    for i in 0..n {
        let start = i * image_size;
        let end = start + image_size;
        let img_slice = &vec_data[start..end];

        // img_slice is CHW
        let r_plane = &img_slice[0..plane_size];
        let g_plane = &img_slice[plane_size..2 * plane_size];
        let b_plane = &img_slice[2 * plane_size..3 * plane_size];

        let mut u8_data = vec![0u8; plane_size * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let r = r_plane[idx] * std[0] + mean[0];
                let g = g_plane[idx] * std[1] + mean[1];
                let b = b_plane[idx] * std[2] + mean[2];

                let dst_idx = (y * w + x) * 3;
                u8_data[dst_idx] = (r * 255.0).clamp(0.0, 255.0) as u8;
                u8_data[dst_idx + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
                u8_data[dst_idx + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        images.push(Image::new(
            ImageSize {
                width: w,
                height: h,
            },
            u8_data,
            CpuAllocator,
        )?);
    }
    Ok(images)
}
