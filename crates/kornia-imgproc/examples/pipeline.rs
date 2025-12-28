// examples/visualize_diff.rs

use kornia_image::{Image, ImageSize};
use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use std::path::Path;

pub struct ClipPreprocessConfig {
    pub target_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ClipPreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: 224,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        }
    }
}

pub fn clip_preprocess_fused<A: ImageAllocator>(
    image: &Image<u8, 3, A>,
    config: &ClipPreprocessConfig,
) -> Image<f32, 3, CpuAllocator>{
    let out_size = config.target_size;
    let w = image.width();
    let h = image.height();
    let src = image.as_slice();

    // 1. Setup Output Vector
    let mut data = Vec::with_capacity(out_size * out_size * 3);
    // Safe initialization of vector length
    unsafe { data.set_len(out_size * out_size * 3) };

    // 2. Calculate Scaling & Offsets
    let scale = if w < h {
        w as f32 / out_size as f32
    } else {
        h as f32 / out_size as f32
    };

    let start_x = (w as f32 - out_size as f32 * scale) * 0.5;
    let start_y = (h as f32 - out_size as f32 * scale) * 0.5;

    // 3. Precompute normalization constants
    let inv255 = 1.0 / 255.0;
    let mul = [
        inv255 / config.std[0],
        inv255 / config.std[1],
        inv255 / config.std[2],
    ];
    let sub = [
        config.mean[0] / config.std[0],
        config.mean[1] / config.std[1],
        config.mean[2] / config.std[2],
    ];

    let stride = w * 3;
    let row_len = out_size * 3;

    data.par_chunks_exact_mut(row_len)
        .enumerate()
        .for_each(|(y, out_row)| {
            let sy = start_y + y as f32 * scale;
            let y0_idx = sy as usize;
            let y1_idx = (y0_idx + 1).min(h - 1);

            let dy = sy - y0_idx as f32;
            let wy0 = 1.0 - dy;
            let wy1 = dy;

            // Unsafe read is safe here because `src` is read-only and Sync
            let row0 = unsafe { src.get_unchecked(y0_idx * stride..) };
            let row1 = unsafe { src.get_unchecked(y1_idx * stride..) };

            // We iterate manually over pixels in this row
            for x in 0..out_size {
                let sx = start_x + x as f32 * scale;
                let x0 = sx as usize;
                let x1 = (x0 + 1).min(w - 1);
                let dx = sx - x0 as f32;

                let idx0 = x0 * 3;
                let idx1 = x1 * 3;
                let wx0 = 1.0 - dx;
                let wx1 = dx;

                // Index into the current row slice
                let out_pixel_idx = x * 3;

                for c in 0..3 {
                    let p00 = unsafe { *row0.get_unchecked(idx0 + c) } as f32;
                    let p01 = unsafe { *row0.get_unchecked(idx1 + c) } as f32;
                    let p10 = unsafe { *row1.get_unchecked(idx0 + c) } as f32;
                    let p11 = unsafe { *row1.get_unchecked(idx1 + c) } as f32;

                    let top_val = p00 * wx0 + p01 * wx1;
                    let bot_val = p10 * wx0 + p11 * wx1;
                    let val = top_val * wy0 + bot_val * wy1;

                    // Write to the output slice for this row
                    out_row[out_pixel_idx + c] = val * mul[c] - sub[c];
                }
            }
        });

    Image::new(
        ImageSize {
            width: out_size,
            height: out_size,
        },
        data,
        CpuAllocator,
    )
    .expect("Failed to create image from buffer")
}

// =======================================================================
// PART 2: Standard Kornia Pipeline
// =======================================================================

use kornia_imgproc::{
    crop,
    interpolation::InterpolationMode,
    normalize,
    resize,
};

fn kornia_pipeline_op(
    image: &Image<u8, 3, CpuAllocator>,
    config: &ClipPreprocessConfig,
) -> Image<f32, 3, CpuAllocator> {
    
    let src_w = image.width();
    let src_h = image.height();
    let target = config.target_size;

    let scale = if src_w < src_h {
        target as f32 / src_w as f32
    } else {
        target as f32 / src_h as f32
    };

    let new_w = (src_w as f32 * scale).ceil() as usize;
    let new_h = (src_h as f32 * scale).ceil() as usize;

    // 2. Cast to f32
    let image_f32 = image
        .clone() 
        .cast_and_scale::<f32>(1.0 / 255.0)
        .unwrap();

    // 3. Resize
    let mut resized = Image::<f32, 3, _>::from_size_val(
        ImageSize { width: new_w, height: new_h },
        0.0f32,
        CpuAllocator
    ).unwrap();

    resize::resize_native(
        &image_f32,
        &mut resized,
        InterpolationMode::Bilinear,
    ).unwrap();

    // 4. Center Crop
    let crop_x = (new_w.saturating_sub(target)) / 2;
    let crop_y = (new_h.saturating_sub(target)) / 2;
    
    let mut cropped = Image::<f32, 3, _>::from_size_val(
        ImageSize { width: target, height: target },
        0.0f32,
        CpuAllocator
    ).unwrap();

    crop::crop_image(
        &resized,
        &mut cropped,
        crop_x, 
        crop_y,
    ).unwrap();

    // 5. Normalize
    let mut normalized = Image::<f32, 3, _>::from_size_val(
        cropped.size(),
        0.0f32,
        CpuAllocator
    ).unwrap();

    normalize::normalize_mean_std(
        &cropped,
        &mut normalized,
        &config.mean,
        &config.std,
    )
    .unwrap();

    normalized
}

fn main() {
    // 1. CONFIGURATION
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let input_path = std::path::Path::new(manifest_dir).join("../../sample.jpg");
    
    // Ensure the file exists
    if !input_path.exists() {
        eprintln!("Error: 'sample.jpg' not found at {:?}", input_path);
        return;
    }

    let config = ClipPreprocessConfig::default();
    
    // 2. LOAD IMAGE
    println!("Loading {:?}...", input_path);
    let img = image::open(&input_path).expect("Failed to open sample.jpg");
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    
    let source = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width: w as usize, height: h as usize },
        rgb.into_vec(),
        CpuAllocator,
    ).unwrap();

    // 3. RUN FUSED KERNEL
    println!("Running Fused Kernel...");
    let fused_raw = clip_preprocess_fused(&source, &config);
    
    // 4. RUN KORNIA PIPELINE
    println!("Running Standard Pipeline...");
    let kornia_img = kornia_pipeline_op(&source, &config);
    let kornia_raw = kornia_img.as_slice();

    // 5. HELPER: Denormalize & Save
    let save_image = |data: &[f32], filename: &str| {
        let mut u8_buffer = Vec::with_capacity(data.len());
        
        for (i, &v) in data.iter().enumerate() {
            let c = i % 3;
            // Reverse normalization: pixel = val * std + mean
            let denorm = (v * config.std[c] + config.mean[c]) * 255.0;
            u8_buffer.push(denorm.clamp(0.0, 255.0) as u8);
        }

        image::save_buffer(
            Path::new(filename),
            &u8_buffer,
            config.target_size as u32,
            config.target_size as u32,
            image::ColorType::Rgb8,
        ).expect("Failed to save image");
        println!("Saved {}", filename);
    };

    save_image(&fused_raw, "output_fused.jpg");
    save_image(kornia_raw, "output_kornia.jpg");

// 6. GENERATE DIFFERENCE MAP & STATS
    println!("Generating Difference Map & Stats...");
    
    let mut diff_buffer = Vec::with_capacity(fused_raw.len());
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let count = fused_raw.len() as f32;

    for (f, k) in fused_raw.iter().zip(kornia_raw.iter()) {
        let diff = (f - k).abs();

        // Update Stats
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;

        // Visual Difference Map
        let pixel_val = (diff * 255.0).clamp(0.0, 255.0) as u8;
        diff_buffer.push(pixel_val); 
    }

    let mean_diff = sum_diff / count;

    println!("---------------------------------------------------");
    println!("Comparison Statistics (Fused vs Kornia):");
    println!("Max Absolute Diff: {:.6}", max_diff);
    println!("Mean Absolute Diff: {:.6}", mean_diff);
    println!("---------------------------------------------------");

    image::save_buffer(
        Path::new("output_diff_amplified.jpg"),
        &diff_buffer,
        config.target_size as u32,
        config.target_size as u32,
        image::ColorType::Rgb8,
    ).expect("Failed to save diff map");
    println!("Saved output_diff_amplified.jpg");
}