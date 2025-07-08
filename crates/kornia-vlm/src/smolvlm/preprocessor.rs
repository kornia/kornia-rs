use candle_core::{Device, Shape, Tensor};

use kornia_image::allocator::ImageAllocator;

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast;
use kornia_tensor::allocator::CpuAllocator;
use std::fs;
use std::path::Path;

// https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/blob/main/preprocessor_config.json
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

pub fn preprocess_image<A: ImageAllocator>(
    img: Image<u8, 3, A>,
    max_size: u32,
    outer_patch_size: u32,
    device: &Device,
) -> (Tensor, Tensor, ImageSize) {
    // resizing image to match the max_size (on the longest edge)
    let img = {
        let (width, height) = (img.width() as u32, img.height() as u32);
        println!("Image size: {}x{} w/ Max size: {}", width, height, max_size);
        let longest_edge = width.max(height);
        // if longest_edge <= max_size {
        //     img.clone()
        // } else {
        let scale_factor = max_size as f32 / longest_edge as f32;
        let new_width = (width as f32 * scale_factor).ceil() as usize;
        let new_height = (height as f32 * scale_factor).ceil() as usize;
        let mut resized = Image::<u8, 3, A>::from_size_val(
            ImageSize {
                width: new_width,
                height: new_height,
            },
            0,
            img.0.storage.alloc().clone(),
        )
        .unwrap();
        resize_fast(&img, &mut resized, InterpolationMode::Bilinear).unwrap();
        resized
        // }
    };
    let global_img = {
        let (width, height) = (img.width() as u32, img.height() as u32);
        println!("Resized image size: {}x{}", width, height);

        let longest_edge = width.max(height);
        if longest_edge <= outer_patch_size {
            img.clone()
        } else {
            let scale_factor = outer_patch_size as f32 / longest_edge as f32;
            let new_width = (width as f32 * scale_factor) as usize;
            let new_height = (height as f32 * scale_factor) as usize;
            let mut resized = Image::<u8, 3, _>::from_size_val(
                ImageSize {
                    width: new_width,
                    height: new_height,
                },
                0,
                img.0.storage.alloc().clone(),
            )
            .unwrap();
            resize_fast(&img, &mut resized, InterpolationMode::Bilinear).unwrap();
            resized
        }
    };

    // padding image for all dimensions to be multiples of the outer_patch_size
    let (img, mask) = {
        let (width, height) = (img.width(), img.height());
        let new_width = (width as u32).div_ceil(outer_patch_size) * outer_patch_size;
        let new_height = (height as u32).div_ceil(outer_patch_size) * outer_patch_size;
        let mut padded_img = Image::<u8, 3, _>::from_size_val(
            ImageSize {
                width: new_width as usize,
                height: new_height as usize,
            },
            0,
            CpuAllocator,
        )
        .unwrap();
        let mut padded_mask = Image::<u8, 1, _>::from_size_val(
            ImageSize {
                width: new_width as usize,
                height: new_height as usize,
            },
            0,
            CpuAllocator,
        )
        .unwrap();
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    padded_img
                        .set_pixel(x, y, c, *img.get_pixel(x, y, c).unwrap())
                        .unwrap();
                }
                padded_mask.set_pixel(x, y, 0, 255).unwrap();
            }
        }
        (padded_img, padded_mask)
    };
    let (global_img, global_mask) = {
        let (width, height) = (global_img.width(), global_img.height());
        println!("Global image size: {}x{}", width, height);

        let new_width = (width as u32).div_ceil(outer_patch_size) * outer_patch_size;
        let new_height = (height as u32).div_ceil(outer_patch_size) * outer_patch_size;
        let mut padded_img = Image::<u8, 3, _>::from_size_val(
            ImageSize {
                width: new_width as usize,
                height: new_height as usize,
            },
            0,
            CpuAllocator,
        )
        .unwrap();
        let mut padded_mask = Image::<u8, 1, _>::from_size_val(
            ImageSize {
                width: new_width as usize,
                height: new_height as usize,
            },
            0,
            CpuAllocator,
        )
        .unwrap();
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    padded_img
                        .set_pixel(x, y, c, *global_img.get_pixel(x, y, c).unwrap())
                        .unwrap();
                }
                padded_mask.set_pixel(x, y, 0, 255).unwrap();
            }
        }
        (padded_img, padded_mask)
    };

    // Debug output before tensor operations
    println!("=== DEBUG: Before Tensor Operations ===");
    println!("img: {}x{} channels", img.width(), img.height());
    println!("mask: {}x{} channels", mask.width(), mask.height());
    println!(
        "global_img: {}x{} channels",
        global_img.width(),
        global_img.height()
    );
    println!(
        "global_mask: {}x{} channels",
        global_mask.width(),
        global_mask.height()
    );

    // Sample some pixel values for debugging
    if img.width() > 0 && img.height() > 0 {
        println!(
            "img sample pixels (0,0): R={}, G={}, B={}",
            img.get_pixel(0, 0, 0).unwrap_or(&0),
            img.get_pixel(0, 0, 1).unwrap_or(&0),
            img.get_pixel(0, 0, 2).unwrap_or(&0)
        );
    }
    if global_img.width() > 0 && global_img.height() > 0 {
        println!(
            "global_img sample pixels (0,0): R={}, G={}, B={}",
            global_img.get_pixel(0, 0, 0).unwrap_or(&0),
            global_img.get_pixel(0, 0, 1).unwrap_or(&0),
            global_img.get_pixel(0, 0, 2).unwrap_or(&0)
        );
    }
    if mask.width() > 0 && mask.height() > 0 {
        println!(
            "mask sample pixel (0,0): {}",
            mask.get_pixel(0, 0, 0).unwrap_or(&0)
        );
    }
    if global_mask.width() > 0 && global_mask.height() > 0 {
        println!(
            "global_mask sample pixel (0,0): {}",
            global_mask.get_pixel(0, 0, 0).unwrap_or(&0)
        );
    }
    println!("=========================================");
    // Save debug images to .vscode folder for visual inspection
    println!("=== Saving debug images to .vscode folder ===");

    // Create .vscode directory if it doesn't exist
    std::fs::create_dir_all(".vscode").unwrap_or_else(|e| {
        println!("Warning: Could not create .vscode directory: {}", e);
    });

    // Save img as PPM (simple format)
    if let Err(e) = save_image_as_ppm(&img, ".vscode/debug_img.ppm") {
        println!("Failed to save img: {}", e);
    } else {
        println!("Saved img to .vscode/debug_img.ppm");
    }

    // Save global_img as PPM
    if let Err(e) = save_image_as_ppm(&global_img, ".vscode/debug_global_img.ppm") {
        println!("Failed to save global_img: {}", e);
    } else {
        println!("Saved global_img to .vscode/debug_global_img.ppm");
    } // Save mask as PPM (convert to RGB for visibility)
    if let Err(e) = save_mask_as_ppm(&mask, ".vscode/debug_mask.ppm") {
        println!("Failed to save mask: {}", e);
    } else {
        println!("Saved mask to .vscode/debug_mask.ppm");
    }

    // Save global_mask as PPM
    if let Err(e) = save_mask_as_ppm(&global_mask, ".vscode/debug_global_mask.ppm") {
        println!("Failed to save global_mask: {}", e);
    } else {
        println!("Saved global_mask to .vscode/debug_global_mask.ppm");
    }
    println!("================================================");

    let img = {
        let (width, height) = (img.width(), img.height());
        let img_data: Vec<u8> = img.as_slice().to_vec();
        Tensor::from_vec(img_data, Shape::from_dims(&[height, width, 3]), device)
            .unwrap()
            .permute(vec![2, 0, 1])
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
    };
    let mask = {
        let (width, height) = (mask.width(), mask.height());
        let img_data: Vec<u8> = mask.as_slice().to_vec();
        Tensor::from_vec(img_data, Shape::from_dims(&[height, width]), device)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
    };
    let global_img = {
        let (width, height) = (global_img.width(), global_img.height());
        let img_data: Vec<u8> = global_img.as_slice().to_vec();
        Tensor::from_vec(img_data, Shape::from_dims(&[height, width, 3]), device)
            .unwrap()
            .permute(vec![2, 0, 1])
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
    };
    let global_mask = {
        let (width, height) = (global_mask.width(), global_mask.height());
        let img_data: Vec<u8> = global_mask.as_slice().to_vec();
        Tensor::from_vec(img_data, Shape::from_dims(&[height, width]), device)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
    };

    // rescaling and normalizing
    let img = {
        let mut img = (img / 255.0).unwrap();
        let m = Tensor::from_slice(&MEAN, (3, 1, 1), device).unwrap();
        let s = Tensor::from_slice(&STD, (3, 1, 1), device).unwrap();

        img = img.broadcast_sub(&m).unwrap();
        img = img.broadcast_div(&s).unwrap();

        img
    };
    let global_img = {
        let mut global_img = (global_img / 255.0).unwrap();
        let m = Tensor::from_slice(&MEAN, (3, 1, 1), device).unwrap();
        let s = Tensor::from_slice(&STD, (3, 1, 1), device).unwrap();

        global_img = global_img.broadcast_sub(&m).unwrap();
        global_img = global_img.broadcast_div(&s).unwrap();

        global_img
    };

    let (c, h, w) = img.dims3().unwrap();
    let cols = w / outer_patch_size as usize;
    let rows = h / outer_patch_size as usize;

    // splitting
    let img_patches = {
        img.unsqueeze(2)
            .unwrap()
            .unsqueeze(4)
            .unwrap()
            .reshape(&[
                c,
                rows,
                outer_patch_size as usize,
                cols,
                outer_patch_size as usize,
            ])
            .unwrap()
            .permute([1, 3, 0, 2, 4])
            .unwrap()
            .reshape(&[
                rows * cols,
                c,
                outer_patch_size as usize,
                outer_patch_size as usize,
            ])
            .unwrap()
    };
    let mask_patches = {
        mask.unsqueeze(1)
            .unwrap()
            .unsqueeze(3)
            .unwrap()
            .reshape(&[
                rows,
                outer_patch_size as usize,
                cols,
                outer_patch_size as usize,
            ])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap()
            .reshape(&[
                rows * cols,
                outer_patch_size as usize,
                outer_patch_size as usize,
            ])
            .unwrap()
    };

    // concatenating global image
    let img_patches = Tensor::cat(&[&img_patches, &global_img.unsqueeze(0).unwrap()], 0).unwrap();
    let mask_patches =
        Tensor::cat(&[&mask_patches, &global_mask.unsqueeze(0).unwrap()], 0).unwrap();

    (
        img_patches,
        mask_patches,
        ImageSize {
            width: cols,
            height: rows,
        },
    )
}

pub fn get_prompt_split_image(img_seq_len: usize, size: ImageSize) -> String {
    let mut s = String::new();

    for h in 0..size.height {
        for w in 0..size.width {
            s += &format!(
                "<fake_token_around_image><row_{}_col_{}>{}",
                h + 1,
                w + 1,
                "<image>".repeat(img_seq_len)
            );
        }
        s += "\n";
    }

    s += &format!(
        "\n<fake_token_around_image><global-img>{}<fake_token_around_image>",
        "<image>".repeat(img_seq_len)
    );

    s
}

// Helper functions for saving debug images
fn save_image_as_ppm<A: ImageAllocator>(
    img: &Image<u8, 3, A>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (img.width(), img.height());
    let mut content = format!("P3\n{} {}\n255\n", width, height);

    for y in 0..height {
        for x in 0..width {
            let r = img.get_pixel(x, y, 0).unwrap_or(&0);
            let g = img.get_pixel(x, y, 1).unwrap_or(&0);
            let b = img.get_pixel(x, y, 2).unwrap_or(&0);
            content.push_str(&format!("{} {} {} ", r, g, b));
        }
        content.push('\n');
    }

    std::fs::write(path, content)?;
    Ok(())
}

fn save_mask_as_ppm<A: ImageAllocator>(
    mask: &Image<u8, 1, A>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (mask.width(), mask.height());
    let mut content = format!("P3\n{} {}\n255\n", width, height);

    for y in 0..height {
        for x in 0..width {
            let val = mask.get_pixel(x, y, 0).unwrap_or(&0);
            // Convert grayscale to RGB (white for valid pixels, black for padding)
            content.push_str(&format!("{} {} {} ", val, val, val));
        }
        content.push('\n');
    }

    std::fs::write(path, content)?;
    Ok(())
}
