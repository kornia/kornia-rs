use candle_core::{Device, Shape, Tensor};

use reqwest;
use std::error::Error;
use std::fs;
use std::path::Path;

use kornia_io::png::{read_image_png_rgb8, write_image_png_rgb8, write_image_png_gray8};
use kornia_io::jpeg::{read_image_jpeg_rgb8};
use kornia_image::{Image, ImageSize};
use kornia_tensor::allocator::CpuAllocator;
use kornia_imgproc::resize::{resize_fast};
use kornia_imgproc::interpolation::InterpolationMode;



// ImageNet mean and std for normalization
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];



pub fn load_image_url(url: &str) -> std::result::Result<Image<u8, 3, CpuAllocator>, Box<dyn Error>> {
    let dir = Path::new(".vscode");

    let file_path = {
        let parsed_url = reqwest::Url::parse(url)?;
        let path = parsed_url.path();
        dir.join(
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown_file.png") // Use PNG as default
                .to_string()
        )
    };

    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    }
    
    // Check if the file exists locally
    if file_path.exists() {
        // Use kornia_io to read the JPEG file
        let img = read_image_jpeg_rgb8(&file_path)?;
        println!("Loaded image from local cache.");
        return Ok(img);
    }

    // If the file does not exist, download it and save it
    println!("Downloading image from URL...");

    // Fetch the image as bytes
    let response = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&file_path, &response)?;

    // Use kornia_io to read the JPEG file
    let img = read_image_jpeg_rgb8(&file_path)?;
    println!("Saved image locally as {}", file_path.to_str().unwrap());
    Ok(img)
}

pub fn preprocess_image(
    img: Image<u8, 3, CpuAllocator>, max_size: u32, outer_patch_size: u32, device: &Device
) -> (Tensor, Tensor, usize, usize) {

    // resizing image to match the max_size (on the longest edge)
    let img = {
        let (width, height) = (img.width() as u32, img.height() as u32);
        let longest_edge = width.max(height);
        if longest_edge <= max_size {
            img.clone()
        } else {
            let scale_factor = max_size as f32 / longest_edge as f32;
            let new_width = (width as f32 * scale_factor) as usize;
            let new_height = (height as f32 * scale_factor) as usize;
            let mut resized = Image::<u8, 3, _>::from_size_val(
                ImageSize { width: new_width, height: new_height }, 0, CpuAllocator
            ).unwrap();
            resize_fast(&img, &mut resized, InterpolationMode::Bilinear).unwrap();
            resized
        }
    };
    let global_img = {
        let (width, height) = (img.width() as u32, img.height() as u32);
        let longest_edge = width.max(height);
        if longest_edge <= outer_patch_size {
            img.clone()
        } else {
            let scale_factor = outer_patch_size as f32 / longest_edge as f32;
            let new_width = (width as f32 * scale_factor) as usize;
            let new_height = (height as f32 * scale_factor) as usize;
            let mut resized = Image::<u8, 3, _>::from_size_val(
                ImageSize { width: new_width, height: new_height }, 0, CpuAllocator
            ).unwrap();
            resize_fast(&img, &mut resized, InterpolationMode::Bilinear).unwrap();
            resized
        }
    };

    // padding image for all dimensions to be multiples of the outer_patch_size
    let (img, mask) = {
        let (width, height) = (img.width(), img.height());
        let new_width = ((width as u32 + outer_patch_size - 1) / outer_patch_size) * outer_patch_size;
        let new_height = ((height as u32 + outer_patch_size - 1) / outer_patch_size) * outer_patch_size;
        let mut padded_img = Image::<u8, 3, _>::from_size_val(
            ImageSize { width: new_width as usize, height: new_height as usize }, 0, CpuAllocator
        ).unwrap();
        let mut padded_mask = Image::<u8, 1, _>::from_size_val(
            ImageSize { width: new_width as usize, height: new_height as usize }, 0, CpuAllocator
        ).unwrap();
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    padded_img.set_pixel(x, y, c, *img.get_pixel(x, y, c).unwrap()).unwrap();
                }
                padded_mask.set_pixel(x, y, 0, 255).unwrap();
            }
        }
        (padded_img, padded_mask)
    };
    let (global_img, global_mask) = {
        let (width, height) = (global_img.width(), global_img.height());
        let new_width = ((width as u32 + outer_patch_size - 1) / outer_patch_size) * outer_patch_size;
        let new_height = ((height as u32 + outer_patch_size - 1) / outer_patch_size) * outer_patch_size;
        let mut padded_img = Image::<u8, 3, _>::from_size_val(
            ImageSize { width: new_width as usize, height: new_height as usize }, 0, CpuAllocator
        ).unwrap();
        let mut padded_mask = Image::<u8, 1, _>::from_size_val(
            ImageSize { width: new_width as usize, height: new_height as usize }, 0, CpuAllocator
        ).unwrap();
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    padded_img.set_pixel(x, y, c, *global_img.get_pixel(x, y, c).unwrap()).unwrap();
                }
                padded_mask.set_pixel(x, y, 0, 255).unwrap();
            }
        }
        (padded_img, padded_mask)
    };


    write_image_png_rgb8(".vscode/padded_img.png", &img).unwrap();
    write_image_png_gray8(".vscode/mask.png", &mask).unwrap();
    write_image_png_rgb8(".vscode/global_padded_img.png", &global_img).unwrap();
    write_image_png_gray8(".vscode/global_mask.png", &global_mask).unwrap();

    let img = {
        let (width, height) = (img.width(), img.height());
        let img_data: Vec<u8> = img.as_slice().to_vec();
        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize, 3]), device
        ).unwrap()
            .permute(vec![2, 0, 1]).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let mask = {
        let (width, height) = (mask.width(), mask.height());
        let img_data: Vec<u8> = mask.as_slice().to_vec();
        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize]), device
        ).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let global_img = {
        let (width, height) = (global_img.width(), global_img.height());
        let img_data: Vec<u8> = global_img.as_slice().to_vec();
        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize, 3]), device
        ).unwrap()
            .permute(vec![2, 0, 1]).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let global_mask = {
        let (width, height) = (global_mask.width(), global_mask.height());
        let img_data: Vec<u8> = global_mask.as_slice().to_vec();
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
