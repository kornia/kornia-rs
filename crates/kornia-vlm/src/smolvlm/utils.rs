use candle_core::{Device, Shape, Tensor};

use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage, GenericImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage};
use reqwest;
use std::error::Error;
use std::fs;
use std::path::Path;

use kornia_io::png::read_image_png_rgb8;
use kornia_image::Image;
use kornia_tensor::allocator::CpuAllocator;



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
        // Use kornia_io to read the PNG file
        let img = read_image_png_rgb8(&file_path)?;
        println!("Loaded image from local cache.");
        return Ok(img);
    }

    // If the file does not exist, download it and save it
    println!("Downloading image from URL...");

    // Fetch the image as bytes
    let response = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&file_path, &response)?;

    // Use kornia_io to read the PNG file
    let img = read_image_png_rgb8(&file_path)?;
    println!("Saved image locally as {}", file_path.to_str().unwrap());
    Ok(img)
}

pub fn preprocess_image(
    img: Image<u8, 3, CpuAllocator>, max_size: u32, outer_patch_size: u32, device: &Device
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


    img.save(".vscode/padded_img.png").unwrap();
    mask.save(".vscode/mask.png").unwrap();
    global_img.save(".vscode/global_padded_img.png").unwrap();
    global_mask.save(".vscode/global_mask.png").unwrap();

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
