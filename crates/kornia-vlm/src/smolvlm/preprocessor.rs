use candle_core::{Device, Shape, Tensor};

use kornia_io::png::{write_image_png_gray8, write_image_png_rgb8};

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast;
use kornia_tensor::allocator::CpuAllocator;

// ImageNet mean and std for normalization
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub fn preprocess_image(
    img: Image<u8, 3, CpuAllocator>,
    max_size: u32,
    outer_patch_size: u32,
    device: &Device,
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
                ImageSize {
                    width: new_width,
                    height: new_height,
                },
                0,
                CpuAllocator,
            )
            .unwrap();
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
                ImageSize {
                    width: new_width,
                    height: new_height,
                },
                0,
                CpuAllocator,
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

    (img_patches, mask_patches, cols, rows)
}

pub fn get_prompt_split_image(img_seq_len: usize, img_rows: usize, img_cols: usize) -> String {
    let mut s = String::new();

    for h in 0..img_rows {
        for w in 0..img_cols {
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
