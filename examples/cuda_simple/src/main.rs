use anyhow::Result;
use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use std::time::Instant;

// kernel to convert image to grayscale
const PTX_SRC: &str = "
extern \"C\" __global__ void rgb_to_gray(const float* rgb, float* gray, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = rgb[3 * i + 0];
        float g = rgb[3 * i + 1];
        float b = rgb[3 * i + 2];
        float gray_val = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[3 * i + 0] = gray_val;
        gray[3 * i + 1] = gray_val;
        gray[3 * i + 2] = gray_val;
    }
}
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = std::path::Path::new("../../tests/data/dog.jpeg");
    let img = F::read_image_jpeg(&image_path)?;
    let img = img.cast::<f32>()?;

    //let img_gray = Image::<u8, 1>::from_size_val(img.size(), 0u8)?;

    let ptx = compile_ptx(PTX_SRC)?;

    let dev = CudaDevice::new(0)?;

    dev.load_ptx(ptx, "kernel", &["rgb_to_gray"])?;
    let f = dev
        .get_func("kernel", "rgb_to_gray")
        .expect("Kernel not found");

    let t0 = Instant::now();

    // copy image to device
    let img_rgb_host = img.data.as_slice().expect("Image data is not contiguous");
    let mut img_gray_host = vec![0f32; img.data.len()];
    let img_rgb_dev = dev.htod_sync_copy(&img_rgb_host)?;
    let mut img_gray_dev = dev.htod_sync_copy(&img_gray_host)?;

    // launch kernel
    let n = img_gray_host.len() as u32;
    //let cfg = LaunchConfig::for_num_elems(n);
    let cfg = LaunchConfig {
        grid_dim: (n / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&img_rgb_dev, &mut img_gray_dev, n))? };

    //dev.dtoh_sync_copy_into(&img_gray_dev, &mut img_gray_host)?;
    //let img_gray_host = dev.sync_reclaim(img_gray_dev)?;
    dev.dtoh_sync_copy_into(&img_gray_dev, &mut img_gray_host)?;

    let img_gray = Image::<f32, 3>::new(img.size(), img_gray_host)?;

    let img2 = img.clone();
    let t1 = Instant::now();

    let img_gray2 = kornia_rs::color::gray_from_rgb(&img2)?;

    let t2 = Instant::now();

    println!("Elapsed time: {:?}", t1 - t0); // 4.9022ms
    println!("Elapsed time: {:?}", t2 - t1); // 5.3233ms

    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;
    rec.log("image", &rerun::Image::try_from(img.data)?)?;
    rec.log("image_gray", &rerun::Image::try_from(img_gray.data)?)?;

    Ok(())
}
