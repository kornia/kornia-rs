use anyhow::Result;
use kornia_rs::cuda::{CudaImage, KorniaScript};
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
    let dev = CudaDevice::new(0)?;

    let img_cuda = CudaImage::from_host(img.clone(), dev.clone())?;
    let mut gray_cuda = img_cuda.zeros_like()?;

    let gray_from_rgb_script = KorniaScript::new(dev.clone(), PTX_SRC, "rgb_to_gray")?;

    for i in 0..100 {
        let t0 = Instant::now();
        let img_gray_cuda = gray_from_rgb_script.run(&img_cuda, &mut gray_cuda)?;

        let t1 = Instant::now();

        let img2 = img.clone();
        let img_gray2 = kornia_rs::color::gray_from_rgb(&img2)?;

        let t2 = Instant::now();

        println!("Elapsed time: {:?}", t1 - t0); // 4.9022ms
        println!("Elapsed time: {:?}", t2 - t1); // 5.3233ms
    }

    //let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;
    //rec.log("image", &rerun::Image::try_from(img.data)?)?;
    //rec.log(
    //    "image_gray",
    //    &rerun::Image::try_from(img_gray_cuda.to_host()?.data)?,
    //)?;

    Ok(())
}
