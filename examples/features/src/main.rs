use argh::FromArgs;
use std::path::PathBuf;

use kornia::{
    image::{ops, Image},
    imgproc,
    io::functional as F,
    tensor::CpuAllocator,
};

#[derive(FromArgs)]
/// Compute Hessian and GFTT responses and log to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let img_rgb = F::read_image_any_rgb8(args.image_path)?;
    let size = img_rgb.size();

    // preallocate images
    let mut img_f32 = Image::from_size_val(size, 0f32, CpuAllocator)?;
    let mut gray = Image::from_size_val(size, 0f32, CpuAllocator)?;
    let mut gftt_response = Image::from_size_val(size, 0f32, CpuAllocator)?;
    let mut hessian_response = Image::from_size_val(size, 0f32, CpuAllocator)?;
    let mut gftt_corners = Image::from_size_val(size, 0f32, CpuAllocator)?;
    let mut hessian_corners = Image::from_size_val(size, 0f32, CpuAllocator)?;

    // convert to gray scale
    ops::cast_and_scale(&img_rgb, &mut img_f32, 1.0 / 255.0)?;
    imgproc::color::gray_from_rgb(&img_f32, &mut gray)?;

    // compute hessian response
    imgproc::features::hessian_response(&gray, &mut hessian_response)?;
    imgproc::threshold::threshold_binary(&hessian_response, &mut hessian_corners, 0.01, 1.0)?;

    // compute gftt response
    imgproc::features::gftt_response(&gray, &mut gftt_response, 3, 5, true)?;
    imgproc::threshold::threshold_binary(&gftt_response, &mut gftt_corners, 0.01, 1.0)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Feature App").spawn()?;

    rec.log(
        "image",
        &rerun::Image::from_elements(
            img_rgb.as_slice(),
            img_rgb.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "corners/hessian",
        &rerun::Image::from_elements(
            hessian_corners.as_slice(),
            hessian_corners.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    rec.log(
        "corners/gftt",
        &rerun::Image::from_elements(
            gftt_corners.as_slice(),
            gftt_corners.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    Ok(())
}
