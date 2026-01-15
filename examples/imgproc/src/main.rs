use argh::FromArgs;
use kornia::tensor::CpuAllocator;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{
    image::{ops, Image, ImageSize},
    imgproc,
};

#[derive(FromArgs)]
/// Perform basic image processing and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image = F::read_image_any_rgb8(args.image_path)?;

    // convert the image to grayscale
    let mut gray = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;
    imgproc::color::gray_from_rgb_u8(&image, &mut gray)?;

    // convert to float
    let mut gray_f32 = Image::<f32, 1, _>::from_size_val(gray.size(), 0.0, CpuAllocator)?;
    ops::cast_and_scale(&gray, &mut gray_f32, 1.0 / 255.0)?;

    let new_size = ImageSize {
        width: 512,
        height: 512,
    };

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the original images
    rec.log(
        "image/original",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "gray/original",
        &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
    )?;

    // Nearest neighbor
    let mut gray_nearest = Image::<f32, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;
    imgproc::resize::resize_native(
        &gray_f32,
        &mut gray_nearest,
        imgproc::interpolation::InterpolationMode::Nearest,
    )?;

    rec.log(
        "gray/nearest",
        &rerun::Image::from_elements(
            gray_nearest.as_slice(),
            gray_nearest.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    // Bilinear
    let mut gray_bilinear = Image::<f32, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;
    imgproc::resize::resize_native(
        &gray_f32,
        &mut gray_bilinear,
        imgproc::interpolation::InterpolationMode::Bilinear,
    )?;

    rec.log(
        "gray/bilinear",
        &rerun::Image::from_elements(
            gray_bilinear.as_slice(),
            gray_bilinear.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    // Bicubic
    let mut gray_bicubic = Image::<f32, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;
    imgproc::resize::resize_native(
        &gray_f32,
        &mut gray_bicubic,
        imgproc::interpolation::InterpolationMode::Bicubic,
    )?;

    rec.log(
        "gray/bicubic",
        &rerun::Image::from_elements(
            gray_bicubic.as_slice(),
            gray_bicubic.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    Ok(())
}
