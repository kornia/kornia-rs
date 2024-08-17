use clap::Parser;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{image::Image, imgproc};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;

    // binarize the image as u8
    let mut bin = Image::<u8, 3>::from_size_val(image.size(), 0)?;
    imgproc::threshold::threshold_binary(&image, &mut bin, 127, 255)?;

    // normalize the image between 0 and 1
    let image_f32: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert to grayscale as floating point
    let mut gray = Image::<f32, 1>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    // binarize the gray image as floating point
    let mut gray_bin = Image::<f32, 1>::from_size_val(gray.size(), 0.0)?;
    imgproc::threshold::threshold_binary(&gray, &mut gray_bin, 0.5, 1.0)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log("image", &rerun::Image::try_from(image_f32.data)?)?;
    rec.log("gray", &rerun::Image::try_from(gray.data)?)?;
    rec.log("gray_bin", &rerun::Image::try_from(gray_bin.data)?)?;

    Ok(())
}
