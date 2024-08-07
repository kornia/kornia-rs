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

    // cast the image to floating point
    let image_f32: Image<f32, 3> = image.clone().cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert to grayscale
    let gray_f32: Image<f32, 1> = imgproc::color::gray_from_rgb(&image_f32)?;

    // normalize the image each channel
    let _image_norm: Image<f32, 3> =
        imgproc::normalize::normalize_mean_std(&image_f32, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])?;

    // normalize the grayscale image
    let _gray_norm: Image<f32, 1> =
        imgproc::normalize::normalize_mean_std(&gray_f32, &[0.5], &[0.5])?;

    // alternative way to normalize the image between 0 and 255
    // let _gray_norm_min_max = imgproc::normalize::normalize_min_max(
    //     &gray, 0.0, 255.0)?;

    let (min, max) = imgproc::normalize::find_min_max(&gray_f32)?;
    println!("min: {:?}, max: {:?}", min, max);

    Ok(())
}
