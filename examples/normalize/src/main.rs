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
    let image_f32: Image<f32, 3> = image.clone().cast::<f32>()?;

    // normalize the image between 0 and 255
    let _image_f32_norm: Image<f32, 3> = imgproc::normalize::normalize_mean_std(
        &image_f32,
        &[127.5, 127.5, 127.5],
        &[127.5, 127.5, 127.5],
    )?;

    // alternative way to normalize the image between 0 and 255
    let _image_f32_norm: Image<f32, 3> = image.clone().cast_and_scale::<f32>(1.0 / 255.0)?;
    // Or: image.cast_and_scale::<f64>(1.0 / 255.0)?;

    // alternative way to normalize the image between 0 and 1
    let _image_f32_norm: Image<f32, 3> = image_f32.mul(1.0 / 255.0);
    // Or: let image_f32_norm = image_f32.div(255.0);

    let (min, max) = imgproc::normalize::find_min_max(&image_f32)?;
    println!("min: {:?}, max: {:?}", min, max);

    Ok(())
}
