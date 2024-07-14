use clap::Parser;
use std::path::PathBuf;

use kornia_rs::io::functional as F;
use kornia_rs::{image::Image, imgproc};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;

    // convert the image to f32 and scale it
    let image: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // modify the image to see the changes
    let image_dirty = imgproc::flip::horizontal_flip(&image)?;

    // compute the mean squared error (mse) between the original and the modified image
    let mse = imgproc::metrics::mse(&image, &image_dirty);
    let psnr = imgproc::metrics::psnr(&image, &image_dirty, 1.0);

    // print the mse error
    println!("MSE error: {:?}", mse);
    println!("PSNR error: {:?}", psnr);

    // or, alternatively, compute the mse using the built-in functions
    let mse_map = image.sub(&image_dirty).powi(2);
    // let mse_ii = mse_map_ii.mean();

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the images
    rec.log("image", &rerun::Image::try_from(image.data)?)?;
    rec.log("flip", &rerun::Image::try_from(image_dirty.data)?)?;
    rec.log("mse_map", &rerun::Image::try_from(mse_map.data)?)?;

    Ok(())
}
