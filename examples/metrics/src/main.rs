use clap::Parser;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{
    image::{ops, Image},
    imgproc,
};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let image: Image<u8, 3> = F::read_image_any(args.image_path)?;

    // convert the image to f32 and scale it
    let mut image_f32 = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
    ops::cast_and_scale(&image, &mut image_f32, 1.0 / 255.0)?;

    // modify the image to see the changes
    let mut image_dirty = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
    imgproc::flip::horizontal_flip(&image_f32, &mut image_dirty)?;

    // compute the mean squared error (mse) between the original and the modified image
    let mse = imgproc::metrics::mse(&image_f32, &image_dirty)?;
    let psnr = imgproc::metrics::psnr(&image_f32, &image_dirty, 1.0)?;

    // print the mse error
    println!("MSE error: {:?}", mse);
    println!("PSNR error: {:?}", psnr);

    // or, alternatively, compute the mse using the built-in functions
    let mse_map = image_f32.sub(&image_dirty).powi(2);
    // let mse_ii = mse_map_ii.mean();

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the images
    rec.log(
        "image",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "flip",
        &rerun::Image::from_elements(
            image_dirty.as_slice(),
            image_dirty.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "mse_map",
        &rerun::Image::from_elements(
            mse_map.as_slice(),
            [mse_map.shape[1] as u32, mse_map.shape[0] as u32],
            rerun::ColorModel::RGB,
        ),
    )?;

    Ok(())
}
