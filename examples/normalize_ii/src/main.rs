use argh::FromArgs;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{image::Image, imgproc};

#[derive(FromArgs)]
/// Normalize an image using the mean and STD; and print the min and max values.
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    // cast the image to floating point
    let image_f32: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert to grayscale
    let mut gray = Image::<f32, 1>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    // normalize the image each channel
    let mut image_norm: Image<f32, 3> = Image::from_size_val(image_f32.size(), 0.0)?;
    imgproc::normalize::normalize_mean_std(
        &image_f32,
        &mut image_norm,
        &[0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5],
    )?;

    // normalize the grayscale image
    let mut gray_norm = Image::<f32, 1>::from_size_val(gray.size(), 0.0)?;
    imgproc::normalize::normalize_mean_std(&gray, &mut gray_norm, &[0.5], &[0.5])?;

    // alternative way to normalize the image between 0 and 255
    // let mut gray_norm_min_max = Image::<f32, 1>::from_size_val(gray.size(), 0.0)?;
    // imgproc::normalize::normalize_min_max(&gray, &mut gray_norm_min_max, 0.0, 255.0)?;

    let (min, max) = imgproc::normalize::find_min_max(&gray)?;
    println!("min: {:?}, max: {:?}", min, max);

    Ok(())
}
