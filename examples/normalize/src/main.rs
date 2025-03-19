use argh::FromArgs;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{image::Image, imgproc};

#[derive(FromArgs)]
/// Normalize an image and print the min and max values.
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
    let image_f32 = image.clone().cast::<f32>()?;

    // normalize the image between 0 and 255
    let mut image_f32_norm = Image::from_size_val(image_f32.size(), 0.0)?;
    imgproc::normalize::normalize_mean_std(
        &image_f32,
        &mut image_f32_norm,
        &[127.5, 127.5, 127.5],
        &[127.5, 127.5, 127.5],
    )?;

    // alternative way to normalize the image between 0 and 255
    let _image_f32_norm = image.clone().cast_and_scale::<f32>(1.0 / 255.0)?;
    // Or: image.cast_and_scale::<f64>(1.0 / 255.0)?;

    let (min, max) = imgproc::normalize::find_min_max(&image_f32)?;
    println!("min: {:?}, max: {:?}", min, max);

    Ok(())
}
