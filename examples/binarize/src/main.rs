use argh::FromArgs;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{image::Image, imgproc};

#[derive(FromArgs)]
/// Binary threshold an image and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    // binarize the image as u8
    let mut bin = Image::<u8, 3>::from_size_val(image.size(), 0)?;
    imgproc::threshold::threshold_binary(&image, &mut bin, 127, 255)?;

    // normalize the image between 0 and 1
    let image_f32 = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert to grayscale as floating point
    let mut gray = Image::<f32, 1>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    // binarize the gray image as floating point
    let mut gray_bin = Image::<f32, 1>::from_size_val(gray.size(), 0.0)?;
    imgproc::threshold::threshold_binary(&gray, &mut gray_bin, 0.5, 1.0)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "image",
        &rerun::Image::from_elements(
            image_f32.as_slice(),
            image_f32.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "gray",
        &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "gray_bin",
        &rerun::Image::from_elements(
            gray_bin.as_slice(),
            gray_bin.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    Ok(())
}
