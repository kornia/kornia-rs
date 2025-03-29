use argh::FromArgs;
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
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    // convert the image to grayscale
    let mut gray = Image::<u8, 1>::from_size_val(image.size(), 0)?;
    imgproc::color::gray_from_rgb_u8(&image, &mut gray)?;

    // convert to float
    let mut gray_f32 = Image::<f32, 1>::from_size_val(gray.size(), 0.0)?;
    ops::cast_and_scale(&gray, &mut gray_f32, 1.0 / 255.0)?;

    let new_size = ImageSize {
        width: 128,
        height: 128,
    };

    let mut gray_resize = Image::<f32, 1>::from_size_val(new_size, 0.0)?;
    imgproc::resize::resize_native(
        &gray_f32,
        &mut gray_resize,
        imgproc::interpolation::InterpolationMode::Bilinear,
    )?;

    println!("gray_resize: {:?}", gray_resize.size());

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
        "gray",
        &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "gray_resize",
        &rerun::Image::from_elements(
            gray_resize.as_slice(),
            gray_resize.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    Ok(())
}
