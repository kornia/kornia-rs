use clap::Parser;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{
    image::{ops, Image, ImageSize},
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
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    // convert the image to f32 and scale it
    let mut image_f32 = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
    ops::cast_and_scale(&image, &mut image_f32, 1.0 / 255.0)?;

    // convert the image to grayscale
    let mut gray = Image::<f32, 1>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    let new_size = ImageSize {
        width: 128,
        height: 128,
    };

    let mut gray_resize = Image::<f32, 1>::from_size_val(new_size, 0.0)?;
    imgproc::resize::resize_native(
        &gray,
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
