use argh::FromArgs;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{
    image::{ops, Image},
    imgproc,
};

#[derive(FromArgs)]
/// Compute the distance transform of an image and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image = F::read_image_any_rgb8(args.image_path)?;

    // convert the image to grayscale
    let mut gray = Image::<u8, 1>::from_size_val(image.size(), 0)?;
    imgproc::color::gray_from_rgb_u8(&image, &mut gray)?;

    // convert to float
    let mut gray_f32 = Image::<f32, 1>::from_size_val(gray.size(), 0.0)?;
    ops::cast_and_scale(&gray, &mut gray_f32, 1.0 / 255.0)?;

    // binarize: pixels brighter than 0.5 are foreground
    let mut mask = Image::<f32, 1>::from_size_val(gray_f32.size(), 0.0)?;
    imgproc::threshold::threshold_binary(&gray_f32, &mut mask, 0.5, 1.0)?;

    // distance from every pixel to the nearest foreground pixel
    let mut executor = imgproc::distance_transform::DistanceTransformExecutor::new();
    let distance = executor.execute(&mask)?;

    println!("distance: {:?}", distance.size());

    // scale the distances to [0, 1] for display
    let max = distance.as_slice().iter().cloned().fold(1.0f32, f32::max);
    let distance_vis: Vec<f32> = distance.as_slice().iter().map(|d| d / max).collect();

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the images
    rec.log(
        "gray",
        &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "mask",
        &rerun::Image::from_elements(mask.as_slice(), mask.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "distance",
        &rerun::Image::from_elements(&distance_vis, distance.size().into(), rerun::ColorModel::L),
    )?;

    Ok(())
}
