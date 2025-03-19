use argh::FromArgs;
use std::path::PathBuf;

use kornia::image::{ops, Image};
use kornia::imgproc;
use kornia::io::functional as F;

#[derive(FromArgs)]
/// Segment the green color in an image and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let rgb = F::read_image_any_rgb8(args.image_path)?;

    // cast the image to f32
    let mut rgb_f32 = Image::<f32, 3>::from_size_val(rgb.size(), 0.0)?;
    ops::cast_and_scale(&rgb, &mut rgb_f32, 1.0)?;

    // binarize the image as u8
    let mut hsv = Image::<f32, 3>::from_size_val(rgb.size(), 0.0)?;
    imgproc::color::hsv_from_rgb(&rgb_f32, &mut hsv)?; // convert to u8 (0-255)

    // create the mask for the green color
    let mut mask = Image::<u8, 1>::from_size_val(hsv.size(), 0)?;
    imgproc::threshold::in_range(&hsv, &mut mask, &[40.0, 110.0, 50.0], &[90.0, 255.0, 255.0])?;

    // apply the mask to the image
    let mut out = Image::<u8, 3>::from_size_val(mask.size(), 0)?;
    imgproc::core::bitwise_and(&rgb, &rgb, &mut out, &mask)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "rgb",
        &rerun::Image::from_elements(rgb.as_slice(), rgb.size().into(), rerun::ColorModel::RGB),
    )?;

    rec.log(
        "hsv",
        &rerun::Image::from_elements(hsv.as_slice(), hsv.size().into(), rerun::ColorModel::RGB),
    )?;

    rec.log(
        "mask",
        &rerun::Image::from_elements(mask.as_slice(), mask.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "output",
        &rerun::Image::from_elements(out.as_slice(), out.size().into(), rerun::ColorModel::RGB),
    )?;

    Ok(())
}
