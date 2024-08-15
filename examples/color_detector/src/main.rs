use clap::Parser;
use kornia::image::Image;
use std::path::PathBuf;

use kornia::imgproc;
use kornia::io::functional as F;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let rgb = F::read_image_any(&args.image_path)?;

    // binarize the image as u8
    let mut hsv = Image::<f32, 3>::from_size_val(rgb.size(), 0.0)?;
    imgproc::color::hsv_from_rgb(&rgb.clone().cast()?, &mut hsv)?; // convert to u8 (0-255)

    // create the mask for the green color
    let mut mask = Image::<u8, 1>::from_size_val(hsv.size(), 0)?;
    imgproc::threshold::in_range(&hsv, &mut mask, &[40.0, 110.0, 50.0], &[90.0, 255.0, 255.0])?;

    // apply the mask to the image
    let mut out = Image::<u8, 3>::from_size_val(mask.size(), 0)?;
    imgproc::core::bitwise_and(&rgb, &rgb, &mut out, &mask)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log("rgb", &rerun::Image::try_from(rgb.data)?)?;
    rec.log("hsv", &rerun::Image::try_from(hsv.data)?)?;
    rec.log("mask", &rerun::Image::try_from(mask.data)?)?;
    rec.log("output", &rerun::Image::try_from(out.data)?)?;

    Ok(())
}
