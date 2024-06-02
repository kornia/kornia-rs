use clap::Parser;

use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let rgb = kornia_rs::io::functional::read_image_any(&args.image_path)?;

    // binarize the image as u8
    let hsv = kornia_rs::color::hsv_from_rgb(&rgb.clone().cast()?)?; // convert to u8 (0-255)

    // create the mask for the green color
    let mask = kornia_rs::threshold::in_range(&hsv, &[40.0, 110.0, 50.0], &[90.0, 255.0, 255.0])?;

    // apply the mask to the image
    let output = kornia_rs::core::bitwise_and(&rgb, &rgb, &mask)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log("rgb", &rerun::Image::try_from(rgb.data)?)?;
    rec.log("hsv", &rerun::Image::try_from(hsv.clone().data)?)?;
    rec.log("mask", &rerun::Image::try_from(mask.clone().data)?)?;
    rec.log("output", &rerun::Image::try_from(output.clone().data)?)?;

    Ok(())
}
