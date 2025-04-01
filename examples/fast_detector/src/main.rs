use argh::FromArgs;
use std::path::PathBuf;

use kornia::{image::Image, imgproc, io::functional as F};

/// Detect FAST features on an image.
#[derive(FromArgs)]
struct Args {
    /// path to the image to detect FAST features on
    #[argh(option)]
    image_path: PathBuf,

    /// threshold for the FAST detector
    #[argh(option, default = "10")]
    threshold: u8,

    /// arc length for the FAST detector
    #[argh(option, default = "5")]
    arc_length: u8,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Fast Detector App").spawn()?;

    // read the image
    let img_rgb8 = F::read_image_any_rgb8(args.image_path)?;

    // convert to grayscale
    let mut img_gray8 = Image::from_size_val(img_rgb8.size(), 0u8)?;
    imgproc::color::gray_from_rgb_u8(&img_rgb8, &mut img_gray8)?;

    // detect the fast features
    let keypoints =
        imgproc::features::fast_feature_detector(&img_gray8, args.threshold, args.arc_length)?;
    println!("Found {} keypoints", keypoints.len());

    // log the image
    rec.log_static(
        "image",
        &rerun::Image::from_elements(
            img_rgb8.as_slice(),
            img_rgb8.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    // log the keypoints
    let points = keypoints
        .iter()
        .map(|k| (k[0] as f32, k[1] as f32))
        .collect::<Vec<_>>();

    rec.log_static(
        "image/keypoints",
        &rerun::Points2D::new(points).with_colors([[0, 0, 255]]),
    )?;

    Ok(())
}
