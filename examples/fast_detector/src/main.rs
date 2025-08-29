use argh::FromArgs;
use std::path::PathBuf;

use kornia::{image::Image, imgproc, io::functional as F, tensor::CpuAllocator};

/// Detect FAST features on an image.
#[derive(FromArgs)]
struct Args {
    /// path to the image to detect FAST features on
    #[argh(option)]
    image_path: PathBuf,

    /// threshold for the FAST detector
    #[argh(option, default = "38")]
    threshold: u8,

    /// arc length for the FAST detector
    #[argh(option, default = "12")]
    arc_length: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Fast Detector App").spawn()?;

    // read the image
    let img_rgb8 = F::read_image_any_rgb8(args.image_path)?;

    // convert to grayscale
    let mut img_gray8 = Image::from_size_val(img_rgb8.size(), 0u8, CpuAllocator)?;
    imgproc::color::gray_from_rgb_u8(&img_rgb8, &mut img_gray8)?;

    // detect the fast features
    let fast_response =
        imgproc::features::corner_fast(&img_gray8, args.threshold, args.arc_length)?;
    let keypoints = imgproc::features::peak_local_max(&fast_response, 1, args.threshold)?;
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
        .map(|k| (k.x as f32, k.y as f32))
        .collect::<Vec<_>>();

    rec.log_static(
        "image/keypoints",
        &rerun::Points2D::new(points).with_colors([[0, 0, 255]]),
    )?;

    Ok(())
}
