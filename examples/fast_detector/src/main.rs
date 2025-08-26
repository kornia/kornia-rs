use argh::FromArgs;
use std::path::PathBuf;

use kornia::{image::Image, imgproc, io::png::read_image_png_mono8, tensor::CpuAllocator};

/// Detect FAST features on an image.
#[derive(FromArgs)]
struct Args {
    /// path to the image to detect FAST features on
    #[argh(option)]
    image_path: PathBuf,

    /// threshold for the FAST detector
    #[argh(option, default = "0.15")]
    threshold: f32,

    /// arc length for the FAST detector
    #[argh(option, default = "12")]
    arc_length: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Fast Detector App").spawn()?;

    // read the image
    let img_gray8 = read_image_png_mono8(args.image_path)?;

    let mut img_gray_f32 = Image::from_size_val(img_gray8.size(), 0.0f32, CpuAllocator)?;
    let img_gray8_slice = img_gray8.as_slice();
    let img_gray_f32_slice = img_gray_f32.as_slice_mut();
    for (dst, src) in img_gray_f32_slice.iter_mut().zip(img_gray8_slice.iter()) {
        *dst = *src as f32 / 255.0;
    }

    // detect the fast features
    let mask =
        imgproc::features::other::corner_fast(&img_gray_f32, args.threshold, args.arc_length)?;
    let keypoints = imgproc::features::other::peak_local_max(&mask, 1)?;
    println!("Found {} keypoints", keypoints.len());

    // log the image
    rec.log_static(
        "image",
        &rerun::Image::from_elements(
            img_gray8.as_slice(),
            img_gray8.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    // log the keypoints
    let points = keypoints
        .iter()
        .map(|k| (k.1 as f32, k.0 as f32))
        .collect::<Vec<_>>();

    rec.log_static(
        "image/keypoints",
        &rerun::Points2D::new(points).with_colors([[0, 255, 0]]),
    )?;

    Ok(())
}
