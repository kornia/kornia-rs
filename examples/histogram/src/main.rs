use argh::FromArgs;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{
    image::{Image, ImageError},
    imgproc,
};

#[derive(FromArgs)]
/// Compute the histogram of an image and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    // compute the histogram per channel
    let histograms = image
        .split_channels()?
        .iter()
        .map(|ch| {
            let mut hist = vec![0; 256];
            imgproc::histogram::compute_histogram(ch, &mut hist, 256)?;
            Ok(hist)
        })
        .collect::<Result<Vec<Vec<_>>, ImageError>>()?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Histogram App").spawn()?;

    // log the image and the histogram
    rec.set_time_sequence("step", 0);
    rec.log(
        "image",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    // show the image and the histogram
    rec.log_static(
        "histogram/red",
        &rerun::SeriesLine::new()
            .with_color([255, 0, 0])
            .with_name("red")
            .with_width(2.0),
    )?;

    rec.log_static(
        "histogram/green",
        &rerun::SeriesLine::new()
            .with_color([0, 255, 0])
            .with_name("green")
            .with_width(2.0),
    )?;

    rec.log_static(
        "histogram/blue",
        &rerun::SeriesLine::new()
            .with_color([0, 0, 255])
            .with_name("blue")
            .with_width(2.0),
    )?;

    // TODO: not sure how to log the histogram properly
    for (i, hist) in histograms.iter().enumerate() {
        for (j, val) in hist.iter().enumerate() {
            rec.set_time_sequence("step", j as i64);
            match i {
                0 => {
                    let _ = rec.log("histogram/red", &rerun::Scalar::new(*val as f64));
                }
                1 => {
                    let _ = rec.log("histogram/green", &rerun::Scalar::new(*val as f64));
                }
                2 => {
                    let _ = rec.log("histogram/blue", &rerun::Scalar::new(*val as f64));
                }
                _ => {}
            }
        }
    }

    Ok(())
}
