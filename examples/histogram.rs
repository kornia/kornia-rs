use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_any(image_path)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // compute the histogram per channel
    let histogram = image
        .split_channels()?
        .iter()
        .map(|ch| kornia_rs::histogram::compute_histogram(ch, 256))
        .collect::<Result<Vec<_>, _>>()?;

    // log the image and the histogram
    rec.set_time_sequence("step", 0);
    rec.log("image", &rerun::Image::try_from(image.clone().data)?)?;

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
    for (i, hist) in histogram.iter().enumerate() {
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
