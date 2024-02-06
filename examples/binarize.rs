use kornia_rs::io::functions as F;
use rerun;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path)?;

    // convert the image to grayscale
    let gray = kornia_rs::color::gray_from_rgb(&image)?;
    let gray_viz = gray.clone();

    // binarize the image
    let gray_bin = kornia_rs::threshold::threshold_binary(&gray, 127, 255)?;

    // Option1: convert the grayscale image to floating point
    let gray_f32 = gray.cast::<f32>()?;

    // Option 2: onvert and normalize the grayscale image to floating point
    // let gray_f32 = gray.cast_and_scale::<f32>(1.0 / 255, 0.0)?;

    // normalize the image between 0 and 1
    let gray_f32 = kornia_rs::normalize::normalize_mean_std(&gray_f32, &[0.0], &[255.0])?;

    // binarize the image as floating point
    let gray_f32 = kornia_rs::threshold::threshold_binary(&gray_f32, 0.5, 1.0)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    let _ = rec.log("image", &rerun::Image::try_from(image.data())?);
    let _ = rec.log("gray", &rerun::Image::try_from(gray_viz.data())?);
    let _ = rec.log("gray_bin", &rerun::Image::try_from(gray_bin.data())?);
    let _ = rec.log("gray_f32", &rerun::Image::try_from(gray_f32.data())?);

    Ok(())
}
