use kornia_rs::io::functions as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path)?;
    let image_viz = image.clone();

    // binarize the image as u8
    let _image_bin = kornia_rs::threshold::threshold_binary(&image.clone(), 127, 255)?;

    // normalize the image between 0 and 1
    let image_f32 = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert to grayscale as floating point
    let gray_f32 = kornia_rs::color::gray_from_rgb(&image_f32)?;
    let gray_viz = gray_f32.clone();

    // binarize the gray image as floating point
    let gray_bin = kornia_rs::threshold::threshold_binary(&gray_f32, 0.5, 1.0)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    let _ = rec.log("image", &rerun::Image::try_from(image_viz.data)?);
    let _ = rec.log("gray", &rerun::Image::try_from(gray_viz.data)?);
    let _ = rec.log("gray_bin", &rerun::Image::try_from(gray_bin.data)?);

    Ok(())
}
