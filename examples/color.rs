use kornia_rs::io::functions as F;
use rerun;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path);

    // convert the image to grayscale
    let gray = kornia_rs::color::gray_from_rgb(image.clone());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    // log the images
    let _ = rec.log("image", &rerun::Image::try_from(image.data)?);
    let _ = rec.log("gray", &rerun::Image::try_from(gray.data)?);

    Ok(())
}
