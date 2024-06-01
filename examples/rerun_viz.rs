use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_any(image_path)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the image
    rec.log("image", &rerun::Image::try_from(image.data)?)?;

    Ok(())
}
