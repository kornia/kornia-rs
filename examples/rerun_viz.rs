use kornia_rs::io::functions as F;
use rerun;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path);

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    // log the image
    let _ = rec.log("image", &rerun::Image::try_from(image.data)?);

    Ok(())
}
