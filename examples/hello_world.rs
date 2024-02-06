use kornia_rs::io::functions as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path)?;

    println!("Image size: {:?}", image.image_size());
    Ok(())
}
