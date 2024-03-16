use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_jpeg(image_path)?;

    println!("Hello, world!");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
