use argh::FromArgs;
use std::path::PathBuf;

use kornia::image::Image;
use kornia::io::functional as F;

#[derive(FromArgs)]
/// read an image and print its size.
struct Args {
    #[argh(option, short = 'i')]
    /// input image path
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    println!("Hello, world! 🦀");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
