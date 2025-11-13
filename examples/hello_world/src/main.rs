use argh::FromArgs;
use kornia::io::functional as F;
use std::path::PathBuf;

#[derive(FromArgs)]
/// Hello world!
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image = F::read_image_any_rgb8(args.image_path)?;

    println!("Hello, world! 🦀");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
