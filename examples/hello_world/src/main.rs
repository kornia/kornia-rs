use clap::Parser;
use std::path::PathBuf;

use kornia::image::Image;
use kornia::io::functional as F;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    println!("Hello, world! 🦀");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
