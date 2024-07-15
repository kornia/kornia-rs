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
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the image
    rec.log("image", &rerun::Image::try_from(image.data)?)?;

    Ok(())
}
