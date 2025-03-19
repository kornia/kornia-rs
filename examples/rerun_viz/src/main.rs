use argh::FromArgs;
use std::path::PathBuf;

use kornia::image::Image;
use kornia::io::functional as F;

#[derive(FromArgs)]
/// rerun recording example.
struct Args {
    #[argh(option, short = 'i')]
    /// input image path
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args:Args = argh::from_env();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the image
    rec.log(
        "image",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    Ok(())
}
