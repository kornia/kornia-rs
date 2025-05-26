use argh::FromArgs;
use kornia::{image::Image, io::functional as F, tensor::CpuAllocator};
use std::path::PathBuf;

#[derive(FromArgs)]
/// Log an image to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image: Image<u8, 3, CpuAllocator> = F::read_image_any_rgb8(args.image_path)?;

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
