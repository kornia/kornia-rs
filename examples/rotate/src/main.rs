use clap::Parser;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{image::Image, imgproc};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;
    let image: Image<f32, 3> = image.cast::<f32>()?;

    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // rotate the image

    let center = (
        image.size().width as f32 / 2.0,
        image.size().height as f32 / 2.0,
    );

    for i in 0..360 {
        let angle = i as f32;
        let scale = i as f32 / 360.0;
        let rotation_matrix = imgproc::warp::get_rotation_matrix2d(center, angle, scale);

        let output = imgproc::warp::warp_affine(
            &image,
            rotation_matrix,
            image.size(),
            imgproc::interpolation::InterpolationMode::Bilinear,
        )?;

        let output = imgproc::normalize::normalize_min_max(&output, 0.0, 255.0)?;

        rec.log("image", &rerun::Image::try_from(output.data)?)?;
    }

    Ok(())
}
