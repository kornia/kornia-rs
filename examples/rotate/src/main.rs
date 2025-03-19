use argh::FromArgs;
use std::path::PathBuf;

use kornia::io::functional as F;
use kornia::{image::Image, imgproc};

#[derive(FromArgs)]
/// Rotate an image and log it to Rerun.
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(args.image_path)?;
    let image: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

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

        let mut output = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
        let mut output_norm = output.clone();

        imgproc::warp::warp_affine(
            &image,
            &mut output,
            &rotation_matrix,
            imgproc::interpolation::InterpolationMode::Bilinear,
        )?;

        imgproc::normalize::normalize_min_max(&output, &mut output_norm, 0.0, 255.0)?;

        rec.log(
            "image",
            &rerun::Image::from_elements(
                output_norm.as_slice(),
                output_norm.size().into(),
                rerun::ColorModel::RGB,
            ),
        )?;
    }

    Ok(())
}
