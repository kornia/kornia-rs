use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");

    let image: Image<u8, 3> = F::read_image_any(image_path)?;
    let image = image.cast::<f32>()?;

    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // rotate the image

    let center = (
        image.size().width as f32 / 2.0,
        image.size().height as f32 / 2.0,
    );

    for i in 0..360 {
        let angle = i as f32;
        let scale = i as f32 / 360.0;
        let rotation_matrix = kornia_rs::warp::get_rotation_matrix2d(center, angle, scale);

        let output = kornia_rs::warp::warp_affine(
            &image,
            rotation_matrix,
            image.size(),
            kornia_rs::interpolation::InterpolationMode::Bilinear,
        )?;

        let output = kornia_rs::normalize::normalize_min_max(&output, 0.0, 255.0)?;

        rec.log("image", &rerun::Image::try_from(output.data)?)?;
    }

    Ok(())
}
