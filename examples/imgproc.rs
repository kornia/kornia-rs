use kornia_rs::io::functions as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path)?;
    let image_viz = image.clone();

    let image_f32 = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert the image to grayscale
    let gray = kornia_rs::color::gray_from_rgb(&image_f32)?;

    let gray_resize = kornia_rs::resize::resize(
        &gray,
        kornia_rs::image::ImageSize {
            width: 128,
            height: 128,
        },
        kornia_rs::resize::ResizeOptions {
            interpolation: kornia_rs::resize::InterpolationMode::Bilinear,
        },
    )?;

    println!("gray_resize: {:?}", gray_resize.image_size());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    // log the images
    let _ = rec.log("image", &rerun::Image::try_from(image_viz.data)?);
    let _ = rec.log("gray", &rerun::Image::try_from(gray.data)?);
    let _ = rec.log("gray_resize", &rerun::Image::try_from(gray_resize.data)?);

    Ok(())
}
