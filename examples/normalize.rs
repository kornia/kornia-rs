use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_jpeg(image_path)?;

    // cast the image to floating point
    let image_f32: Image<f32, 3> = image.clone().cast::<f32>()?;

    // normalize the image between 0 and 255
    let _image_f32_norm: Image<f32, 3> = kornia_rs::normalize::normalize_mean_std(
        &image_f32,
        &[127.5, 127.5, 127.5],
        &[127.5, 127.5, 127.5],
    )?;

    // alternative way to normalize the image between 0 and 255
    let _image_f32_norm: Image<f32, 3> = image.clone().cast_and_scale::<f32>(1.0 / 255.0)?;
    // Or: image.cast_and_scale::<f64>(1.0 / 255.0)?;

    // alternative way to normalize the image between 0 and 1
    let _image_f32_norm: Image<f32, 3> = image_f32.mul(1.0 / 255.0);
    // Or: let image_f32_norm = image_f32.div(255.0);

    let (min, max) = kornia_rs::normalize::find_min_max(&image_f32)?;
    println!("min: {:?}, max: {:?}", min, max);

    Ok(())
}
