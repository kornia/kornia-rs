use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_jpeg(image_path)?;

    // convert the image to f32 and scale it
    let image: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // modify the image to see the changes
    let image_dirty = kornia_rs::flip::horizontal_flip(&image)?;

    // compute the mean squared error (mse) between the original and the modified image
    let mse = kornia_rs::metrics::mse(&image, &image_dirty);
    let psnr = kornia_rs::metrics::psnr(&image, &image_dirty, 1.0);

    // print the mse error
    println!("MSE error: {:?}", mse);
    println!("PSNR error: {:?}", psnr);

    // or, alternatively, compute the mse using the built-in functions
    let mse_map = image.sub(&image_dirty).powi(2);
    // let mse_ii = mse_map_ii.mean();

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    // log the images
    let _ = rec.log("image", &rerun::Image::try_from(image.data)?);
    let _ = rec.log("flip", &rerun::Image::try_from(image_dirty.data)?);
    let _ = rec.log("mse_map", &rerun::Image::try_from(mse_map.data)?);

    Ok(())
}
