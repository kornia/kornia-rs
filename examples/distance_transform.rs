use kornia_rs::io::functions as F;
use rerun;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image = F::read_image_jpeg(image_path);

    // convert the image to grayscale
    let gray = kornia_rs::color::gray_from_rgb(&image);

    // binarize the image
    let gray_bin = kornia_rs::threshold::threshold_binary(&gray, 127, 255);

    // compute the distance transform
    let gray_dt = kornia_rs::distance_transform::distance_transform(&gray_bin.cast());
    println!("{:?}", gray_dt.data);

    // normalize the distance transform
    //let gray_norm = kornia_rs::normalize::normalize_min_max(&gray_dt.cast(), 0.0, 1.0);

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    let _ = rec.log("image", &rerun::Image::try_from(image.data)?);
    let _ = rec.log("gray", &rerun::Image::try_from(gray.data)?);
    let _ = rec.log("gray_bin", &rerun::Image::try_from(gray_bin.data)?);
    let _ = rec.log("gray_dt", &rerun::Image::try_from(gray_dt.data)?);
    //let _ = rec.log("gray_norm", &rerun::Image::try_from(gray_norm.data)?);

    Ok(())
}
