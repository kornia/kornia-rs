use std::env;

use kornia_io::jpeg::{read_image_jpeg_rgb8, write_image_jpeg_rgb8};
use kornia_io::metadata::read_image_metadata;
use kornia_io::read_image_jpeg_auto_orient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args.next().ok_or_else(|| {
        "Usage: cargo run -p exif_auto_orient -- <input.jpg> [raw_out.jpg] [fixed_out.jpg]"
            .to_string()
    })?;
    let raw_out = args
        .next()
        .unwrap_or_else(|| "/tmp/exif_before_raw.jpg".to_string());
    let fixed_out = args
        .next()
        .unwrap_or_else(|| "/tmp/exif_after_auto_orient.jpg".to_string());

    let metadata = read_image_metadata(&input)?;
    println!(
        "EXIF orientation: {:?}",
        metadata.exif_orientation.map(|v| v.get())
    );

    let raw = read_image_jpeg_rgb8(&input)?;
    let fixed = read_image_jpeg_auto_orient(&input)?;

    println!("raw dims  : {}x{}", raw.cols(), raw.rows());
    println!("fixed dims: {}x{}", fixed.cols(), fixed.rows());

    let fixed_u8 = fixed
        .cast::<u8>()
        .map_err(kornia_io::error::IoError::ImageCreationError)?;

    write_image_jpeg_rgb8(raw_out.as_str(), &raw, 95)?;
    write_image_jpeg_rgb8(fixed_out.as_str(), &fixed_u8, 95)?;

    println!("wrote raw  : {}", raw_out);
    println!("wrote fixed: {}", fixed_out);

    Ok(())
}
