use clap::{Arg, ArgAction, Command};
use kornia::image::Image;
use kornia::io::functional as F;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("decode_from_bytes")
        .arg(
            Arg::new("image-path")
                .short('i')
                .long("image-path")
                .action(ArgAction::Set)
                .required(true)
                .help("Path to the input image"),
        )
        .get_matches();

    let image_path = matches
        .get_one::<String>("image-path")
        .expect("Required argument");

    println!("Reading image from: {}", image_path);

    // Read the image file as raw bytes
    let image_data = fs::read(image_path)?;
    println!("Image data size: {} bytes", image_data.len());

    // Extract format from file extension
    let format_hint = image_path.split('.').last();
    println!("Format hint: {:?}", format_hint);

    // Decode image in RGB format from bytes
    let image_rgb: Image<u8, 3> = F::decode_image_bytes_rgb8(&image_data, format_hint)?;
    println!("Decoded RGB image size: {:?}", image_rgb.size());

    // Decode image in grayscale format from bytes
    let image_gray: Image<u8, 1> = F::decode_image_bytes_gray8(&image_data, format_hint)?;
    println!("Decoded grayscale image size: {:?}", image_gray.size());

    // If it's a JPEG image, also try using the specialized JPEG decoder
    if format_hint.map_or(false, |fmt| fmt.eq_ignore_ascii_case("jpg") || fmt.eq_ignore_ascii_case("jpeg")) {
        println!("Using specialized JPEG decoder");
        
        let image_jpeg_rgb: Image<u8, 3> = F::decode_image_jpegturbo_rgb8(&image_data)?;
        println!("Decoded JPEG RGB image size: {:?}", image_jpeg_rgb.size());
        
        let image_jpeg_gray: Image<u8, 1> = F::decode_image_jpegturbo_gray8(&image_data)?;
        println!("Decoded JPEG grayscale image size: {:?}", image_jpeg_gray.size());
    }

    println!("All decoding methods successful!");

    Ok(())
} 