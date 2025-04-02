use anyhow::Result;
use argh::FromArgs;
use kornia_image::Image;
use kornia_io::{
     read_image_any, read_image_any_gray8, read_image_any_rgb8, read_image_any_rgba8,
    write_image_png_gray8, write_image_png_rgb8, write_image_png_rgba8,
};
use std::path::PathBuf;
#[cfg(feature = "turbojpeg")]
use kornia_io::{
    write_image_jpegturbo_gray8, write_image_jpegturbo_rgb8,
};

/// Command line arguments for the image_io example
#[derive(FromArgs, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input image file
    #[argh(option, short = 'i')]
    input_path: PathBuf,

    /// Path where the output image will be saved
   #[argh(option, short = 'o')]
    output_path: PathBuf,

    /// Output format (png or jpeg)
    #[argh(option, short = 'f', default = "String::from(\"png\")")]
    format: String,

    /// Number of channels for the output image (1 for grayscale, 3 for RGB, 4 for RGBA - PNG only)
    #[argh(option, short = 'c', default = "3")]
    channels: usize,
    
    /// Input reading mode (auto, gray, rgb, rgba)
    #[argh(option, short = 'm', default = "String::from(\"auto\")")]
    input_mode: String,
    
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Args = argh::from_env();

    // Validate input format
    if !["png", "jpeg"].contains(&args.format.as_str()) {
        anyhow::bail!("Unsupported format: {}. Only 'png' and 'jpeg' are supported.", args.format);
    }

    // Validate channels
    if ![1, 3, 4].contains(&args.channels) {
        anyhow::bail!("Unsupported number of channels: {}. Only 1, 3, and 4 are supported.", args.channels);
    }

    // Check if RGBA is requested with JPEG (which is not supported)
    if args.format == "jpeg" && args.channels == 4 {
        anyhow::bail!("RGBA (4 channels) is not supported for JPEG format.");
    }

    // Validate input mode
    if !["auto", "gray", "rgb", "rgba"].contains(&args.input_mode.as_str()) {
        anyhow::bail!("Unsupported input mode: {}. Only 'auto', 'gray', 'rgb', or 'rgba' are supported.", args.input_mode);
    }
    
    // Read the input image
    println!("Reading image from: {}", args.input_path.display());
    // Read the input image based on the specified mode
    let result = match args.input_mode.as_str() {
        "auto" => {
            // Auto-detect format and read accordingly
            let image = read_image_any(&args.input_path)?;
            println!("Auto-detected image with {} channels", image.num_channels());
            process_image(image, &args)
        },
        "gray" => {
            let gray_image = read_image_any_gray8(&args.input_path)?;
            println!("Read as grayscale: {}x{}, 1 channel", gray_image.rows(), gray_image.cols());
            process_grayscale_image(gray_image, &args)
        },
        "rgb" => {
            let rgb_image = read_image_any_rgb8(&args.input_path)?;
            println!("Read as RGB: {}x{}, 3 channels", rgb_image.rows(), rgb_image.cols());
            process_rgb_image(rgb_image, &args)
        },
        "rgba" => {
            let rgba_image = read_image_any_rgba8(&args.input_path)?;
            println!("Read as RGBA: {}x{}, 4 channels", rgba_image.rows(), rgba_image.cols());
            process_rgba_image(rgba_image, &args)
        },
        _ => unreachable!(),
    };

    match result {
        Ok(_) => {
            println!("Image successfully written to: {}", args.output_path.display());
            Ok(())
        },
        Err(e) => Err(e),
    }
}

// Process image of any type
fn process_image(image: Image<u8>, args: &Args) -> Result<()> {
    match image.num_channels() {
        1 => process_grayscale_image(image.into_typed::<1>(), args),
        3 => process_rgb_image(image.into_typed::<3>(), args),
        4 => process_rgba_image(image.into_typed::<4>(), args),
        n => anyhow::bail!("Unsupported number of channels in input image: {}", n),
    }
}

// Process grayscale image
fn process_grayscale_image(gray_image: Image<u8, 1>, args: &Args) -> Result<()> {
    match (args.format.as_str(), args.channels) {
        ("png", 1) => {
            write_image_png_gray8(&args.output_path, &gray_image)?;
        },
        ("png", 3) => {
            let rgb_image = to_rgb_from_gray(&gray_image);
            write_image_png_rgb8(&args.output_path, &rgb_image)?;
        },
        ("png", 4) => {
            let rgba_image = to_rgba_from_gray(&gray_image);
            write_image_png_rgba8(&args.output_path, &rgba_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 1) => {
            write_image_jpegturbo_gray8(&args.output_path, &gray_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 3) => {
            let rgb_image = to_rgb_from_gray(&gray_image);
            write_image_jpegturbo_rgb8(&args.output_path, &rgb_image)?;
        },
        #[cfg(not(feature = "turbojpeg"))]
        ("jpeg", _) => {
            anyhow::bail!("JPEG writing is not supported because the 'turbojpeg' feature is not enabled.");
        },
        _ => unreachable!(),
    }
    Ok(())
}

// Process RGB image
fn process_rgb_image(rgb_image: Image<u8, 3>, args: &Args) -> Result<()> {
    match (args.format.as_str(), args.channels) {
        ("png", 1) => {
            // Use the kornia_imgproc function to convert RGB to grayscale
            let gray_image = gray_from_rgb_u8(&rgb_image);
            write_image_png_gray8(&args.output_path, &gray_image)?;
        },
        ("png", 3) => {
            write_image_png_rgb8(&args.output_path, &rgb_image)?;
        },
        ("png", 4) => {
            let rgba_image = to_rgba(&rgb_image);
            write_image_png_rgba8(&args.output_path, &rgba_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 1) => {
            // Use the kornia_imgproc function to convert RGB to grayscale
            let gray_image = gray_from_rgb_u8(&rgb_image);
            write_image_jpegturbo_gray8(&args.output_path, &gray_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 3) => {
            write_image_jpegturbo_rgb8(&args.output_path, &rgb_image)?;
        },
        #[cfg(not(feature = "turbojpeg"))]
        ("jpeg", _) => {
            anyhow::bail!("JPEG writing is not supported because the 'turbojpeg' feature is not enabled.");

        _ => unreachable!(),
    }
    Ok(())
}

// Process RGBA image
fn process_rgba_image(rgba_image: Image<u8, 4>, args: &Args) -> Result<()> {
    match (args.format.as_str(), args.channels) {
        ("png", 1) => {
            // First convert RGBA to RGB, then use kornia_imgproc for RGB to grayscale
            let rgb_image = to_rgb_from_rgba(&rgba_image);
            let gray_image = gray_from_rgb_u8(&rgb_image);
            write_image_png_gray8(&args.output_path, &gray_image)?;
        },
        ("png", 3) => {
            let rgb_image = to_rgb_from_rgba(&rgba_image);
            write_image_png_rgb8(&args.output_path, &rgb_image)?;
        },
        ("png", 4) => {
            write_image_png_rgba8(&args.output_path, &rgba_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 1) => {
            // First convert RGBA to RGB, then use kornia_imgproc for RGB to grayscale
            let rgb_image = to_rgb_from_rgba(&rgba_image);
            let gray_image = gray_from_rgb_u8(&rgb_image);
            write_image_jpegturbo_gray8(&args.output_path, &gray_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 3) => {
            let rgb_image = to_rgb_from_rgba(&rgba_image);
            write_image_jpegturbo_rgb8(&args.output_path, &rgb_image)?;
        },
        #[cfg(not(feature = "turbojpeg"))]
        ("jpeg", _) => {
            anyhow::bail!("JPEG writing is not supported because the 'turbojpeg' feature is not enabled.");
        },
        _ => unreachable!(),
    }
    Ok(())
}
