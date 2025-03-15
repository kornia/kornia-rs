use anyhow::Result;
use clap::Parser;
use kornia_image::Image;
use kornia_io::{
    read_image_any_rgb8,
    write_image_png_gray8, write_image_png_rgb8, write_image_png_rgba8,
};

#[cfg(feature = "turbojpeg")]
use kornia_io::{
    write_image_jpegturbo_gray8, write_image_jpegturbo_rgb8,
};

/// Command line arguments for the image_io example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input image file
    #[arg(short, long)]
    input_path: String,

    /// Path where the output image will be saved
    #[arg(short, long)]
    output_path: String,

    /// Output format (png or jpeg)
    #[arg(short, long, default_value = "png")]
    format: String,

    /// Number of channels for the output image (1 for grayscale, 3 for RGB, 4 for RGBA - PNG only)
    #[arg(short, long, default_value_t = 3)]
    channels: usize,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

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

    // Read the input image
    println!("Reading image from: {}", args.input_path);
    let input_image = read_image_any_rgb8(&args.input_path)?;
    println!("Image dimensions: {}x{}, channels: {}", input_image.rows(), input_image.cols(), input_image.num_channels());

    // Process and write the output image based on format and channels
    match (args.format.as_str(), args.channels) {
        ("png", 1) => {
            // Convert to grayscale and write as PNG
            let gray_image = to_grayscale(&input_image);
            write_image_png_gray8(&args.output_path, &gray_image)?;
        },
        ("png", 3) => {
            // Write as RGB PNG
            write_image_png_rgb8(&args.output_path, &input_image)?;
        },
        ("png", 4) => {
            // Convert to RGBA and write as PNG
            let rgba_image = to_rgba(&input_image);
            write_image_png_rgba8(&args.output_path, &rgba_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 1) => {
            // Convert to grayscale and write as JPEG
            let gray_image = to_grayscale(&input_image);
            write_image_jpegturbo_gray8(&args.output_path, &gray_image)?;
        },
        #[cfg(feature = "turbojpeg")]
        ("jpeg", 3) => {
            // Write as RGB JPEG
            write_image_jpegturbo_rgb8(&args.output_path, &input_image)?;
        },
        #[cfg(not(feature = "turbojpeg"))]
        ("jpeg", _) => {
            anyhow::bail!("JPEG writing is not supported because the 'turbojpeg' feature is not enabled.");
        },
        _ => unreachable!(), // Already validated above
    }

    println!("Image written to: {}", args.output_path);
    Ok(())
}

/// Convert an RGB image to grayscale
fn to_grayscale(rgb_image: &Image<u8, 3>) -> Image<u8, 1> {
    let (rows, cols) = (rgb_image.rows(), rgb_image.cols());
    let mut gray_image = Image::<u8, 1>::new(rows, cols);
    
    for r in 0..rows {
        for c in 0..cols {
            // Standard RGB to grayscale conversion formula: 0.299*R + 0.587*G + 0.114*B
            let pixel = rgb_image.pixel(r, c);
            let gray_value = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            gray_image.set_pixel(r, c, [gray_value]);
        }
    }
    
    gray_image
}

/// Convert an RGB image to RGBA (with alpha = 255)
fn to_rgba(rgb_image: &Image<u8, 3>) -> Image<u8, 4> {
    let (rows, cols) = (rgb_image.rows(), rgb_image.cols());
    let mut rgba_image = Image::<u8, 4>::new(rows, cols);
    
    for r in 0..rows {
        for c in 0..cols {
            let pixel = rgb_image.pixel(r, c);
            rgba_image.set_pixel(r, c, [pixel[0], pixel[1], pixel[2], 255]);
        }
    }
    
    rgba_image
} 
