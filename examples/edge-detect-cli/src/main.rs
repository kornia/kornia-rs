//! Edge Detection CLI Tool
//!
//! A command-line utility for applying Sobel edge detection to images.
//!
//! Usage:
//! ```bash
//! cargo run --example edge-detect-cli -- \
//!   --input image.png \
//!   --output edges.png \
//!   --kernel-size 3 \
//!   --show-stats
//! ```

use anyhow::Result;
use clap::Parser;
use kornia_image::Image;
use kornia_imgproc::filter::sobel;
use kornia_io::functional as F;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "edge-detect-cli")]
#[command(about = "Sobel edge detection tool for robotics and image processing")]
struct Args {
    /// Input image file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output edge map file path
    #[arg(short, long)]
    output: PathBuf,

    /// Sobel kernel size (1, 3, 5, or 7)
    #[arg(short, long, default_value = "3")]
    kernel_size: usize,

    /// Edge magnitude threshold (0-255)
    #[arg(short, long)]
    threshold: Option<f32>,

    /// Whether to display statistics
    #[arg(long, default_value = "true")]
    show_stats: bool,
}

/// Apply Sobel edge detection to an image
fn apply_edge_detection(
    input_path: &PathBuf,
    output_path: &PathBuf,
    kernel_size: usize,
    threshold: Option<f32>,
    show_stats: bool,
) -> Result<()> {
    println!("🚀 Starting edge detection pipeline...");
    println!("📁 Input: {}", input_path.display());

    let start_load = Instant::now();

    // Load image as RGB
    let img_rgb8 = F::read_image_any_rgb8(input_path)?;
    let load_time = start_load.elapsed();

    println!("✅ Image loaded: {}x{} pixels in {:.2}ms",
        img_rgb8.size().width,
        img_rgb8.size().height,
        load_time.as_secs_f64() * 1000.0
    );

    let start_preprocess = Instant::now();

    // Convert RGB to grayscale (u8)
    let mut gray8 = Image::from_size_val(img_rgb8.size(), 0u8, kornia_tensor::CpuAllocator)?;
    
    // Simple grayscale conversion (average of R, G, B channels)
    let rgb_data = img_rgb8.as_slice();
    for i in 0..img_rgb8.size().width * img_rgb8.size().height {
        let r = rgb_data[i * 3] as f32;
        let g = rgb_data[i * 3 + 1] as f32;
        let b = rgb_data[i * 3 + 2] as f32;
        gray8.as_slice_mut()[i] = ((r + g + b) / 3.0) as u8;
    }

    // Convert u8 to f32 for Sobel
    let gray_f32 = gray8
        .map(|&x| x as f32 / 255.0)
        .map_err(|e| anyhow::anyhow!("Failed to convert image: {}", e))?;

    let preprocess_time = start_preprocess.elapsed();
    println!("✅ Preprocessing complete in {:.2}ms", preprocess_time.as_secs_f64() * 1000.0);

    let start_sobel = Instant::now();

    // Apply Sobel edge detection
    let mut edges_f32 = Image::from_size_val(gray_f32.size(), 0.0f32, kornia_tensor::CpuAllocator)?;
    sobel(&gray_f32, &mut edges_f32, kernel_size)?;

    let sobel_time = start_sobel.elapsed();
    println!("✅ Sobel edge detection complete in {:.2}ms", sobel_time.as_secs_f64() * 1000.0);

    let start_postprocess = Instant::now();

    // Convert edges back to u8 and apply threshold
    let mut edges_u8 = Image::from_size_val(edges_f32.size(), 0u8, kornia_tensor::CpuAllocator)?;
    
    let edges_data = edges_f32.as_slice();
    let max_edge = edges_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    for i in 0..edges_u8.size().width * edges_u8.size().height {
        let edge_val = edges_data[i] / max_edge.max(1.0);
        let normalized = (edge_val * 255.0) as u8;
        
        edges_u8.as_slice_mut()[i] = if let Some(th) = threshold {
            if (normalized as f32) > th { 255 } else { 0 }
        } else {
            normalized
        };
    }

    let postprocess_time = start_postprocess.elapsed();
    println!("✅ Post-processing complete in {:.2}ms", postprocess_time.as_secs_f64() * 1000.0);

    let start_save = Instant::now();

    // Save output
    F::write_image(&edges_u8, output_path)?;
    let save_time = start_save.elapsed();

    println!("✅ Output saved: {} in {:.2}ms", output_path.display(), save_time.as_secs_f64() * 1000.0);

    if show_stats {
        let total_time = load_time + preprocess_time + sobel_time + postprocess_time + save_time;
        let height = img_rgb8.size().height;
        let width = img_rgb8.size().width;
        let pixels = width * height;
        
        // Calculate FPS equivalents at common resolutions
        let fps_1080p = if height > 0 && width > 0 {
            (1920.0 * 1080.0 / pixels as f64) / sobel_time.as_secs_f64()
        } else {
            0.0
        };
        
        let edge_pixels = edges_u8.as_slice().iter().filter(|&&x| x > 0).count();
        let edge_percentage = (edge_pixels as f64 / pixels as f64) * 100.0;
        
        println!("\n📊 Statistics:");
        println!("─────────────────────────────────────────");
        println!("  Input size: {}x{} ({}M pixels)", width, height, pixels / 1_000_000);
        println!("  Kernel size: {}x{}", kernel_size, kernel_size);
        if let Some(th) = threshold {
            println!("  Threshold: {}", th);
        }
        println!("─────────────────────────────────────────");
        println!("  Total processing time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
        println!("    ├─ Load: {:.2}ms", load_time.as_secs_f64() * 1000.0);
        println!("    ├─ Preprocess: {:.2}ms", preprocess_time.as_secs_f64() * 1000.0);
        println!("    ├─ Sobel: {:.2}ms", sobel_time.as_secs_f64() * 1000.0);
        println!("    ├─ Post-process: {:.2}ms", postprocess_time.as_secs_f64() * 1000.0);
        println!("    └─ Save: {:.2}ms", save_time.as_secs_f64() * 1000.0);
        println!("─────────────────────────────────────────");
        println!("  Performance:");
        println!("    ├─ Edge pixels: {} ({:.2}%)", edge_pixels, edge_percentage);
        println!("    ├─ Pixels/second: {:.0}M", (pixels as f64 / sobel_time.as_secs_f64()) / 1_000_000.0);
        println!("    └─ FPS (at 1080p): {:.1}", fps_1080p);
        println!("─────────────────────────────────────────");
        println!("\n✨ Edge detection complete!");
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    apply_edge_detection(
        &args.input,
        &args.output,
        args.kernel_size,
        args.threshold,
        args.show_stats,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = Args::parse_from(&[
            "edge-detect-cli",
            "--input", "input.png",
            "--output", "output.png",
            "--kernel-size", "5",
        ]);
        
        assert_eq!(args.input, PathBuf::from("input.png"));
        assert_eq!(args.output, PathBuf::from("output.png"));
        assert_eq!(args.kernel_size, 5);
    }
}
