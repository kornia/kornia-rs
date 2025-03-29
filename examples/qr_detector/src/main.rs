use anyhow::Result;
use kornia::{
    io::functional as F,
    qr::QrDetectionExt,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path-to-image-with-qr-code>", args[0]);
        return Ok(());
    }

    let image_path = PathBuf::from(&args[1]);
    
    println!("Reading image from: {}", image_path.display());
    
    // Load the image
    let image = F::read_image_any_rgb8(&image_path)?;
    println!("Image loaded. Size: {}x{}", image.width(), image.height());
    
    // Detect QR codes
    println!("Detecting QR codes...");
    match image.detect_qr_codes() {
        Ok(detections) => {
            println!("Detection complete. Found {} QR codes:", detections.len());
            
            for (i, detection) in detections.iter().enumerate() {
                println!("QR Code #{}", i + 1);
                println!("  Content: {}", detection.content);
                println!("  Version: {}", detection.version);
                println!("  ECC Level: {}", detection.ecc_level);
                println!("  Corners:");
                for (j, corner) in detection.corners.iter().enumerate() {
                    println!("    Corner {}: ({:.1}, {:.1})", j + 1, corner[0], corner[1]);
                }
                println!();
            }
        },
        Err(e) => {
            println!("QR detection failed: {}", e);
        }
    }
    
    Ok(())
} 