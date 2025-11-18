//! Comprehensive example demonstrating the kornia-image Image API
//!
//! This example shows:
//! - Creating images from fill values or data
//! - Zero-copy data access
//! - Error handling
//! - Different image types (u8, f32) and channels (1, 3, 4)
//!
//! Run with: cargo run --example image_api

use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};

fn print_separator() {
    println!("================================================");
}

fn example_create_from_value() -> Result<(), ImageError> {
    println!("Example 1: Creating images from fill values");
    print_separator();

    // Create a 640x480 RGB image filled with value 128
    let size = ImageSize {
        width: 640,
        height: 480,
    };
    let alloc = CpuAllocator::default();
    let img = Image::<u8, 3, _>::from_size_val(size, 128, alloc)?;

    println!("✓ Created {}x{}x{} image", img.width(), img.height(), 3);
    println!("  Fill value: {}", img.as_slice()[0]);
    println!("  Total bytes: {}", img.as_slice().len());
    println!();

    Ok(())
}

fn example_create_from_data() -> Result<(), ImageError> {
    println!("Example 2: Creating images from data vectors");
    print_separator();

    // Create data: 10x10 grayscale gradient
    let mut data = Vec::with_capacity(100);
    for i in 0..100 {
        data.push((i * 255 / 100) as u8);
    }

    let size = ImageSize {
        width: 10,
        height: 10,
    };
    let alloc = CpuAllocator::default();
    let img = Image::<u8, 1, _>::new(size, data, alloc)?;

    println!("✓ Created {}x{} grayscale image from data", img.width(), img.height());
    println!("  First pixel: {}", img.as_slice()[0]);
    println!("  Last pixel: {}", img.as_slice()[img.as_slice().len() - 1]);
    println!();

    Ok(())
}

fn example_error_handling() {
    println!("Example 3: Error handling (wrong data size)");
    print_separator();

    // Deliberately create wrong-sized data
    let data = vec![0u8; 100]; // Need 300 for 10x10x3 image
    let size = ImageSize {
        width: 10,
        height: 10,
    };
    let alloc = CpuAllocator::default();

    match Image::<u8, 3, _>::new(size, data, alloc) {
        Ok(_) => println!("✓ Created image (should not reach here)"),
        Err(e) => {
            println!("✓ Caught expected error:");
            println!("  {}", e);
        }
    }
    println!();
}

fn example_zero_copy_access() -> Result<(), ImageError> {
    println!("Example 4: Zero-copy data access");
    print_separator();

    // Create small RGB image
    let size = ImageSize { width: 5, height: 5 };
    let alloc = CpuAllocator::default();
    let img = Image::<u8, 3, _>::from_size_val(size, 42, alloc)?;

    // Zero-copy access to underlying data
    let data = img.as_slice(); // &[u8] - zero copy!

    println!("✓ Image dimensions: {}x{}x{}", img.width(), img.height(), 3);
    println!("  Data is zero-copy reference to underlying memory");
    println!("  Total elements: {}", data.len());

    // Access specific pixels (row-major, interleaved RGB)
    let pixel_idx = (2 * img.width() + 3) * 3; // Row 2, Col 3
    println!(
        "  Pixel (2,3) RGB: ({}, {}, {})",
        data[pixel_idx],
        data[pixel_idx + 1],
        data[pixel_idx + 2]
    );
    println!();

    Ok(())
}

fn example_owned_copy() -> Result<(), ImageError> {
    println!("Example 5: Creating owned copy of image data");
    print_separator();

    // Create image
    let size = ImageSize { width: 3, height: 3 };
    let alloc = CpuAllocator::default();
    let img = Image::<f32, 3, _>::from_size_val(size, 0.5, alloc)?;

    // Zero-copy view
    let data_view = img.as_slice();
    println!("✓ Zero-copy view size: {} elements", data_view.len());

    // Create owned copy
    let data_copy = img.to_vec();
    println!("✓ Owned copy size: {} elements", data_copy.len());
    println!("  First element: {}", data_copy[0]);
    println!("  Can modify copy independently of original");
    println!();

    Ok(())
}

fn example_different_types() -> Result<(), ImageError> {
    println!("Example 6: Different image types");
    print_separator();

    let size = ImageSize {
        width: 100,
        height: 100,
    };
    let alloc = CpuAllocator::default();

    // U8 images (8-bit unsigned)
    let _gray_u8 = Image::<u8, 1, _>::from_size_val(size, 255, alloc.clone())?;
    let _rgb_u8 = Image::<u8, 3, _>::from_size_val(size, 128, alloc.clone())?;
    let _rgba_u8 = Image::<u8, 4, _>::from_size_val(size, 64, alloc.clone())?;

    // F32 images (32-bit float, common for ML/processing)
    let _gray_f32 = Image::<f32, 1, _>::from_size_val(size, 1.0, alloc.clone())?;
    let _rgb_f32 = Image::<f32, 3, _>::from_size_val(size, 0.5, alloc.clone())?;
    let _rgba_f32 = Image::<f32, 4, _>::from_size_val(size, 0.25, alloc)?;

    println!("✓ Created 6 different image types:");
    println!("  U8:  Grayscale (C1), RGB (C3), RGBA (C4)");
    println!("  F32: Grayscale (C1), RGB (C3), RGBA (C4)");
    println!();
    println!("All types support:");
    println!("  - width(), height(), size(), cols(), rows()");
    println!("  - as_slice() for zero-copy access");
    println!("  - to_vec() for owned copy");
    println!("  - get_pixel() and set_pixel() for pixel-level access");
    println!();

    Ok(())
}

fn example_pixel_access() -> Result<(), ImageError> {
    println!("Example 7: Pixel-level access and manipulation");
    print_separator();

    let size = ImageSize { width: 5, height: 5 };
    let alloc = CpuAllocator::default();
    let mut img = Image::<u8, 3, _>::from_size_val(size, 0, alloc)?;

    // Set individual pixels
    img.set_pixel(0, 0, 0, 255)?; // Red channel at (0, 0)
    img.set_pixel(0, 0, 1, 0)?; // Green channel at (0, 0)
    img.set_pixel(0, 0, 2, 0)?; // Blue channel at (0, 0)

    img.set_pixel(2, 2, 0, 0)?; // Center pixel: green
    img.set_pixel(2, 2, 1, 255)?;
    img.set_pixel(2, 2, 2, 0)?;

    // Get individual pixels
    let r = img.get_pixel(0, 0, 0)?;
    let g = img.get_pixel(0, 0, 1)?;
    let b = img.get_pixel(0, 0, 2)?;

    println!("✓ Set and retrieved individual pixels");
    println!("  Pixel (0,0) RGB: ({}, {}, {})", r, g, b);
    println!("  Pixel (2,2) RGB: ({}, {}, {})", 
        img.get_pixel(2, 2, 0)?, 
        img.get_pixel(2, 2, 1)?, 
        img.get_pixel(2, 2, 2)?
    );
    println!();

    Ok(())
}

fn example_image_from_slice() -> Result<(), ImageError> {
    println!("Example 8: Creating images from slices (zero-copy)");
    print_separator();

    // Create data
    let data = vec![128u8; 10 * 10 * 3];
    
    let size = ImageSize { width: 10, height: 10 };
    let alloc = CpuAllocator::default();
    
    // Create image from slice (copies data)
    let img = Image::<u8, 3, _>::from_size_slice(size, &data, alloc)?;

    println!("✓ Created {}x{}x{} image from slice", img.width(), img.height(), 3);
    println!("  Original data length: {} bytes", data.len());
    println!("  Image data length: {} bytes", img.as_slice().len());
    println!();

    Ok(())
}

fn example_channel_operations() -> Result<(), ImageError> {
    println!("Example 9: Channel extraction and splitting");
    print_separator();

    let size = ImageSize { width: 10, height: 10 };
    let alloc = CpuAllocator::default();
    
    // Create RGB image with different values per channel
    let mut data = Vec::with_capacity(10 * 10 * 3);
    for _ in 0..100 {
        data.push(255); // R
        data.push(128); // G
        data.push(64);  // B
    }
    
    let img = Image::<u8, 3, _>::new(size, data, alloc)?;

    // Extract single channel
    let red_channel = img.channel(0)?;
    println!("✓ Extracted red channel ({}x{})", red_channel.width(), red_channel.height());
    println!("  First pixel value: {}", red_channel.as_slice()[0]);

    // Split into all channels
    let channels = img.split_channels()?;
    println!("✓ Split into {} channels", channels.len());
    for (i, ch) in channels.iter().enumerate() {
        println!("  Channel {}: first value = {}", i, ch.as_slice()[0]);
    }
    println!();

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("╔════════════════════════════════════════════╗");
    println!("║   Kornia Rust Image API Examples          ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    example_create_from_value()?;
    example_create_from_data()?;
    example_error_handling();
    example_zero_copy_access()?;
    example_owned_copy()?;
    example_different_types()?;
    example_pixel_access()?;
    example_image_from_slice()?;
    example_channel_operations()?;

    println!("╔════════════════════════════════════════════╗");
    println!("║   All examples completed successfully!     ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    Ok(())
}

