//! Tour of the kornia-image `Image` API: construction, zero-copy access,
//! typed channels, and construction-time validation.
//!
//! Run with: cargo run -p image_api

use kornia_image::{Image, ImageSize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fill a 640x480 RGB image with a constant.
    let img = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: 640,
            height: 480,
        },
        128,
    )?;
    println!(
        "filled {}x{}x3, first byte {}",
        img.width(),
        img.height(),
        img.as_slice()[0]
    );

    // Build a 10x10 grayscale gradient from a data vector.
    let gradient: Vec<u8> = (0..100).map(|i| (i * 255 / 100) as u8).collect();
    let grad = Image::<u8, 1>::new(
        ImageSize {
            width: 10,
            height: 10,
        },
        gradient,
    )?;
    println!(
        "gradient {}x{}, last pixel {}",
        grad.width(),
        grad.height(),
        grad.as_slice()[99]
    );

    // as_slice() borrows the backing buffer — no copy. Pixels are interleaved.
    let rgb = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: 5,
            height: 5,
        },
        42,
    )?;
    let data = rgb.as_slice();
    let px = (2 * rgb.width() + 3) * 3; // row 2, col 3
    println!(
        "pixel (2,3) = ({}, {}, {})",
        data[px],
        data[px + 1],
        data[px + 2]
    );

    // f32 and 4-channel images use the same API through the type parameters.
    let rgba = Image::<f32, 4>::from_size_val(
        ImageSize {
            width: 8,
            height: 8,
        },
        0.25,
    )?;
    println!(
        "rgba f32 {}x{}x4, {} elements",
        rgba.width(),
        rgba.height(),
        rgba.as_slice().len()
    );

    // Construction checks the buffer length against the shape.
    let wrong = vec![0u8; 100]; // a 10x10x3 image needs 300 bytes
    match Image::<u8, 3>::new(
        ImageSize {
            width: 10,
            height: 10,
        },
        wrong,
    ) {
        Ok(_) => unreachable!("wrong-sized data must not construct"),
        Err(e) => println!("rejected wrong-sized data: {e}"),
    }

    Ok(())
}
