//! Integration test: Run Canny + Hough on real EuRoC and dog images.
//!
//! Run with: cargo test -p kornia-imgproc --test real_canny_hough -- --nocapture

use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_imgproc::features::hough_lines;
use kornia_imgproc::filter::canny;
use kornia_tensor::CpuAllocator;

fn u8_to_f32(src: &Image<u8, 1, CpuAllocator>) -> Image<f32, 1, CpuAllocator> {
    let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
    src.as_slice()
        .iter()
        .zip(dst.as_slice_mut())
        .for_each(|(&s, d)| *d = s as f32);
    dst
}

#[test]
fn test_canny_hough_euroc() {
    let data_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/data");

    // EuRoC MH01 Frame 1 (already grayscale mono8)
    let img_gray = kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame1.png")).unwrap();
    let img_f32 = u8_to_f32(&img_gray);

    // Note: spatial_gradient_float uses a normalized Sobel kernel, so
    // gradient magnitudes are in the range [0, ~125] for 0-255 input.
    // Thresholds must be chosen accordingly.
    let mut edges = Image::<u8, 1, _>::from_size_val(img_f32.size(), 0, CpuAllocator).unwrap();
    canny(&img_f32, &mut edges, 10.0, 40.0).unwrap();

    let edge_count: usize = edges.as_slice().iter().filter(|&&v| v == 255).count();
    let total = img_f32.rows() * img_f32.cols();
    let edge_pct = 100.0 * edge_count as f64 / total as f64;

    println!(
        "=== EuRoC MH01 Frame 1 ({}x{}) ===",
        img_f32.cols(),
        img_f32.rows()
    );
    println!(
        "  Canny edges: {} pixels ({:.2}% of image)",
        edge_count, edge_pct
    );

    // Hough on the Canny output
    let lines = hough_lines(&edges, 80, 1.0, std::f32::consts::PI / 180.0).unwrap();
    println!("  Hough lines (threshold=80): {} detected", lines.len());
    for (i, line) in lines.iter().take(10).enumerate() {
        println!(
            "    Line {}: rho={:.1}, theta={:.1}°",
            i,
            line.rho,
            line.theta.to_degrees()
        );
    }
    assert!(
        edge_count > 100,
        "expected many edges on a real scene, got {edge_count}"
    );
}

#[test]
fn test_canny_hough_dog_rgb() {
    let data_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/data");

    // Dog image (RGB) → convert to grayscale first
    let img_rgb = kornia_io::png::read_image_png_rgb8(data_dir.join("dog-rgb8.png")).unwrap();

    let mut img_gray = Image::<u8, 1, _>::from_size_val(img_rgb.size(), 0, CpuAllocator).unwrap();
    gray_from_rgb_u8(&img_rgb, &mut img_gray).unwrap();
    let img_f32 = u8_to_f32(&img_gray);

    let mut edges = Image::<u8, 1, _>::from_size_val(img_f32.size(), 0, CpuAllocator).unwrap();
    canny(&img_f32, &mut edges, 8.0, 30.0).unwrap();

    let edge_count: usize = edges.as_slice().iter().filter(|&&v| v == 255).count();
    let total = img_f32.rows() * img_f32.cols();
    let edge_pct = 100.0 * edge_count as f64 / total as f64;

    println!("\n=== Dog RGB ({}x{}) ===", img_f32.cols(), img_f32.rows());
    println!(
        "  Canny edges: {} pixels ({:.2}% of image)",
        edge_count, edge_pct
    );

    let lines = hough_lines(&edges, 30, 1.0, std::f32::consts::PI / 180.0).unwrap();
    println!("  Hough lines (threshold=30): {} detected", lines.len());
    for (i, line) in lines.iter().take(10).enumerate() {
        println!(
            "    Line {}: rho={:.1}, theta={:.1}°",
            i,
            line.rho,
            line.theta.to_degrees()
        );
    }
    assert!(
        edge_count > 50,
        "expected edges on the dog image, got {edge_count}"
    );
}
