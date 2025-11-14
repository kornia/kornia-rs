use kornia_image::allocator::CpuAllocator;
use kornia_image::ImageSize;
use kornia_imgproc::color::{
    Bgr8, Bgra8, ConvertColor, ConvertColorWithBackground, Gray8, Grayf32, Hsvf32, Rgb8, Rgba8,
    Rgbf32,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Type-Safe Color Conversion API Demo ===\n");

    // === Example 1: RGB to Grayscale (u8) ===
    println!("1. RGB (u8) -> Grayscale");
    let rgb = Rgb8::from_size_vec(
        ImageSize {
            width: 4,
            height: 1,
        },
        vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            128, 128, 128, // Gray
        ],
        CpuAllocator,
    )?;

    let mut gray = Gray8::from_size_val(rgb.size(), 0, CpuAllocator)?;

    rgb.convert(&mut gray)?;

    println!("   Input: RGB image {}x{}", rgb.width(), rgb.height());
    println!("   Output: Grayscale values: {:?}\n", gray.as_slice());

    // === Example 2: RGB to Grayscale (f32) ===
    println!("2. RGB (f32) -> Grayscale");
    let rgb_f32 = Rgbf32::from_size_vec(
        ImageSize {
            width: 3,
            height: 1,
        },
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        CpuAllocator,
    )?;

    let mut gray_f32 = Grayf32::from_size_val(rgb_f32.size(), 0.0, CpuAllocator)?;

    rgb_f32.convert(&mut gray_f32)?;

    println!("   Grayscale values: {:?}\n", gray_f32.as_slice());

    // === Example 3: Grayscale to RGB ===
    println!("3. Grayscale -> RGB");
    let mut rgb_from_gray = Rgb8::from_size_val(gray.size(), 0, CpuAllocator)?;
    gray.convert(&mut rgb_from_gray)?;

    println!(
        "   Converted back to RGB with {} channels\n",
        rgb_from_gray.num_channels()
    );

    // === Example 4: RGB to BGR ===
    println!("4. RGB -> BGR");
    let rgb_test = Rgb8::from_size_vec(
        ImageSize {
            width: 1,
            height: 1,
        },
        vec![255, 128, 64], // R=255, G=128, B=64
        CpuAllocator,
    )?;

    let mut bgr = Bgr8::from_size_val(rgb_test.size(), 0, CpuAllocator)?;

    rgb_test.convert(&mut bgr)?;

    let bgr_data = bgr.as_slice();
    println!(
        "   RGB [255, 128, 64] -> BGR [{}, {}, {}]\n",
        bgr_data[0], bgr_data[1], bgr_data[2]
    );

    // === Example 5: BGR to RGB ===
    println!("5. BGR -> RGB");
    let mut rgb_back = Rgb8::from_size_val(bgr.size(), 0, CpuAllocator)?;
    bgr.convert(&mut rgb_back)?;

    let rgb_data = rgb_back.as_slice();
    println!(
        "   BGR -> RGB back: [{}, {}, {}]\n",
        rgb_data[0], rgb_data[1], rgb_data[2]
    );

    // === Example 6: RGB to HSV ===
    println!("6. RGB -> HSV");
    let rgb_for_hsv = Rgbf32::from_size_vec(
        ImageSize {
            width: 1,
            height: 1,
        },
        vec![255.0, 0.0, 0.0], // Pure red
        CpuAllocator,
    )?;

    let mut hsv = Hsvf32::from_size_val(rgb_for_hsv.size(), 0.0, CpuAllocator)?;

    rgb_for_hsv.convert(&mut hsv)?;

    println!("   Pure red in HSV: {:?}\n", hsv.as_slice());

    // === Example 7: RGBA to RGB (drop alpha) ===
    println!("7. RGBA -> RGB (drop alpha)");
    let rgba = Rgba8::from_size_vec(
        ImageSize {
            width: 1,
            height: 1,
        },
        vec![255, 128, 64, 200], // RGB + alpha
        CpuAllocator,
    )?;

    let mut rgb_no_alpha = Rgb8::from_size_val(rgba.size(), 0, CpuAllocator)?;

    rgba.convert(&mut rgb_no_alpha)?;

    let rgb_result = rgb_no_alpha.as_slice();
    println!(
        "   RGBA [255, 128, 64, 200] -> RGB [{}, {}, {}]\n",
        rgb_result[0], rgb_result[1], rgb_result[2]
    );

    // === Example 8: RGBA to RGB with background blending ===
    println!("8. RGBA -> RGB (with background blending)");
    let rgba_blend = Rgba8::from_size_vec(
        ImageSize {
            width: 1,
            height: 1,
        },
        vec![255, 0, 0, 128], // Red with 50% alpha
        CpuAllocator,
    )?;

    let mut rgb_blended = Rgb8::from_size_val(rgba_blend.size(), 0, CpuAllocator)?;

    // Blend with gray background [100, 100, 100]
    rgba_blend.convert_with_bg(&mut rgb_blended, Some([100, 100, 100]))?;

    let blended_result = rgb_blended.as_slice();
    println!("   Red (50% alpha) over gray background:");
    println!(
        "   Result: [{}, {}, {}]\n",
        blended_result[0], blended_result[1], blended_result[2]
    );

    // === Example 9: BGRA to RGB ===
    println!("9. BGRA -> RGB");
    let bgra = Bgra8::from_size_vec(
        ImageSize {
            width: 1,
            height: 1,
        },
        vec![64, 128, 255, 255], // BGRA
        CpuAllocator,
    )?;

    let mut rgb_from_bgra = Rgb8::from_size_val(bgra.size(), 0, CpuAllocator)?;

    bgra.convert(&mut rgb_from_bgra)?;

    let rgb_bgra_result = rgb_from_bgra.as_slice();
    println!(
        "   BGRA [64, 128, 255, 255] -> RGB [{}, {}, {}]\n",
        rgb_bgra_result[0], rgb_bgra_result[1], rgb_bgra_result[2]
    );

    // === Example 10: Deref coercion with existing APIs ===
    println!("10. Using with existing APIs (Deref)");
    let test_rgb = Rgb8::from_size_vec(
        ImageSize {
            width: 10,
            height: 10,
        },
        vec![0; 10 * 10 * 3],
        CpuAllocator,
    )?;

    // Deref allows direct access to Image methods
    println!("   Width: {}", test_rgb.width());
    println!("   Height: {}", test_rgb.height());
    println!("   Channels: {}", test_rgb.num_channels());
    println!("   Total pixels: {}", test_rgb.as_slice().len() / 3);

    println!("\n=== All conversions successful! ===");

    Ok(())
}
