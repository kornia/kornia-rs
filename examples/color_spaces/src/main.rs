use kornia_image::allocator::CpuAllocator;
use kornia_image::{color_spaces::Rgb8, ImageSize};
use kornia_imgproc::color::{ConvertColor, Gray8};
use kornia_io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Type-Safe Color Space API Demo\n");

    // Load an image - now returns Rgb8 directly!
    let rgb: Rgb8<CpuAllocator> = F::read_image_any_rgb8("../../tests/data/dog.jpeg")?;
    println!("✓ Loaded RGB8 image: {}x{}", rgb.width(), rgb.height());

    // Convert to grayscale with type safety
    let mut gray = Gray8::from_size_val(rgb.size(), 0, CpuAllocator)?;
    rgb.convert(&mut gray)?;
    println!("✓ Converted to Gray8: {}x{}", gray.width(), gray.height());

    // Convert back to RGB
    let mut rgb_back = Rgb8::from_size_val(gray.size(), 0, CpuAllocator)?;
    gray.convert(&mut rgb_back)?;
    println!("✓ Converted back to RGB8");

    // Create a simple image from scratch
    let small_rgb = Rgb8::from_size_vec(
        ImageSize {
            width: 2,
            height: 2,
        },
        vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            128, 128, 128, // Gray
        ],
        CpuAllocator,
    )?;
    println!("\n✓ Created 2x2 RGB8 image from scratch");

    // Type-safe conversions work seamlessly
    let mut small_gray = Gray8::from_size_val(small_rgb.size(), 0, CpuAllocator)?;
    small_rgb.convert(&mut small_gray)?;

    println!("✓ Converted to grayscale: {:?}", small_gray.as_slice());

    // Works with existing APIs via Deref
    println!("\n📊 Image properties (via Deref):");
    println!("  Width: {}", rgb.width());
    println!("  Height: {}", rgb.height());
    println!("  Channels: {}", rgb.num_channels());
    println!("  Size: {}", rgb.size());

    println!("\n✨ All conversions type-safe at compile time!");

    Ok(())
}
