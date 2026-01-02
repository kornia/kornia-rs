# Type-Safe Color Space Example

This example demonstrates the modern, type-safe color space API in kornia-rs.

## Features

- **Explicit Types**: Use `Rgb8`, `Gray8`, `Bgr8` instead of generic `Image<u8, 3, _>`
- **Compile-Time Safety**: Can't mix RGB with BGR accidentally
- **Clean API**: `rgb.convert(&mut gray)?` instead of function calls
- **Zero-Cost**: `#[repr(transparent)]` wrappers with Deref

## Run

```bash
cargo run -p color_spaces
```

## Key API Highlights

### Creating Images

```rust
// From file - returns typed Rgb8
let rgb = F::read_image_any_rgb8("dog.jpeg")?;

// From scratch - explicit type
let rgb = Rgb8::from_size_vec(
    ImageSize { width: 640, height: 480 },
    vec![128; 640 * 480 * 3],
    CpuAllocator
)?;
```

### Converting Colors

```rust
// Type-safe conversions
rgb.convert(&mut gray)?;  // ✅ Works
gray.convert(&mut bgr)?;  // ❌ Compile error - no impl!
```

### Deref Coercion

```rust
// Works seamlessly with existing APIs
let width = rgb.width();  // Deref to Image
resize::resize_native(&rgb, (320, 240), ...)?;  // Automatic coercion
```

## Available Types

- **8-bit**: `Rgb8`, `Bgr8`, `Gray8`, `Rgba8`, `Bgra8`
- **16-bit**: `Rgb16`, `Bgr16`, `Gray16`, `Rgba16`, `Bgra16`
- **Float**: `Rgbf32`, `Rgbf64`, `Grayf32`, `Grayf64`, `Hsvf32`, `Hsvf64`
