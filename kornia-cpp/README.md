# kornia-cpp

Thin C++ wrapper for the Kornia Rust library using [CXX](https://cxx.rs/).

## Overview

Lightweight, zero-overhead C++ interface to kornia-rs. All functions are inline wrappers with no data copies or performance penalties.

## Features

- ✅ Zero-copy via `rust::Slice` (direct pointer into Rust memory)
- ✅ Header-only C++ wrapper (inline functions only)
- ✅ Exception-based error handling (Rust `Result` → C++ exceptions)
- ✅ CMake integration with `find_package` support
- ✅ Namespaced image types (`kornia::image::ImageU8C3`, `kornia::image::ImageF32C1`, etc.)
- ✅ Thread-safe concurrent reads (standard C++ object lifetime rules)
- ✅ Move semantics (no copy constructors for image types)
- ✅ Cross-platform (Linux, macOS, Windows)

## Quick Start

```cpp
#include <kornia.hpp>
#include <iostream>

int main() {
    // Read JPEG image with zero-copy data access
    auto image = kornia::io::read_jpeg_rgb8("image.jpg");
    
    std::cout << "Loaded: " << image.width() << "x" << image.height() 
              << " (" << image.channels() << " channels)\n";
    
    // Access raw pixel data (rust::Slice - no copy!)
    auto data = image.data();
    std::cout << "First pixel R: " << (int)data[0] << "\n";
    
    return 0;
}
```

## Building

### Using CMake directly

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix ~/local  # optional
```

### Using Just

> **Note:** If you are running commands from the `kornia-cpp` directory, use the local commands below. If running from the project root, use the `cpp-` prefixed commands (e.g., `just cpp-test`), which delegate to the local justfile.

```bash
just test            # Build and run tests
just build-examples  # Build examples
just format          # Format code with clang-format
just format-check    # Check format without modifying
just clean           # Clean build artifacts
```

## API

### Image Types

```cpp
// Image types in kornia::image namespace
kornia::image::ImageU8C1   // Grayscale u8 (1 channel)
kornia::image::ImageU8C3   // RGB u8 (3 channels)
kornia::image::ImageU8C4   // RGBA/BGRA u8 (4 channels)
kornia::image::ImageF32C1  // Grayscale f32 (1 channel)
kornia::image::ImageF32C3  // RGB f32 (3 channels)
kornia::image::ImageF32C4  // RGBA f32 (4 channels)
```

### Image Construction

Images can be constructed in multiple ways:

```cpp
// 1. From I/O functions (zero-copy)
auto image = kornia::io::jpeg::read_jpeg_rgb8("image.jpg");

// 2. Create filled with a value
kornia::image::ImageU8C3 image(width, height, 255);  // All pixels = 255

// 3. From std::vector
std::vector<uint8_t> data(width * height * 3, 128);
kornia::image::ImageU8C3 image(width, height, data);

// 4. From raw pointer (explicit constructor - useful for graphics APIs)
const uint8_t* pixels = /* from Unreal, OpenGL, etc. */;
kornia::image::ImageU8C4 image(width, height, pixels);
```

### Image Methods

All methods are `const` and thread-safe for concurrent reads:

```cpp
auto image = kornia::io::jpeg::read_jpeg_rgb8("image.jpg");

size_t w = image.width();          // Image width in pixels
size_t h = image.height();         // Image height in pixels
size_t c = image.channels();       // Number of channels
ImageSize size = image.size();     // {width, height} struct
rust::Slice<const uint8_t> data = image.data();  // Zero-copy data view!
std::vector<uint8_t> vec = image.to_vec();       // Explicit copy to vector
```

**Data Layout:** Row-major, interleaved channels (e.g., `RGBRGBRGB...`)

**Pixel Access:**
```cpp
// Access pixel at (row, col), channel ch:
size_t idx = (row * width + col) * channels + ch;
uint8_t value = data[idx];
```

### I/O Functions

#### Reading Images

```cpp
namespace kornia::io::jpeg {
    // Read JPEG as RGB u8 (uses libjpeg-turbo)
    ImageU8C3 read_jpeg_rgb8(const std::string& file_path);
    
    // Read JPEG as grayscale u8 (auto-converts if needed)
    ImageU8C1 read_jpeg_mono8(const std::string& file_path);
}
```

#### Encoding Images

```cpp
namespace kornia::io::jpeg {
    // Encode RGB image to JPEG bytes (zero-copy buffer)
    void encode_image_jpeg_rgb8(const ImageU8C3& image, uint8_t quality, ImageBuffer& buffer);
    
    // Encode BGRA image to JPEG bytes (zero-copy buffer)
    // Perfect for Unreal Engine, DirectX, and other BGRA-based graphics APIs
    void encode_image_jpeg_bgra8(const ImageU8C4& image, uint8_t quality, ImageBuffer& buffer);
    
    // Decode JPEG bytes back to RGB image
    ImageU8C3 decode_image_jpeg_rgb8(const std::vector<uint8_t>& jpeg_bytes);
}
```

**ImageBuffer** is a reusable, zero-copy buffer for encoded data:
```cpp
kornia::image::ImageBuffer buffer;

// Encode multiple images with the same buffer (efficient!)
for (const auto& img : images) {
    buffer.clear();  // Retains capacity
    kornia::io::jpeg::encode_image_jpeg_rgb8(img, 95, buffer);
    
    // Zero-copy access: no std::vector allocation
    send_to_network(buffer.data(), buffer.size());
}
```

All I/O functions throw `rust::Error` on failure (file not found, invalid format, decode error, etc.).

## Integration

### Option 1: CMake Subdirectory

```cmake
add_subdirectory(kornia-cpp)
target_link_libraries(myapp PRIVATE kornia_cpp)
```

### Option 2: find_package (after installation)

```cmake
find_package(kornia REQUIRED PATHS ~/local)
target_link_libraries(myapp PRIVATE kornia::kornia_cpp)
```

## Thread Safety & Memory Model

### Thread Safety
- ✅ **Safe:** Concurrent reads from multiple threads
- ✅ **Safe:** Concurrent calls to I/O functions
- ⚠️ **Unsafe:** Accessing image after move or destruction
- ⚠️ **Unsafe:** Concurrent writes (not supported - images are immutable)

### Memory Management
- **Ownership:** Rust owns all image data via `rust::Box`
- **Views:** C++ accesses data via `rust::Slice` (non-owning pointer)
- **Lifetime:** Data valid as long as image wrapper exists
- **Cleanup:** Automatic via RAII (`rust::Box` destructor)

### Move Semantics
Image types support move but not copy:
```cpp
auto img1 = kornia::io::read_jpeg_rgb8("a.jpg");
auto img2 = std::move(img1);  // ✅ OK - move
// auto img3 = img2;           // ❌ Error - copy deleted
```

## Testing

```bash
just cpp-test           # Run all tests (from project root)
just test              # Run all tests (from kornia-cpp/)
just test-sanitizers   # Run with ASAN/UBSAN enabled
```

Tests use [Catch2](https://github.com/catchorg/Catch2) v3 framework.

## Use Cases

### Graphics API Integration (Unreal Engine, DirectX, OpenGL)

Perfect for encoding BGRA framebuffers from graphics APIs:

```cpp
// Unreal Engine example: encode FColor buffer (BGRA) to JPEG
const FColor* unreal_pixels = /* from UTextureRenderTarget2D */;
const uint8_t* pixel_data = reinterpret_cast<const uint8_t*>(unreal_pixels);

kornia::image::ImageU8C4 frame(width, height, pixel_data);
kornia::image::ImageBuffer jpeg_buffer;

kornia::io::jpeg::encode_image_jpeg_bgra8(frame, 85, jpeg_buffer);

// Send over network (zero-copy)
zenoh_publisher.put(jpeg_buffer.data(), jpeg_buffer.size());
```

### High-Performance Image Processing Pipeline

Reuse buffers across frames for minimal allocations:

```cpp
kornia::image::ImageBuffer buffer;

while (running) {
    auto frame = capture_camera_frame();
    
    buffer.clear();  // Reuse same buffer
    kornia::io::jpeg::encode_image_jpeg_rgb8(frame, 90, buffer);
    
    // Process encoded data (zero-copy)
    process_jpeg(buffer.data(), buffer.size());
}
```

## Examples

See `examples/` directory for complete examples demonstrating:
- Reading JPEG images
- Encoding images to JPEG (RGB and BGRA)
- Accessing image properties
- Zero-copy data access
- Buffer reuse for performance
- Error handling with try/catch

## Requirements

- CMake 3.15 or later
- C++14 compatible compiler (GCC 5+, Clang 3.4+, MSVC 2015+)
- Rust toolchain (for building the Rust library)
- clang-format (optional, for code formatting)

## Formatting

The project uses `.clang-format` (LLVM style, 100 column limit):

```bash
just format        # Auto-format all C++ files
just format-check  # Check format without modifying
```

CI enforces formatting - make sure to run `just format` before committing.

## Design Principles

1. **Zero-copy:** No data copies on FFI boundary (use `rust::Slice`)
2. **Thin wrapper:** All C++ code is inline, delegates to Rust
3. **Type-safe:** CXX bridge ensures FFI safety at compile time
4. **Header-only:** C++ wrapper is header-only (except Rust lib)
5. **Const-correct:** All accessor methods are `const`
6. **Modern C++:** Use RAII, move semantics, `= delete` for safety

## Contributing

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Coding standards and style guidelines
- How to add new image types or I/O functions
- Testing requirements
- CI checks

## License

Apache-2.0
